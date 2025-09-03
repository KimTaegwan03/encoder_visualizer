import av, os, cv2, numpy as np, torch, math
import json
from fractions import Fraction
from collections import Counter
from openface.multitask_model import MultitaskPredictor
from subprocess import run

VIDEO_IN = "/mnt/data1/videos/-0k881ES7Bo.mp4"
BBOX_JSON = "/mnt/data1/asd_bbox_per_sec_jsons/-0k881ES7Bo.json"
VIDEO_OUT = "-0k881ES7Bo_annotated.mp4"
WEIGHTS_DIR = "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PADDING_FRAC = 0.1
BOX_THICK = 2
TEXT_SCALE = 0.7
TEXT_THICK = 2

_PREDICTOR = None
_DEVICE = None


# AffectNet 8-class order (OpenFace 3.0 commonly uses)
EMO_LABELS = [
    "Neutral", "Happy", "Sad", "Surprise",
    "Fear", "Disgust", "Anger", "Contempt"
]

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.max()
    ex = np.exp(x)
    return ex / (ex.sum() + 1e-12)

def draw_label_below(img, text, x1, y2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, TEXT_SCALE, TEXT_THICK)
    cx = (x1 + tw // 2)
    tx = max(3, x1)
    ty = min(img.shape[0] - 5, y2 + th + 12)
    # background
    cv2.rectangle(img, (tx - 4, ty - th - 6), (tx + tw + 6, ty + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (tx, ty), font, TEXT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)

def clip_box(x1, y1, x2, y2, w, h):
    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def expand_box(x1, y1, x2, y2, w, h, frac):
    bw, bh = x2 - x1 + 1, y2 - y1 + 1
    px, py = int(round(bw * frac)), int(round(bh * frac))
    return clip_box(x1 - px, y1 - py, x2 + px, y2 + py, w, h)

def load_bbox_map_exact(bbox_json):
    with open(bbox_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    sec2box={}
    for item in data:
        s = item["start"]
        e = item["end"]
        sec = int(round(s))
        x1, y1, x2, y2 = item["bboxes"]
        sec2box[sec] = (x1, y1, x2, y2)
    return sec2box

def save_bbox_with_label(bbox_json,output_json,labels):
    with open(bbox_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_dict = []

    for item in data:
        try:
            s = item["start"]
            sec = int(round(s))
            label_data = labels[sec][1]
            if isinstance(label_data, np.ndarray):
                item['label'] = label_data.tolist()
            else:
                item['label'] = label_data
            output_dict.append(item)
        except:
            continue

    with open(output_json,"w", encoding="utf-8") as f:
        json.dump(output_dict,f,indent=4)

def get_predictor(device: str, weights_dir: str):
    global _PREDICTOR, _DEVICE
    if _PREDICTOR is None or _DEVICE != device:
        # 이전에 로드된 predictor가 없거나 device가 변경된 경우 새로 로드
        mtl_path = os.path.join(weights_dir, "MTL_backbone.pth")
        if not os.path.isfile(mtl_path):
            raise FileNotFoundError(
                f"Missing weights: {mtl_path}\n"
                "Install: `pip install openface-test` then `openface download`."
            )
        _DEVICE = device
        _PREDICTOR = MultitaskPredictor(model_path=mtl_path, device=device)
    return _PREDICTOR
        

def annotate_facial_emotion(
        video_in: str,
        bbox_json: str,
        video_out: str,
        device: str,
        weights_dir: str = "weights",
        padding_frac: float = 0.1,
        box_thick: int = 2,
        add_audio:bool = True,
        audio_source:str = None,
        return_video:bool = True
):
    """
    Produce an annotated MP4 with facial emotion labels drawn on active speaker boxes.
    Returns the output path.
    """
    mtl_path = os.path.join(weights_dir, "MTL_backbone.pth")
    if not os.path.isfile(mtl_path):
        raise FileNotFoundError(
            f"Missing weights: {mtl_path}\n"
            "Install: `pip install openface-test` then `openface download`."
        )
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = get_predictor(dev, weights_dir)

    # bbox per second
    sec2box = load_bbox_map_exact(bbox_json)

    in_container = av.open(video_in)
    vstream = next(s for s in in_container.streams if s.type == "video")
    src_rate = vstream.average_rate if vstream.average_rate is not None else Fraction(30, 1)
    time_base = vstream.time_base if vstream.time_base is not None else Fraction(1, max(1, int(src_rate)))
    fps_float = float(src_rate)
    W = vstream.codec_context.width
    H = vstream.codec_context.height

    if return_video:
        os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)
        out_container = av.open(video_out, mode="w")
        try:
            out_stream = out_container.add_stream("libx264", rate=src_rate)
        except av.AVError:
            out_stream = out_container.add_stream("mpeg4", rate=src_rate)
        out_stream.width = W
        out_stream.height = H
        out_stream.pix_fmt = "yuv420p"

    pred_cache: dict[int, tuple[str, float]] = {}
    face_counts = Counter()
    per_sec_labels: dict[int, str] = {}

    for frame in in_container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")
        t = float(frame.pts * time_base) if frame.pts is not None else (frame.index / fps_float)
        sec = int(math.floor(t + 1e-9))

        if sec in sec2box:
            x1, y1, x2, y2= sec2box[sec]
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, W, H)
            ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, W, H, padding_frac)
            crop = img[ey1:ey2, ex1:ex2]
            if crop.size == 0:
                continue
            if sec in pred_cache:
                label, p = pred_cache[sec]
            else:
                emo_logits, _, _ = predictor.predict(crop)
                probs = softmax(emo_logits.detach().cpu().numpy().reshape(-1))
                idx = int(probs.argmax())
                label, p = EMO_LABELS[idx], float(probs[idx])
                pred_cache[sec] = (label, p)

            face_counts[label] += 1
            per_sec_labels[sec] = (label,probs)
            if return_video:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 175, 255), box_thick)
                draw_label_below(img, f"{label}  {p*100:.1f}%", x1, y2)

        if return_video:
            out_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
            for packet in out_stream.encode(out_frame):
                out_container.mux(packet)

    if return_video:
        for packet in out_stream.encode(None):
            out_container.mux(packet)
        out_container.close()
    in_container.close()
    if add_audio and return_video:
        src = audio_source or video_in
        tmp_out = f"{video_out}.tmp.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-i", video_out,          # annotated video (video-only)
            "-i", src,                # <-- can be your 16kHz mono WAV
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k",
            "-ar", "48000", "-ac", "2",   # resample WAV → player-friendly AAC
            "-shortest", "-movflags", "+faststart",
            tmp_out,
        ]
        proc = run(cmd, capture_output=True, text=True)
        if proc.returncode ==0 and os.path.exists(tmp_out):
            os.replace(tmp_out, video_out)
        else:
            if os.path.exists(tmp_out):
                os.remove(tmp_out)
    if return_video:
        return video_out, dict(face_counts), per_sec_labels
    else:
        return per_sec_labels

def main():
    mtl_path = os.path.join(WEIGHTS_DIR, "MTL_backbone.pth")
    if not os.path.isfile(mtl_path):
        raise FileNotFoundError(
            f"Missing weights: {mtl_path}\n"
            "Install: `pip install openface-test` then `openface download`."
        )
    predictor = MultitaskPredictor(model_path=mtl_path, device=DEVICE)

    # Load mapping second -> bbox
    sec2box = load_bbox_map_exact(BBOX_JSON)

    # IO
    in_container = av.open(VIDEO_IN)
    vstream = next(s for s in in_container.streams if s.type == "video")
    src_rate = vstream.average_rate if vstream.average_rate is not None else Fraction(30,1)
    time_base = vstream.time_base if vstream.time_base is not None else Fraction(1, max(1, int(src_rate)))
    fps_float = float(src_rate)

    W = vstream.codec_context.width
    H = vstream.codec_context.height

    out_container = av.open(VIDEO_OUT, mode="w")
    try:
        out_stream = out_container.add_stream("libx264", rate=src_rate)
    except av.AVError:
        out_stream = out_container.add_stream("mpeg4", rate=src_rate)
    out_stream.width = W
    out_stream.height = H
    out_stream.pix_fmt = "yuv420p"

    # cache predictions per second
    pred_cache = {}  # sec -> (label, prob)

    for frame in in_container.decode(video=0):
        img = frame.to_ndarray(format="bgr24")

        # timestamp -> second index
        if frame.pts is not None:
            t = float(frame.pts * time_base)
        else:
            t = frame.index / fps_float
        sec = int(math.floor(t + 1e-9))

        if sec in sec2box:
            x1, y1, x2, y2 = sec2box[sec]
            # clip + optional padding
            x1, y1, x2, y2 = clip_box(x1, y1, x2, y2, W, H)
            ex1, ey1, ex2, ey2 = expand_box(x1, y1, x2, y2, W, H, PADDING_FRAC)
            crop = img[ey1:ey2, ex1:ex2]

            # classify once per second
            if sec in pred_cache:
                label, p = pred_cache[sec]
            else:
                emo_logits, _, _ = predictor.predict(crop)
                emo_logits = emo_logits.detach().cpu().numpy().reshape(-1)
                probs = softmax(emo_logits)
                idx = int(probs.argmax())
                label, p = EMO_LABELS[idx], float(probs[idx])
                pred_cache[sec] = (label, p)

            # draw
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 175, 255), BOX_THICK)
            draw_label_below(img, f"{label}  {p*100:.1f}%", x1, y2)

        # write frame
        out_frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

    # flush & close
    for packet in out_stream.encode(None):
        out_container.mux(packet)
    out_container.close()
    in_container.close()
    print(f"[OK] Wrote: {VIDEO_OUT}")

if __name__ == "__main__":
    main()