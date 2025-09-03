from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from functools import lru_cache
from pathlib import Path
import json
from typing import List, Dict, Any
from datetime import datetime
import os
from pathlib import Path
from subprocess import run
import tempfile
from collections import Counter
from utils.utils import *
from active_speaker_match import generate_query_embedding_from_ndarray, get_max_similarity_one
from faceDB import search, insert_records, load_json, ensure_collection, connect
from vocal_emotion import wav2vec2_dim_on_sentences, adv_to_label
from facial_expression import annotate_facial_emotion
from torch_geometric.data import Data
from pyvis.network import Network
import networkx as nx
import torch
from io import BytesIO
import base64
from PIL import Image
import numpy as np
import math
app = FastAPI()
#Directory paths
VIDEO_DIR = Path("/mnt/data1/videos")
ASD_BBOX_DIR = Path("/mnt/data1/asd_bbox_per_sec_jsons")
ORCH_DIR = Path("/mnt/data1/orch_jsons")
ASD_ORCH_DIR = Path("/mnt/data1/asd_orch_jsons")
ASD_DIR = Path("/mnt/data1/asd_jsons")
LABELS_JSON = Path("/mnt/data1/labeled_videos.json")
GRAPH_DIR = Path("/mnt/data1/emb/graph/log")

AUDIO_ROOT = Path("audio")
AUDIO_CLIPS_DIR = AUDIO_ROOT / "clips"
os.makedirs(AUDIO_CLIPS_DIR, exist_ok=True)
app.mount("/audio", StaticFiles(directory=str(AUDIO_ROOT)), name="audio")
app.mount("/static", StaticFiles(directory="static"), name="static")

GENERATED_DIR = (Path(__file__).parent / "generated").resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = (GENERATED_DIR / "cache").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR = "templates"
os.makedirs(TEMPLATES_DIR, exist_ok=True)

#Serve raw files in case you want in-app preview
app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
app.mount("/asd_bbox", StaticFiles(directory=str(ASD_BBOX_DIR)), name ="asd_bbox")
app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")

#Jinja templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

def _coerce_int(v, default=None):
    try:
        return int(v)
    except Exception:
        return default

def _fmt_duration(seconds: int | None) -> str:
    if seconds is None:
        return "-"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"

def _fmt_date(iso:str | None) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return iso
    
def _encode_face_b64(img_bgr: np.ndarray) -> str:
    if not isinstance(img_bgr, np.ndarray):
        return ""
    img_rgb = img_bgr[..., ::-1]
    im = Image.fromarray(img_rgb)
    buf = BytesIO()
    im.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

    
def _entry_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    vid = meta.get("source")
    title = meta.get("title")
    channel = meta.get("channel_name") or meta.get("author")
    length = _coerce_int(meta.get("length"))
    views = _coerce_int(meta.get("view_count"))
    publish_date = meta.get("publish_date")
    persons = meta.get("persons_found") or []
    thumb = meta.get("thumbnail_url")

    video_path = VIDEO_DIR / f"{vid}.mp4"
    asd_path = ASD_BBOX_DIR / f"{vid}.json"
    ready = video_path.is_file() and asd_path.is_file()

    return {
        "id": vid,
        "title": title,
        "channel": channel,
        "length": length,
        "views": views,
        "publish_date": publish_date,
        "persons": persons,
        "thumbnail": thumb,
        "video_rel": f"/videos/{vid}.mp4",
        "asd_rel": f"/asd_bbox/{vid}.json",
        "ready": ready,
        "length_str": _fmt_duration(length),
        "publish_date_str": _fmt_date(publish_date),
    }

@lru_cache(maxsize=1)
def load_catalog() -> List[Dict[str, Any]]:
    """Load and normalize the labeled video catalog."""
    if not LABELS_JSON.is_file():
        return []
    with LABELS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for item in data.get("all_transcripts", []):
        meta = item.get("metadata", {})
        if not meta or not meta.get("source"):
            continue
        items.append(_entry_from_meta(meta))

    return items

def search_entries(q: str | None, person: str | None, channel: str | None, ready_only: bool = True):
    q = (q or "").strip().lower()
    person = (person or "").strip().lower()
    channel = (channel or "").strip().lower()

    def match(e: Dict[str, Any]) -> bool:
        if ready_only and not e["ready"]:
            return False
        if q:
            hay = " ".join([
                e["id"] or "",
                e["title"] or "",
                e["channel"] or "",
                " ".join(e["persons"] or []),
            ]).lower()
            # all terms must appear
            terms = q.split()
            if not all(t in hay for t in terms):
                return False
        if person:
            people = " ".join(e["persons"]).lower()
            if person not in people:
                return False
        if channel:
            ch = (e["channel"] or "").lower()
            if channel not in ch:
                return False
        return True

    return [e for e in load_catalog() if match(e)]

@app.get("/", response_class=HTMLResponse)
async def search_page(request: Request, q: str | None = None, person: str | None = None, channel: str | None = None, page: int = 1):
    per_page = 24
    page = max(1, int(page or 1))
    results = search_entries(q, person, channel, ready_only=True)
    total = len(results)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = results[start:end]

    # collect unique people/channels for quick filters (top-N)
    top_people = {}
    top_channels = {}
    for e in load_catalog():
        for p in e["persons"]:
            top_people[p] = top_people.get(p, 0) + 1
        if e["channel"]:
            top_channels[e["channel"]] = top_channels.get(e["channel"], 0) + 1

    # simple top lists
    people_list = sorted(top_people.items(), key=lambda x: (-x[1], x[0]))[:20]
    channel_list = sorted(top_channels.items(), key=lambda x: (-x[1], x[0]))[:20]

    return templates.TemplateResponse(
        "search.html",
        {
            "request": request,
            "q": q or "",
            "person": person or "",
            "channel": channel or "",
            "items": page_items,
            "total": total,
            "page": page,
            "per_page": per_page,
            "has_prev": page > 1,
            "has_next": end < total,
            "next_page": page + 1,
            "prev_page": page - 1,
            "people_list": people_list,
            "channel_list": channel_list,
        },
    )
def _norm_emo(label: str) -> str:
    return (label or "").strip().lower()

@app.get("/video/{video_id}", response_class=HTMLResponse)
async def video_detail(request: Request, video_id: str):
    entries = [e for e in load_catalog() if e["id"] == video_id]
    if not entries:
        return templates.TemplateResponse(
            "not_found.html",
            {"request": request, "message": f"Video id '{video_id}' not found in catalog."},
            status_code=404,
        )
    e = entries[0]
    if not e["ready"]:
        return templates.TemplateResponse(
            "not_ready.html",
            {
                "request": request,
                "message": "This video cannot be processed yet (missing MP4 or ASD JSON).",
                "entry": e,
            },
            status_code=409,
        )

    return templates.TemplateResponse("detail.html", {"request": request, "entry": e})


@app.get("/process", response_class=HTMLResponse)
async def process_stub(request: Request, video_id: str):
    entries = [e for e in load_catalog() if e["id"] == video_id]
    if not entries:
        return templates.TemplateResponse(
            "not_found.html",
            {"request": request, "message": f"Video id '{video_id}' not found in catalog."},
            status_code=404,
        )
    e = entries[0]
    if not e["ready"]:
        return templates.TemplateResponse(
            "not_ready.html",
            {
                "request": request,
                "message": "This video cannot be processed yet (missing MP4 or ASD JSON).",
                "entry": e,
            },
            status_code=409,
        )

    ###-------------------------------------------------PIPELINE LOGIC BEGINS-------------------------------------------------
    video_path = str(VIDEO_DIR / f"{video_id}.mp4")
    orch_file_path = str(ORCH_DIR / f"{video_id}.json")
    asd_orch_file_path = str(ASD_ORCH_DIR / f"{video_id}.json")
    asd_file_path = str(ASD_DIR / f"{video_id}.json")
    asd_bbox_file_path = str(ASD_BBOX_DIR / f"{video_id}.json")
    graph_path = str(GRAPH_DIR / f"{video_id}.pt")

    graph_data = torch.load(graph_path,weights_only=False)

    G = nx.DiGraph()

    for edge in graph_data.edge_index.t().tolist():
        G.add_edge(edge[0], edge[1])

    net = Network(notebook=False, height='900px', width='100%')
    net.from_nx(G)

    for node in net.nodes:
        node["label"] = str(node["id"])

    try:
        net.write_html("static/graph.html", open_browser=False)   # pyvis >= 0.3
    except TypeError:
        net.save_graph("static/graph.html")
    
    overlap = merge_asr_asd(orch_file_path, asd_file_path)
    merged_script = merge_words_by_speaker(orch_file_path, sentence_level=False)

    imgs = extract_faces_from_video(video_path, overlap)

    #FaceDB
    client = connect()
    coll = ensure_collection(client)
    records = load_json("antelopev2_embeddings.json")
    insert_records(coll, records)

    vote: Dict[str, list] = {}
    for img in imgs:
        qemb = generate_query_embedding_from_ndarray(img["image"])
        _, filtered = search(coll, qemb, 0.5)
        best = get_max_similarity_one(filtered)
        if best:
            img["name"] = best["name"]
            img["retrieved"] = best["image_path"]
            img["score"] = best["cosine_similarity"]
            vote.setdefault(img["speaker"], []).append(img["name"])
        else:
            img["retrieved"] = None
            img["score"] = 0.0
            vote.setdefault(img["speaker"], []).append(img["speaker"])
    
    mapping = {}
    for spk, names in vote.items():
        counts = Counter(names)
        mapping[spk] = max(counts, key=counts.get)

    for seg in merged_script:
        seg["speaker"] = mapping.get(seg["speaker"], seg["speaker"])
    
    for img in imgs:
        img["speaker"] = mapping.get(img["speaker"], img["speaker"])

    speaker_faces = []
    _seen = set()
    for img in imgs:
        spk = img.get("speaker", "unknown")
        if spk in _seen:
            continue
        _seen.add(spk)
        b64 = _encode_face_b64(img["image"])
        if b64:
            speaker_faces.append({"speaker": spk, "img_b64": b64})
    speaker_face_map = {f["speaker"]: f["img_b64"] for f in speaker_faces}

    
    ###VOCAL EMOTION BEGINS -----------------------------------------------------------------------------------------------------------------------
    
    tmp = tempfile.NamedTemporaryFile(prefix=f"{video_id}_", suffix=".wav", delete=False)
    tmp.close()
    wav_path = tmp.name

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        wav_path
    ]

    proc = run(cmd, capture_output=True, text=True)
    if proc.returncode !=0 or not os.path.exists(wav_path):
        return templates.TemplateResponse(
            "process.html.j2",
            {
                "request": request,
                "entry": e,
                "active_speaker_script": merged_script,
                "asd_vocal": [],
                "asd_vocal_error": "Failed to extract runtime WAV. Check ffmpeg installation/permissions." + (f"stderr: {proc.stderr[:300]}" if proc.stderr else ""),
            },
            status_code=200,
        )

    
    emo_rows = wav2vec2_dim_on_sentences(asd_orch_file_path, wav_path, l2norm=True)

    clip_dir = AUDIO_CLIPS_DIR / video_id
    os.makedirs(clip_dir, exist_ok=True)

    asd_vocal = []
    for i, r in enumerate(emo_rows):
        start = float(r.get("start", 0.0))
        end = float(r.get("end", start))
        dur = max(0.0, end - start)
        clip_file = clip_dir / f"{i:05d}.wav"
        if dur>0 and not clip_file.exists():
            os.system(
                f'ffmpeg -y -ss {start:.3f} -t {dur:.3f} -i "{video_path}" '
                f'-vn -ac 1 -ar 16000 -acodec pcm_s16le "{clip_file}" -loglevel error'
            )
        label = adv_to_label(float(r["arousal"]), float(r["dominance"]), float(r["valence"]))
        asd_vocal.append({
            "speaker": r["speaker"],
            "text": (r.get("text") or "").strip(),
            "emotion": label,
            "audio_url": f"/audio/clips/{video_id}/{clip_file.name}",
        })

    counts = Counter((row.get("emotion") or row.get("label") or "unknown") for row in asd_vocal)
    total = sum(counts.values()) or 1
    _color_map = {
        "joy/excited": "#F59E0B",     # amber
        "content/calm": "#10B981",    # emerald
        "sad/low": "#3B82F6",         # blue
        "angry": "#EF4444",           # red
        "anxious": "#8B5CF6",         # violet
        "neutral/mixed": "#9CA3AF",   # gray
        "unknown": "#D1D5DB",
    }
    vocal_emotion_stats = [
        {"emotion": emo, "count": c, "pct": (c/total)*100.0, "color":_color_map.get(emo, "#D1D5DB")}
        for emo, c in counts.items()
    ]
    vocal_emotion_stats.sort(key=lambda x: -x["pct"])
    from collections import defaultdict, Counter as C
    per_spk_counts_v = defaultdict(C)
    for row in asd_vocal:
        spk= row.get("speaker")
        emo = row.get("emotion") or row.get("label") or "unknown"
        if spk:
            per_spk_counts_v[spk][emo] +=1
    
    vocal_emotion_stats_by_speaker = {}
    for spk, cnts in per_spk_counts_v.items():
        tot = sum(cnts.values()) or 1
        stats = [
            {
                "emotion": emo,
                "count": n,
                "pct": (n/tot) * 100.0,
                "color": _color_map.get(_norm_emo(emo), "#D1D5DB")
            }
            for emo, n in cnts.items()
        ]
        stats.sort(key=lambda x: -x["pct"])
        vocal_emotion_stats_by_speaker[spk] = stats
    ###Facial Expression BEGINS------------------------------------------------------------------------------------------------------------------------------------------
    facial_video_url=None
    facial_error=None
    facial_emotion_stats= []
    facial_emotion_stats_by_speaker = {}

    annotated_path = GENERATED_DIR / f"{video_id}_annotated.mp4"
    counts_path = CACHE_DIR / f"{video_id}_facial_counts.json"
    labels_path = CACHE_DIR / f"{video_id}_facial_labels_sec.json"
    try:
        facial_counts = None
        facial_labels_sec = None
        if annotated_path.exists() and counts_path.exists() and labels_path.exists():
            with open(counts_path, "r", encoding="utf-8") as f:
                facial_counts = json.load(f)
            with open(labels_path, "r", encoding="utf-8") as f:
                facial_labels_sec = json.load(f)
        else:

            out_path, facial_counts, facial_labels_sec = annotate_facial_emotion(
                video_in=video_path,
                bbox_json=asd_bbox_file_path,
                video_out=str(annotated_path),
                weights_dir="weights",
                add_audio=True,
                audio_source=wav_path
            )
            with open(counts_path, "w", encoding="utf-8") as f:
                json.dump(facial_counts, f, ensure_ascii=False, indent=2)
            with open(labels_path, "w", encoding="utf-8") as f:
                json.dump(facial_labels_sec, f, ensure_ascii=False, indent=2)
        
        if annotated_path.exists():
            facial_video_url = f"/generated/{annotated_path.name}"

        total_f = sum(facial_counts.values()) or 1
        facial_color_map = {
            "neutral":  "#9CA3AF",
            "happy":    "#F59E0B",
            "sad":      "#3B82F6",
            "surprise": "#10B981",
            "fear":     "#8B5CF6",
            "disgust":  "#14B8A6",
            "anger":    "#EF4444",
            "contempt": "#F43F5E",
        }
        facial_emotion_stats = [
            {
                "emotion": k,
                "count": v,
                "pct": (v/ total_f)* 100.0,
                "color": facial_color_map.get(_norm_emo(k), "#D1D5DB"),
            }
            for k, v in facial_counts.items()
        ]
        facial_emotion_stats.sort(key=lambda x: -x["pct"])


        spk_by_sec = {}

        for seg in merged_script:
            spk = seg.get("speaker")
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start))
            if end < start:
                continue
            s0 = int(math.floor(start))
            s1 = int(math.floor(end - 1e-6))
            for sec in range(s0, s1 + 1):
                spk_by_sec[sec] = spk
        
        per_spk_counts = defaultdict(C)
        for sec_key, label in (facial_labels_sec or {}).items():
            sec = int(sec_key)
            spk = spk_by_sec.get(sec)
            if spk:
                per_spk_counts[spk][label] += 1

        facial_emotion_stats_by_speaker = {}
        for spk, cnts in per_spk_counts.items():
            tot = sum(cnts.values()) or 1
            stats = [
                {
                    "emotion": emo,
                    "count": n,
                    "pct": (n/ tot) * 100.0,
                    "color": facial_color_map.get(_norm_emo(emo), "#D1D5DB"),
                }
                for emo, n in cnts.items()
            ]
            stats.sort(key=lambda x: -x["pct"])
            facial_emotion_stats_by_speaker[spk] = stats

    except Exception as ex:
        facial_error = str(ex)[:400]


    try:
        os.remove(wav_path)
    except Exception:
        pass




    client.close()

    return templates.TemplateResponse("process.html.j2", 
                                      {
                                            "request": request, 
                                            "entry": e, 
                                            "active_speaker_script": merged_script, 
                                            "speaker_faces": speaker_faces,
                                            "asd_vocal": asd_vocal, 
                                            "facial_video_url": facial_video_url, 
                                            "facial_error": facial_error,
                                            "vocal_emotion_stats": vocal_emotion_stats,
                                            "vocal_emotion_stats_by_speaker": vocal_emotion_stats_by_speaker,
                                            "facial_emotion_stats": facial_emotion_stats,
                                            "facial_emotion_stats_by_speaker": facial_emotion_stats_by_speaker,
                                            "speaker_face_map": speaker_face_map
                                            
                                       },
                                    )


@app.get("/healthz")
async def healthz():
    ok = VIDEO_DIR.is_dir() and ASD_BBOX_DIR.is_dir() and LABELS_JSON.is_file()
    return {"ok": ok}