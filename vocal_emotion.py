from __future__ import annotations
import math, os, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
import torchaudio
from torch.nn.utils.rnn import pad_sequence
from torchaudio.functional import resample
from transformers import Wav2Vec2Processor
from typing import List, Dict, Any
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel
json_dir = "/mnt/data1/asd_orch_jsons"
vids_dir = "/mnt/data1/videos"

MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
TARGET_SR = 16000
MIN_DUR_S = 0.40 # pad short sentences to this for stable embeddings
DEVICE ="cuda" if torch.cuda.is_available() else "cpu"

CHUNK_SEC = 12.0        # <=12s windows keep mem low
HOP_SEC   = 12.0        # no overlap; use <CHUNK_SEC for overlap+average
USE_FP16  = (DEVICE == "cuda")

class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense=nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, features):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)
# --- tiny tweak: length-aware pooling inside EmotionModel.forward ---
class EmotionModel(Wav2Vec2PreTrainedModel):
    """Returns (embedding [B,D], logits [B,3]) for A/D/V."""
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()
    def forward(self, input_values, attention_mask=None):
        out = self.wav2vec2(input_values, attention_mask=attention_mask, return_dict=True)
        hs = out.last_hidden_state                  # [B, T', D]

        dtype = hs.dtype
        B, T, D = hs.shape

        if attention_mask is not None:

            # keep computat
            valid = attention_mask.sum(dim=1).to(dtype)                 # [B]
            t_lens = (valid / attention_mask.shape[1] * T).round().clamp_(1, T).long()
            idx = torch.arange(T, device=hs.device)[None, :]            # [1, T]
            mask = (idx < t_lens[:, None]).to(dtype).unsqueeze(-1)      # [B, T, 1]
            denom = mask.sum(dim=1).clamp_min(1.0).to(dtype)            # [B, 1]
            emb = (hs * mask).sum(dim=1) / denom                         # [B, D]
        else:
            emb = hs.mean(dim=1)                                         # [B, D]

        logits = self.classifier(emb)                                    # [B, 3]
        return emb, logits
    
processor = None
model = None 

def _run_chunk(x_np: np.ndarray):
    """Run one audio chunk -> (emb[D], adv[3], nsamp)."""
    global processor, model
    if processor is None:
        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

    inputs = processor(x_np, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    input_values = inputs["input_values"].to(DEVICE)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(DEVICE)
    if USE_FP16:
        input_values = input_values.half()

    with torch.inference_mode():
        if model is None:
            model = EmotionModel.from_pretrained(MODEL_NAME).to(DEVICE).eval() 
        emb, logits = model(input_values, attention_mask=attention_mask)   # [1,D], [1,3]
        emb = emb[0]
        # L2-normalize per-chunk to stabilize averaging
        emb = emb / emb.norm(p=2).clamp_min(1e-9)
        adv = torch.sigmoid(logits)[0]                                     # [3] in [0,1]

    return emb.detach().float().cpu(), adv.detach().float().cpu(), int(inputs["input_values"].shape[-1])
def read_word_json(json_path: str | Path) ->List[Dict[str, Any]]:
    """Read the JSON file and return a clean, time-sorted word_list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    cleaned = []
    for d in data:
        #BAsic validation /coercion
        if not all(k in d for k in ("word", "start", "end", "speaker")):
            continue
        word = str(d["word"]).strip()
        if word == "":
            continue
        try: 
            start = float(d["start"])
            end = float(d["end"])
        except Exception:
            continue
        cleaned.append({"word": word, "start": start, "end": end, "speaker": d["speaker"]})

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    return cleaned

    
def merge_words_by_speaker(word_list, sentence_level=True):
    """
    같은 화자의 연속된 단어들을 문장 단위로 병합하고, speaker를 숫자 ID로 변환.

    Parameters:
        word_list (list): 각 단어에 대한 정보가 담긴 딕셔너리 리스트

    Returns:
        list: 문장 단위로 병합된 딕셔너리 리스트 (speaker는 int ID)
    """
    if not word_list:
        return []

    sentence_endings = {".", "?", "!"}
    merged = []
    current = {
        "speaker": word_list[0]["speaker"],
        "word": word_list[0]["word"],
        "start": word_list[0]["start"],
        "end": word_list[0]["end"]
    }

    for word_info in word_list[1:]:
        speaker_id = word_info["speaker"]
        word = word_info["word"]
        is_sentence_end = word and word[-1] in sentence_endings

        if speaker_id == current["speaker"]:
            current["word"] += " " + word_info["word"]
            current["end"] = word_info["end"]

            if sentence_level and is_sentence_end:
                merged.append(current)
                current = {
                    "speaker": speaker_id,
                    "word": "",
                    "start": word_info["end"],  # 다음 문장은 이후 시간부터 시작
                    "end": word_info["end"]
                }

        else:
            merged.append(current)
            current = {
                "speaker": speaker_id,
                "word": word_info["word"],
                "start": word_info["start"],
                "end": word_info["end"]
            }
    if current["word"]:
        merged.append(current)

    return merged

def _load_mono_16k(wav_path: str | Path, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Returns mono waveform at target_sr shaped [T] (float32, -1..1).
    """
    wav, sr = torchaudio.load(str(wav_path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True) #conversion to mono
    else:
        wav = wav
    if sr != target_sr:
        wav = resample(wav, sr, target_sr)
    return wav.squeeze(0).contiguous()

def _sentences(json_path: str | Path):
    words = read_word_json(json_path)
    sents = merge_words_by_speaker(words, sentence_level = True)
    return [s for s in sents if s.get("word", "").strip() and (s["end"] - s["start"]) > 0]

def _l2(x: torch.Tensor, eps=1e-9):
    return x/x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)

def adv_to_label(a: float, d: float, v: float) -> str:
    # simple, interpretable thresholds; tweak as you like
    HI_V, LO_V = 0.65, 0.40
    HI_A, LO_A = 0.60, 0.50

    if v >= HI_V and a >= HI_A:
        return "joy/excited"
    if v >= HI_V and a < HI_A:
        return "content/calm"
    if v <= LO_V and a <= LO_A:
        return "sad/low"
    if v <= LO_V and a > HI_A:
        return "angry" if d >= 0.55 else "anxious"
    return "neutral/mixed"

def wav2vec2_dim_on_sentences(json_path: str, wav_path: str, l2norm=True):
    global processor, model
    if USE_FP16:
        if model is None:
            model = EmotionModel.from_pretrained(MODEL_NAME).to(DEVICE).eval() 
        model.half()

    sents = _sentences(json_path)
    if not sents:
        return []
    
    wav = _load_mono_16k(wav_path)
    sr = TARGET_SR
    chunk_len = int(CHUNK_SEC * sr)
    hop_len = int(HOP_SEC * sr)
    # slice = list of tensors
    out = []
    for s in sents:
        a = max(0, int(s["start"] * sr))
        b = min(wav.numel(), int(math.ceil(s["end"] * sr)))
       
        if b<=a:
            continue
        dur = b-a
        if dur <= chunk_len:
            x =wav[a:b]
            need = int(MIN_DUR_S * sr) -x.numel()
            if need > 0:
                x = torch.nn.functional.pad(x, (0, need))
            emb, adv, nsamp = _run_chunk(x.numpy())
            w_total = float(nsamp)
            emb_sum = emb * w_total
            adv_sum = adv * w_total
        else:
            emb_sum = torch.zeros(model.config.hidden_size)
            adv_sum = torch.zeros(3)
            w_total = 0.0
            pos = a
            while pos < b:
                epos = min(pos + chunk_len, b)
                x = wav[pos:epos]
                emb, adv, nsamp = _run_chunk(x.numpy())
                w = float(nsamp)
                emb_sum += emb * w
                adv_sum += adv * w
                w_total += w
                pos += hop_len
        
        #Duration -weighted mean across chunks
        emb_mean = emb_sum / max(1.0, w_total)
        if l2norm:
            emb_mean = emb_mean / emb_mean.norm(p=2).clamp_min(1e-9)
        adv_mean = (adv_sum / max(1.0, w_total)).numpy()

        arousal, dominance, valence = adv_mean.tolist()
        out.append({
            "speaker": s["speaker"],
            "start": float(s["start"]),
            "end": float(s["end"]),
            "text": s.get("word", "").strip(),
            "arousal": float(arousal),
            "dominance": float(dominance),
            "valence": float(valence),
            "embedding": emb_mean.numpy().tolist()
        })

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return out

def save_sentence_with_label(json_path,output_path,labels):
    # read json
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    output = []

    # for label
    for label in labels:
        label_start = label["start"]
        label_end = label["end"]
        # for json_path
        for item in data:
            data_start = item["start"]
            data_end = item["end"]
            # if in start end
            if data_start >= label_start and data_end <= label_end:
                item["arousal"] = label["arousal"]
                item["dominance"] = label["dominance"]
                item["valence"] = label["valence"]
                item["embedding"] = label["embedding"]
                output.append(item)

    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(output,f,indent=4)



if __name__ == "__main__":
    json_path = os.path.join(json_dir, "Ez6phBoRvpc.json")
    wav_path = "audio.wav"
    rows = wav2vec2_dim_on_sentences(json_path, wav_path)
    for r in rows[:5]:
        label = adv_to_label(r["arousal"], r["dominance"], r["valence"])
        print(f"{r['speaker']} {r['start']:.2f} {r['end']:.2f} {r['text']} "
            f"[{label}] arousal={r['arousal']:.2f}, dominance={r['dominance']:.2f}, valence={r['valence']:.2f}")
        print(f"Embedding: {np.array(r['embedding']).shape}")
    print(f"Total {len(rows)} sentences processed.")
