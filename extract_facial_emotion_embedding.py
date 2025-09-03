import os 
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
from openface.multitask_model import MultitaskPredictor

IMAGE_PATH = "unnamed.png" #test image (single pre-cropped face)
WEIGHTS_DIR = "weights"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = None

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n

def find_model_module(obj) -> nn.Module:
    """Best-effort: find the underlying nn.Module inside the predictor wrapper."""
    # Common attribute names first:
    for name in ("model", "net", "mtl", "backbone", "_model", "module"):
        if hasattr(obj, name) and isinstance(getattr(obj, name), nn.Module):
            return getattr(obj, name)

    # Fallback: scan attributes for the first nn.Module
    for name in dir(obj):
        try:
            val = getattr(obj, name)
        except Exception:
            continue
        if isinstance(val, nn.Module):
            return val
    raise RuntimeError("Could not find an nn.Module inside MultitaskPredictor")

def find_emotion_linear(net: nn.Module) -> tuple[str, nn.Module]:
    """
    Heuristic: pick the Linear layer whose out_features==8 (AffectNet 8 classes),
    prefer names containing 'emo'/'emotion'.
    """
    candid = []
    for name, m in net.named_modules():
        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == 8:
            candid.append((name, m))

    if not candid:
        raise RuntimeError("No Linear(out_features=8) found. Inspect the model structure.")

    # Prefer something like 'emotion_head.fc' etc.
    for name, m in candid:
        low = name.lower()
        if "emo" in low or "emotion" in low:
            return name, m

    # Otherwise take the first match
    return candid[0]

def main():
    mtl_path = os.path.join(WEIGHTS_DIR, "MTL_backbone.pth")
    if not os.path.isfile(mtl_path):
        raise FileNotFoundError(
            f"Missing weights: {mtl_path}\n"
            "Fetch with: `pip install openface-test && openface download` "
            "or point WEIGHTS_DIR to your weights."
        )

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {IMAGE_PATH}")

    predictor = MultitaskPredictor(model_path=mtl_path, device=DEVICE)

    # Locate the underlying PyTorch model and the emotion classifier layer
    net = find_model_module(predictor)
    emo_name, emo_linear = find_emotion_linear(net)

    # Forward hook: capture the INPUT to the classifier (penultimate features)
    cache = {"embedding": None}
    def hook(module, inputs, output):
        # inputs is a tuple; take inputs[0] => shape [B, D]
        cache["embedding"] = inputs[0].detach().cpu().numpy().squeeze()

    handle = emo_linear.register_forward_hook(hook)

    try:
        # Run a normal inference through the wrapper (this will trigger hooks)
        emo_logits, _, _ = predictor.predict(img)
    finally:
        handle.remove()  # always clean up hooks

    if cache["embedding"] is None:
        raise RuntimeError(
            "Hook did not fire. Model structure may differ — print(net) and inspect named_modules()."
        )

    emb = cache["embedding"].astype(np.float32)
    emb_norm = l2_normalize(emb)

    print(f"[OK] Emotion embedding shape: {emb.shape} | L2 norm: {np.linalg.norm(emb_norm):.4f}")
    # Show a short preview
    print("First 10 dims (normalized):", np.array2string(emb_norm[:10], precision=4))

    if OUT_JSON:
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump({"image": IMAGE_PATH, "embedding": emb_norm.tolist()}, f, ensure_ascii=False, indent=2)
        print(f"Saved to {OUT_JSON}")

if __name__ == "__main__":
    main()