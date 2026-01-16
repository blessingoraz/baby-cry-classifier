import os
import json
import tempfile
from urllib.parse import urlparse

import numpy as np
import requests
import librosa
import onnxruntime as ort

# ONNX model artifact:
# https://github.com/blessingoraz/baby-cry-classifier/releases/tag/v1.0.0

# ---------- Config ----------
MODEL_NAME = os.getenv("MODEL_NAME", "baby_cry_classification_resnet18.onnx")
LABEL_MAP_PATH = os.getenv("LABEL_MAP_PATH", "data/splits/label_map.json")

print("Lambda file:", __file__)
print("Directory:", os.getcwd())
print("Files:", os.listdir("."))
print("MODEL_NAME:", MODEL_NAME)


TARGET_SR = 8000
CLIP_SECONDS = 7.0
FIXED_LEN = int(TARGET_SR * CLIP_SECONDS)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

OUT_SIZE = (224, 224)

# ---------- Load labels once ----------
with open(LABEL_MAP_PATH) as f:
    label_info = json.load(f)
id2label = {int(k): v for k, v in label_info["id2label"].items()}
classes = [id2label[i] for i in range(len(id2label))]

# ---------- Load ONNX once (cold start only) ----------
session = ort.InferenceSession(MODEL_NAME, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def _safe_url(url: str) -> None:
    """Basic URL validation to avoid weird schemes."""
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        raise ValueError("Only http/https URLs are allowed")


def _download_to_temp(url: str) -> str:
    _safe_url(url)

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    suffix = os.path.splitext(urlparse(url).path)[1] or ".wav"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def _load_and_fix_length(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=TARGET_SR, mono=True)

    if len(y) > FIXED_LEN:
        y = y[:FIXED_LEN]
    elif len(y) < FIXED_LEN:
        y = np.pad(y, (0, FIXED_LEN - len(y)))
    return y


def _mel_to_model_input(path: str) -> np.ndarray:
    """
    Returns input array for ONNX model:
      shape: (1, 1, 224, 224), dtype float32
    """
    y = _load_and_fix_length(path)

    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to 0-1 (match training approach)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Resize to (224,224) using simple numpy approach
    # We'll use librosa.util.fix_length + repeat for time axis then crop/pad
    # But easiest: use np.interp for both axes
    H, W = mel_db.shape  # (128, time)
    target_h, target_w = OUT_SIZE

    # Resize H dimension
    x_h = np.linspace(0, H - 1, target_h)
    x_w = np.linspace(0, W - 1, target_w)

    mel_resized = np.zeros((target_h, target_w), dtype=np.float32)
    for i, hh in enumerate(x_h):
        row = np.interp(x_w, np.arange(W), mel_db[int(round(hh))])
        mel_resized[i] = row

    # Shape: (1,1,224,224)
    X = mel_resized[np.newaxis, np.newaxis, :, :].astype(np.float32)
    return X


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / (np.sum(exp) + 1e-12)


def predict_from_url(url: str) -> dict:
    tmp_path = _download_to_temp(url)
    try:
        X = _mel_to_model_input(tmp_path)
        logits = session.run([output_name], {input_name: X})[0][0]  # (8,)
        probs = _softmax(logits)

        return dict(zip(classes, probs.tolist()))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def lambda_handler(event, context):
    """
    Expects:
      event = {"url": "https://.../audio.wav"}
    Returns:
      {"classA": 0.12, "classB": 0.03, ...}
    """
    try:
        url = event["url"]
        return predict_from_url(url)
    except Exception as e:
        return {"error": str(e)}
