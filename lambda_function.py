import os
import requests

import json
import tempfile
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
from scipy.signal import stft

import onnxruntime as ort


# ONNX model artifact:
# https://github.com/blessingoraz/baby-cry-classifier/releases/tag/v1.0.0

# NOTE: Lambda inference uses a lightweight DSP pipeline (numpy-based) instead of librosa
# to avoid numba JIT compilation and caching overhead, which can cause cold start issues
# in AWS Lambda. Training notebooks use librosa for preprocessing; inference is optimized.

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
    # Read wav/pcm
    y, sr = sf.read(path, dtype="float32", always_2d=False)

    # If stereo, convert to mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # Resample if needed (simple, good enough for 8k target)
    if sr != TARGET_SR:
        # linear resample (fast + dependency-light)
        x_old = np.linspace(0, 1, num=len(y), endpoint=False)
        x_new = np.linspace(0, 1, num=int(len(y) * TARGET_SR / sr), endpoint=False)
        y = np.interp(x_new, x_old, y).astype(np.float32)

    # Fix length
    if len(y) > FIXED_LEN:
        y = y[:FIXED_LEN]
    elif len(y) < FIXED_LEN:
        y = np.pad(y, (0, FIXED_LEN - len(y)))

    return y

def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2

    # FFT bin frequencies
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)

    # Mel points
    mel_min = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mel_max = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hz = _mel_to_hz(mels)

    # Convert Hz to FFT bin numbers
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)

    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center == left:
            center += 1
        if right == center:
            right += 1
    
        # Rising slope
        fb[i, left:center] = (np.arange(left, center) - left) / (center - left + 1e-9)
        # Falling slope
        fb[i, center:right] = (right - np.arange(center, right)) / (right - center + 1e-9)

    return fb

def _resize_2d(m: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    # Simple bilinear-ish resize using interpolation on each axis
    H, W = m.shape
    x_h = np.linspace(0, H - 1, target_h)
    x_w = np.linspace(0, W - 1, target_w)

    out = np.zeros((target_h, target_w), dtype=np.float32)
    for i, hh in enumerate(x_h):
        out[i] = np.interp(x_w, np.arange(W), m[int(round(hh))])
    return out


def _mel_to_model_input(path: str) -> np.ndarray:
    y = _load_and_fix_length(path)

    # STFT -> magnitude spectrogram
    _, _, Zxx = stft(y, fs=TARGET_SR, nperseg=N_FFT, noverlap=N_FFT - HOP_LENGTH, nfft=N_FFT, padded=False, boundary=None)
    S = np.abs(Zxx).astype(np.float32) ** 2  # power spectrogram, shape (freq_bins, frames)

    # Mel filterbank
    fb = _mel_filterbank(TARGET_SR, N_FFT, N_MELS)  # (n_mels, freq_bins)
    mel = fb @ S  # (n_mels, frames)

    # dB scale (approx)
    mel = np.maximum(mel, 1e-10)
    mel_db = 10.0 * np.log10(mel)

    # Normalize 0-1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # Resize to 224x224
    mel_resized = _resize_2d(mel_db.astype(np.float32), OUT_SIZE[0], OUT_SIZE[1])

    # Shape (1,1,224,224)
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
    try:
        # API Gateway/Lambda Function URL, payload is in event["body"]
        if isinstance(event, dict) and "body" in event and event["body"] is not None:
            body = event["body"]
            if isinstance(body, str):
                body = json.loads(body)
            url = body["url"]
        else:
            # Direct invoke (Lambda console test, local runtime invoke)
            url = event["url"]

        result = predict_from_url(url)

        # If API Gateway expects a proxy response, return a proper HTTP response
        if isinstance(event, dict) and "body" in event:
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result)
            }

        return result

    except Exception as e:
        err = {"error": str(e)}

        if isinstance(event, dict) and "body" in event:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(err)
            }

        return err
