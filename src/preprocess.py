

import numpy as np
import librosa
import torch
import torch.nn.functional as F

TARGET_SR = 8000
CLIP_SECONDS = 7.0
FIXED_LEN = int(TARGET_SR * CLIP_SECONDS)

N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

def load_and_fix_length(path, target_sr=TARGET_SR, fixed_len=FIXED_LEN):
    y, _ = librosa.load(path, sr=target_sr, mono=True)

    if len(y) > fixed_len:
        y = y[:fixed_len]
    elif len(y) < fixed_len:
        y = np.pad(y, (0, fixed_len - len(y)))
    return y

def audio_to_mel_tensor(path):
    y = load_and_fix_length(path)

    mel = librosa.feature.melspectrogram(
        y=y, sr=TARGET_SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # normalize 0-1 (same idea as training)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)

    # (1, n_mels, time)
    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0)

    # resize to (1, 224, 224)
    x = x.unsqueeze(0)  # (1,1,H,W)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.squeeze(0)    # (1,224,224)

    # add batch dimension -> (1,1,224,224)
    x = x.unsqueeze(0)
    return x