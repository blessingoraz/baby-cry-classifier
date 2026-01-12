
import json
import torch
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.model import model, device
from src.preprocess import audio_to_mel_tensor

with open("data/splits/label_map.json") as f:
   label_info = json.load(f)
id2label = {int(k): v for k, v in label_info["id2label"].items()}
classes = [id2label[i] for i in range(len(id2label))]

@torch.no_grad()
def predict_audio(path):
    x = audio_to_mel_tensor(path).to(device)     # (1,1,224,224)
    logits = model(x)[0]                         # (num_classes,)
    probs = torch.softmax(logits, dim=0).cpu().numpy()

    return dict(zip(classes, probs.tolist()))

