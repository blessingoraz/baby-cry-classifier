
import json
import torch
import numpy as np

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

# Example:
audio_path = "data/raw/hungry/4be720ce-a5e5-4a48-930f-a212f8a239f6-1434737694572-1.7-f-48-hu.wav"

result = predict_audio(audio_path)
print(result)
