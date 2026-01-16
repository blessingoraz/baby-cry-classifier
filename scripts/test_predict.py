from src.predict import predict_audio
from src.utils import format_prediction
import requests

# audio_path = "data/raw/hungry/4be720ce-a5e5-4a48-930f-a212f8a239f6-1434737694572-1.7-f-48-hu.wav"

# probs = predict_audio(audio_path)
# print(format_prediction(probs, top_k=3))


print("Invoking local Lambda function...")
invoke_url = "http://localhost:9000/2015-03-31/functions/function/invocations"

event = {
    "url": "https://raw.githubusercontent.com/blessingoraz/baby-cry-classifier/main/data/raw/belly_pain/549a46d8-9c84-430e-ade8-97eae2bef787-1430130772174-1.7-m-48-bp.wav"
}

result = requests.post(invoke_url, json=event).json()

print(result)