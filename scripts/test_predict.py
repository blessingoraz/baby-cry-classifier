from src.predict import predict_audio
from src.utils import format_prediction
import requests

# audio_path = "data/raw/hungry/4be720ce-a5e5-4a48-930f-a212f8a239f6-1434737694572-1.7-f-48-hu.wav"

# probs = predict_audio(audio_path)
# print(format_prediction(probs, top_k=3))


print("Invoking local Lambda function...")
invoke_url = "https://ikfuba8us0.execute-api.eu-north-1.amazonaws.com/default/babycry-lambda"

event = {
    "url": "https://raw.githubusercontent.com/blessingoraz/baby-cry-classifier/main/data/raw/tired/7A22229D-06C2-4AAA-9674-DE5DF1906B3A-1436891944-1.1-m-72-ti.wav"
}

result = requests.post(invoke_url, json=event).json()

print(result)