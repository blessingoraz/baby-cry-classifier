from src.predict import predict_audio
from src.utils import format_prediction

audio_path = "data/raw/hungry/4be720ce-a5e5-4a48-930f-a212f8a239f6-1434737694572-1.7-f-48-hu.wav"

probs = predict_audio(audio_path)
print(format_prediction(probs, top_k=3))
