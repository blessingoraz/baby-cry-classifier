import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException

from src.predict import predict_audio
from src.utils import format_prediction

app = FastAPI(title="Baby Cry Classifier", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3):
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        probs = predict_audio(tmp_path)
        result = format_prediction(probs, top_k=top_k)
        return result
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
