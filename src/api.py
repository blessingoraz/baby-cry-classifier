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
    
    # Validate top_k parameter
    if top_k < 0:
        raise HTTPException(status_code=422, detail="top_k must be non-negative")

    # Read file content
    file_content = await file.read()
    if not file_content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Save to temp file
    suffix = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_content)
        tmp_path = tmp.name

    try:
        probs = predict_audio(tmp_path)
        result = format_prediction(probs, top_k=top_k)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    finally:
        # cleanup temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass
