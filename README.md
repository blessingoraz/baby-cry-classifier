# Baby Cry Classification using CNNs

## Overview
This project implements a multi-class baby cry classification system using Convolutional Neural Networks (CNNs) trained on mel-spectrogram representations of audio signals.

The goal is to automatically classify baby cries into meaningful categories (e.g. hunger, pain, discomfort), enabling downstream applications in healthcare, parenting support tools, and intelligent assistants.

## Problem Statement
Interpreting baby cries can be challenging, especially for new parents. Different cry types may correspond to different needs such as hunger, pain, or discomfort.

This project addresses the problem of **automatically classifying baby cries** from raw audio recordings into predefined categories using deep learning. The task is framed as a **multi-class classification problem**.

Challenges include:
- High class imbalance
- Short and noisy audio samples
- Limited labeled data


## Dataset
The dataset consists of labeled baby cry audio recordings grouped into 8 distinct classes:

| Class ID | Class Name | Samples | Description |
|----------|-----------|---------|-------------|
| 0 | **belly_pain** | 16 | Cry indicating abdominal discomfort or digestive issues |
| 1 | **burping** | 18 | Cry with burping or wind-related sounds |
| 2 | **cold_hot** | 7 | Cry indicating temperature discomfort (too cold or too hot) |
| 3 | **discomfort** | 30 | General discomfort cry (wet diaper, clothing irritation, etc.) |
| 4 | **hungry** | 382 | Cry indicating hunger or feeding time |
| 5 | **lonely** | 11 | Cry indicating loneliness or need for attention/comfort |
| 6 | **scared** | 20 | Cry indicating fear or startle response |
| 7 | **tired** | 28 | Cry indicating fatigue or need for sleep |

**Total samples:** 512

Each audio file is:
- Resampled to 8 kHz
- Truncated or padded to a fixed duration (7 seconds)
- Converted into a mel-spectrogram representation

### Class Distribution

The dataset is imbalanced, with some classes significantly underrepresented. Class weighting and macro-F1 were therefore used during training and evaluation.


## Model Architecture

### Pipeline Overview

```
┌─────────────────────────────────────────┐
│   Raw Baby Cry Audio (8 classes)        │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Audio Preprocessing                     │
│ • Resample to 8 kHz                     │
│ • Clip/Pad to 7 seconds (56,000 samples)│
│ • Convert to mono                       │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Mel-Spectrogram Extraction              │
│ • n_mels=128, n_fft=1024, hop=256       │
│ • Convert to dB scale                   │
│ • Normalize to [0, 1]                   │
│ • Output: (128, time_steps)             │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Resize to 224×224 (ResNet input)        │
│ • Bilinear interpolation                │
│ • Replicate to 3 channels (RGB)         │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ ResNet18 Backbone (ImageNet pretrained) │
│ • Frozen weights                        │
│ • Extract features (512-dim)            │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Classification Head                     │
│ • ReLU(x)                               │
│ • Dropout(0.8)                          │
│ • FC: 512 → 512                         │
│ • FC: 512 → 8                           │
│ • Softmax                               │
└──────────────┬──────────────────────────┘
               ↓
┌─────────────────────────────────────────┐
│ Output: Class Probabilities             │
│ {hungry: 0.566, discomfort: 0.391, ...} │
└─────────────────────────────────────────┘
```

### Key Design Choices

- **Backbone:** ResNet18 with ImageNet pretraining (transfer learning)
- **Input:** Mel-spectrograms resized to 224×224 with 3 channels
- **Classification Head:** 
  - Fully connected inner layer (512 units)
  - Dropout (p = 0.8) for regularization
  - Output layer with 8 classes + softmax
- **Single-channel spectrograms:** Converted to 3-channel inputs to match ResNet18 architecture

## Training & Evaluation
The model was trained using supervised learning with the following setup:

- **Loss function:** Cross-Entropy Loss with class weights (to handle imbalance)
- **Optimizer:** AdamW
- **Hyperparameters tested:**
  - Learning rates: 0.0001, 0.001, 0.01
  - Inner layer sizes: 64, 128, 256, 512
  - Dropout rates: 0.0, 0.5, 0.8
- **Evaluation metric:** **Macro-F1 score**
  - Chosen due to class imbalance (ensures fair evaluation across all classes)


## Results
The final model achieved its best performance with the following configuration:

- Learning rate: 0.01
- Inner layer size: 512
- Dropout: 0.8

The selected checkpoint maximized macro-F1 on the validation set.

Model artifacts are versioned and available via GitHub Releases.


## Inference & Deployment

The trained model is exported to two formats:

- **PyTorch (.pt)** — used for local inference and FastAPI-based services
- **ONNX** — optimized for CPU inference and serverless deployment

### Supported deployment options
- FastAPI service (Dockerized)
- AWS Lambda using ONNX Runtime


## Project Structure
src/
  __init__.py           # Package initialization
  model.py              # CryResNet architecture and model loading
  preprocess.py         # Audio preprocessing and spectrogram generation
  predict.py            # Inference entrypoint
  api.py                # FastAPI application
  utils.py              # Helper utilities (formatting predictions)
  export_onnx.py        # ONNX model export
scripts/
  test_predict.py       # CLI script for testing predictions
notebooks/
  01_exploration.ipynb  # EDA and audio analysis
  02_preprocessing.ipynb  # Data preprocessing pipeline
  03_training.ipynb     # Model training and evaluation
data/
  raw/                  # Original labeled audio files (8 cry classes)
  processed/            # Preprocessed mel-spectrograms (not tracked in git)
  splits/               # Train/val/test split indices and label mappings
models/
  checkpoints/          # (ignored in git, PyTorch model weights)
  onnx/                 # (ignored in git, ONNX model exports)
tests/
  test_predict.py       # Unit tests for inference pipeline
  test_preprocess.py    # Unit tests for audio preprocessing
  test_utils.py         # Unit tests for formatting utilities
  test_api.py           # FastAPI endpoint integration tests
  test_predict_e2e.py   # End-to-end tests with real audio
lambda_function.py      # AWS Lambda handler for serverless deployment


## Running Locally (FastAPI)
1. Install dependencies
`pip install -r requirements.txt`

2. Download model artifacts

Download the .pt file from the GitHub Release and place it in:
`models/checkpoints`

3. Start the API
`uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload`

### Testing the API
FastAPI Docs 

Open your browser:
`http://localhost:8000/docs`
- Use the `/predict` endpoint
- Upload a `.wav` file
- Receive predicted label + probabilities

Example response:
```
{
  "label": "belly_pain",
  "probability": 0.62,
  "top_k": [
    {"label": "belly_pain", "probability": 0.62},
    {"label": "hungry", "probability": 0.21},
    {"label": "discomfort", "probability": 0.09}
  ]
}
```

## Unit Tests

The project includes **97 comprehensive unit and integration tests** covering:
- Audio preprocessing pipeline (`test_preprocess.py` - 24 tests)
- Inference pipeline (`test_predict.py` - 12 tests)
- Prediction formatting (`test_utils.py` - 14 tests)
- FastAPI endpoints (`test_api.py` - 30 tests)
- End-to-end integration with real audio (`test_predict_e2e.py` - 17 tests)

### Running Tests

Run all tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest tests/ -v
```

Run a specific test file:
```bash
pytest tests/test_api.py -v
```

Run a specific test class:
```bash
pytest tests/test_api.py::TestPredictEndpoint -v
```

Run a specific test:
```bash
pytest tests/test_api.py::TestPredictEndpoint::test_predict_returns_200_with_valid_file -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Reproducibility
- All experiments were run with fixed random seeds
- Model configuration and hyperparameters are documented
- Trained artifacts are versioned via GitHub Releases


## Demo (Screenshots / Video)
### Local Inference
![Prediction Output](docs/images/prediction_output.png)

### API Demo
![FastAPI Swagger UI](docs/images/swagger_ui.png)

🎥 Demo video:
https://youtu.be/your-video-link


## Model Artifacts

Due to size constraints, trained model files are not stored directly in Git.

The best-performing model is available via **GitHub Releases**:

- **Release:** `v1.0.0`
- **PyTorch checkpoint:** `best_lr_0.01_inner_512_drop_0.8.pt`
- **ONNX model:** `baby_cry_classification_resnet18.onnx`

👉 Download from:  
https://github.com/blessingoraz/baby-cry-classifier/releases/tag/v1.0.0

## Limitations & Future Work

- Dataset size and imbalance limit generalization
- Real-world noise robustness can be improved
- Future improvements:
  - Larger datasets
  - Temporal models (CNN + LSTM)
  - Store model artifacts in S3 with versioning
  - Automated CI pipeline for model promotion


## References
- Librosa: Audio analysis library
- PyTorch & TorchVision
- ONNX Runtime

This project was built to demonstrate end-to-end ML engineering skills,
including data preprocessing, model training, evaluation, deployment,
and artifact management.
