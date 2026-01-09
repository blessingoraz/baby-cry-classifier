# Baby Cry Classifier 👶🔊

A neural-network powered system that classifies baby cries into emotional/physical categories
(e.g., hungry, tired, discomfort). Built as my ML Zoomcamp capstone, deployed using AWS + Kubernetes.

## Features
- Audio preprocessing with mel-spectrograms
- CNN/LSTM model for classification
- Handles class imbalance with augmentation + weighted loss
- FastAPI inference service
- Dockerized + deployed to AWS
- Kubernetes manifests for scalable deployment

## Model Download
Download the trained model from [Releases](https://github.com/blessingoraz/baby-cry-classifier/releases):
- `best_lr_0.01_inner_512_drop_0.8.pt` (model weights)