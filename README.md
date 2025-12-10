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
