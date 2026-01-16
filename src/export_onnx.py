# Export PyTorch model to ONNX format for cross-platform deployment
import os
import torch
import onnxruntime as ort
import numpy as np

from src.model import load_model

model, device = load_model("models/checkpoints/best_lr_0.01_inner_512_drop_0.8.pt", device=torch.device("cpu"))

# Create dummy input with same shape as real predictions: (batch_size=1, channels=1, height=224, width=224)
dummy_input = torch.randn(1, 1, 224, 224)

# Define output path for ONNX model file
onnx_path = os.path.abspath("baby_cry_classification_resnet18.onnx")
print("Predicting: is something here")

# Export PyTorch model to ONNX format
# opset_version=17: ONNX operator set version (use 17 for modern ops)
# dynamic_axes: allows variable batch_size at inference time
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    opset_version=17,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
)

print("Exported:", onnx_path)

# Verify the exported ONNX model by running inference with ONNX Runtime
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
# Test with random input (same shape as real data)
x = np.random.randn(1, 1, 224, 224).astype(np.float32)
# Run inference and check output shape should be (batch_size=1, num_classes=8)
out = sess.run(None, {"input": x})
print("Output shape:", out[0].shape)  # (1, 8)
