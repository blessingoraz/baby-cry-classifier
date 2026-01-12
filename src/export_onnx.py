import os
import torch
import onnxruntime as ort
import numpy as np

from src.model import model  # loads checkpoint + eval
# or: from src.model import load_model and call it

model.to("cpu").eval()

dummy_input = torch.randn(1, 1, 224, 224)

onnx_path = os.path.abspath("baby_cry_classification_resnet18.onnx")

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

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
x = np.random.randn(1, 1, 224, 224).astype(np.float32)
out = sess.run(None, {"input": x})
print("Output shape:", out[0].shape)  # (1, 8)
