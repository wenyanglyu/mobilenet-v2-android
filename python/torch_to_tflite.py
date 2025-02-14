import torch
import torchvision.models as models
import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# ✅ 1. Load MobileNetV2 with correct preprocessing
pytorch_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
pytorch_model.eval()

# ✅ 2. Create proper normalized input (PyTorch-style preprocessing)
dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = (dummy_input - torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) / torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)

# ✅ 3. Export to ONNX with dynamic axes (better for deployment)
onnx_path = "mobilenet_v2.onnx"
torch.onnx.export(
    pytorch_model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=13,  # Updated from 11 to 13
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

# ✅ 4. Convert ONNX to TensorFlow with explicit shape
onnx_model = onnx.load(onnx_path)
tf_rep = prepare(onnx_model, strict=True)
tf_rep.export_graph("mobilenet_tf")

# ✅ 5. Convert to TFLite with optimizations
converter = tf.lite.TFLiteConverter.from_saved_model("mobilenet_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Add optimizations
tflite_model = converter.convert()

# ✅ 6. Save with metadata
tflite_path = "mobilenet_v2.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Conversion successful: {tflite_path}")