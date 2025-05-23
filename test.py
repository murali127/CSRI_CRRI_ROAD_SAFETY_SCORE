import onnx
import yolox
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is properly configured
print(yolox.__version__)
print(onnx.__version__)