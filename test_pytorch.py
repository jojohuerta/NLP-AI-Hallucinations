import torch

print("Hello world (PyTorch)")
print(f"Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    print("Executing in CPU.")