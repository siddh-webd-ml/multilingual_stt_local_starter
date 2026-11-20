import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(props.total_memory / 1024**3, 2))
else:
    print("CUDA GPU not detected. Install the CUDA-enabled PyTorch build.")
