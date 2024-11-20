import torch
from sentence_transformers import SentenceTransformer

# Check CUDA
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f}MB")

# Test with SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"\nModel device: {next(model.parameters()).device}")

# Test embedding generation
text = "This is a test sentence."
embedding = model.encode(text)
print(f"Generated embedding shape: {embedding.shape}")