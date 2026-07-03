
## Troubleshooting

### FFmpeg Not Found / Torchcodec Error
```bash
# The error "Could not load libtorchcodec" means FFmpeg is missing
# Install FFmpeg at system level:

# Ubuntu/Debian:
sudo apt update && sudo apt install -y ffmpeg

# macOS:
brew install ffmpeg

# Verify installation:
ffmpeg -version

# Then reinstall torchcodec:
pip install --force-reinstall torchcodec==0.8.1
```
> \[!Note]
> Conda environments include FFmpeg automatically. UV/pip environments require manual FFmpeg installation.

### FFmpeg Version Mismatch

If you see errors about missing symbols or incompatible FFmpeg libraries:

```bash
# Check FFmpeg version (requires 6.x or 7.x for torchcodec)
ffmpeg -version

# Ubuntu 22.04 has FFmpeg 4.x by default - use Conda or upgrade:
conda install -c conda-forge ffmpeg=7.*

# Or reinstall torchcodec to match your FFmpeg version:
pip install --force-reinstall torchcodec
```

### Out of Memory
```python
# Reduce batch sizes
batch_size=15.0, # sec
```

### GPU Not Detected
```python
# Check PyTorch CUDA
import torch
print(torch.cuda.is_available())

# Force CPU if needed
whisper_device="cpu"
```

### Package Version Conflicts
```bash
# Use UV for better dependency resolution
uv pip install -r requirements-lock-uv-gpu.txt --upgrade  # or requirements-lock-uv-cpu.txt
```
