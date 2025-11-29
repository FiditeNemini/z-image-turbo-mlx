# Z-Image-Turbo MLX

High-quality image generation on Apple Silicon using the Z-Image-Turbo model, ported to MLX.

## Overview

Z-Image-Turbo is a 6B parameter diffusion transformer model that generates high-quality 1024×1024 images in just 9 steps. This repository provides an MLX implementation optimized for Apple Silicon Macs, along with the original PyTorch reference implementation.

### Key Features

- **Fast generation**: 9 inference steps for high-quality results
- **Apple Silicon optimized**: Native MLX implementation for M1/M2/M3/M4 Macs
- **Bit-perfect accuracy**: MLX output matches PyTorch within 1 pixel per channel

## Installation

### Requirements

- macOS 12.3+ with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~20GB disk space for model weights

### Setup

```bash
# Create conda environment
conda create -n z-image-mlx python=3.12
conda activate z-image-mlx

# Install dependencies
pip install -r requirements.txt

# Convert model weights (auto-downloads from Hugging Face if not found)
cd src
python convert_to_mlx.py
```

The conversion script will automatically download the Z-Image-Turbo model from Hugging Face (~20GB) if it's not already present in `models/Z-Image-Turbo/`.

## Usage

### Quick Start (MLX)

```bash
cd src
python generate_mlx.py --prompt "A beautiful sunset over the ocean" --output sunset.png
```

### Full Options

```bash
python src/generate_mlx.py \
    --prompt "Your detailed prompt here" \
    --output output.png \
    --seed 42 \
    --steps 9 \
    --height 1024 \
    --width 1024
```

### PyTorch Reference

For comparison or on non-Apple hardware:

```bash
python src/generate_pytorch.py --prompt "Your prompt" --output output.png
```

## Project Structure

```
z-image-turbo-mlx/
├── src/                    # Essential source files
│   ├── generate_mlx.py     # MLX image generation
│   ├── generate_pytorch.py # PyTorch reference
│   ├── z_image_mlx.py      # MLX transformer model
│   ├── text_encoder.py     # MLX Qwen3-4B encoder
│   ├── vae.py              # MLX VAE decoder
│   └── convert_to_mlx.py   # Weight converter
├── models/                 # Model weights
│   ├── Z-Image-Turbo/      # Original PyTorch weights
│   └── mlx_model/          # Converted MLX weights
├── debugging/              # Debug & diagnostic tools
└── requirements.txt
```

## Model Architecture

| Component | Details |
|-----------|---------|
| **Transformer** | S3-DiT (Scalable Sparse DiT), 6B parameters |
| **Text Encoder** | Qwen3-4B (hidden_size=2560, 36 layers) |
| **VAE** | FLUX.1-dev compatible (16 latent channels) |
| **Scheduler** | FlowMatchEulerDiscreteScheduler (shift=3.0) |
| **Resolution** | 1024×1024 (128×128 latents) |

## Performance

| Device | Generation Time (9 steps) |
|--------|---------------------------|
| M2 Ultra | ~XX seconds |
| M3 Max | ~XX seconds |
| M1 Max | ~XX seconds |

*(Performance numbers to be updated)*

## Troubleshooting

### Out of Memory
The model requires significant RAM. If you encounter memory issues:
- Close other applications
- Use a smaller resolution (e.g., 512×512)
- Consider using CPU offloading (PyTorch only)

### Model Path Errors
Ensure you're running from the correct directory:
```bash
cd z-image-turbo-mlx
python src/generate_mlx.py --prompt "..."
```

Or specify the model path explicitly:
```bash
python src/generate_mlx.py --model_path /full/path/to/models/mlx_model --prompt "..."
```

## License

This project is for research and personal use. Please refer to the original Z-Image-Turbo model license for usage terms.

## Acknowledgments

- Original Z-Image-Turbo model from Tongyi-MAI
- MLX framework by Apple
- Diffusers library by Hugging Face
