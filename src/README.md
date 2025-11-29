# Z-Image-Turbo Source Code

This folder contains the essential source files for running Z-Image-Turbo on both PyTorch and MLX (Apple Silicon).

## Files

### Image Generation
- **`generate_mlx.py`** - Generate images using MLX (Apple Silicon optimized)
- **`generate_pytorch.py`** - Generate images using PyTorch (reference implementation)

### MLX Model Components
- **`z_image_mlx.py`** - MLX implementation of the Z-Image transformer (S3-DiT architecture)
- **`text_encoder.py`** - MLX implementation of the Qwen3-4B text encoder
- **`vae.py`** - MLX implementation of the FLUX.1-dev VAE

### Utilities
- **`convert_to_mlx.py`** - Convert PyTorch weights to MLX format

## Usage

### MLX Generation (Recommended for Apple Silicon)

```bash
python src/generate_mlx.py --prompt "Your prompt here" --output output.png
```

Options:
- `--prompt` - Text description of the image to generate
- `--output` - Output file path (default: `generated_mlx.png`)
- `--seed` - Random seed for reproducibility (default: 42)
- `--steps` - Number of inference steps (default: 9)
- `--height` - Image height (default: 1024)
- `--width` - Image width (default: 1024)
- `--model_path` - Path to MLX model weights (default: `./models/mlx_model`)

### PyTorch Generation

```bash
python src/generate_pytorch.py --prompt "Your prompt here" --output output.png
```

Uses the same options as MLX generation.

## Model Architecture

- **Transformer**: S3-DiT (Scalable Sparse Transformer for Diffusion) with 6B parameters
- **Text Encoder**: Qwen3-4B (hidden_size=2560, 36 layers)
- **VAE**: FLUX.1-dev compatible (16 latent channels, scaling_factor=0.3611)
- **Scheduler**: FlowMatchEulerDiscreteScheduler with shift=3.0

## Converting Weights

If you need to re-convert PyTorch weights to MLX format:

```bash
python src/convert_to_mlx.py
```

This reads from `models/Z-Image-Turbo/` and writes to `models/mlx_model/`.
