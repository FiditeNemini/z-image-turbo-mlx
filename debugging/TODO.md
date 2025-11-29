# TODO: Z-Image-Turbo MLX Port

## Objective
Port Z-Image-Turbo to MLX for efficient inference on Apple Silicon.

## Current Status
- [x] Locate PyTorch source code
- [x] Implement `ZImageTransformer2DModel` in MLX (`z_image_mlx.py`)
- [x] Create conversion script (`convert_to_mlx.py`)
- [x] Port `ZImageTransformer2DModel` to MLX (`z_image_mlx.py`)
- [x] Create weight conversion script (`convert_to_mlx.py`)
- [x] Create inference script (`generate_mlx.py`)
- [x] Port VAE to MLX (`vae.py`)
- [x] Port Text Encoder to MLX (`text_encoder.py`)
- [x] Make `generate_mlx.py` self-contained (using MLX VAE/TextEncoder)
- [ ] Optimize performance (compilation, quantization)
- [ ] Add support for other schedulers
- [ ] Refactor code (remove hardcoded paths)
