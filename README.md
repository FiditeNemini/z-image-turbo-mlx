# Z-Image-Turbo MLX

High-quality image generation on Apple Silicon using the Z-Image-Turbo model, ported to MLX.

## Overview

Z-Image-Turbo is a 6B parameter diffusion transformer model that generates high-quality 1024×1024 images in just 9 steps. This repository provides an MLX implementation optimized for Apple Silicon Macs, along with the original PyTorch reference implementation.

### Key Features

- **Fast generation**: 9 inference steps for high-quality results
- **LeMiCa speed acceleration**: Training-free caching for up to 30% faster generation
- **Apple Silicon optimized**: Native MLX implementation for M1/M2/M3/M4 Macs
- **Bit-perfect accuracy**: MLX output matches PyTorch within 1 pixel per channel
- **Model management**: Support for multiple models and fine-tuned variants
- **LoRA support**: Apply style and concept customizations with adjustable weights
- **LoRA fusion**: Permanently fuse LoRAs and export to MLX, PyTorch, or ComfyUI formats
- **Model merging**: Combine multiple models using Weighted Sum or Add Difference methods
- **Latent upscaling**: Upscale in latent space for enhanced detail before decoding
- **ESRGAN upscaling**: 4× pixel-space resolution enhancement with RRDB-based models
- **Random prompt generation**: Auto-generates creative prompts when input is empty
- **Comprehensive logging**: Detailed logs in `./logs/` for troubleshooting
- **Scrollable LoRA list**: Browse all installed LoRAs with improved UI
- **LoRA Training**: Train custom LoRAs using PyTorch with MPS (Apple Silicon) or CUDA (NVIDIA)
- **Aspect Ratio Bucketing**: Train on varied image aspect ratios without cropping
- **Training Export**: Export trained LoRAs as standalone files or merged models (MLX/PyTorch/ComfyUI)
- **Gradio UI**: User-friendly web interface for image generation

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [**WALKTHROUGH-DOCUMENTATION.md**](WALKTHROUGH-DOCUMENTATION.md) | **Start here!** Beginner-friendly guide covering installation, GUI usage, CLI commands, and tips for getting the best results |
| [**TECHNICAL_DOCUMENTATION.md**](TECHNICAL_DOCUMENTATION.md) | In-depth technical reference covering architecture, weight formats, quantization, model loading, and implementation details |

### Quick Start

New to the project? Follow these steps:
1. Read the **Installation** section below
2. Follow the [User Walkthrough Guide](WALKTHROUGH-DOCUMENTATION.md) for step-by-step usage instructions
3. Consult the [Technical Documentation](TECHNICAL_DOCUMENTATION.md) if you need to understand the internals or debug issues

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

The conversion script will automatically download the Z-Image-Turbo model from Hugging Face (~20GB) if it's not already present in `models/pytorch/Z-Image-Turbo/`.

## Usage

### Web UI (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

This opens a browser-based UI with:
- **Generation Tab**: Create images with full control over parameters
- **Model Settings Tab**: Manage, import, and switch between models

### Command Line

#### Quick Start (MLX)

```bash
cd src
python generate_mlx.py --prompt "A beautiful sunset over the ocean" --output sunset.png
```

#### Full Options

```bash
python src/generate_mlx.py \
    --prompt "Your detailed prompt here" \
    --output output.png \
    --seed 42 \
    --steps 9 \
    --height 1024 \
    --width 1024 \
    --cache medium
```

#### Speed Acceleration (LeMiCa)

Use the `--cache` option for faster generation with minimal quality impact:

| Mode | Steps Computed | Speed Gain | Quality |
|------|----------------|------------|----------|
| `slow` | 7/9 | ~14% faster | Highest |
| `medium` | 6/9 | ~22% faster | Excellent |
| `fast` | 5/9 | ~30% faster | Very Good |

```bash
# Fast mode for quick iterations
python src/generate_mlx.py --prompt "..." --cache fast

# Medium mode for balanced speed/quality
python src/generate_mlx.py --prompt "..." --cache medium
```

#### PyTorch Reference

For comparison or on non-Apple hardware:

```bash
python src/generate_pytorch.py --prompt "Your prompt" --output output.png
```

## Model Management

### Directory Structure

Models are organized by platform:

```
models/
├── mlx/                    # MLX-converted models (used for generation)
│   ├── mlx_model/          # Default converted model
│   └── RedCraft-AIO/       # Example: fine-tuned variant
└── pytorch/                # PyTorch/Diffusers format models
    └── Z-Image-Turbo/      # Original model (for conversion)
```

### Migrating from Previous Versions

If you're upgrading from a version that used the old directory structure (`models/mlx_model/`, `models/Z-Image-Turbo/`), run the migration script:

```bash
# Preview what will be migrated (dry run)
python migrate_models.py

# Apply the migration
python migrate_models.py --apply
```

The script will:
- Create the new `models/mlx/` and `models/pytorch/` directories
- Move existing models to the correct locations
- Validate that models are still accessible after migration

### Supported Model Formats

#### MLX Models (Ready to Use)
Place pre-converted MLX models in `models/mlx/<model_name>/`. Each model folder should contain:
- `weights.safetensors` - Transformer weights
- `text_encoder.safetensors` - Text encoder weights
- `vae.safetensors` - VAE decoder weights
- `config.json`, `vae_config.json`, `text_encoder_config.json`

#### Hugging Face Models (Diffusers Format)
Download from Hugging Face and convert:
1. Place in `models/pytorch/<model_name>/`
2. Run `python src/convert_to_mlx.py`

#### ComfyUI Checkpoints (Single-File)
The app supports ComfyUI-style all-in-one `.safetensors` files for Z-Image-Turbo architecture:
- These use `diffusion_` prefix for transformer weights
- Text encoder weights use `text_encoders.qwen3_4b.` prefix
- Import via Model Settings → Import Single-File Checkpoint

> **Note**: Only Z-Image-Turbo architecture checkpoints are compatible. Other architectures (SDXL, SD1.5, Flux, Hunyuan) will be detected and show an error message.

### Adding New Models

#### From Hugging Face
1. Go to **Model Settings** → **Import from Hugging Face**
2. Enter the repository ID (e.g., `username/model-name`)
3. Select format: "Diffusers (requires conversion)" or "Pre-converted MLX"
4. Click **Download Model**

#### From Local Checkpoint
1. Go to **Model Settings** → **Import Single-File Checkpoint**
2. Click **Browse** and select your `.safetensors` file
3. The app will validate compatibility before import
4. Enter a name for the model and click **Import**

### Switching Models

Use the dropdown in **Model Settings** → **Select Model** to switch between available models. Click the refresh button (🔄) to rescan for newly added models.

## LoRA Support

LoRAs (Low-Rank Adaptations) allow you to customize the generation style without modifying the base model.

### Adding LoRAs

1. Place `.safetensors` LoRA files in `models/loras/`
2. Optionally organize in subfolders: `styles/`, `concepts/`, `characters/`

```
models/loras/
├── anime_style.safetensors
├── styles/
│   └── watercolor.safetensors
└── concepts/
    └── cyberpunk.safetensors
```

### Using LoRAs

1. In the **Generate** tab, expand **🎨 LoRA Settings**
2. Enable desired LoRAs with the checkbox
3. Adjust weight (0.0-2.0) using the spinner
4. Include trigger words in your prompt if the LoRA has them

### LoRA Features

| Feature | Description |
|---------|-------------|
| **Multiple LoRAs** | Stack multiple LoRAs with independent weights |
| **Per-LoRA Weights** | Fine-tune each LoRA's influence (0.05 increments) |
| **Trigger Words** | Auto-displayed from LoRA metadata |
| **Subfolder Support** | Organize LoRAs in categories |
| **Live Tags** | See active LoRAs as `<lora:name:weight>` |

> **Note**: Only Z-Image-Turbo compatible LoRAs work. LoRAs trained for SDXL, SD1.5, Flux, etc. are NOT compatible.

## Speed Acceleration (LeMiCa)

LeMiCa (Lexicographic Minimax Path Caching) is a training-free acceleration technique that caches transformer residuals between denoising steps instead of recomputing from scratch.

### How It Works

1. On "compute" steps, the full transformer forward pass runs and stores the residual
2. On "skip" steps, the cached residual is reused: `output = input + cached_residual`
3. The schedule determines which steps compute vs skip

### Speed Modes

| Mode | Computed Steps | Speedup | Quality |
|------|----------------|---------|----------|
| **None** | 9/9 | Baseline | Reference |
| **slow** | 7/9 | ~14% faster | Highest |
| **medium** | 6/9 | ~22% faster | Excellent |
| **fast** | 5/9 | ~30% faster | Very Good |

### Usage

**GUI**: Use the "⚡ LeMiCa Speed" dropdown below the Steps slider

**CLI**:
```bash
python src/generate_mlx.py --prompt "..." --cache medium
```

### Technical Details

Based on [LeMiCa: Lexicographic Minimax Path Caching](https://github.com/UnicomAI/LeMiCa) (NeurIPS 2025 Spotlight). The Z-Image implementation uses optimized step schedules derived from the original research:

- `slow`: Steps 0,1,2,3,5,7,8 compute (skip 4,6)
- `medium`: Steps 0,1,2,4,6,8 compute (skip 3,5,7)
- `fast`: Steps 0,1,2,5,8 compute (skip 3,4,6,7)

### Saving Fused Models

You can permanently fuse loaded LoRAs into the base model and export to multiple formats:

1. Configure your LoRAs with desired weights in the LoRA Settings panel
2. Enter a name for your fused model
3. Select output formats:
   - **MLX**: Ready for use in this app (`models/mlx/`)
   - **PyTorch**: Diffusers format for sharing (`models/pytorch/`)
   - **ComfyUI**: Single-file checkpoint (`models/comfyui/`)
4. Click **Save Fused Model**

## Model Merging

Combine multiple Z-Image-Turbo models to create novel blends using the **Merge** tab.

### Merge Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **Weighted Sum** | `(1-α)A + αB` | Blend two models proportionally |
| **Add Difference** | `A + α(B-C)` | Extract fine-tune changes from B relative to C, apply to A |

### Features

- **Sequential merging** for 3+ models: `((A⊕B)⊕C)⊕D...`
- **Memory-safe mode**: Auto-chunked processing for systems with <32GB RAM
- **Base model integration**: Uses Generate tab's selected model (with fused LoRAs) as Model A
- **FP16+ only**: Excludes FP8 quantized models from merging

### Usage

1. Select your base model in the **Generate** tab
2. Go to the **Merge** tab
3. Choose a merge method (Weighted Sum or Add Difference)
4. Enable models to merge and set their weights (0.0-1.0)
5. For Add Difference: Also select Model C (the original model B was fine-tuned from)
6. Enter an output model name
7. Select output formats:
   - **MLX**: Ready for use in this app (`models/mlx/`)
   - **PyTorch**: Diffusers format for sharing (`models/pytorch/`)
   - **ComfyUI**: Single-file checkpoint (`models/comfyui/`)
8. Click **Merge Models**

The merged model will be saved in the selected format directories and can be used immediately.

## Project Structure

```
z-image-turbo-mlx/
├── app.py                  # Gradio web UI (with LeMiCa & upscaling)
├── migrate_models.py       # Migration script for directory structure
├── src/                    # Core source files
│   ├── generate_mlx.py     # MLX image generation (--cache for LeMiCa)
│   ├── generate_pytorch.py # PyTorch reference
│   ├── z_image_mlx.py      # MLX transformer model (LeMiCa caching)
│   ├── text_encoder.py     # MLX Qwen3-4B encoder
│   ├── vae.py              # MLX VAE decoder
│   ├── lora.py             # LoRA loading and application
│   ├── merge.py            # Model merging algorithms
│   ├── convert_to_mlx.py   # Weight converter
│   ├── training_ui.py      # Training tab UI components
│   └── training/           # Training module
│       ├── trainer.py      # LoRA training with MPS/CUDA support
│       ├── dataset.py      # Dataset management & bucketing
│       ├── config.py       # Training configurations
│       └── ...             # Additional training utilities
├── models/                 # Model weights
│   ├── mlx/                # MLX-converted models
│   ├── pytorch/            # PyTorch/Diffusers models
│   ├── loras/              # LoRA files (.safetensors)
│   ├── training_adapters/  # Training adapters (de-distillation)
│   └── upscalers/          # ESRGAN upscaler models
├── datasets/               # Training datasets
├── outputs/                # Training outputs
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

### Model Not Showing in Dropdown
- Ensure the model is in `models/mlx/<model_name>/`
- Check that all required files are present (weights.safetensors, etc.)
- Click the refresh button (🔄) to rescan

### Incompatible Checkpoint Error
When importing a single-file checkpoint, you may see an error like:
- "SDXL checkpoint detected" 
- "Flux model detected"

This means the checkpoint is not a Z-Image-Turbo model. Only checkpoints fine-tuned from Z-Image-Turbo are compatible.

### Model Path Errors
Ensure you're running from the correct directory:
```bash
cd z-image-turbo-mlx
python src/generate_mlx.py --prompt "..."
```

Or specify the model path explicitly:
```bash
python src/generate_mlx.py --model_path /full/path/to/models/mlx/mlx_model --prompt "..."
```

## License

This project is for research and personal use. Please refer to the original Z-Image-Turbo model license for usage terms.

## Acknowledgments

- Original Z-Image-Turbo model from Tongyi-MAI
- MLX framework by Apple
- Diffusers library by Hugging Face
- LeMiCa acceleration from UnicomAI

```bibtex
@article{team2025zimage,
  title={Z-Image: An Efficient Image Generation Foundation Model with Single-Stream Diffusion Transformer},
  author={Z-Image Team},
  journal={arXiv preprint arXiv:2511.22699},
  year={2025}
}

@inproceedings{gao2025lemica,
  title={LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation},
  author={Huanlin Gao and Ping Chen and Fuyuan Shi and Chao Tan and Zhaoxiang Liu and Fang Zhao and Kai Wang and Shiguo Lian},
  journal={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  url={https://arxiv.org/abs/2511.00090}
}

@article{liu2025decoupled,
  title={Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield},
  author={Dongyang Liu and Peng Gao and David Liu and Ruoyi Du and Zhen Li and Qilong Wu and Xin Jin and Sihan Cao and Shifeng Zhang and Hongsheng Li and Steven Hoi},
  journal={arXiv preprint arXiv:2511.22677},
  year={2025}
}

@article{jiang2025distribution,
  title={Distribution Matching Distillation Meets Reinforcement Learning},
  author={Jiang, Dengyang and Liu, Dongyang and Wang, Zanyi and Wu, Qilong and Jin, Xin and Liu, David and Li, Zhen and Wang, Mengmeng and Gao, Peng and Yang, Harry},
  journal={arXiv preprint arXiv:2511.13649},
  year={2025}
}
```