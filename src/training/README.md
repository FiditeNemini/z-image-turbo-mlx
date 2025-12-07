# Z-Image-Turbo Training Module

This module provides LoRA training capabilities for Z-Image-Turbo models.

## Requirements

Training requires one of the following GPU backends:

### macOS (Apple Silicon) - Recommended
- **Apple Silicon Mac** (M1/M2/M3/M4) with MPS support
- **macOS 12.3+** (Monterey or later)
- **16GB+ unified memory** recommended (32GB+ for larger batches)
- **PyTorch 2.0+**: Already included in requirements.txt

### Linux/Windows (NVIDIA)
- **NVIDIA GPU** with CUDA support (24GB+ VRAM recommended)
- **PyTorch with CUDA**: `pip install torch --index-url https://download.pytorch.org/whl/cu121`

### Common Requirements
- **Additional packages**: `pip install diffusers transformers torchvision`

> **Note**: MLX is used for inference only. Training uses PyTorch with MPS (Apple Silicon) or CUDA (NVIDIA). Trained LoRAs work seamlessly with MLX inference.

## Features

### LoRA Training
- Train custom LoRA adapters for Z-Image-Turbo
- Configurable rank, alpha, and target modules
- Support for DoRA (Weight-Decomposed LoRA)

### Training Adapter Support
- Integrates Ostris's de-distillation adapter
- Preserves turbo model capabilities during fine-tuning
- Adapter merging during training, inversion during inference

### Dataset Management
- Create and manage training datasets
- Automatic caption file handling
- **Aspect ratio bucketing** for training on varied image dimensions:
  - 9 predefined buckets from 768×1344 to 1344×768
  - Maintains ~1 megapixel per image regardless of aspect ratio
  - No cropping required - images resized to nearest bucket
  - Enable/disable via checkbox in training UI

## Quick Start

### 1. Prepare Your Dataset

Create a dataset directory structure:
```
datasets/
└── my_dataset/
    ├── dataset.json       # Auto-created metadata
    └── images/
        ├── image1.png
        ├── image1.txt     # Caption (optional)
        ├── image2.jpg
        └── image2.txt
```

Or use the Gradio UI to create and manage datasets.

### 2. Configure Training

Using presets:
```python
from src.training import TrainingConfig, get_preset_config

# Use a preset
config = get_preset_config("character_lora",
    output_name="my_character",
    dataset={"dataset_path": "datasets/my_dataset"}
)
```

Manual configuration:
```python
config = TrainingConfig(
    model_path="models/pytorch/Z-Image-Turbo",
    output_dir="outputs/my_lora",
    output_name="my_lora",
    
    # Training settings
    max_train_steps=1500,
    learning_rate=1e-4,
    batch_size=1,
    gradient_accumulation_steps=4,
    
    # LoRA settings
    lora=LoRAConfig(
        rank=32,
        alpha=32.0,
    ),
    
    # Dataset
    dataset=DatasetConfig(
        dataset_path="datasets/my_dataset",
        resolution=1024,
    ),
    
    # Use training adapter
    use_training_adapter=True,
)
```

### 3. Train

```python
from src.training import LoRATrainer

trainer = LoRATrainer(config)
trainer.setup()
trainer.train()
```

Or use the Training tab in the Gradio UI.

### 4. Export Your LoRA

After training, use the **💾 Export Trained LoRA** section in the Training tab to save your LoRA:

- **📁 LoRA Only**: Copy to `models/loras/custom/` for use in Generate tab
- **🍎 MLX**: Bake LoRA into base model, save to `models/mlx/`
- **🔥 PyTorch**: Bake LoRA into base model, save to `models/pytorch/`
- **🎨 ComfyUI**: Bake LoRA into base model, save to `models/comfyui/`

You can select multiple formats simultaneously.

## Training Presets

| Preset | Steps | Rank | Best For |
|--------|-------|------|----------|
| `quick_test` | 100 | 8 | Testing setup |
| `character_lora` | 1500 | 32 | Character/subject training |
| `style_lora` | 2000 | 64 | Artistic styles |
| `concept_lora` | 1000 | 16 | General concepts |

## Understanding the Training Adapter

The training adapter (from [ostris/zimage_turbo_training_adapter](https://huggingface.co/ostris/zimage_turbo_training_adapter)) is a de-distillation LoRA that helps preserve the turbo model's fast-generation capabilities during fine-tuning.

### How It Works

1. **During Training**: The adapter is merged into the model (+1.0 weight)
2. **Your LoRA Learns**: Against the adapter-modified model
3. **During Inference**: The adapter can be inverted (-1.0) or omitted

### When to Use

- **Short training runs (<1500 steps)**: Use adapter, generate with turbo settings (8 steps, CFG 1.0)
- **Longer training (>3000 steps)**: Consider inverting adapter during inference

## Module Structure

```
src/training/
├── __init__.py          # Package exports
├── config.py            # TrainingConfig, LoRAConfig, DatasetConfig
├── dataset.py           # DatasetManager, TrainingDataset
├── trainer.py           # LoRATrainer main class
├── lora_network.py      # LoRA injection and weight management
├── adapter.py           # Training adapter handling
└── utils.py             # Utilities (timers, memory estimation, etc.)
```

## Memory Optimization

For systems with limited memory:

1. **Lower resolution**: 512px uses ~40% less memory than 1024px
2. **Smaller LoRA rank**: rank=8 vs rank=32 saves ~50% LoRA memory
3. **Gradient accumulation**: Use grad_accum=8, batch_size=1 instead of batch=8
4. **Enable gradient checkpointing**: Trades compute for memory (enabled by default)

### MPS (Apple Silicon) Notes
- MPS uses unified memory shared with the system
- Close other applications to free up memory
- Mixed precision (fp16) is supported but without AMP GradScaler
- Some operations may fall back to CPU automatically

### CUDA Notes
- Use `torch.cuda.empty_cache()` if you encounter fragmentation
- Monitor VRAM with `nvidia-smi`

## Troubleshooting

### Out of Memory (MPS or CUDA)
- Reduce batch size or resolution
- Lower LoRA rank
- Increase gradient accumulation instead of batch size
- Close other applications (especially on macOS)

### Training Not Converging
- Check that captions are correct
- Ensure trigger word is in all captions
- Try different learning rates (1e-5 to 1e-3)
- Increase training steps

### LoRA Not Working in Inference
- Verify LoRA saved correctly (check file size)
- Ensure using correct model (MLX vs PyTorch)
- Try adjusting LoRA weight (0.5-1.5)
