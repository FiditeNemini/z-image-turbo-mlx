"""
Training UI functions for Z-Image-Turbo Gradio interface.

This module provides Gradio UI components and callbacks for:
- Dataset management (create, import, validate)
- Training configuration
- LoRA training execution
- Training progress monitoring
"""

import gradio as gr
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
import shutil
import threading
from datetime import datetime

# Training module imports (delayed to avoid import errors if PyTorch not available)
TRAINING_AVAILABLE = False
try:
    import torch
    # Check for MPS (Apple Silicon) or CUDA availability
    if torch.backends.mps.is_available() or torch.cuda.is_available():
        TRAINING_AVAILABLE = True
        from src.training import (
            TrainingConfig,
            LoRAConfig,
            DatasetConfig,
            DatasetManager,
            LoRATrainer,
            TrainingAdapterManager,
            check_cuda_available,
            check_mps_available,
            check_training_available,
            get_device_info,
        )
        from src.training.config import PRESET_CONFIGS, get_preset_config
except ImportError:
    pass

# Directories
DATASETS_DIR = Path("./datasets")
TRAINING_ADAPTERS_DIR = Path("./models/training_adapters")
OUTPUTS_DIR = Path("./outputs/training")
LORAS_DIR = Path("./models/loras")
MLX_MODELS_DIR = Path("./models/mlx")
PYTORCH_MODELS_DIR = Path("./models/pytorch")
COMFYUI_MODELS_DIR = Path("./models/comfyui")

# Global training state
_training_thread: Optional[threading.Thread] = None
_training_stop_flag = False
_training_progress: Dict[str, Any] = {
    "status": "idle",
    "step": 0,
    "total_steps": 0,
    "loss": 0.0,
    "lr": 0.0,
    "eta": "",
    "message": "",
}


def get_available_datasets() -> List[str]:
    """Get list of available datasets.
    
    Supports both:
    - Structured datasets: datasets/<name>/images/ (from Training tab)
    - Flat datasets: datasets/<name>/ with images directly (from Generate tab)
    """
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    datasets = []
    
    for d in DATASETS_DIR.iterdir():
        if not d.is_dir():
            continue
        
        # Check for structured dataset (has images/ subdirectory)
        if (d / "images").exists():
            datasets.append(d.name)
        else:
            # Check for flat dataset (has images directly in folder)
            has_images = any(
                f.suffix.lower() in image_extensions 
                for f in d.iterdir() if f.is_file()
            )
            if has_images:
                datasets.append(d.name)
    
    return sorted(datasets)


def get_dataset_info(dataset_name: str) -> str:
    """Get formatted info about a dataset."""
    if not dataset_name:
        return "Select a dataset to see info"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"Dataset not found: {dataset_name}"
    
    # Determine images directory (supports flat and structured layouts)
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path  # Flat layout
    is_flat = images_dir == dataset_path
    
    # Load metadata
    metadata_path = dataset_path / "dataset.json"
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    # Count images
    image_files = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    num_images = len(image_files)
    
    # Count captions
    caption_files = list(images_dir.glob("*.txt"))
    num_captions = len(caption_files)
    
    info_lines = [
        f"📁 **{dataset_name}**",
        f"*{'Saved from Generate tab' if is_flat else 'Structured dataset'}*",
        "",
        f"📷 Images: {num_images}",
        f"📝 Captions: {num_captions} ({num_captions/num_images*100:.0f}% coverage)" if num_images > 0 else "📝 Captions: 0",
        "",
    ]
    
    if metadata.get("trigger_word"):
        info_lines.append(f"🏷️ Trigger Word: `{metadata['trigger_word']}`")
    
    if metadata.get("description"):
        info_lines.append(f"📋 Description: {metadata['description']}")
    
    if metadata.get("resolution"):
        info_lines.append(f"📐 Resolution: {metadata['resolution']}px")
    
    return "\n".join(info_lines)


def load_dataset_metadata(dataset_name: str) -> Tuple[str, str, str, int]:
    """Load metadata for a dataset (returns trigger_word, description, default_caption, resolution)."""
    if not dataset_name:
        return "", "", "", 1024
    
    dataset_path = DATASETS_DIR / dataset_name
    metadata_path = dataset_path / "dataset.json"
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return (
            metadata.get("trigger_word", ""),
            metadata.get("description", ""),
            metadata.get("default_caption", ""),
            metadata.get("resolution", 1024),
        )
    
    return "", "", "", 1024


def save_dataset_metadata(
    dataset_name: str,
    trigger_word: str,
    description: str,
    default_caption: str,
    resolution: int,
) -> str:
    """Save or update metadata for an existing dataset."""
    if not dataset_name:
        return "❌ Select a dataset first"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"❌ Dataset not found: {dataset_name}"
    
    # Determine images directory to count images
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path  # Flat layout
    
    # Count images
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_count = sum(1 for f in images_dir.iterdir() if f.is_file() and f.suffix.lower() in image_extensions)
    
    # Load existing metadata or create new
    metadata_path = dataset_path / "dataset.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    else:
        metadata = {
            "name": dataset_name,
            "created_at": datetime.now().isoformat(),
        }
    
    # Update fields
    metadata.update({
        "trigger_word": trigger_word,
        "description": description,
        "default_caption": default_caption,
        "resolution": resolution,
        "num_images": image_count,
    })
    
    # Save
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return f"✅ Metadata saved for {dataset_name}"


def create_dataset_simple(name: str) -> Tuple[str, List[str]]:
    """Create a new empty dataset (just the name).
    
    Returns: (status message, updated dataset choices list)
    """
    if not name:
        return "❌ Please enter a dataset name", get_available_datasets()
    
    # Sanitize name
    name = "".join(c for c in name if c.isalnum() or c in "._- ")
    name = name.strip()
    
    if not name:
        return "❌ Invalid dataset name", get_available_datasets()
    
    dataset_path = DATASETS_DIR / name
    if dataset_path.exists():
        return f"❌ Dataset already exists: {name}", get_available_datasets()
    
    try:
        # Create directories
        images_dir = dataset_path / "images"
        images_dir.mkdir(parents=True)
        
        # Create minimal metadata
        metadata = {
            "name": name,
            "description": "",
            "trigger_word": "",
            "default_caption": "",
            "resolution": 1024,
            "created_at": datetime.now().isoformat(),
            "num_images": 0,
        }
        
        with open(dataset_path / "dataset.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return f"✅ Created dataset: {name}", get_available_datasets()
    
    except Exception as e:
        return f"❌ Error creating dataset: {e}", get_available_datasets()


def create_dataset(
    name: str,
    description: str,
    trigger_word: str,
    default_caption: str,
    resolution: int,
) -> str:
    """Create a new dataset (legacy function for compatibility)."""
    if not name:
        return "❌ Please enter a dataset name"
    
    # Sanitize name
    name = "".join(c for c in name if c.isalnum() or c in "._- ")
    name = name.strip()
    
    if not name:
        return "❌ Invalid dataset name"
    
    dataset_path = DATASETS_DIR / name
    if dataset_path.exists():
        return f"❌ Dataset already exists: {name}"
    
    try:
        # Create directories
        images_dir = dataset_path / "images"
        images_dir.mkdir(parents=True)
        
        # Create metadata
        metadata = {
            "name": name,
            "description": description,
            "trigger_word": trigger_word,
            "default_caption": default_caption,
            "resolution": resolution,
            "created_at": datetime.now().isoformat(),
            "num_images": 0,
        }
        
        with open(dataset_path / "dataset.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return f"✅ Created dataset: {name}\n\nAdd images to: {images_dir}"
    
    except Exception as e:
        return f"❌ Error creating dataset: {e}"


def add_images_to_dataset(
    dataset_name: str,
    uploaded_files: List[str],
    auto_caption: bool,
) -> str:
    """Add images and caption files to an existing dataset."""
    if not dataset_name:
        return "❌ Select a dataset first"
    
    if not uploaded_files:
        return "❌ No files selected"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"❌ Dataset not found: {dataset_name}"
    
    # Determine images directory (supports flat and structured layouts)
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path  # Flat layout
    added_images = 0
    added_captions = 0
    errors = []
    
    # Separate images and caption files
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    image_files = []
    caption_files = {}
    
    for src_path in uploaded_files:
        src = Path(src_path)
        if not src.exists():
            errors.append(f"Not found: {src.name}")
            continue
        
        if src.suffix.lower() in image_extensions:
            image_files.append(src)
        elif src.suffix.lower() == ".txt":
            # Store by stem for matching with images
            caption_files[src.stem] = src
    
    # Process images and their paired captions
    for src in image_files:
        dst = images_dir / src.name
        counter = 1
        orig_stem = src.stem
        new_stem = orig_stem
        while dst.exists():
            new_stem = f"{orig_stem}_{counter}"
            dst = images_dir / f"{new_stem}{src.suffix}"
            counter += 1
        
        try:
            shutil.copy2(src, dst)
            added_images += 1
            
            # Check if there's a matching caption file uploaded
            if orig_stem in caption_files:
                caption_src = caption_files[orig_stem]
                caption_dst = dst.with_suffix(".txt")
                shutil.copy2(caption_src, caption_dst)
                added_captions += 1
            elif not auto_caption:
                # Create empty caption file if requested
                caption_path = dst.with_suffix(".txt")
                if not caption_path.exists():
                    caption_path.touch()
        except Exception as e:
            errors.append(f"Error copying {src.name}: {e}")
    
    # Update metadata
    metadata_path = dataset_path / "dataset.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["num_images"] = len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.webp")))
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    result = f"✅ Added {added_images} images"
    if added_captions > 0:
        result += f" and {added_captions} captions"
    result += f" to {dataset_name}"
    
    if errors:
        result += f"\n\n⚠️ Errors:\n" + "\n".join(errors[:5])
        if len(errors) > 5:
            result += f"\n... and {len(errors) - 5} more"
    
    return result


def import_folder_to_dataset(
    dataset_name: str,
    folder_path: str,
) -> str:
    """Import a folder of image/caption pairs to a dataset."""
    if not dataset_name:
        return "❌ Select a dataset first"
    
    if not folder_path or not folder_path.strip():
        return "❌ Please enter a folder path"
    
    folder = Path(folder_path.strip()).expanduser()
    if not folder.exists():
        return f"❌ Folder not found: {folder}"
    if not folder.is_dir():
        return f"❌ Not a folder: {folder}"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"❌ Dataset not found: {dataset_name}"
    
    # Determine images directory (supports flat and structured layouts)
    images_dir = dataset_path / "images"
    if not images_dir.exists():
        images_dir = dataset_path  # Flat layout
    
    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    
    added_images = 0
    added_captions = 0
    errors = []
    
    # Find all images in the folder
    for src in folder.iterdir():
        if not src.is_file():
            continue
        if src.suffix.lower() not in image_extensions:
            continue
        
        # Determine destination path
        dst = images_dir / src.name
        counter = 1
        orig_stem = src.stem
        new_stem = orig_stem
        while dst.exists():
            new_stem = f"{orig_stem}_{counter}"
            dst = images_dir / f"{new_stem}{src.suffix}"
            counter += 1
        
        try:
            shutil.copy2(src, dst)
            added_images += 1
            
            # Check for accompanying caption file
            caption_src = src.with_suffix(".txt")
            if caption_src.exists():
                caption_dst = dst.with_suffix(".txt")
                shutil.copy2(caption_src, caption_dst)
                added_captions += 1
        except Exception as e:
            errors.append(f"Error copying {src.name}: {e}")
    
    # Update metadata
    metadata_path = dataset_path / "dataset.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        metadata["num_images"] = len(
            list(images_dir.glob("*.png")) + 
            list(images_dir.glob("*.jpg")) + 
            list(images_dir.glob("*.jpeg")) + 
            list(images_dir.glob("*.webp"))
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    if added_images == 0:
        return f"❌ No images found in {folder}"
    
    result = f"✅ Imported {added_images} images"
    if added_captions > 0:
        result += f" and {added_captions} captions"
    result += f" from {folder.name}"
    
    if errors:
        result += f"\n\n⚠️ Errors:\n" + "\n".join(errors[:5])
        if len(errors) > 5:
            result += f"\n... and {len(errors) - 5} more"
    
    return result


def validate_dataset(dataset_name: str) -> str:
    """Validate a dataset and report issues."""
    if not dataset_name:
        return "Select a dataset to validate"
    
    if not TRAINING_AVAILABLE:
        return "❌ Training module not available (requires MPS or CUDA)"
    
    try:
        manager = DatasetManager(DATASETS_DIR)
        result = manager.validate_dataset(dataset_name)
        
        lines = []
        if result["valid"]:
            lines.append(f"✅ **{dataset_name}** is valid!")
        else:
            lines.append(f"❌ **{dataset_name}** has issues")
        
        lines.append("")
        lines.append(f"📷 Total images: {result['total_images']}")
        lines.append(f"✓ Valid images: {result['valid_images']}")
        lines.append(f"📝 Caption coverage: {result['caption_coverage']}%")
        
        if result.get("warnings"):
            lines.append("")
            lines.append("⚠️ **Warnings:**")
            for w in result["warnings"][:5]:
                lines.append(f"  • {w}")
        
        if result.get("issues"):
            lines.append("")
            lines.append("❌ **Issues:**")
            for i in result["issues"][:5]:
                lines.append(f"  • {i}")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"❌ Validation error: {e}"


def get_available_adapters() -> List[str]:
    """Get list of available training adapters for dropdown."""
    TRAINING_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    adapters = list(TRAINING_ADAPTERS_DIR.glob("*.safetensors"))
    return [a.name for a in sorted(adapters)]


def get_trained_loras() -> List[str]:
    """Get list of trained LoRAs from the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    loras = []
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir():
            # Look for final LoRA (same name as directory)
            lora_file = d / f"{d.name}.safetensors"
            if lora_file.exists():
                loras.append(d.name)
            else:
                # Also check for old naming convention
                final_lora = d / f"{d.name}_final.safetensors"
                if final_lora.exists():
                    loras.append(d.name)
                else:
                    # Check for checkpoint subdirectories with lora.safetensors
                    checkpoint_dirs = [p for p in d.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
                    if checkpoint_dirs:
                        # Check if any checkpoint has lora.safetensors
                        for ckpt_dir in checkpoint_dirs:
                            if (ckpt_dir / "lora.safetensors").exists():
                                loras.append(d.name)
                                break
    return sorted(loras)


def get_available_checkpoints() -> List[str]:
    """Get list of available checkpoints for continue training.
    
    Returns checkpoints in format: "run_name (step N)"
    """
    import re
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    
    for d in OUTPUTS_DIR.iterdir():
        if not d.is_dir():
            continue
        
        # Look for checkpoint-N subdirectories
        checkpoint_dirs = [p for p in d.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if checkpoint_dirs:
            # Extract step numbers and get the latest
            latest_step = 0
            for ckpt_dir in checkpoint_dirs:
                match = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
                if match and (ckpt_dir / "lora.safetensors").exists():
                    step = int(match.group(1))
                    if step > latest_step:
                        latest_step = step
            
            if latest_step > 0:
                checkpoints.append(f"{d.name} (step {latest_step})")
    
    return sorted(checkpoints)


def get_lora_path_for_export(lora_name: str) -> Optional[Path]:
    """Get the path to a trained LoRA file for export."""
    import re
    lora_dir = OUTPUTS_DIR / lora_name
    if not lora_dir.exists():
        return None
    
    # Prefer final LoRA (same name as directory)
    final_lora = lora_dir / f"{lora_name}.safetensors"
    if final_lora.exists():
        return final_lora
    
    # Also check old naming convention
    final_lora_old = lora_dir / f"{lora_name}_final.safetensors"
    if final_lora_old.exists():
        return final_lora_old
    
    # Fall back to latest checkpoint directory
    checkpoint_dirs = [p for p in lora_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if checkpoint_dirs:
        # Find the highest step checkpoint
        latest_step = 0
        latest_path = None
        for ckpt_dir in checkpoint_dirs:
            match = re.search(r"checkpoint-(\d+)", ckpt_dir.name)
            if match:
                step = int(match.group(1))
                lora_file = ckpt_dir / "lora.safetensors"
                if step > latest_step and lora_file.exists():
                    latest_step = step
                    latest_path = lora_file
        if latest_path:
            return latest_path
    
    return None


def get_training_adapter_info() -> str:
    """Get info about available training adapters."""
    adapters = get_available_adapters()
    
    if not adapters:
        return "No training adapters found.\n\nDownload from: https://huggingface.co/ostris/zimage_turbo_training_adapter"
    
    lines = ["**Available Training Adapters:**", ""]
    for adapter in adapters:
        lines.append(f"• {adapter}")
    
    lines.append("")
    lines.append("*Training adapters help preserve turbo model capabilities during fine-tuning.*")
    
    return "\n".join(lines)


def get_preset_description(preset_name: str) -> str:
    """Get description of a training preset."""
    descriptions = {
        "quick_test": "**Quick Test** - Fast 100-step run for testing setup",
        "character_lora": "**Character LoRA** - Optimized for training character/subject concepts (1500 steps, no horizontal flip)",
        "style_lora": "**Style LoRA** - For artistic styles (2000 steps, high rank for style details)",
        "concept_lora": "**Concept LoRA** - General concept training (1000 steps, balanced settings)",
        "custom": "**Custom** - Configure all settings manually",
    }
    return descriptions.get(preset_name, "")


def update_config_from_preset(preset_name: str):
    """Update UI values from preset."""
    if preset_name == "custom":
        # Return current values unchanged
        return [gr.update() for _ in range(13)]
    
    preset = PRESET_CONFIGS.get(preset_name, {})
    lora = preset.get("lora", {})
    dataset = preset.get("dataset", {})
    
    return [
        gr.update(value=preset.get("max_train_steps", 1000)),
        gr.update(value=preset.get("learning_rate", 1e-4)),
        gr.update(value=preset.get("batch_size", 1)),
        gr.update(value=preset.get("gradient_accumulation_steps", 1)),
        gr.update(value=lora.get("rank", 16)),
        gr.update(value=lora.get("alpha", 16.0)),
        gr.update(value=preset.get("lr_scheduler", "cosine")),
        gr.update(value=preset.get("save_every_n_steps", 500)),
        gr.update(value=preset.get("validation_every_n_steps", 250)),
        gr.update(value=dataset.get("flip_horizontal", True)),
        gr.update(value=dataset.get("resolution", 1024)),
        gr.update(value=dataset.get("use_bucketing", True)),
        gr.update(value=preset.get("use_training_adapter", True)),
    ]


def estimate_vram_usage(
    rank: int,
    batch_size: int,
    resolution: int,
    grad_accum: int,
    grad_checkpointing: bool,
) -> str:
    """Estimate memory usage for training."""
    if not TRAINING_AVAILABLE:
        return "Memory estimation requires MPS or CUDA"
    
    try:
        from src.training.utils import estimate_memory_usage
        
        # Z-Image-Turbo has ~3.4B params
        model_params = 3_400_000_000
        
        estimate = estimate_memory_usage(
            model_params=model_params,
            lora_rank=rank,
            batch_size=batch_size,
            resolution=resolution,
            gradient_checkpointing=grad_checkpointing,
        )
        
        total = estimate["total_estimated_gb"]
        
        # Get actual available memory
        device_info = get_device_info()
        
        # Check for MPS or CUDA memory
        if device_info.get("mps_available"):
            available = device_info.get("mps_memory_total", 0)
            memory_type = "System Memory (unified)"
        else:
            available = device_info.get("cuda_memory_total", 0)
            memory_type = "VRAM"
        
        lines = [
            f"**Estimated Memory: {total:.1f} GB**",
            "",
            f"• Model: {estimate['model_memory_gb']:.1f} GB",
            f"• LoRA: {estimate['lora_memory_gb']:.2f} GB",
            f"• Optimizer: {estimate['optimizer_memory_gb']:.2f} GB",
            f"• Activations: {estimate['activation_memory_gb']:.1f} GB",
            f"• VAE: {estimate['vae_memory_gb']:.1f} GB",
            f"• Text Encoder: {estimate['text_encoder_memory_gb']:.1f} GB",
        ]
        
        if available > 0:
            lines.append("")
            if total > available:
                lines.append(f"⚠️ Exceeds available {memory_type} ({available:.1f} GB)")
                lines.append("Try: lower resolution, smaller batch, lower rank")
            else:
                lines.append(f"✅ Fits in available {memory_type} ({available:.1f} GB)")
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error estimating memory: {e}"


def check_training_requirements() -> Tuple[bool, str]:
    """Check if training requirements are met."""
    issues = []
    device_type = None
    device_name = "Unknown"
    memory_gb = 0
    
    # Check for GPU backend (MPS or CUDA)
    try:
        import torch
        if torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple Silicon (MPS)"
            # Get system memory for MPS (unified memory)
            import subprocess
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
            memory_gb = int(result.stdout.strip()) / (1024**3)
        elif torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            issues.append("❌ No GPU available - training requires MPS (Apple Silicon) or CUDA (NVIDIA)")
        
        if memory_gb > 0 and memory_gb < 16:
            issues.append(f"⚠️ Low memory ({memory_gb:.1f}GB) - may need to reduce settings")
    except ImportError:
        issues.append("❌ PyTorch not installed")
    
    # Check diffusers
    try:
        import diffusers
    except ImportError:
        issues.append("❌ diffusers not installed")
    
    # Check transformers
    try:
        import transformers
    except ImportError:
        issues.append("❌ transformers not installed")
    
    # Check training module
    if not TRAINING_AVAILABLE:
        issues.append("❌ Training module not loaded")
    
    if not issues:
        memory_type = "Memory (unified)" if device_type == "mps" else "VRAM"
        return True, f"✅ Ready for training\n\nDevice: {device_name}\n{memory_type}: {memory_gb:.1f} GB"
    
    return False, "\n".join(issues)


def start_training(
    dataset_name: str,
    output_name: str,
    model_path: str,
    max_steps: int,
    learning_rate: float,
    batch_size: int,
    grad_accum: int,
    lora_rank: int,
    lora_alpha: float,
    lr_scheduler: str,
    save_every: int,
    validate_every: int,
    flip_horizontal: bool,
    resolution: int,
    use_bucketing: bool,
    use_adapter: bool,
    adapter_name: str,
    adapter_weight: float,
    use_grad_checkpointing: bool,
    validation_prompts: str,
) -> str:
    """Start the training process."""
    global _training_thread, _training_stop_flag, _training_progress
    
    if not TRAINING_AVAILABLE:
        return "**❌ Error:** Training not available (requires MPS or CUDA)"
    
    if _training_thread and _training_thread.is_alive():
        return "**❌ Error:** Training already in progress. Stop current training first."
    
    if not dataset_name:
        return "**❌ Error:** Please select a dataset"
    
    if not output_name:
        return "**❌ Error:** Please enter an output name"
    
    # Parse validation prompts
    prompts = [p.strip() for p in validation_prompts.split("\n") if p.strip()]
    if not prompts:
        prompts = ["a photograph of a beautiful sunset over the ocean"]
    
    # Create config
    config = TrainingConfig(
        model_path=model_path,
        output_dir=str(OUTPUTS_DIR / output_name),
        output_name=output_name,
        
        # Training
        max_train_steps=max_steps,
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        gradient_checkpointing=use_grad_checkpointing,
        lr_scheduler=lr_scheduler,
        
        # LoRA
        lora=LoRAConfig(
            rank=lora_rank,
            alpha=lora_alpha,
        ),
        
        # Dataset
        dataset=DatasetConfig(
            dataset_path=str(DATASETS_DIR / dataset_name),
            resolution=resolution,
            flip_horizontal=flip_horizontal,
            use_bucketing=use_bucketing,
        ),
        
        # Adapter
        use_training_adapter=use_adapter,
        training_adapter_path=str(TRAINING_ADAPTERS_DIR / adapter_name) if adapter_name else "",
        training_adapter_weight=adapter_weight,
        
        # Checkpointing
        save_every_n_steps=save_every,
        validation_every_n_steps=validate_every,
        validation_prompts=prompts,
    )
    
    # Reset state
    _training_stop_flag = False
    _training_progress = {
        "status": "starting",
        "step": 0,
        "total_steps": max_steps,
        "loss": 0.0,
        "lr": learning_rate,
        "eta": "calculating...",
        "message": "Initializing trainer...",
    }
    
    def progress_callback(step, loss, lr, eta):
        global _training_progress
        _training_progress.update({
            "status": "training",
            "step": step,
            "loss": loss,
            "lr": lr,
            "eta": eta,
            "message": f"Step {step}/{max_steps} - Loss: {loss:.4f}",
        })
    
    def training_thread():
        global _training_progress, _training_stop_flag
        try:
            trainer = LoRATrainer(config)
            
            _training_progress["message"] = "Loading models..."
            trainer.setup(progress_callback=lambda msg, prog: _training_progress.update({"message": msg}))
            
            _training_progress["message"] = "Starting training..."
            trainer.train(progress_callback=progress_callback)
            
            _training_progress.update({
                "status": "complete",
                "message": f"✅ Training complete! LoRA saved to {config.output_dir}",
            })
        
        except Exception as e:
            _training_progress.update({
                "status": "error",
                "message": f"❌ Training error: {e}",
            })
    
    # Start training thread
    _training_thread = threading.Thread(target=training_thread, daemon=True)
    _training_thread.start()
    
    return "**✅ Training started!** Monitor progress below."


def stop_training() -> str:
    """Stop the current training."""
    global _training_stop_flag, _training_progress
    
    _training_stop_flag = True
    _training_progress["status"] = "stopping"
    _training_progress["message"] = "Stopping training..."
    
    return "**⏹️ Stopping...** Training will stop after current step."


def continue_training(
    checkpoint_selection: str,
    additional_steps: int,
    new_resolution: int,
    dataset_name: str,
    model_path: str,
) -> str:
    """Continue training from a checkpoint with optional resolution change.
    
    This enables progressive training (e.g., 512px -> 1024px).
    """
    global _training_thread, _training_stop_flag, _training_progress
    import re
    
    if not TRAINING_AVAILABLE:
        return "**❌ Error:** Training not available (requires MPS or CUDA)"
    
    if _training_thread and _training_thread.is_alive():
        return "**❌ Error:** Training already in progress. Stop current training first."
    
    if not checkpoint_selection:
        return "**❌ Error:** Please select a checkpoint"
    
    if not dataset_name:
        return "**❌ Error:** Please select a dataset"
    
    # Parse checkpoint selection: "run_name (step N)"
    match = re.match(r"(.+) \(step (\d+)\)", checkpoint_selection)
    if not match:
        return f"**❌ Error:** Invalid checkpoint format: {checkpoint_selection}"
    
    run_name = match.group(1)
    current_step = int(match.group(2))
    checkpoint_dir = OUTPUTS_DIR / run_name
    
    if not checkpoint_dir.exists():
        return f"**❌ Error:** Checkpoint directory not found: {checkpoint_dir}"
    
    # Find the checkpoint file (checkpoint-N/lora.safetensors)
    ckpt_subdir = checkpoint_dir / f"checkpoint-{current_step}"
    ckpt_file = ckpt_subdir / "lora.safetensors"
    if not ckpt_file.exists():
        return f"**❌ Error:** Checkpoint file not found: {ckpt_file}"
    
    total_steps = current_step + additional_steps
    
    # Load original config and update - could be in run dir or checkpoint subdir
    config_path = checkpoint_dir / "training_config.json"
    if not config_path.exists():
        config_path = ckpt_subdir / "training_config.json"
    
    if config_path.exists():
        base_config = TrainingConfig.load(str(config_path))
        # Update with new parameters
        base_config.max_train_steps = total_steps
        base_config.resume_from_checkpoint = str(ckpt_subdir)
        base_config.dataset.resolution = new_resolution
        base_config.dataset.dataset_path = str(DATASETS_DIR / dataset_name)
    else:
        return "**❌ Error:** Could not find training config for checkpoint"
    
    # Reset state
    _training_stop_flag = False
    _training_progress = {
        "status": "starting",
        "step": current_step,
        "total_steps": total_steps,
        "loss": 0.0,
        "lr": base_config.learning_rate,
        "eta": "calculating...",
        "message": f"Resuming from step {current_step}...",
    }
    
    def progress_callback(step, loss, lr, eta):
        global _training_progress
        _training_progress.update({
            "status": "training",
            "step": step,
            "loss": loss,
            "lr": lr,
            "eta": eta,
            "message": f"Step {step}/{total_steps} - Loss: {loss:.4f}",
        })
    
    def training_thread():
        global _training_progress, _training_stop_flag
        try:
            trainer = LoRATrainer(base_config)
            
            _training_progress["message"] = "Loading models and checkpoint..."
            trainer.setup(progress_callback=lambda msg, prog: _training_progress.update({"message": msg}))
            
            _training_progress["message"] = f"Continuing training from step {current_step}..."
            trainer.train(progress_callback=progress_callback)
            
            _training_progress.update({
                "status": "complete",
                "message": f"✅ Training complete! LoRA saved to {base_config.output_dir}",
            })
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            _training_progress.update({
                "status": "error",
                "message": f"❌ Training error: {e}",
            })
    
    # Start training thread
    _training_thread = threading.Thread(target=training_thread, daemon=True)
    _training_thread.start()
    
    return f"**✅ Continuing from step {current_step}!** Training {additional_steps} more steps at {new_resolution}px."


def save_trained_lora_as_model(
    trained_lora_name: str,
    output_model_name: str,
    lora_scale: float,
    save_lora_only: bool,
    save_mlx: bool,
    save_pytorch: bool,
    save_comfyui: bool,
    progress=None,
) -> str:
    """
    Save a trained LoRA in various formats.
    
    Args:
        trained_lora_name: Name of the trained LoRA in outputs directory
        output_model_name: Name for the output model/LoRA
        lora_scale: Scale/strength to apply when baking into model
        save_lora_only: If True, just copy the LoRA to models/loras/
        save_mlx: Save as MLX model with LoRA baked in
        save_pytorch: Save as PyTorch model with LoRA baked in
        save_comfyui: Save as ComfyUI model with LoRA baked in
        progress: Optional Gradio progress callback
    
    Returns:
        Status message
    """
    import re
    
    if not trained_lora_name:
        return "❌ Please select a trained LoRA"
    
    if not output_model_name or not output_model_name.strip():
        return "❌ Please enter an output name"
    
    output_model_name = output_model_name.strip()
    
    # Validate name
    if not re.match(r'^[\w\-]+$', output_model_name):
        return "❌ Name can only contain letters, numbers, hyphens, and underscores"
    
    # Check at least one format selected
    if not save_lora_only and not save_mlx and not save_pytorch and not save_comfyui:
        return "❌ Please select at least one output format"
    
    # Get the LoRA path
    lora_path = get_lora_path_for_export(trained_lora_name)
    if not lora_path:
        return f"❌ Could not find LoRA file for '{trained_lora_name}'"
    
    results = []
    
    try:
        # Export LoRA and/or provide instructions for model fusion
        result = _save_lora_as_merged_model(
            lora_path, output_model_name, lora_scale,
            save_lora_only, save_mlx, save_pytorch, save_comfyui, progress
        )
        return result
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"


def _copy_lora_to_models(lora_path: Path, output_name: str, progress=None) -> str:
    """Copy a trained LoRA to the models/loras/custom/ directory."""
    output_dir = LORAS_DIR / "custom"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{output_name}.safetensors"
    if output_path.exists():
        return f"❌ LoRA '{output_name}.safetensors' already exists in models/loras/custom/"
    
    if progress:
        progress(0.5, desc="Copying LoRA file...")
    
    shutil.copy2(lora_path, output_path)
    
    if progress:
        progress(1.0, desc="Done!")
    
    return f"✅ Saved LoRA to models/loras/custom/{output_name}.safetensors"


def _save_lora_as_merged_model(
    lora_path: Path,
    output_name: str,
    lora_scale: float,
    save_lora_only: bool,
    save_mlx: bool,
    save_pytorch: bool,
    save_comfyui: bool,
    progress=None,
) -> str:
    """
    Export a trained LoRA in multiple formats.
    
    - save_lora_only: Copy LoRA to models/loras/ for use in generation
    - save_mlx: Fuse LoRA into MLX model
    - save_pytorch: Fuse LoRA into PyTorch model
    - save_comfyui: Fuse LoRA into ComfyUI single-file checkpoint
    """
    results = []
    
    # Progress helper
    def update_progress(pct, desc):
        if progress:
            try:
                progress(pct, desc=desc)
            except:
                pass
    
    try:
        # 1. Export standalone LoRA
        if save_lora_only:
            update_progress(0.1, "Exporting LoRA...")
            lora_output = LORAS_DIR / f"{output_name}.safetensors"
            LORAS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(lora_path, lora_output)
            results.append(f"✅ LoRA: {lora_output}")
        
        # 2. Export MLX model with fused LoRA
        if save_mlx:
            update_progress(0.2, "Exporting MLX model...")
            result = _export_fused_mlx_model(lora_path, output_name, lora_scale, update_progress)
            results.append(result)
        
        # 3. Export PyTorch model with fused LoRA
        if save_pytorch:
            update_progress(0.5, "Exporting PyTorch model...")
            result = _export_fused_pytorch_model(lora_path, output_name, lora_scale, update_progress)
            results.append(result)
        
        # 4. Export ComfyUI checkpoint with fused LoRA
        if save_comfyui:
            update_progress(0.7, "Exporting ComfyUI checkpoint...")
            result = _export_fused_comfyui_model(lora_path, output_name, lora_scale, update_progress)
            results.append(result)
        
        update_progress(1.0, "Done!")
        return "\n".join(results) if results else "❌ No export options selected"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Export error: {e}"


def _export_fused_mlx_model(lora_path: Path, output_name: str, lora_scale: float, progress_fn) -> str:
    """Export an MLX model with LoRA fused into weights."""
    try:
        import mlx.core as mx
        import numpy as np
        from safetensors.numpy import save_file
        import json
        
        # Add src to path for lora import
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from lora import load_lora, apply_lora_to_model
        from z_image_mlx import ZImageTransformer2DModel
    except ImportError as e:
        return f"❌ MLX export requires: {e}"
    
    # Source model path - try multiple common locations
    source_path = None
    for model_name in ["Z-Image-Turbo-MLX", "mlx_model"]:
        candidate = MLX_MODELS_DIR / model_name
        if candidate.exists() and (candidate / "weights.safetensors").exists():
            source_path = candidate
            break
    
    if source_path is None:
        return f"❌ MLX source model not found in: {MLX_MODELS_DIR}"
    
    # Output path
    output_path = MLX_MODELS_DIR / output_name
    if output_path.exists():
        return f"❌ Model '{output_name}' already exists"
    
    progress_fn(0.25, "Loading MLX model...")
    
    # Load model config
    with open(source_path / "config.json", "r") as f:
        config = json.load(f)
    
    # Create and load model
    model = ZImageTransformer2DModel(config)
    weights = mx.load(str(source_path / "weights.safetensors"))
    
    # MLX model weights are already in correct format, load directly
    # Filter out any None keys or pad_token weights
    processed_weights = {k: v for k, v in weights.items() if k and "pad_token" not in k}
    model.load_weights(list(processed_weights.items()), strict=False)
    
    progress_fn(0.35, "Applying LoRA...")
    
    # Apply LoRA
    lora_weights = load_lora(lora_path)
    apply_lora_to_model(model, lora_weights, scale=lora_scale, verbose=False)
    
    progress_fn(0.4, "Saving fused MLX model...")
    
    # Save fused weights
    output_path.mkdir(parents=True, exist_ok=True)
    
    fused_weights = {}
    for name, param in model.named_modules():
        if hasattr(param, 'weight'):
            fused_weights[f"{name}.weight"] = np.array(param.weight)
        if hasattr(param, 'bias') and param.bias is not None:
            fused_weights[f"{name}.bias"] = np.array(param.bias)
    
    save_file(fused_weights, str(output_path / "weights.safetensors"))
    
    # Copy auxiliary files
    for file in ["config.json", "vae_config.json", "vae.safetensors", 
                 "text_encoder_config.json", "text_encoder.safetensors"]:
        if (source_path / file).exists():
            shutil.copy(source_path / file, output_path / file)
    
    for folder in ["tokenizer", "scheduler"]:
        if (source_path / folder).exists():
            shutil.copytree(source_path / folder, output_path / folder)
    
    # Create metadata
    metadata = {
        "base_model": "Z-Image-Turbo-MLX",
        "fused_lora": str(lora_path.name),
        "lora_scale": lora_scale,
        "created": datetime.now().isoformat(),
    }
    with open(output_path / "lora_fusion_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return f"✅ MLX: {output_path}"


def _export_fused_pytorch_model(lora_path: Path, output_name: str, lora_scale: float, progress_fn) -> str:
    """Export a PyTorch model with LoRA fused into weights."""
    try:
        import torch
        import numpy as np
        import json
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from lora import load_lora, parse_lora_weights
    except ImportError as e:
        return f"❌ PyTorch export requires: {e}"
    
    # Source model path
    source_path = PYTORCH_MODELS_DIR / "Z-Image-Turbo"
    if not source_path.exists():
        return f"❌ PyTorch source model not found: {source_path}"
    
    # Output path
    output_path = PYTORCH_MODELS_DIR / output_name
    if output_path.exists():
        return f"❌ Model '{output_name}' already exists"
    
    progress_fn(0.55, "Loading PyTorch model...")
    
    try:
        from diffusers import DiffusionPipeline
        from src.training.models import ZImagePipeline
    except ImportError:
        try:
            from training.models import ZImagePipeline
        except ImportError:
            # Fallback: try loading directly from diffusers
            from diffusers import DiffusionPipeline as ZImagePipeline
    
    pipe = ZImagePipeline.from_pretrained(
        str(source_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    
    progress_fn(0.6, "Applying LoRA to PyTorch model...")
    
    # Apply LoRA
    lora_weights = load_lora(lora_path)
    _apply_lora_to_pytorch_transformer(pipe.transformer, lora_weights, lora_scale)
    
    progress_fn(0.65, "Saving PyTorch model...")
    
    # Save pipeline
    pipe.save_pretrained(str(output_path))
    
    # Create metadata
    metadata = {
        "base_model": "Z-Image-Turbo",
        "fused_lora": str(lora_path.name),
        "lora_scale": lora_scale,
        "created": datetime.now().isoformat(),
    }
    with open(output_path / "lora_fusion_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return f"✅ PyTorch: {output_path}"


def _apply_lora_to_pytorch_transformer(model, lora_weights, scale=1.0):
    """Apply MLX LoRA weights to a PyTorch transformer model."""
    import torch
    import numpy as np
    
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from lora import parse_lora_weights
    
    lora_pairs = parse_lora_weights(lora_weights)
    
    for model_key, (lora_a, lora_b) in lora_pairs.items():
        layer_path = model_key.rsplit(".weight", 1)[0]
        
        try:
            parts = layer_path.split(".")
            layer = model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            
            if not hasattr(layer, 'weight'):
                continue
            
            a_np = np.array(lora_a)
            b_np = np.array(lora_b)
            
            if len(a_np.shape) == 2 and len(b_np.shape) == 2:
                delta_np = b_np @ a_np
            else:
                continue
            
            with torch.no_grad():
                delta_torch = torch.from_numpy(delta_np * scale).to(layer.weight.dtype).to(layer.weight.device)
                if delta_torch.shape == layer.weight.shape:
                    layer.weight.add_(delta_torch)
                    
        except (AttributeError, IndexError, KeyError):
            continue


def _export_fused_comfyui_model(lora_path: Path, output_name: str, lora_scale: float, progress_fn) -> str:
    """Export a ComfyUI single-file checkpoint with LoRA fused."""
    try:
        import torch
        import numpy as np
        import json
        from safetensors.torch import save_file as torch_save_safetensors
        
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from lora import load_lora
        from convert_pytorch_to_comfyui import (
            convert_transformer_key_to_comfyui,
            convert_text_encoder_key_to_comfyui,
            convert_vae_key_to_comfyui,
        )
    except ImportError as e:
        return f"❌ ComfyUI export requires: {e}"
    
    # Source model path
    source_path = PYTORCH_MODELS_DIR / "Z-Image-Turbo"
    if not source_path.exists():
        return f"❌ PyTorch source model not found: {source_path}"
    
    # Output path
    COMFYUI_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = COMFYUI_MODELS_DIR / f"{output_name}.safetensors"
    if output_file.exists():
        return f"❌ ComfyUI model '{output_name}.safetensors' already exists"
    
    progress_fn(0.75, "Loading PyTorch model for ComfyUI...")
    
    try:
        from src.training.models import ZImagePipeline
    except ImportError:
        try:
            from training.models import ZImagePipeline
        except ImportError:
            from diffusers import DiffusionPipeline as ZImagePipeline
    
    # Load in bfloat16 for smaller file size (matches typical ComfyUI Z-Image checkpoints)
    pipe = ZImagePipeline.from_pretrained(
        str(source_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    progress_fn(0.8, "Applying LoRA and converting...")
    
    # Apply LoRA (convert to float32 for computation, then back to bfloat16)
    pipe.transformer = pipe.transformer.to(torch.float32)
    lora_weights = load_lora(lora_path)
    _apply_lora_to_pytorch_transformer(pipe.transformer, lora_weights, lora_scale)
    pipe.transformer = pipe.transformer.to(torch.bfloat16)
    
    # Convert TRANSFORMER ONLY to ComfyUI format (no VAE/TextEncoder)
    # VAE and TextEncoder are loaded separately by ComfyUI
    all_weights = {}
    
    # Transformer with QKV fusion
    transformer = pipe.transformer.state_dict()
    qkv_groups = {}
    other_keys = []
    
    for key in transformer.keys():
        if ".attention.to_q.weight" in key:
            base_key = key.replace(".to_q.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["q"] = key
        elif ".attention.to_k.weight" in key:
            base_key = key.replace(".to_k.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["k"] = key
        elif ".attention.to_v.weight" in key:
            base_key = key.replace(".to_v.weight", "")
            if base_key not in qkv_groups:
                qkv_groups[base_key] = {}
            qkv_groups[base_key]["v"] = key
        else:
            other_keys.append(key)
    
    for base_key, qkv_keys in qkv_groups.items():
        if "q" in qkv_keys and "k" in qkv_keys and "v" in qkv_keys:
            q = transformer[qkv_keys["q"]]
            k = transformer[qkv_keys["k"]]
            v = transformer[qkv_keys["v"]]
            fused = torch.cat([q, k, v], dim=0)
            comfyui_key = convert_transformer_key_to_comfyui(f"{base_key}.qkv.weight")
            all_weights[comfyui_key] = fused
    
    for key in other_keys:
        comfyui_key = convert_transformer_key_to_comfyui(key)
        all_weights[comfyui_key] = transformer[key]
    
    # NOTE: Not including VAE and TextEncoder - ComfyUI loads these separately
    # This matches the typical 12GB Z-Image checkpoint format
    
    progress_fn(0.9, "Saving ComfyUI checkpoint (float16, transformer only)...")
    
    torch_save_safetensors(all_weights, str(output_file))
    file_size = output_file.stat().st_size / (1024 * 1024 * 1024)
    
    return f"✅ ComfyUI: {output_file} ({file_size:.2f}GB, transformer only)"


def get_training_progress() -> Tuple[str, float, str]:
    """Get current training progress."""
    global _training_progress
    
    status = _training_progress.get("status", "idle")
    step = _training_progress.get("step", 0)
    total = _training_progress.get("total_steps", 1)
    message = _training_progress.get("message", "")
    
    progress = step / max(total, 1)
    
    if status == "idle":
        progress_text = "No training in progress"
    elif status == "training":
        loss = _training_progress.get("loss", 0)
        lr = _training_progress.get("lr", 0)
        eta = _training_progress.get("eta", "")
        progress_text = f"Step {step}/{total} ({progress*100:.1f}%)\nLoss: {loss:.4f} | LR: {lr:.2e} | ETA: {eta}"
    else:
        progress_text = message
    
    return progress_text, progress, status


def create_training_tab():
    """Create the Training tab UI components."""
    
    with gr.Tab("🎓 Training"):
        # Check requirements first
        ready, requirements_status = check_training_requirements()
        
        if not ready:
            gr.Markdown("## ⚠️ Training Requirements Not Met")
            gr.Markdown(requirements_status)
            gr.Markdown("""
            ### Requirements for Training:
            - **macOS:** Apple Silicon Mac with MPS support (macOS 12.3+, 16GB+ unified memory recommended)
            - **Linux/Windows:** NVIDIA GPU with CUDA support (24GB+ VRAM recommended)
            - PyTorch 2.0+: `pip install torch`
            - Additional packages: `pip install diffusers transformers`
            """)
            return {}
        
        gr.Markdown(
            """
            ### LoRA Training for Z-Image-Turbo
            
            Train custom LoRA adapters for Z-Image-Turbo models. Training uses PyTorch with
            MPS (Apple Silicon) or CUDA (NVIDIA) and can leverage Ostris's de-distillation adapter.
            
            **Workflow:** 1️⃣ Prepare Dataset → 2️⃣ Configure Training → 3️⃣ Train → 4️⃣ Use LoRA in Generate tab
            """
        )
        
        with gr.Accordion("📁 Dataset Management", open=True):
            # Row 1: Select existing OR Create new
            with gr.Row():
                dataset_dropdown = gr.Dropdown(
                    choices=get_available_datasets(),
                    label="Select Dataset",
                    info="Existing datasets (or create new below)",
                    scale=2,
                )
                refresh_datasets_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                new_dataset_name = gr.Textbox(
                    label="New Dataset Name",
                    placeholder="my_character",
                    scale=2,
                )
                create_dataset_btn = gr.Button("📁 Create", variant="secondary", scale=0, min_width=80)
            create_dataset_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
            
            # Row 2: Dataset Info display
            dataset_info_display = gr.Markdown("*Select a dataset to view/edit settings*")
            
            # Row 3: Metadata fields (all in one row)
            with gr.Row():
                meta_trigger_word = gr.Textbox(
                    label="Trigger Word",
                    placeholder="ohwx person",
                    scale=1,
                )
                meta_default_caption = gr.Textbox(
                    label="Default Caption",
                    placeholder="a photo of ohwx person",
                    scale=2,
                )
                meta_resolution = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=1024,
                    label="Resolution",
                    scale=0,
                    min_width=100,
                )
            with gr.Row():
                meta_description = gr.Textbox(
                    label="Description (optional)",
                    placeholder="Dataset description",
                    scale=3,
                )
                save_metadata_btn = gr.Button("💾 Save", variant="primary", scale=0, min_width=80)
                metadata_status = gr.Textbox(show_label=False, interactive=False, max_lines=1, scale=1)
            
            # Row 4: Add images
            with gr.Row():
                with gr.Column(scale=1):
                    image_upload = gr.File(
                        label="Upload Images (.png/.jpg + .txt captions)",
                        file_count="multiple",
                        file_types=[".png", ".jpg", ".jpeg", ".webp", ".txt"],
                        height=80,
                    )
                    with gr.Row():
                        auto_caption_checkbox = gr.Checkbox(label="Create empty captions", value=True, scale=2)
                        add_images_btn = gr.Button("➕ Add", variant="secondary", scale=1)
                    add_images_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
                
                with gr.Column(scale=1):
                    folder_path_input = gr.Textbox(
                        label="Import from Folder",
                        placeholder="/path/to/images",
                    )
                    import_folder_btn = gr.Button("📥 Import Folder", variant="secondary")
                    import_folder_status = gr.Textbox(show_label=False, interactive=False, max_lines=1)
            
            # Row 5: Validate
            with gr.Row():
                validate_dataset_btn = gr.Button("✅ Validate Dataset", size="sm")
                validate_status = gr.Markdown("")
        
        with gr.Accordion("⚙️ Training Configuration", open=True):
            # Preset selector
            with gr.Row():
                training_preset = gr.Dropdown(
                    choices=["quick_test", "character_lora", "style_lora", "concept_lora", "custom"],
                    value="character_lora",
                    label="Preset",
                    scale=1,
                )
                output_name = gr.Textbox(
                    label="Output Name",
                    placeholder="my_character_lora",
                    scale=2,
                )
                model_path_input = gr.Textbox(
                    label="Model Path",
                    value="models/pytorch/Z-Image-Turbo",
                    scale=2,
                )
            preset_description = gr.Markdown(get_preset_description("character_lora"))
            
            # Main settings - 3 columns
            with gr.Row():
                max_train_steps = gr.Slider(100, 10000, 1500, step=100, label="Steps")
                learning_rate = gr.Number(value=1e-4, label="LR")
                lora_rank = gr.Slider(4, 128, 32, step=4, label="LoRA Rank")
            
            with gr.Row():
                lora_alpha = gr.Number(value=32.0, label="LoRA Alpha")
                batch_size = gr.Slider(1, 4, 1, step=1, label="Batch Size")
                grad_accum = gr.Slider(1, 16, 4, step=1, label="Grad Accum")
            
            with gr.Row():
                lr_scheduler = gr.Dropdown(
                    choices=["cosine", "constant", "linear", "cosine_with_restarts"],
                    value="cosine", label="LR Scheduler",
                )
                save_every = gr.Slider(100, 2000, 500, step=100, label="Save Every N")
                validate_every = gr.Slider(100, 1000, 250, step=50, label="Validate Every N")
            
            # Dataset & adapter options in compact rows
            with gr.Row():
                train_resolution = gr.Dropdown(choices=[512, 768, 1024], value=1024, label="Resolution")
                flip_horizontal = gr.Checkbox(label="Horizontal Flip", value=True)
                use_bucketing = gr.Checkbox(label="Aspect Bucketing", value=True)
                grad_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=False)
            
            available_adapters = get_available_adapters()
            with gr.Row():
                use_adapter = gr.Checkbox(
                    label="Use Training Adapter",
                    value=len(available_adapters) > 0,
                    interactive=len(available_adapters) > 0,
                )
                adapter_dropdown = gr.Dropdown(
                    choices=available_adapters,
                    value=available_adapters[0] if available_adapters else None,
                    label="Adapter",
                    visible=len(available_adapters) > 0,
                )
                adapter_weight = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Adapter Weight", visible=len(available_adapters) > 0)
            
            if not available_adapters:
                gr.Markdown("⚠️ No adapters found. [Download](https://huggingface.co/ostris/zimage_turbo_training_adapter)")
            
            with gr.Accordion("🔮 Validation Prompts", open=False):
                validation_prompts = gr.Textbox(
                    label="Prompts (one per line)",
                    value="a photograph of [trigger], professional lighting\na portrait of [trigger], detailed",
                    lines=2,
                )
            
            with gr.Row():
                estimate_vram_btn = gr.Button("📊 Estimate VRAM", size="sm")
                vram_estimate = gr.Markdown("*Select settings to see estimate*")
        
        with gr.Accordion("🚀 Training", open=True):
            with gr.Row():
                start_training_btn = gr.Button("▶️ Start", variant="primary", scale=1)
                stop_training_btn = gr.Button("⏹️ Stop", variant="stop", scale=0, min_width=80)
                refresh_progress_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
            
            training_status = gr.Markdown("**Status:** Ready to train")
            training_progress_text = gr.Markdown("")
        
        with gr.Accordion("ℹ️ Training Adapter Info", open=False):
            adapter_info = gr.Markdown(get_training_adapter_info())
        
        with gr.Accordion("🔄 Continue Training", open=False):
            gr.Markdown("*Resume from a checkpoint with optional resolution change (e.g., 512px → 1024px)*")
            with gr.Row():
                continue_checkpoint = gr.Dropdown(
                    choices=get_available_checkpoints(),
                    label="Checkpoint",
                    scale=2,
                )
                refresh_checkpoints_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
            with gr.Row():
                continue_additional_steps = gr.Number(
                    value=200,
                    label="Additional Steps",
                    precision=0,
                    minimum=10,
                    maximum=10000,
                    scale=1,
                )
                continue_resolution = gr.Dropdown(
                    choices=[512, 768, 1024],
                    value=1024,
                    label="New Resolution",
                    scale=1,
                )
            continue_btn = gr.Button("▶️ Continue Training", variant="primary")
        
        with gr.Accordion("💾 Export LoRA", open=False):
            with gr.Row():
                trained_lora_dropdown = gr.Dropdown(
                    choices=get_trained_loras(),
                    label="Trained LoRA",
                    scale=2,
                )
                refresh_trained_loras_btn = gr.Button("🔄", size="sm", scale=0, min_width=40)
                export_output_name = gr.Textbox(label="Output Name", placeholder="my_lora", scale=2)
                export_lora_scale = gr.Slider(0.1, 2.0, 1.0, step=0.1, label="Strength", scale=1)
            
            with gr.Row():
                export_lora_only = gr.Checkbox(label="📁 LoRA", value=True)
                export_mlx = gr.Checkbox(label="🍎 MLX", value=False)
                export_pytorch = gr.Checkbox(label="🔥 PyTorch", value=False)
                export_comfyui = gr.Checkbox(label="🎨 ComfyUI", value=False)
                export_btn = gr.Button("💾 Export", variant="primary")
            
            export_status = gr.Textbox(show_label=False, interactive=False, max_lines=2)
        
        # Event handlers
        
        # Create new dataset (simple flow - just name, then edit metadata)
        def on_create_dataset(name):
            status, choices = create_dataset_simple(name)
            # Return status, updated dropdown choices, and select the new dataset if created
            if status.startswith("✅"):
                # Extract the created dataset name
                new_name = name.strip()
                new_name = "".join(c for c in new_name if c.isalnum() or c in "._- ")
                return status, gr.update(choices=choices, value=new_name)
            return status, gr.update(choices=choices)
        
        create_dataset_btn.click(
            fn=on_create_dataset,
            inputs=[new_dataset_name],
            outputs=[create_dataset_status, dataset_dropdown],
        )
        
        # Helper to load dataset info and metadata together
        def refresh_and_load(current_dataset):
            choices = get_available_datasets()
            if current_dataset and current_dataset in choices:
                info = get_dataset_info(current_dataset)
                trigger, desc, caption, res = load_dataset_metadata(current_dataset)
                return gr.update(choices=choices), info, trigger, desc, caption, res
            return gr.update(choices=choices), "*Select a dataset above to see info and edit settings*", "", "", "", 1024
        
        refresh_datasets_btn.click(
            fn=refresh_and_load,
            inputs=[dataset_dropdown],
            outputs=[dataset_dropdown, dataset_info_display, meta_trigger_word, meta_description, meta_default_caption, meta_resolution],
        )
        
        # When dataset is selected, load its info AND populate metadata fields
        def on_dataset_select(dataset_name):
            info = get_dataset_info(dataset_name)
            trigger, desc, caption, res = load_dataset_metadata(dataset_name)
            return info, trigger, desc, caption, res
        
        dataset_dropdown.change(
            fn=on_dataset_select,
            inputs=[dataset_dropdown],
            outputs=[dataset_info_display, meta_trigger_word, meta_description, meta_default_caption, meta_resolution],
        )
        
        # Save metadata button
        save_metadata_btn.click(
            fn=save_dataset_metadata,
            inputs=[dataset_dropdown, meta_trigger_word, meta_description, meta_default_caption, meta_resolution],
            outputs=[metadata_status],
        ).then(
            fn=get_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info_display],
        )
        
        add_images_btn.click(
            fn=lambda ds, files, auto: add_images_to_dataset(ds, [f.name for f in files] if files else [], auto),
            inputs=[dataset_dropdown, image_upload, auto_caption_checkbox],
            outputs=[add_images_status],
        ).then(
            fn=get_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info_display],
        )
        
        validate_dataset_btn.click(
            fn=validate_dataset,
            inputs=[dataset_dropdown],
            outputs=[validate_status],
        )
        
        import_folder_btn.click(
            fn=import_folder_to_dataset,
            inputs=[dataset_dropdown, folder_path_input],
            outputs=[import_folder_status],
        ).then(
            fn=get_dataset_info,
            inputs=[dataset_dropdown],
            outputs=[dataset_info_display],
        )
        
        training_preset.change(
            fn=get_preset_description,
            inputs=[training_preset],
            outputs=[preset_description],
        ).then(
            fn=update_config_from_preset,
            inputs=[training_preset],
            outputs=[
                max_train_steps, learning_rate, batch_size, grad_accum,
                lora_rank, lora_alpha, lr_scheduler, save_every, validate_every,
                flip_horizontal, train_resolution, use_bucketing, use_adapter,
            ],
        )
        
        estimate_vram_btn.click(
            fn=estimate_vram_usage,
            inputs=[lora_rank, batch_size, train_resolution, grad_accum, grad_checkpointing],
            outputs=[vram_estimate],
        )
        
        use_adapter.change(
            fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
            inputs=[use_adapter],
            outputs=[adapter_dropdown, adapter_weight],
        )
        
        start_training_btn.click(
            fn=start_training,
            inputs=[
                dataset_dropdown, output_name, model_path_input,
                max_train_steps, learning_rate, batch_size, grad_accum,
                lora_rank, lora_alpha, lr_scheduler, save_every, validate_every,
                flip_horizontal, train_resolution, use_bucketing,
                use_adapter, adapter_dropdown, adapter_weight,
                grad_checkpointing,
                validation_prompts,
            ],
            outputs=[training_status],
        )
        
        stop_training_btn.click(
            fn=stop_training,
            outputs=[training_status],
        )
        
        def refresh_progress():
            text, progress, status = get_training_progress()
            return text
        
        refresh_progress_btn.click(
            fn=refresh_progress,
            outputs=[training_progress_text],
        )
        
        # Export section event handlers
        refresh_trained_loras_btn.click(
            fn=lambda: gr.update(choices=get_trained_loras()),
            outputs=[trained_lora_dropdown],
        )
        
        export_btn.click(
            fn=save_trained_lora_as_model,
            inputs=[
                trained_lora_dropdown,
                export_output_name,
                export_lora_scale,
                export_lora_only,
                export_mlx,
                export_pytorch,
                export_comfyui,
            ],
            outputs=[export_status],
        )
        
        # Continue Training event handlers
        refresh_checkpoints_btn.click(
            fn=lambda: gr.update(choices=get_available_checkpoints()),
            outputs=[continue_checkpoint],
        )
        
        continue_btn.click(
            fn=continue_training,
            inputs=[
                continue_checkpoint,
                continue_additional_steps,
                continue_resolution,
                dataset_dropdown,
                model_path_input,
            ],
            outputs=[training_status],
        )
        
        # Return components for potential external access
        return {
            "dataset_dropdown": dataset_dropdown,
            "output_name": output_name,
            "training_status": training_status,
        }
