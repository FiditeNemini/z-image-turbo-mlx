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
    """Get list of available datasets."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = []
    for d in DATASETS_DIR.iterdir():
        if d.is_dir() and (d / "images").exists():
            datasets.append(d.name)
    return sorted(datasets)


def get_dataset_info(dataset_name: str) -> str:
    """Get formatted info about a dataset."""
    if not dataset_name:
        return "Select a dataset to see info"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"Dataset not found: {dataset_name}"
    
    images_dir = dataset_path / "images"
    
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


def create_dataset(
    name: str,
    description: str,
    trigger_word: str,
    default_caption: str,
    resolution: int,
) -> str:
    """Create a new dataset."""
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
    image_files: List[str],
    auto_caption: bool,
) -> str:
    """Add images to an existing dataset."""
    if not dataset_name:
        return "❌ Select a dataset first"
    
    if not image_files:
        return "❌ No images selected"
    
    dataset_path = DATASETS_DIR / dataset_name
    if not dataset_path.exists():
        return f"❌ Dataset not found: {dataset_name}"
    
    images_dir = dataset_path / "images"
    added = 0
    errors = []
    
    for src_path in image_files:
        src = Path(src_path)
        if not src.exists():
            errors.append(f"Not found: {src.name}")
            continue
        
        # Copy image
        dst = images_dir / src.name
        counter = 1
        while dst.exists():
            dst = images_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        
        try:
            shutil.copy2(src, dst)
            added += 1
            
            # Create empty caption file if auto_caption disabled
            if not auto_caption:
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
        metadata["num_images"] = len(list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg")))
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    
    result = f"✅ Added {added} images to {dataset_name}"
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


def get_training_adapter_info() -> str:
    """Get info about available training adapters."""
    TRAINING_ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)
    
    adapters = list(TRAINING_ADAPTERS_DIR.glob("*.safetensors"))
    
    if not adapters:
        return "No training adapters found.\n\nDownload from: https://huggingface.co/ostris/zimage_turbo_training_adapter"
    
    lines = ["**Available Training Adapters:**", ""]
    for adapter in adapters:
        lines.append(f"• {adapter.name}")
    
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
        return [gr.update() for _ in range(12)]
    
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
        gr.update(value=preset.get("use_training_adapter", True)),
    ]


def estimate_vram_usage(
    rank: int,
    batch_size: int,
    resolution: int,
    grad_accum: int,
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
            gradient_checkpointing=True,
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
    use_adapter: bool,
    adapter_weight: float,
    validation_prompts: str,
) -> str:
    """Start the training process."""
    global _training_thread, _training_stop_flag, _training_progress
    
    if not TRAINING_AVAILABLE:
        return "❌ Training not available (requires MPS or CUDA)"
    
    if _training_thread and _training_thread.is_alive():
        return "❌ Training already in progress. Stop current training first."
    
    if not dataset_name:
        return "❌ Please select a dataset"
    
    if not output_name:
        return "❌ Please enter an output name"
    
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
        ),
        
        # Adapter
        use_training_adapter=use_adapter,
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
    
    return "✅ Training started! Monitor progress below."


def stop_training() -> str:
    """Stop the current training."""
    global _training_stop_flag, _training_progress
    
    _training_stop_flag = True
    _training_progress["status"] = "stopping"
    _training_progress["message"] = "Stopping training..."
    
    return "⏹️ Stop signal sent. Training will stop after current step."


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
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Create New Dataset**")
                    new_dataset_name = gr.Textbox(
                        label="Dataset Name",
                        placeholder="e.g., my_character",
                    )
                    new_dataset_description = gr.Textbox(
                        label="Description",
                        placeholder="Optional description",
                    )
                    new_dataset_trigger = gr.Textbox(
                        label="Trigger Word",
                        placeholder="e.g., ohwx person",
                        info="Unique word to activate the concept",
                    )
                    new_dataset_caption = gr.Textbox(
                        label="Default Caption",
                        placeholder="e.g., a photo of ohwx person",
                    )
                    new_dataset_resolution = gr.Dropdown(
                        choices=[512, 768, 1024],
                        value=1024,
                        label="Training Resolution",
                    )
                    create_dataset_btn = gr.Button("📁 Create Dataset", variant="secondary")
                    create_dataset_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("**Select & Manage Dataset**")
                    dataset_dropdown = gr.Dropdown(
                        choices=get_available_datasets(),
                        label="Select Dataset",
                        info="Choose a dataset to train on",
                    )
                    refresh_datasets_btn = gr.Button("🔄 Refresh", size="sm")
                    dataset_info_display = gr.Markdown("Select a dataset to see info")
                    
                    gr.Markdown("---")
                    gr.Markdown("**Add Images**")
                    image_upload = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    auto_caption_checkbox = gr.Checkbox(
                        label="Create empty caption files",
                        value=True,
                        info="You can edit captions manually later",
                    )
                    add_images_btn = gr.Button("➕ Add Images to Dataset", variant="secondary")
                    add_images_status = gr.Textbox(label="Status", interactive=False)
                    
                    validate_dataset_btn = gr.Button("✅ Validate Dataset", variant="secondary")
                    validate_status = gr.Markdown("")
        
        with gr.Accordion("⚙️ Training Configuration", open=True):
            with gr.Row():
                training_preset = gr.Dropdown(
                    choices=["quick_test", "character_lora", "style_lora", "concept_lora", "custom"],
                    value="character_lora",
                    label="Training Preset",
                    info="Select a preset or choose 'custom' for manual configuration",
                )
                preset_description = gr.Markdown(get_preset_description("character_lora"))
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Basic Settings**")
                    output_name = gr.Textbox(
                        label="Output LoRA Name",
                        placeholder="e.g., my_character_lora",
                    )
                    model_path_input = gr.Textbox(
                        label="Base Model Path",
                        value="models/pytorch/Z-Image-Turbo",
                        info="Path to PyTorch model directory",
                    )
                    max_train_steps = gr.Slider(
                        minimum=100,
                        maximum=10000,
                        value=1500,
                        step=100,
                        label="Max Training Steps",
                    )
                    learning_rate = gr.Number(
                        value=1e-4,
                        label="Learning Rate",
                    )
                
                with gr.Column():
                    gr.Markdown("**LoRA Settings**")
                    lora_rank = gr.Slider(
                        minimum=4,
                        maximum=128,
                        value=32,
                        step=4,
                        label="LoRA Rank",
                        info="Higher = more capacity, more VRAM",
                    )
                    lora_alpha = gr.Number(
                        value=32.0,
                        label="LoRA Alpha",
                        info="Typically same as rank",
                    )
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=1,
                        step=1,
                        label="Batch Size",
                    )
                    grad_accum = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Gradient Accumulation Steps",
                        info="Effective batch = batch_size × grad_accum",
                    )
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Training Options**")
                    lr_scheduler = gr.Dropdown(
                        choices=["cosine", "constant", "linear", "cosine_with_restarts"],
                        value="cosine",
                        label="LR Scheduler",
                    )
                    save_every = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        value=500,
                        step=100,
                        label="Save Checkpoint Every N Steps",
                    )
                    validate_every = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        value=250,
                        step=50,
                        label="Generate Validation Images Every N Steps",
                    )
                
                with gr.Column():
                    gr.Markdown("**Dataset Options**")
                    train_resolution = gr.Dropdown(
                        choices=[512, 768, 1024],
                        value=1024,
                        label="Training Resolution",
                    )
                    flip_horizontal = gr.Checkbox(
                        label="Random Horizontal Flip",
                        value=True,
                        info="Disable for asymmetric subjects",
                    )
                    
                    gr.Markdown("**Training Adapter**")
                    use_adapter = gr.Checkbox(
                        label="Use De-distillation Adapter",
                        value=True,
                        info="Preserves turbo capabilities during training",
                    )
                    adapter_weight = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Adapter Weight",
                        visible=True,
                    )
            
            with gr.Accordion("🔮 Validation Prompts", open=False):
                validation_prompts = gr.Textbox(
                    label="Validation Prompts (one per line)",
                    value="a photograph of [trigger], professional lighting\na portrait of [trigger], detailed, high quality",
                    lines=4,
                    info="Replace [trigger] with your trigger word",
                )
            
            vram_estimate = gr.Markdown("Select settings to see VRAM estimate")
            estimate_vram_btn = gr.Button("📊 Estimate VRAM", variant="secondary")
        
        with gr.Accordion("🚀 Training", open=True):
            with gr.Row():
                start_training_btn = gr.Button("▶️ Start Training", variant="primary", scale=2)
                stop_training_btn = gr.Button("⏹️ Stop Training", variant="stop", scale=1)
            
            training_status = gr.Textbox(
                label="Status",
                value="Ready to train",
                interactive=False,
                lines=2,
            )
            
            training_progress_bar = gr.Progress()
            training_progress_text = gr.Markdown("No training in progress")
            
            # Auto-refresh progress (using Gradio's built-in mechanism)
            refresh_progress_btn = gr.Button("🔄 Refresh Progress", size="sm")
        
        with gr.Accordion("ℹ️ Training Adapter Info", open=False):
            adapter_info = gr.Markdown(get_training_adapter_info())
        
        # Event handlers
        create_dataset_btn.click(
            fn=create_dataset,
            inputs=[new_dataset_name, new_dataset_description, new_dataset_trigger, new_dataset_caption, new_dataset_resolution],
            outputs=[create_dataset_status],
        ).then(
            fn=lambda: gr.update(choices=get_available_datasets()),
            outputs=[dataset_dropdown],
        )
        
        refresh_datasets_btn.click(
            fn=lambda: gr.update(choices=get_available_datasets()),
            outputs=[dataset_dropdown],
        )
        
        dataset_dropdown.change(
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
                flip_horizontal, train_resolution, use_adapter,
            ],
        )
        
        estimate_vram_btn.click(
            fn=estimate_vram_usage,
            inputs=[lora_rank, batch_size, train_resolution, grad_accum],
            outputs=[vram_estimate],
        )
        
        use_adapter.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_adapter],
            outputs=[adapter_weight],
        )
        
        start_training_btn.click(
            fn=start_training,
            inputs=[
                dataset_dropdown, output_name, model_path_input,
                max_train_steps, learning_rate, batch_size, grad_accum,
                lora_rank, lora_alpha, lr_scheduler, save_every, validate_every,
                flip_horizontal, train_resolution, use_adapter, adapter_weight,
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
        
        # Return components for potential external access
        return {
            "dataset_dropdown": dataset_dropdown,
            "output_name": output_name,
            "training_status": training_status,
        }
