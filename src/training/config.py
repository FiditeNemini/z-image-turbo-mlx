"""
Training configuration classes for Z-Image-Turbo.

This module defines dataclasses for configuring:
- LoRA parameters (rank, alpha, target modules)
- Dataset settings (resolution, augmentation)
- Training hyperparameters (batch size, learning rate, steps)
- Output and logging settings
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import json


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training."""
    
    # LoRA architecture
    rank: int = 16
    """Rank of the LoRA decomposition. Higher = more capacity but more VRAM."""
    
    alpha: float = 16.0
    """LoRA alpha for scaling. Effective scale = alpha / rank."""
    
    dropout: float = 0.0
    """Dropout probability for LoRA layers (0.0 = no dropout)."""
    
    # Target modules - which layers to add LoRA to
    target_modules: List[str] = field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out",  # Attention projections
        "ff.0", "ff.2",  # Feed-forward layers (proj_in, proj_out)
    ])
    """List of module name patterns to apply LoRA to."""
    
    # Training settings
    train_norm_layers: bool = False
    """Whether to also train normalization layer parameters."""
    
    use_dora: bool = False
    """Use DoRA (Weight-Decomposed LoRA) for potentially better results."""
    
    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
            "train_norm_layers": self.train_norm_layers,
            "use_dora": self.use_dora,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LoRAConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DatasetConfig:
    """Configuration for training dataset."""
    
    # Paths
    dataset_path: str = ""
    """Path to dataset directory containing images folder and dataset.json."""
    
    # Image processing
    resolution: int = 1024
    """Target resolution for training images (will be center-cropped/resized)."""
    
    min_resolution: int = 512
    """Minimum resolution - images smaller than this will be upscaled."""
    
    max_aspect_ratio: float = 2.0
    """Maximum aspect ratio allowed (width/height or height/width)."""
    
    # Augmentation
    flip_horizontal: bool = True
    """Enable random horizontal flipping."""
    
    flip_vertical: bool = False
    """Enable random vertical flipping (usually False for most subjects)."""
    
    random_crop: bool = False
    """Enable random cropping instead of center crop."""
    
    color_jitter: float = 0.0
    """Amount of random color jittering (0.0 = disabled)."""
    
    # Caching
    cache_latents: bool = True
    """Pre-encode images to latents and cache them (saves VRAM during training)."""
    
    cache_text_embeddings: bool = True
    """Pre-encode captions to text embeddings and cache them."""
    
    # Bucketing for variable aspect ratios
    use_bucketing: bool = True
    """Use aspect ratio bucketing for more efficient training."""
    
    bucket_no_upscale: bool = True
    """Don't upscale images in bucketing (only downscale)."""
    
    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "resolution": self.resolution,
            "min_resolution": self.min_resolution,
            "max_aspect_ratio": self.max_aspect_ratio,
            "flip_horizontal": self.flip_horizontal,
            "flip_vertical": self.flip_vertical,
            "random_crop": self.random_crop,
            "color_jitter": self.color_jitter,
            "cache_latents": self.cache_latents,
            "cache_text_embeddings": self.cache_text_embeddings,
            "use_bucketing": self.use_bucketing,
            "bucket_no_upscale": self.bucket_no_upscale,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "DatasetConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Main training configuration combining all settings."""
    
    # Paths
    model_path: str = "models/pytorch/Z-Image-Turbo"
    """Path to the base Z-Image-Turbo model (PyTorch format)."""
    
    output_dir: str = "outputs/training"
    """Directory to save checkpoints and final model."""
    
    output_name: str = "my_lora"
    """Name for the output LoRA file (without extension)."""
    
    # Training mode
    training_mode: Literal["lora", "full_finetune"] = "lora"
    """Training mode: 'lora' for LoRA training, 'full_finetune' for full model fine-tuning."""
    
    # Training adapter (Ostris's de-distillation adapter)
    use_training_adapter: bool = True
    """Use the training adapter to preserve turbo capabilities during training."""
    
    training_adapter_path: str = "models/training_adapters/zimage_turbo_training_adapter_v2.safetensors"
    """Path to the training adapter file."""
    
    training_adapter_weight: float = 1.0
    """Weight to apply to the training adapter (1.0 = full strength)."""
    
    # Training hyperparameters
    batch_size: int = 1
    """Training batch size per GPU."""
    
    gradient_accumulation_steps: int = 1
    """Number of gradient accumulation steps (effective batch = batch_size * grad_accum)."""
    
    max_train_steps: int = 1000
    """Maximum number of training steps."""
    
    learning_rate: float = 1e-4
    """Base learning rate."""
    
    lr_scheduler: Literal["constant", "cosine", "linear", "cosine_with_restarts"] = "cosine"
    """Learning rate scheduler type."""
    
    lr_warmup_steps: int = 100
    """Number of warmup steps for learning rate scheduler."""
    
    lr_warmup_ratio: float = 0.0
    """Warmup ratio (alternative to warmup_steps, overrides if > 0)."""
    
    # Optimizer
    optimizer: Literal["adamw", "adamw8bit", "prodigy", "adafactor"] = "adamw"
    """Optimizer to use."""
    
    weight_decay: float = 0.01
    """Weight decay for optimizer."""
    
    adam_beta1: float = 0.9
    """Adam beta1 parameter."""
    
    adam_beta2: float = 0.999
    """Adam beta2 parameter."""
    
    adam_epsilon: float = 1e-8
    """Adam epsilon parameter."""
    
    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping (0 = no clipping)."""
    
    # Flow matching settings (Z-Image-Turbo specific)
    num_train_timesteps: int = 1000
    """Number of timesteps for flow matching scheduler."""
    
    shift: float = 3.0
    """Shift parameter for flow matching (affects noise schedule)."""
    
    use_dynamic_shifting: bool = False
    """Whether to use dynamic shifting in the scheduler."""
    
    # Inference settings during training
    sample_steps: int = 8
    """Number of sampling steps for validation images (turbo mode)."""
    
    guidance_scale: float = 1.0
    """CFG scale for validation images (1.0 = no CFG for turbo)."""
    
    # Precision
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    """Mixed precision training mode."""
    
    # Checkpointing
    save_every_n_steps: int = 500
    """Save a checkpoint every N steps."""
    
    save_total_limit: int = 3
    """Maximum number of checkpoints to keep."""
    
    resume_from_checkpoint: Optional[str] = None
    """Path to checkpoint to resume training from."""
    
    # Validation
    validation_prompts: List[str] = field(default_factory=lambda: [
        "a photograph of a beautiful sunset over the ocean",
        "a portrait of a woman with flowing hair",
        "a detailed macro photo of a flower",
    ])
    """Prompts to use for validation image generation."""
    
    validation_every_n_steps: int = 250
    """Generate validation images every N steps."""
    
    num_validation_images: int = 4
    """Number of validation images to generate per prompt."""
    
    # Logging
    log_every_n_steps: int = 10
    """Log training metrics every N steps."""
    
    use_wandb: bool = False
    """Enable Weights & Biases logging."""
    
    wandb_project: str = "zimage-turbo-training"
    """W&B project name."""
    
    # Hardware
    seed: int = 42
    """Random seed for reproducibility."""
    
    num_workers: int = 4
    """Number of data loading workers."""
    
    pin_memory: bool = True
    """Pin memory for data loading (faster GPU transfer)."""
    
    gradient_checkpointing: bool = False
    """Enable gradient checkpointing to save memory at the cost of slower training."""
    
    # Sub-configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    """LoRA configuration."""
    
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set warmup steps from ratio if specified
        if self.lr_warmup_ratio > 0:
            self.lr_warmup_steps = int(self.max_train_steps * self.lr_warmup_ratio)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary (for saving)."""
        data = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, (LoRAConfig, DatasetConfig)):
                data[field_name] = value.to_dict()
            else:
                data[field_name] = value
        return data
    
    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        # Handle nested configs
        if "lora" in data and isinstance(data["lora"], dict):
            data["lora"] = LoRAConfig.from_dict(data["lora"])
        if "dataset" in data and isinstance(data["dataset"], dict):
            data["dataset"] = DatasetConfig.from_dict(data["dataset"])
        
        # Filter out unknown keys
        known_keys = set(cls.__dataclass_fields__.keys())
        filtered_data = {k: v for k, v in data.items() if k in known_keys}
        
        return cls(**filtered_data)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


# Preset configurations for common use cases
PRESET_CONFIGS = {
    "quick_test": {
        "max_train_steps": 100,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "save_every_n_steps": 50,
        "validation_every_n_steps": 50,
        "lora": {"rank": 8, "alpha": 8.0},
    },
    "character_lora": {
        "max_train_steps": 1500,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_every_n_steps": 500,
        "validation_every_n_steps": 250,
        "lora": {"rank": 32, "alpha": 32.0},
        "dataset": {"flip_horizontal": False},  # Don't flip for characters
    },
    "style_lora": {
        "max_train_steps": 2000,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "save_every_n_steps": 500,
        "validation_every_n_steps": 250,
        "lora": {"rank": 64, "alpha": 64.0},
        "dataset": {"flip_horizontal": True},
    },
    "concept_lora": {
        "max_train_steps": 1000,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "save_every_n_steps": 250,
        "validation_every_n_steps": 250,
        "lora": {"rank": 16, "alpha": 16.0},
    },
}


def get_preset_config(preset_name: str, **overrides) -> TrainingConfig:
    """
    Get a preset training configuration with optional overrides.
    
    Args:
        preset_name: Name of the preset (quick_test, character_lora, style_lora, concept_lora)
        **overrides: Additional parameters to override in the config
        
    Returns:
        TrainingConfig with preset values and overrides applied
    """
    if preset_name not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESET_CONFIGS.keys())}")
    
    preset = PRESET_CONFIGS[preset_name].copy()
    
    # Handle nested configs
    lora_overrides = {}
    dataset_overrides = {}
    
    for key, value in list(overrides.items()):
        if key == "lora" and isinstance(value, dict):
            lora_overrides.update(value)
            del overrides[key]
        elif key == "dataset" and isinstance(value, dict):
            dataset_overrides.update(value)
            del overrides[key]
    
    # Merge preset lora config with overrides
    if "lora" in preset:
        preset["lora"].update(lora_overrides)
    else:
        preset["lora"] = lora_overrides
    
    # Merge preset dataset config with overrides  
    if "dataset" in preset:
        preset["dataset"].update(dataset_overrides)
    else:
        preset["dataset"] = dataset_overrides
    
    # Apply remaining overrides
    preset.update(overrides)
    
    return TrainingConfig.from_dict(preset)
