"""
Z-Image-Turbo Training Module

This module provides training functionality for the Z-Image-Turbo model including:
- LoRA (Low-Rank Adaptation) training
- Model fine-tuning with de-distillation support
- Dataset management and preprocessing
- Training adapter support (Ostris's zimage_turbo_training_adapter)

Training is performed using PyTorch with either:
- **MPS (Metal Performance Shaders)** on Apple Silicon Macs
- **CUDA** on NVIDIA GPUs (Linux/Windows)

The trained LoRAs can then be used with both MLX and PyTorch inference pipelines.

Usage:
    from src.training import TrainingConfig, LoRATrainer, DatasetManager
    
    # Configure training
    config = TrainingConfig(
        model_path="models/pytorch/Z-Image-Turbo",
        output_dir="outputs/my_lora",
        dataset_path="datasets/my_dataset",
        use_training_adapter=True,
    )
    
    # Create trainer and start training
    trainer = LoRATrainer(config)
    trainer.train()
"""

# Configuration (always available)
from .config import TrainingConfig, LoRAConfig, DatasetConfig

# Dataset management (always available - uses PIL fallback if no torchvision)
from .dataset import DatasetManager, TrainingDataset

# Utilities (always available)
from .utils import (
    check_cuda_available,
    check_mps_available,
    check_training_available,
    get_training_device,
    get_device_info,
    estimate_memory_usage,
    format_training_time,
)

# Training components (may fail if dependencies missing)
try:
    from .trainer import LoRATrainer
    from .adapter import TrainingAdapterManager
    from .lora_network import LoRANetwork, LoRAModule
    TRAINER_AVAILABLE = True
except ImportError as e:
    LoRATrainer = None
    TrainingAdapterManager = None
    LoRANetwork = None
    LoRAModule = None
    TRAINER_AVAILABLE = False
    import warnings
    warnings.warn(f"Training components not available: {e}")

__all__ = [
    # Configuration
    "TrainingConfig",
    "LoRAConfig", 
    "DatasetConfig",
    # Dataset
    "DatasetManager",
    "TrainingDataset",
    # Training (may be None)
    "LoRATrainer",
    "TrainingAdapterManager",
    "LoRANetwork",
    "LoRAModule",
    "TRAINER_AVAILABLE",
    # Utilities
    "check_cuda_available",
    "check_mps_available",
    "check_training_available",
    "get_training_device",
    "get_device_info",
    "estimate_memory_usage",
    "format_training_time",
]

__version__ = "0.1.0"
