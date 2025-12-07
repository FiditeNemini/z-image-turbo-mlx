"""
Training utilities for Z-Image-Turbo.

This module provides utility functions for:
- CUDA/device detection
- Memory estimation
- Training progress formatting
- Model parameter counting
"""

import torch
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import time
from datetime import timedelta


def check_cuda_available() -> bool:
    """
    Check if CUDA is available for training.
    
    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def check_mps_available() -> bool:
    """
    Check if MPS (Metal Performance Shaders) is available for training on macOS.
    
    Returns:
        True if MPS is available, False otherwise
    """
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def check_training_available() -> bool:
    """
    Check if any GPU backend is available for training.
    
    Returns:
        True if CUDA or MPS is available, False otherwise
    """
    return check_cuda_available() or check_mps_available()


def get_training_device() -> torch.device:
    """
    Get the best available device for training.
    
    Priority: CUDA > MPS > CPU
    
    Returns:
        torch.device for the best available backend
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about the training device.
    
    Returns:
        Dictionary with device name, memory info, etc.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
        "device": str(get_training_device()),
        "device_name": "CPU",
        "memory_total": 0,
    }
    
    if torch.cuda.is_available():
        info.update({
            "device_name": torch.cuda.get_device_name(0),
            "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),  # GB
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_memory_allocated": torch.cuda.memory_allocated(0) / (1024**3),  # GB
            "cuda_memory_reserved": torch.cuda.memory_reserved(0) / (1024**3),  # GB
        })
    elif info["mps_available"]:
        # MPS on Apple Silicon
        import subprocess
        try:
            # Get total system memory as proxy (MPS shares unified memory)
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            total_mem = int(result.stdout.strip()) / (1024**3)
            info.update({
                "device_name": "Apple Silicon (MPS)",
                "memory_total": total_mem,
                "memory_type": "unified",
            })
        except Exception:
            info.update({
                "device_name": "Apple Silicon (MPS)",
                "memory_total": 0,
            })
    
    return info


def estimate_memory_usage(
    model_params: int,
    lora_rank: int,
    batch_size: int,
    resolution: int = 1024,
    mixed_precision: str = "bf16",
    gradient_checkpointing: bool = True,
) -> Dict[str, float]:
    """
    Estimate VRAM usage for training.
    
    This is a rough estimate based on typical memory patterns.
    
    Args:
        model_params: Number of parameters in the base model
        lora_rank: LoRA rank
        batch_size: Training batch size
        resolution: Training resolution
        mixed_precision: Precision mode (no, fp16, bf16)
        gradient_checkpointing: Whether gradient checkpointing is used
        
    Returns:
        Dictionary with memory estimates in GB
    """
    # Bytes per parameter
    param_bytes = 2 if mixed_precision in ["fp16", "bf16"] else 4
    
    # Base model (frozen, only needs forward pass memory)
    model_memory = model_params * param_bytes / (1024**3)
    
    # LoRA parameters (rough estimate: ~1-5% of model size depending on rank)
    # Assuming targeting ~10% of layers with LoRA
    lora_params = model_params * 0.1 * (lora_rank / 128)  # Scale by rank
    lora_memory = lora_params * param_bytes / (1024**3)
    
    # Optimizer states (Adam: 2 states per trainable param)
    optimizer_memory = lora_params * param_bytes * 2 / (1024**3)
    
    # Gradients
    gradient_memory = lora_params * param_bytes / (1024**3)
    
    # Activations (rough estimate based on resolution and batch)
    # Z-Image-Turbo: 16 channels, resolution/8 latent size
    latent_size = resolution // 8
    latent_elements = batch_size * 16 * latent_size * latent_size
    activation_memory = latent_elements * param_bytes * 30 / (1024**3)  # ~30 layers
    
    if gradient_checkpointing:
        activation_memory *= 0.3  # Checkpointing reduces activation memory significantly
    
    # VAE memory (for encoding training images)
    vae_memory = 1.5  # GB, rough estimate
    
    # Text encoder memory
    text_encoder_memory = 2.5  # GB for Qwen2.5-3B
    
    total_memory = (
        model_memory +
        lora_memory +
        optimizer_memory +
        gradient_memory +
        activation_memory +
        vae_memory +
        text_encoder_memory
    )
    
    return {
        "model_memory_gb": round(model_memory, 2),
        "lora_memory_gb": round(lora_memory, 2),
        "optimizer_memory_gb": round(optimizer_memory, 2),
        "gradient_memory_gb": round(gradient_memory, 2),
        "activation_memory_gb": round(activation_memory, 2),
        "vae_memory_gb": round(vae_memory, 2),
        "text_encoder_memory_gb": round(text_encoder_memory, 2),
        "total_estimated_gb": round(total_memory, 2),
    }


def format_training_time(seconds: float) -> str:
    """
    Format training time in a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string like "1h 23m 45s"
    """
    td = timedelta(seconds=int(seconds))
    
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if td.days > 0:
        return f"{td.days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def format_eta(current_step: int, total_steps: int, elapsed_time: float) -> str:
    """
    Format estimated time remaining.
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        elapsed_time: Time elapsed so far in seconds
        
    Returns:
        Formatted ETA string
    """
    if current_step == 0:
        return "calculating..."
    
    steps_remaining = total_steps - current_step
    time_per_step = elapsed_time / current_step
    eta_seconds = steps_remaining * time_per_step
    
    return format_training_time(eta_seconds)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(num_params: int) -> str:
    """
    Format parameter count in human-readable format.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string like "1.2B" or "345M"
    """
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_training_steps: int,
    num_warmup_steps: int = 0,
    num_cycles: int = 1,
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Create a learning rate scheduler.
    
    Args:
        optimizer: The optimizer
        scheduler_type: Type of scheduler (constant, linear, cosine, cosine_with_restarts)
        num_training_steps: Total number of training steps
        num_warmup_steps: Number of warmup steps
        num_cycles: Number of cycles for cosine_with_restarts
        
    Returns:
        Learning rate scheduler
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def constant_schedule_with_warmup(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    
    def linear_schedule_with_warmup(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    def cosine_schedule_with_warmup(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    def cosine_with_restarts_schedule_with_warmup(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((progress * num_cycles) % 1.0))))
    
    schedulers = {
        "constant": constant_schedule_with_warmup,
        "linear": linear_schedule_with_warmup,
        "cosine": cosine_schedule_with_warmup,
        "cosine_with_restarts": cosine_with_restarts_schedule_with_warmup,
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}. Available: {list(schedulers.keys())}")
    
    return LambdaLR(optimizer, schedulers[scheduler_type])


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic operations (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class TrainingTimer:
    """Simple timer for tracking training duration."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.step_times: list = []
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        self.step_times = []
    
    def step(self):
        """Record a step time."""
        if self.start_time is not None:
            self.step_times.append(time.time())
    
    @property
    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time
    
    @property
    def elapsed_formatted(self) -> str:
        """Get formatted elapsed time."""
        return format_training_time(self.elapsed)
    
    @property
    def avg_step_time(self) -> float:
        """Get average time per step in seconds."""
        if len(self.step_times) < 2:
            return 0.0
        # Calculate from differences between consecutive step times
        diffs = [self.step_times[i] - self.step_times[i-1] for i in range(1, len(self.step_times))]
        return sum(diffs) / len(diffs)
    
    def get_eta(self, current_step: int, total_steps: int) -> str:
        """Get estimated time remaining."""
        return format_eta(current_step, total_steps, self.elapsed)


def save_training_state(
    output_dir: str,
    step: int,
    model_state: Optional[dict] = None,
    optimizer_state: Optional[dict] = None,
    scheduler_state: Optional[dict] = None,
    config: Optional[dict] = None,
    metrics: Optional[dict] = None,
):
    """
    Save training checkpoint.
    
    Args:
        output_dir: Directory to save checkpoint
        step: Current training step
        model_state: Model state dict (LoRA weights)
        optimizer_state: Optimizer state dict
        scheduler_state: LR scheduler state dict
        config: Training configuration
        metrics: Training metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
    }
    
    if model_state is not None:
        checkpoint["model_state"] = model_state
    if optimizer_state is not None:
        checkpoint["optimizer_state"] = optimizer_state
    if scheduler_state is not None:
        checkpoint["scheduler_state"] = scheduler_state
    if config is not None:
        checkpoint["config"] = config
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    checkpoint_path = output_dir / f"checkpoint-{step}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Also save a "latest" symlink
    latest_path = output_dir / "checkpoint-latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(checkpoint_path.name)
    
    return checkpoint_path


def load_training_state(checkpoint_path: str) -> dict:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint dictionary
    """
    return torch.load(checkpoint_path, map_location="cpu")
