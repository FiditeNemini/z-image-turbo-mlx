"""
LoRA Trainer for Z-Image-Turbo.

This module provides the main training loop and orchestration for
LoRA training on Z-Image-Turbo models.

Training is performed using PyTorch with:
- MPS (Metal Performance Shaders) on Apple Silicon Macs
- CUDA on NVIDIA GPUs  
- CPU fallback (very slow, not recommended)

Requirements:
- Apple Silicon Mac with MPS support (macOS 12.3+), OR
- NVIDIA GPU with CUDA support (recommended 24GB+ VRAM)
- PyTorch 2.0+ with MPS or CUDA support
- diffusers, transformers

Usage:
    from src.training import TrainingConfig, LoRATrainer
    
    config = TrainingConfig(
        model_path="models/pytorch/Z-Image-Turbo",
        output_dir="outputs/my_lora",
        dataset=DatasetConfig(dataset_path="datasets/my_dataset"),
    )
    
    trainer = LoRATrainer(config)
    trainer.train(progress_callback=my_callback)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import json
import math
from datetime import datetime
from tqdm import tqdm

from .config import TrainingConfig, LoRAConfig, DatasetConfig
from .dataset import (
    DatasetManager,
    TrainingDataset,
    ImageEntry,
    AspectRatioBucket,
    BucketBatchSampler,
    create_dataloader,
    create_bucketed_dataloader,
)
from .lora_network import LoRANetwork, get_target_modules_for_zimage
from .adapter import TrainingAdapterManager, AdapterAwareTraining
from .utils import (
    check_cuda_available,
    check_mps_available,
    check_training_available,
    get_training_device,
    get_device_info,
    seed_everything,
    count_parameters,
    format_params,
    format_training_time,
    get_lr_scheduler,
    TrainingTimer,
    save_training_state,
    load_training_state,
)


class LoRATrainer:
    """
    Main trainer class for Z-Image-Turbo LoRA training.
    
    Handles:
    - Model loading (PyTorch Z-Image-Turbo)
    - LoRA injection and management
    - Training loop with flow matching loss
    - Checkpoint saving and resumption
    - Validation image generation
    - Training adapter integration
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = get_training_device()
        
        # Will be initialized in setup()
        self.model = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.lora_network = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")
        
        # Adapter support
        self.adapter_helper = None
        if config.use_training_adapter:
            self.adapter_helper = AdapterAwareTraining(
                adapter_path=config.training_adapter_path,
                adapter_weight=config.training_adapter_weight,
            )
        
        # Timer
        self.timer = TrainingTimer()
        
        # Mixed precision - only use GradScaler on CUDA (not supported on MPS)
        self.scaler = None
        self.use_amp = False
        if config.mixed_precision != "no" and self.device.type == "cuda":
            try:
                from torch.cuda.amp import GradScaler
                self.scaler = GradScaler()
                self.use_amp = True
            except ImportError:
                print("Warning: GradScaler not available, training without AMP")
        elif config.mixed_precision != "no" and self.device.type == "mps":
            # MPS supports float16/bfloat16 but not GradScaler
            # We'll use manual dtype casting instead
            self.use_amp = False
            print("Note: Using MPS backend - AMP with GradScaler not supported, using manual dtype casting")
    
    def setup(self, progress_callback: Optional[Callable] = None):
        """
        Set up training components.
        
        Args:
            progress_callback: Optional callback for progress updates
                Signature: callback(message: str, progress: float)
        """
        if progress_callback:
            progress_callback("Checking GPU availability...", 0.0)
        
        if not check_training_available():
            raise RuntimeError(
                "No GPU backend available for training. "
                "Please ensure you have either:\n"
                "- Apple Silicon Mac with MPS support (macOS 12.3+), or\n"
                "- NVIDIA GPU with CUDA support\n"
                "And PyTorch is installed with the appropriate backend."
            )
        
        device_info = get_device_info()
        
        # Print device info based on backend
        if self.device.type == "cuda":
            print(f"Training device: {device_info.get('cuda_device_name', 'CUDA GPU')}")
            print(f"VRAM: {device_info.get('cuda_memory_total', 0):.1f} GB")
        elif self.device.type == "mps":
            print(f"Training device: Apple Silicon (MPS)")
            print(f"System Memory: {device_info.get('mps_memory_total', 0):.1f} GB (unified memory)")
        else:
            print(f"Training device: CPU (Warning: training will be very slow!)")
        
        # Set random seed
        seed_everything(self.config.seed)
        
        # Load models
        if progress_callback:
            progress_callback("Loading transformer model...", 0.1)
        self._load_transformer()
        
        if progress_callback:
            progress_callback("Loading text encoder...", 0.3)
        self._load_text_encoder()
        
        if progress_callback:
            progress_callback("Loading VAE...", 0.5)
        self._load_vae()
        
        if progress_callback:
            progress_callback("Setting up scheduler...", 0.6)
        self._setup_scheduler()
        
        # Apply training adapter if enabled
        if self.adapter_helper and progress_callback:
            progress_callback("Applying training adapter...", 0.65)
        
        if self.adapter_helper:
            self.adapter_helper.prepare_model_for_training(self.model, verbose=True)
        
        # Set up LoRA
        if progress_callback:
            progress_callback("Initializing LoRA network...", 0.7)
        self._setup_lora()
        
        # Set up optimizer and scheduler
        if progress_callback:
            progress_callback("Setting up optimizer...", 0.8)
        self._setup_optimizer()
        
        # Load dataset
        if progress_callback:
            progress_callback("Loading dataset...", 0.9)
        self._setup_dataset()
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            if progress_callback:
                progress_callback("Resuming from checkpoint...", 0.95)
            self._resume_from_checkpoint()
        
        if progress_callback:
            progress_callback("Setup complete!", 1.0)
        
        print(f"\nTraining setup complete:")
        print(f"  - Model parameters: {format_params(count_parameters(self.model))}")
        print(f"  - LoRA parameters: {format_params(self.lora_network.num_trainable_params())}")
        print(f"  - Dataset size: {len(self.dataloader.dataset)} images")
        print(f"  - Effective batch size: {self.config.effective_batch_size}")
        print(f"  - Training steps: {self.config.max_train_steps}")
    
    def _load_transformer(self):
        """Load the Z-Image-Turbo transformer model."""
        model_path = Path(self.config.model_path)
        
        # Try to load from diffusers format or safetensors
        try:
            # Z-Image uses its own custom transformer architecture, not Flux
            from diffusers import ZImageTransformer2DModel
            
            # Check if it's a diffusers-format model
            # First check if config.json exists at model_path (direct transformer path)
            # Then check if it's a pipeline directory with transformer subdirectory
            if (model_path / "config.json").exists():
                transformer_path = model_path
            elif (model_path / "transformer" / "config.json").exists():
                # Diffusers pipeline format - transformer is in subdirectory
                transformer_path = model_path / "transformer"
            else:
                # Load from safetensors directly
                # This requires the model architecture to be defined
                raise NotImplementedError(
                    "Direct safetensors loading not yet implemented. "
                    "Please provide a diffusers-format model (with config.json in the model "
                    "directory or in a 'transformer' subdirectory)."
                )
            
            self.model = ZImageTransformer2DModel.from_pretrained(
                transformer_path,
                torch_dtype=self._get_dtype(),
            )
        except ImportError:
            raise ImportError(
                "diffusers library is required for training. "
                "Install with: pip install git+https://github.com/huggingface/diffusers"
            )
        
        self.model = self.model.to(self.device)
        self.model.requires_grad_(False)  # Freeze base model
    
    def _load_text_encoder(self):
        """Load the text encoder (Qwen2.5-3B)."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Text encoder path - inside the model directory
        model_path = Path(self.config.model_path)
        encoder_path = model_path / "text_encoder"
        tokenizer_path = model_path / "tokenizer"  # Tokenizer is in separate folder
        
        if not encoder_path.exists():
            # Try parent directory structure
            encoder_path = model_path.parent / "text_encoder"
            tokenizer_path = model_path.parent / "tokenizer"
        
        use_local = encoder_path.exists() and tokenizer_path.exists()
        
        if not use_local:
            # Fallback to HuggingFace (with warning)
            print("⚠️ Local text encoder or tokenizer not found, downloading from HuggingFace...")
            encoder_path = "Qwen/Qwen2.5-3B"
            tokenizer_path = "Qwen/Qwen2.5-3B"
        else:
            print(f"✓ Using local text encoder: {encoder_path}")
            print(f"✓ Using local tokenizer: {tokenizer_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=use_local)
        self.text_encoder = AutoModelForCausalLM.from_pretrained(
            encoder_path,
            torch_dtype=self._get_dtype(),
            local_files_only=use_local,
        ).to(self.device)
        
        self.text_encoder.requires_grad_(False)  # Freeze text encoder
    
    def _load_vae(self):
        """Load the VAE for encoding images to latents."""
        from diffusers import AutoencoderKL
        
        # VAE path - inside the model directory
        model_path = Path(self.config.model_path)
        vae_path = model_path / "vae"
        
        if not vae_path.exists():
            # Try parent directory structure
            vae_path = model_path.parent / "vae"
        
        if not vae_path.exists():
            # Fallback to HuggingFace (with warning)
            print("⚠️ Local VAE not found, downloading from HuggingFace...")
            vae_path = "stabilityai/sdxl-vae"
        else:
            print(f"✓ Using local VAE: {vae_path}")
        
        self.vae = AutoencoderKL.from_pretrained(
            vae_path,
            torch_dtype=self._get_dtype(),
            local_files_only=isinstance(vae_path, Path),
        ).to(self.device)
        
        self.vae.requires_grad_(False)  # Freeze VAE
    
    def _setup_scheduler(self):
        """Set up the flow matching scheduler."""
        from diffusers import FlowMatchEulerDiscreteScheduler
        
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.config.num_train_timesteps,
            shift=self.config.shift,
            use_dynamic_shifting=self.config.use_dynamic_shifting,
        )
    
    def _setup_lora(self):
        """Set up LoRA network on the transformer."""
        target_modules = self.config.lora.target_modules or get_target_modules_for_zimage()
        
        self.lora_network = LoRANetwork(
            model=self.model,
            target_modules=target_modules,
            rank=self.config.lora.rank,
            alpha=self.config.lora.alpha,
            dropout=self.config.lora.dropout,
            use_dora=self.config.lora.use_dora,
        )
        
        print(f"LoRA initialized with rank {self.config.lora.rank}, "
              f"alpha {self.config.lora.alpha}")
        print(f"Target modules: {target_modules}")
        print(f"Trainable parameters: {format_params(self.lora_network.num_trainable_params())}")
    
    def _setup_optimizer(self):
        """Set up optimizer and learning rate scheduler."""
        trainable_params = self.lora_network.get_trainable_params()
        
        # Additional trainable params if training norms
        if self.config.lora.train_norm_layers:
            for name, param in self.model.named_parameters():
                if "norm" in name.lower():
                    param.requires_grad = True
                    trainable_params.append(param)
        
        # Create optimizer
        if self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw8bit":
            try:
                import bitsandbytes as bnb
                self.optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    weight_decay=self.config.weight_decay,
                )
            except ImportError:
                print("bitsandbytes not available, falling back to AdamW")
                self.optimizer = torch.optim.AdamW(
                    trainable_params,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.adam_epsilon,
                    weight_decay=self.config.weight_decay,
                )
        elif self.config.optimizer == "prodigy":
            try:
                from prodigyopt import Prodigy
                self.optimizer = Prodigy(
                    trainable_params,
                    lr=1.0,  # Prodigy sets its own LR
                    weight_decay=self.config.weight_decay,
                )
            except ImportError:
                raise ImportError("prodigyopt not installed. Install with: pip install prodigyopt")
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Learning rate scheduler
        self.lr_scheduler = get_lr_scheduler(
            self.optimizer,
            self.config.lr_scheduler,
            self.config.max_train_steps,
            self.config.lr_warmup_steps,
        )
    
    def _setup_dataset(self):
        """Set up training dataset and dataloader."""
        dataset_config = self.config.dataset
        
        # Load dataset entries
        dataset_manager = DatasetManager()
        
        if not dataset_config.dataset_path:
            raise ValueError("No dataset path specified in config")
        
        # Extract dataset name from path
        dataset_path = Path(dataset_config.dataset_path)
        if dataset_path.is_absolute():
            # Full path provided, load directly
            entries = self._load_entries_from_path(dataset_path)
        else:
            # Relative name, use dataset manager
            entries = dataset_manager.load_dataset(dataset_config.dataset_path)
        
        if not entries:
            raise ValueError(f"No images found in dataset: {dataset_config.dataset_path}")
        
        # Set up bucketing if enabled
        bucket_manager = None
        if dataset_config.use_bucketing:
            bucket_manager = AspectRatioBucket(
                base_resolution=dataset_config.resolution,
                max_resolution=dataset_config.resolution,
                min_resolution=dataset_config.min_resolution,
                step_size=64,
                max_aspect_ratio=dataset_config.max_aspect_ratio,
            )
            print(f"\nAspect ratio bucketing enabled:")
            print(f"  Base resolution: {dataset_config.resolution}px")
            print(f"  Min resolution: {dataset_config.min_resolution}px")
            print(f"  Max aspect ratio: {dataset_config.max_aspect_ratio}")
            print(f"  Available buckets: {len(bucket_manager.buckets)}")
        
        # Create dataset
        dataset = TrainingDataset(
            entries=entries,
            resolution=dataset_config.resolution,
            flip_horizontal=dataset_config.flip_horizontal,
            flip_vertical=dataset_config.flip_vertical,
            random_crop=dataset_config.random_crop,
            bucket_manager=bucket_manager,
        )
        
        # Create dataloader (bucketed or standard)
        if dataset_config.use_bucketing and bucket_manager is not None:
            self.dataloader, self.bucket_sampler = create_bucketed_dataloader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
            # Print bucket distribution
            stats = self.bucket_sampler.get_bucket_stats()
            print(f"\nBucket distribution ({stats['num_buckets']} buckets used):")
            for bucket_name, info in stats["buckets"].items():
                print(f"  {bucket_name}: {info['count']} images, {info['batches']} batches")
        else:
            self.dataloader = create_dataloader(
                dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
            )
            self.bucket_sampler = None
    
    def _load_entries_from_path(self, dataset_path: Path) -> List[ImageEntry]:
        """Load dataset entries from a direct path."""
        entries = []
        images_dir = dataset_path / "images" if (dataset_path / "images").exists() else dataset_path
        
        # Load metadata
        metadata_path = dataset_path / "dataset.json"
        default_caption = ""
        trigger_word = ""
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
                default_caption = metadata.get("default_caption", "")
                trigger_word = metadata.get("trigger_word", "")
        
        # Find images
        from .dataset import SUPPORTED_FORMATS
        for f in images_dir.iterdir():
            if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS:
                # Load caption
                caption_path = f.with_suffix(".txt")
                if caption_path.exists():
                    with open(caption_path) as cf:
                        caption = cf.read().strip()
                else:
                    caption = default_caption
                
                if trigger_word and trigger_word not in caption:
                    caption = f"{trigger_word}, {caption}" if caption else trigger_word
                
                entries.append(ImageEntry(image_path=f, caption=caption))
        
        return entries
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype based on mixed precision config."""
        if self.config.mixed_precision == "fp16":
            return torch.float16
        elif self.config.mixed_precision == "bf16":
            return torch.bfloat16
        return torch.float32
    
    def _encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images to latents using VAE."""
        with torch.no_grad():
            latents = self.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents
    
    def _encode_text(self, captions: List[str]) -> torch.Tensor:
        """Encode captions to text embeddings."""
        with torch.no_grad():
            inputs = self.tokenizer(
                captions,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            
            outputs = self.text_encoder(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True,
            )
            
            # Use last hidden state
            text_embeds = outputs.hidden_states[-1]
        
        return text_embeds
    
    def _compute_loss(
        self,
        latents: torch.Tensor,
        text_embeds: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute flow matching loss.
        
        The loss is computed as:
        loss = MSE(predicted_velocity, target_velocity)
        
        where target_velocity = noise - latents (the direction from latents to noise)
        """
        batch_size = latents.shape[0]
        
        # Add frames dimension if not present: [B, C, H, W] -> [B, C, 1, H, W]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)  # Add F=1 dimension
        
        # Sample noise (same shape as latents)
        noise = torch.randn_like(latents)
        
        # Get sigmas for timesteps (scheduler.sigmas is on CPU, so index with CPU tensor)
        sigmas = self.scheduler.sigmas[timesteps.cpu()].to(self.device).view(-1, 1, 1, 1, 1)
        
        # Create noisy latents (interpolate between latents and noise)
        noisy_latents = (1 - sigmas) * latents + sigmas * noise
        
        # Scale timesteps to [0, 1] as ZImage expects (based on generate_mlx.py)
        # The scheduler gives timesteps in [0, num_train_timesteps]
        # ZImage uses: t = (1000 - t_scheduler) / 1000 for inference
        # For training with random timesteps, just normalize to [0, 1]
        t_normalized = timesteps.float() / self.config.num_train_timesteps
        
        # ZImage expects List[Tensor] for x and cap_feats (one tensor per batch item)
        # Split batch into list of individual samples
        x_list = [noisy_latents[i] for i in range(batch_size)]
        cap_feats_list = [text_embeds[i] for i in range(batch_size)]
        
        # Forward pass
        # ZImage forward returns: (output_list, {}) not an object with .sample
        output, _ = self.model(
            x=x_list,
            t=t_normalized,
            cap_feats=cap_feats_list,
        )
        
        # Reconstruct batch from list output and negate (as done in inference)
        # output is List[Tensor], stack back to batch
        model_pred = torch.stack(output, dim=0)
        model_pred = -model_pred  # Negate like inference does
        
        # Target is the velocity from latents to noise
        target = noise - latents
        
        # Compute loss
        loss = F.mse_loss(model_pred, target, reduction="mean")
        
        return loss
    
    def train(
        self,
        progress_callback: Optional[Callable] = None,
        image_callback: Optional[Callable] = None,
    ):
        """
        Run the training loop.
        
        Args:
            progress_callback: Callback for progress updates
                Signature: callback(step, loss, lr, eta)
            image_callback: Callback for validation images
                Signature: callback(images, step, prompt)
        """
        print("\n" + "="*60)
        print("Starting LoRA Training")
        print("="*60)
        
        self.timer.start()
        self.model.train()
        
        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.config.save(output_dir / "training_config.json")
        
        # Training loop
        total_steps = self.config.max_train_steps
        accumulated_loss = 0.0
        
        progress_bar = tqdm(
            total=total_steps,
            initial=self.global_step,
            desc="Training",
        )
        
        while self.global_step < total_steps:
            for batch in self.dataloader:
                if self.global_step >= total_steps:
                    break
                
                # Move batch to device
                pixel_values = batch["pixel_values"].to(self.device, dtype=self._get_dtype())
                captions = batch["captions"]
                
                # Encode images and text
                with torch.no_grad():
                    latents = self._encode_images(pixel_values)
                    text_embeds = self._encode_text(captions)
                
                # Sample timesteps
                batch_size = latents.shape[0]
                timesteps = torch.randint(
                    0, self.scheduler.config.num_train_timesteps,
                    (batch_size,),
                    device=self.device,
                )
                
                # Forward pass with mixed precision
                if self.scaler is not None:
                    # CUDA with GradScaler
                    with torch.amp.autocast(device_type="cuda", dtype=self._get_dtype()):
                        loss = self._compute_loss(latents, text_embeds, timesteps)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.config.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                elif self.device.type == "mps" and self.config.mixed_precision != "no":
                    # MPS with manual dtype (no GradScaler support)
                    # Compute loss with model in correct dtype
                    loss = self._compute_loss(latents, text_embeds, timesteps)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                else:
                    # CPU or full precision
                    loss = self._compute_loss(latents, text_embeds, timesteps)
                    loss = loss / self.config.gradient_accumulation_steps
                    loss.backward()
                
                accumulated_loss += loss.item()
                
                # Gradient accumulation step
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.lora_network.get_trainable_params(),
                            self.config.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                self.timer.step()
                
                # Logging
                if self.global_step % self.config.log_every_n_steps == 0:
                    avg_loss = accumulated_loss / self.config.log_every_n_steps
                    lr = self.lr_scheduler.get_last_lr()[0]
                    eta = self.timer.get_eta(self.global_step, total_steps)
                    
                    progress_bar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{lr:.2e}",
                        "ETA": eta,
                    })
                    
                    if progress_callback:
                        progress_callback(self.global_step, avg_loss, lr, eta)
                    
                    accumulated_loss = 0.0
                
                # Save checkpoint
                if self.global_step % self.config.save_every_n_steps == 0:
                    self._save_checkpoint()
                
                # Validation
                if self.global_step % self.config.validation_every_n_steps == 0:
                    if image_callback and self.config.validation_prompts:
                        images = self._generate_validation_images()
                        for i, prompt in enumerate(self.config.validation_prompts):
                            if i < len(images):
                                image_callback(images[i], self.global_step, prompt)
                
                progress_bar.update(1)
        
        progress_bar.close()
        
        # Final save
        self._save_final_lora()
        
        print("\n" + "="*60)
        print(f"Training complete! Total time: {self.timer.elapsed_formatted}")
        print(f"Final LoRA saved to: {output_dir / f'{self.config.output_name}.safetensors'}")
        print("="*60)
    
    def _save_checkpoint(self):
        """Save training checkpoint."""
        output_dir = Path(self.config.output_dir)
        
        # Save LoRA weights
        lora_path = output_dir / f"checkpoint-{self.global_step}" / "lora.safetensors"
        lora_path.parent.mkdir(exist_ok=True)
        self.lora_network.save_weights(str(lora_path))
        
        # Save training state
        save_training_state(
            output_dir=str(lora_path.parent),
            step=self.global_step,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.lr_scheduler.state_dict(),
            config=self.config.to_dict(),
        )
        
        # Clean up old checkpoints
        self._cleanup_checkpoints(output_dir)
        
        print(f"\nCheckpoint saved at step {self.global_step}")
    
    def _cleanup_checkpoints(self, output_dir: Path):
        """Remove old checkpoints to maintain save_total_limit."""
        checkpoints = sorted(
            [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda x: int(x.name.split("-")[1]),
        )
        
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
    
    def _save_final_lora(self):
        """Save the final LoRA weights."""
        output_dir = Path(self.config.output_dir)
        output_path = output_dir / f"{self.config.output_name}.safetensors"
        
        # Prepare metadata
        metadata = {
            "name": self.config.output_name,
            "base_model": str(self.config.model_path),
            "training_steps": str(self.global_step),
            "lora_rank": str(self.config.lora.rank),
            "lora_alpha": str(self.config.lora.alpha),
            "learning_rate": str(self.config.learning_rate),
            "training_date": datetime.now().isoformat(),
            "used_training_adapter": str(self.config.use_training_adapter),
        }
        
        self.lora_network.save_weights(str(output_path), metadata=metadata)
    
    def _resume_from_checkpoint(self):
        """Resume training from a checkpoint."""
        checkpoint_path = Path(self.config.resume_from_checkpoint)
        
        # Load LoRA weights
        lora_path = checkpoint_path / "lora.safetensors"
        if lora_path.exists():
            state_dict, _ = LoRANetwork.load_weights(str(lora_path))
            self.lora_network.load_lora_state_dict(state_dict)
        
        # Load training state
        state_path = checkpoint_path / "checkpoint-latest.pt"
        if state_path.exists():
            state = load_training_state(str(state_path))
            self.global_step = state.get("step", 0)
            if "optimizer_state" in state:
                self.optimizer.load_state_dict(state["optimizer_state"])
            if "scheduler_state" in state:
                self.lr_scheduler.load_state_dict(state["scheduler_state"])
        
        print(f"Resumed from checkpoint at step {self.global_step}")
    
    def _generate_validation_images(self) -> List[Any]:
        """Generate validation images for progress tracking."""
        self.model.eval()
        images = []
        
        with torch.no_grad():
            for prompt in self.config.validation_prompts[:self.config.num_validation_images]:
                # Generate image using the current model state
                # This is a simplified version - full pipeline would be more complex
                try:
                    # Encode prompt
                    text_embeds = self._encode_text([prompt])
                    
                    # Generate latents
                    latents = torch.randn(
                        1, 16, 128, 128,
                        device=self.device,
                        dtype=self._get_dtype(),
                    )
                    
                    # Denoise
                    self.scheduler.set_timesteps(self.config.sample_steps)
                    for t in self.scheduler.timesteps:
                        noise_pred = self.model(
                            hidden_states=latents,
                            encoder_hidden_states=text_embeds,
                            timestep=t.unsqueeze(0).to(self.device),
                        ).sample
                        
                        latents = self.scheduler.step(noise_pred, t, latents).prev_sample
                    
                    # Decode
                    latents = latents / self.vae.config.scaling_factor
                    image = self.vae.decode(latents).sample
                    
                    # Convert to PIL
                    image = (image / 2 + 0.5).clamp(0, 1)
                    image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
                    image = (image * 255).astype("uint8")
                    
                    from PIL import Image
                    images.append(Image.fromarray(image))
                    
                except Exception as e:
                    print(f"Warning: Failed to generate validation image: {e}")
        
        self.model.train()
        return images
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "global_step": self.global_step,
            "max_steps": self.config.max_train_steps,
            "progress": self.global_step / self.config.max_train_steps,
            "elapsed_time": self.timer.elapsed_formatted,
            "eta": self.timer.get_eta(self.global_step, self.config.max_train_steps),
            "learning_rate": self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0,
            "lora_params": self.lora_network.num_trainable_params() if self.lora_network else 0,
        }
