"""
Z-Image-Turbo - Gradio Web Interface
Supports both MLX (Apple Silicon) and PyTorch backends
"""
import gradio as gr
import torch
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from transformers import AutoTokenizer
from diffusers import FlowMatchEulerDiscreteScheduler, ZImagePipeline
import json
import random
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global model cache
_mlx_models = None
_pytorch_pipe = None
_current_mlx_model_path = None  # Track which model is currently loaded

# Session image gallery storage
_image_gallery = []  # List of dicts: {"image": PIL.Image, "prompt": str, "seed": int, "png_path": str, "jpg_path": str}

# Model paths - organized by platform
MODELS_DIR = "./models"  # Base directory for all models
MLX_MODELS_DIR = "./models/mlx"  # MLX models directory
PYTORCH_MODELS_DIR = "./models/pytorch"  # PyTorch models directory
PYTORCH_MODEL_PATH = "./models/pytorch/Z-Image-Turbo"  # Default PyTorch model
MLX_MODEL_PATH = "./models/mlx/mlx_model"  # Default MLX model
SINGLE_FILE_MODEL_PATH = "./models/single_file"  # For single-file .safetensors models

# Z-Image-Turbo architecture signature keys
# These patterns identify Z-Image-Turbo compatible models
ZIMAGE_SIGNATURE_KEYS = {
    # Transformer keys (with or without diffusion_ prefix)
    "transformer": [
        ("x_embedder", "diffusion_x_embedder"),
        ("t_embedder", "diffusion_t_embedder"),
        ("layers.0.attention", "diffusion_layers.0.attention"),
        ("final_layer", "diffusion_final_layer"),
        ("context_refiner", "diffusion_context_refiner"),
        ("noise_refiner", "diffusion_noise_refiner"),
    ],
    # Text encoder signature (Qwen2.5-3B based)
    "text_encoder": [
        ("model.layers.0.self_attn", "text_encoders.qwen3_4b"),
    ],
}

# Architecture detection patterns for error messages
KNOWN_ARCHITECTURES = {
    "sdxl": ["down_blocks.0.attentions", "mid_block.attentions", "conditioner.embedders"],
    "sd15": ["model.diffusion_model.input_blocks", "cond_stage_model"],
    "flux": ["double_blocks", "single_blocks", "img_in"],
    "hunyuan": ["pooler", "blocks.0.attn1"],
}

# Default configs for single-file models (when config is not embedded)
DEFAULT_TRANSFORMER_CONFIG = {
    "hidden_size": 3840,
    "num_attention_heads": 30,
    "intermediate_size": 10240,
    "num_hidden_layers": 30,
    "n_refiner_layers": 2,
    "in_channels": 16,
    "text_embed_dim": 2560,
    "patch_size": 2,
    "rope_theta": 256.0,
    "axes_dims": [32, 48, 48],
    "axes_lens": [1536, 512, 512],
}

DEFAULT_VAE_CONFIG = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": [128, 256, 512, 512],
    "layers_per_block": 2,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
}

DEFAULT_TEXT_ENCODER_CONFIG = {
    "hidden_size": 2560,
    "intermediate_size": 13696,
    "max_position_embeddings": 32768,
    "num_attention_heads": 20,
    "num_hidden_layers": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "rms_norm_eps": 1e-5,
    "rope_theta": 1000000.0,
    "use_sliding_window": False,
    "sliding_window": 32768,
    "tie_word_embeddings": True,
}

# Image dimension presets organized by base resolution
# Using traditional photographic/video aspect ratios
# Base resolution is the SMALLER dimension
# Format: {base_resolution: {aspect_ratio_name: (width, height)}}
DIMENSION_PRESETS = {
    "1024": {
        # Square
        "1:1 — 1024×1024": (1024, 1024),
        # Landscape (width is larger)
        "3:2 — 1536×1024 (Landscape)": (1536, 1024),
        "4:3 — 1368×1024 (Landscape)": (1368, 1024),
        "5:4 — 1280×1024 (Landscape)": (1280, 1024),
        "16:9 — 1824×1024 (Landscape)": (1824, 1024),
        "21:9 — 2392×1024 (Landscape)": (2392, 1024),
        # Portrait (height is larger)
        "2:3 — 1024×1536 (Portrait)": (1024, 1536),
        "3:4 — 1024×1368 (Portrait)": (1024, 1368),
        "4:5 — 1024×1280 (Portrait)": (1024, 1280),
        "9:16 — 1024×1824 (Portrait)": (1024, 1824),
        "9:21 — 1024×2392 (Portrait)": (1024, 2392),
    },
    "1280": {
        # Square
        "1:1 — 1280×1280": (1280, 1280),
        # Landscape (width is larger)
        "3:2 — 1920×1280 (Landscape)": (1920, 1280),
        "4:3 — 1712×1280 (Landscape)": (1712, 1280),
        "5:4 — 1600×1280 (Landscape)": (1600, 1280),
        "16:9 — 2280×1280 (Landscape)": (2280, 1280),
        "21:9 — 2992×1280 (Landscape)": (2992, 1280),
        # Portrait (height is larger)
        "2:3 — 1280×1920 (Portrait)": (1280, 1920),
        "3:4 — 1280×1712 (Portrait)": (1280, 1712),
        "4:5 — 1280×1600 (Portrait)": (1280, 1600),
        "9:16 — 1280×2280 (Portrait)": (1280, 2280),
        "9:21 — 1280×2992 (Portrait)": (1280, 2992),
    },
}

# Default values
DEFAULT_BASE_RESOLUTION = "1024"
DEFAULT_ASPECT_RATIO = "1:1 — 1024×1024"


def get_aspect_ratio_choices(base_resolution):
    """Get aspect ratio choices, filtering out section headers"""
    all_items = list(DIMENSION_PRESETS[base_resolution].keys())
    # Filter out None values (section headers) but keep them for display
    return all_items


def detect_model_architecture(keys):
    """
    Detect what architecture a model uses based on its weight keys.
    Returns (architecture_name, is_zimage_compatible, details)
    """
    keys_str = " ".join(keys)
    
    # Check for Z-Image-Turbo signature
    zimage_matches = 0
    for key_pair in ZIMAGE_SIGNATURE_KEYS["transformer"]:
        if any(k in keys_str for k in key_pair):
            zimage_matches += 1
    
    if zimage_matches >= 4:  # Most signature keys found
        # Determine if it's the original format or all-in-one format
        has_diffusion_prefix = any("diffusion_layers" in k for k in keys)
        format_type = "all-in-one (ComfyUI compatible)" if has_diffusion_prefix else "standard"
        return "Z-Image-Turbo", True, f"Format: {format_type}, {len(keys)} parameters"
    
    # Check for other known architectures
    for arch_name, patterns in KNOWN_ARCHITECTURES.items():
        if any(pattern in keys_str for pattern in patterns):
            return arch_name.upper(), False, f"This is a {arch_name.upper()} model, not Z-Image-Turbo"
    
    return "Unknown", False, "Could not identify model architecture"


def validate_mlx_model(model_path):
    """Validate that a model is compatible with Z-Image-Turbo architecture"""
    from safetensors import safe_open
    
    weights_file = Path(model_path) / "weights.safetensors"
    if not weights_file.exists():
        return False, "Missing weights.safetensors"
    
    try:
        with safe_open(str(weights_file), framework="numpy") as f:
            keys = list(f.keys())
            arch_name, is_compatible, details = detect_model_architecture(keys)
            
            if is_compatible:
                return True, f"Z-Image-Turbo compatible ({details})"
            else:
                return False, f"Incompatible: {arch_name} - {details}"
    except Exception as e:
        return False, f"Error reading weights: {str(e)}"


def validate_safetensors_file(file_path):
    """
    Validate a single safetensors file for Z-Image-Turbo compatibility.
    Returns (is_valid, architecture, details, has_all_components)
    """
    from safetensors import safe_open
    
    try:
        with safe_open(str(file_path), framework="numpy") as f:
            keys = list(f.keys())
            
            arch_name, is_compatible, details = detect_model_architecture(keys)
            
            # Check what components are present
            has_transformer = any("layers.0.attention" in k or "diffusion_layers.0.attention" in k for k in keys)
            has_vae = any("decoder." in k or "encoder." in k for k in keys)
            has_text_encoder = any("model.layers" in k or "text_encoders" in k for k in keys)
            
            components = []
            if has_transformer:
                components.append("Transformer")
            if has_vae:
                components.append("VAE")
            if has_text_encoder:
                components.append("Text Encoder")
            
            has_all = has_transformer and has_vae and has_text_encoder
            
            return is_compatible, arch_name, details, components, has_all
    except Exception as e:
        return False, "Error", str(e), [], False


def get_available_mlx_models():
    """Scan the MLX models directory for available models"""
    models_path = Path(MLX_MODELS_DIR)
    available_models = []
    
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)
        return available_models
    
    # Check each subdirectory for MLX model files
    for item in models_path.iterdir():
        if item.is_dir():
            # Check if it has the required MLX model files
            weights_file = item / "weights.safetensors"
            config_file = item / "config.json"
            
            if weights_file.exists() and config_file.exists():
                # Validate the model architecture
                is_valid, reason = validate_mlx_model(item)
                if is_valid:
                    model_name = item.name
                    available_models.append(model_name)
                else:
                    print(f"Skipping incompatible model '{item.name}': {reason}")
    
    return sorted(available_models)


def select_mlx_model(model_name):
    """Select and load a specific MLX model"""
    global _mlx_models, _current_mlx_model_path
    
    if not model_name:
        return False
    
    model_path = Path(MLX_MODELS_DIR) / model_name
    
    if not model_path.exists():
        return False
    
    weights_file = model_path / "weights.safetensors"
    if not weights_file.exists():
        return False
    
    # Validate model compatibility
    is_valid, reason = validate_mlx_model(model_path)
    if not is_valid:
        print(f"Cannot select incompatible model: {reason}")
        return False
    
    # Clear cached model to force reload
    _mlx_models = None
    _current_mlx_model_path = str(model_path)
    
    return True


def get_current_model_name():
    """Get the name of the currently selected model"""
    global _current_mlx_model_path
    
    if _current_mlx_model_path:
        return Path(_current_mlx_model_path).name
    
    # Default to mlx_model if it exists
    if Path(MLX_MODEL_PATH).exists():
        return "mlx_model"
    
    return None


def convert_single_file_to_mlx(single_file_path, output_path=None, progress=None):
    """
    Convert a single-file .safetensors model to MLX multi-file format.
    
    Single-file models should have weights with prefixes like:
    - transformer.* or model.* for the main transformer
    - vae.* for the VAE
    - text_encoder.* for the text encoder
    
    Args:
        single_file_path: Path to the source .safetensors file
        output_path: Where to save the converted model (defaults to MLX_MODELS_DIR/model_name)
        progress: Optional Gradio progress callback
    """
    if output_path is None:
        # Use the source filename as the model name in the MLX directory
        model_name = Path(single_file_path).stem
        output_path = str(Path(MLX_MODELS_DIR) / model_name)
    import mlx.core as mx
    from safetensors.torch import load_file
    from tqdm import tqdm
    
    if progress:
        progress(0.1, desc="Loading single-file model...")
    
    print(f"Loading single-file model from {single_file_path}...")
    all_weights = load_file(single_file_path)
    
    # Separate weights by component
    transformer_weights = {}
    vae_weights = {}
    text_encoder_weights = {}
    
    if progress:
        progress(0.2, desc="Separating model components...")
    
    for key, value in all_weights.items():
        if key.startswith("transformer.") or key.startswith("model."):
            # Remove prefix
            new_key = key.replace("transformer.", "").replace("model.", "")
            transformer_weights[new_key] = value
        elif key.startswith("vae."):
            new_key = key.replace("vae.", "")
            vae_weights[new_key] = value
        elif key.startswith("text_encoder."):
            new_key = key.replace("text_encoder.", "")
            text_encoder_weights[new_key] = value
        else:
            # Try to infer from key structure
            if "x_embedder" in key or "t_embedder" in key or "final_layer" in key or "transformer_blocks" in key:
                transformer_weights[key] = value
            elif "encoder." in key or "decoder." in key or "quant_conv" in key or "post_quant_conv" in key:
                vae_weights[key] = value
            elif "embed_tokens" in key or "self_attn" in key or "mlp." in key:
                text_encoder_weights[key] = value
            else:
                # Default to transformer
                transformer_weights[key] = value
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert transformer weights
    if progress:
        progress(0.3, desc="Converting transformer weights...")
    
    if transformer_weights:
        print(f"Converting {len(transformer_weights)} transformer tensors...")
        mlx_transformer = {}
        for key, value in tqdm(transformer_weights.items(), desc="Transformer"):
            new_key = key
            # Apply key mappings (same as convert_to_mlx.py)
            if key.startswith("all_final_layer.2-1."):
                new_key = key.replace("all_final_layer.2-1.", "final_layer.")
            elif key.startswith("all_x_embedder.2-1."):
                new_key = key.replace("all_x_embedder.2-1.", "x_embedder.")
            elif "t_embedder.mlp." in key:
                new_key = key.replace("t_embedder.mlp.", "t_embedder.mlp.layers.")
            if "adaLN_modulation.0." in new_key:
                new_key = new_key.replace("adaLN_modulation.0.", "adaLN_modulation.")
            if "adaLN_modulation.1." in new_key:
                new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.")
            if "to_out.0." in new_key:
                new_key = new_key.replace("to_out.0.", "to_out.")
            if "cap_embedder.0." in new_key:
                new_key = new_key.replace("cap_embedder.0.", "cap_embedder.layers.0.")
            if "cap_embedder.1." in new_key:
                new_key = new_key.replace("cap_embedder.1.", "cap_embedder.layers.1.")
            
            # Convert to float32 first (handles bfloat16), then to float16 for MLX
            mlx_transformer[new_key] = mx.array(value.float().numpy().astype("float16"))
        
        mx.save_safetensors(str(output_dir / "weights.safetensors"), mlx_transformer)
        print(f"Saved transformer weights to {output_dir / 'weights.safetensors'}")
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
    
    # Convert VAE weights
    if progress:
        progress(0.5, desc="Converting VAE weights...")
    
    if vae_weights:
        print(f"Converting {len(vae_weights)} VAE tensors...")
        mlx_vae = {}
        layers_per_block = DEFAULT_VAE_CONFIG["layers_per_block"]
        
        for key, value in tqdm(vae_weights.items(), desc="VAE"):
            new_key = key
            
            # Map down_blocks/up_blocks structure
            if "down_blocks" in new_key:
                if "resnets" in new_key:
                    parts = new_key.split(".")
                    block_idx = parts.index("down_blocks") + 1
                    resnet_pos = parts.index("resnets")
                    parts[resnet_pos] = "layers"
                    new_key = ".".join(parts)
                elif "downsamplers" in new_key:
                    parts = new_key.split(".")
                    ds_pos = parts.index("downsamplers")
                    parts[ds_pos] = "layers"
                    parts[ds_pos + 1] = str(layers_per_block)
                    new_key = ".".join(parts)
            
            if "up_blocks" in new_key:
                if "resnets" in new_key:
                    parts = new_key.split(".")
                    resnet_pos = parts.index("resnets")
                    parts[resnet_pos] = "layers"
                    new_key = ".".join(parts)
                elif "upsamplers" in new_key:
                    parts = new_key.split(".")
                    us_pos = parts.index("upsamplers")
                    parts[us_pos] = "layers"
                    parts[us_pos + 1] = str(layers_per_block + 1)
                    new_key = ".".join(parts)
            
            if "mid_block" in new_key:
                if "resnets.0" in new_key:
                    new_key = new_key.replace("resnets.0", "layers.0")
                elif "attentions.0" in new_key:
                    new_key = new_key.replace("attentions.0", "layers.1")
                elif "resnets.1" in new_key:
                    new_key = new_key.replace("resnets.1", "layers.2")
            
            if "to_out.0" in new_key:
                new_key = new_key.replace("to_out.0", "to_out")
            
            # Transpose conv weights
            if "conv" in new_key and "weight" in new_key and len(value.shape) == 4:
                value = value.permute(0, 2, 3, 1)
            
            # Convert to float32 first (handles bfloat16)
            mlx_vae[new_key] = mx.array(value.float().numpy())
        
        mx.save_safetensors(str(output_dir / "vae.safetensors"), mlx_vae)
        with open(output_dir / "vae_config.json", "w") as f:
            json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
        print(f"Saved VAE weights to {output_dir / 'vae.safetensors'}")
    
    # Convert text encoder weights
    if progress:
        progress(0.7, desc="Converting text encoder weights...")
    
    if text_encoder_weights:
        print(f"Converting {len(text_encoder_weights)} text encoder tensors...")
        mlx_te = {}
        for key, value in tqdm(text_encoder_weights.items(), desc="Text Encoder"):
            # Convert to float32 first (handles bfloat16)
            mlx_te[key] = mx.array(value.float().numpy())
        
        mx.save_safetensors(str(output_dir / "text_encoder.safetensors"), mlx_te)
        with open(output_dir / "text_encoder_config.json", "w") as f:
            json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
        print(f"Saved text encoder weights to {output_dir / 'text_encoder.safetensors'}")
    
    # Create default tokenizer and scheduler configs if they don't exist
    if progress:
        progress(0.9, desc="Setting up tokenizer and scheduler...")
    
    tokenizer_dir = output_dir / "tokenizer"
    scheduler_dir = output_dir / "scheduler"
    
    # Check if we need to copy from HF model or create defaults
    hf_tokenizer = Path(PYTORCH_MODEL_PATH) / "tokenizer"
    hf_scheduler = Path(PYTORCH_MODEL_PATH) / "scheduler"
    
    if hf_tokenizer.exists() and not tokenizer_dir.exists():
        shutil.copytree(hf_tokenizer, tokenizer_dir)
        print(f"Copied tokenizer from {hf_tokenizer}")
    elif not tokenizer_dir.exists():
        print("Warning: Tokenizer not found. You may need to download it separately.")
    
    if hf_scheduler.exists() and not scheduler_dir.exists():
        shutil.copytree(hf_scheduler, scheduler_dir)
        print(f"Copied scheduler from {hf_scheduler}")
    elif not scheduler_dir.exists():
        # Create default scheduler config
        scheduler_dir.mkdir(parents=True, exist_ok=True)
        scheduler_config = {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "max_image_seq_len": 4096,
            "max_shift": 1.15,
            "num_train_timesteps": 1000,
            "shift": 3.0,
        }
        with open(scheduler_dir / "scheduler_config.json", "w") as f:
            json.dump(scheduler_config, f, indent=2)
        print("Created default scheduler config")
    
    if progress:
        progress(1.0, desc="Conversion complete!")
    
    print(f"\n✓ Single-file model converted to {output_dir}")
    return True


def check_and_setup_models():
    """Check if models exist, download and convert if necessary."""
    from convert_to_mlx import ensure_model_downloaded
    
    # Ensure directory structure exists
    Path(MLX_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    Path(PYTORCH_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    
    pytorch_path = Path(PYTORCH_MODEL_PATH)
    mlx_path = Path(MLX_MODEL_PATH)
    
    # Check if MLX model exists
    mlx_weights = mlx_path / "weights.safetensors"
    if mlx_weights.exists():
        print(f"✓ MLX model found at {mlx_path}")
        return True
    
    print("MLX model not found. Setting up models...")
    
    # Check if PyTorch model exists, download if not
    transformer_path = pytorch_path / "transformer"
    if not transformer_path.exists() or len(list(transformer_path.glob("*.safetensors"))) == 0:
        print("PyTorch model not found. Downloading from Hugging Face...")
        ensure_model_downloaded(str(pytorch_path))
    else:
        print(f"✓ PyTorch model found at {pytorch_path}")
    
    # Convert to MLX
    print("\nConverting PyTorch model to MLX format...")
    print("This may take a few minutes...\n")
    
    # Import and run conversion
    from convert_to_mlx import convert_weights
    
    # Create output directory
    mlx_path.mkdir(parents=True, exist_ok=True)
    
    # We need to set up args for convert_weights
    class Args:
        model_path = str(pytorch_path / "transformer")
        output_path = str(mlx_path)
    
    # Temporarily set global args for the conversion script
    import convert_to_mlx
    convert_to_mlx.args = Args()
    
    convert_weights(Args.model_path, Args.output_path)
    
    print("\n✓ Model conversion complete!")
    return True


def load_mlx_models(model_path=None):
    """Load MLX models (cached globally)"""
    global _mlx_models, _current_mlx_model_path
    
    # Use current selected model path if not specified
    if model_path is None:
        if _current_mlx_model_path:
            model_path = _current_mlx_model_path
        else:
            model_path = MLX_MODEL_PATH
    
    # Check if we need to reload (different model selected)
    if _mlx_models is not None and _current_mlx_model_path == model_path:
        return _mlx_models
    
    # Clear cache if switching models
    if _current_mlx_model_path != model_path:
        _mlx_models = None
        _current_mlx_model_path = model_path
    
    import mlx.core as mx
    from z_image_mlx import ZImageTransformer2DModel
    from vae import AutoencoderKL
    from text_encoder import TextEncoder
    
    print(f"Loading MLX Models from {model_path}...")
    
    # Load Transformer
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model = ZImageTransformer2DModel(config)
    weights = mx.load(f"{model_path}/weights.safetensors")
    
    # Handle weights that may have prefixes (e.g., "diffusion_" from all-in-one safetensors)
    # Strip the prefix to match model parameter names
    processed_weights = []
    for key, value in weights.items():
        new_key = key
        # Strip common prefixes from all-in-one format
        if key.startswith("diffusion_"):
            new_key = key[len("diffusion_"):]
        processed_weights.append((new_key, value))
    
    model.load_weights(processed_weights)
    model.eval()
    
    # Load VAE
    with open(f"{model_path}/vae_config.json", "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL(vae_config)
    vae_weights = mx.load(f"{model_path}/vae.safetensors")
    vae.load_weights(list(vae_weights.items()), strict=False)
    vae.eval()
    
    # Load Text Encoder
    with open(f"{model_path}/text_encoder_config.json", "r") as f:
        te_config = json.load(f)
    text_encoder = TextEncoder(te_config)
    te_weights = mx.load(f"{model_path}/text_encoder.safetensors")
    
    # Handle text encoder weights that may have prefixes
    processed_te_weights = []
    for key, value in te_weights.items():
        new_key = key
        # Strip text encoder prefixes from all-in-one format
        if key.startswith("text_encoders.qwen3_4b.transformer."):
            new_key = key[len("text_encoders.qwen3_4b.transformer."):]
        elif key.startswith("text_encoders.qwen3_4b."):
            new_key = key[len("text_encoders.qwen3_4b."):]
        processed_te_weights.append((new_key, value))
    
    text_encoder.load_weights(processed_te_weights, strict=False)
    text_encoder.eval()
    
    # Load Tokenizer and Scheduler
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}/tokenizer", trust_remote_code=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{model_path}/scheduler")
    
    _mlx_models = {
        "model": model,
        "vae": vae,
        "vae_config": vae_config,
        "text_encoder": text_encoder,
        "tokenizer": tokenizer,
        "scheduler": scheduler,
    }
    
    print("MLX Models loaded successfully!")
    return _mlx_models


def load_pytorch_pipeline(model_path=None):
    """Load PyTorch pipeline (cached globally)"""
    global _pytorch_pipe
    
    if model_path is None:
        model_path = PYTORCH_MODEL_PATH
    
    if _pytorch_pipe is not None:
        return _pytorch_pipe
    
    print("Loading PyTorch Pipeline...")
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    _pytorch_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    _pytorch_pipe.to(device)
    
    print("PyTorch Pipeline loaded successfully!")
    return _pytorch_pipe


# Global cache for prompt enhancer model
_prompt_enhancer = None

PROMPT_ENHANCER_PATH = "./models/prompt_enhancer"
PROMPT_ENHANCER_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"


def load_prompt_enhancer():
    """Load prompt enhancer model, downloading if necessary"""
    global _prompt_enhancer
    
    if _prompt_enhancer is not None:
        return _prompt_enhancer
    
    try:
        from mlx_lm import load
    except ImportError:
        raise gr.Error("mlx-lm not installed. Run: pip install mlx-lm")
    
    enhancer_path = Path(PROMPT_ENHANCER_PATH)
    
    # Check if local model exists
    if enhancer_path.exists() and (enhancer_path / "config.json").exists():
        print(f"Loading prompt enhancer from {PROMPT_ENHANCER_PATH}...")
        model, tokenizer = load(str(enhancer_path))
    else:
        # Download and save locally
        print(f"Prompt enhancer not found. Downloading {PROMPT_ENHANCER_MODEL}...")
        print("This will be saved to models/prompt_enhancer/ for future use.")
        
        from huggingface_hub import snapshot_download
        
        # Download to local path
        enhancer_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=PROMPT_ENHANCER_MODEL,
            local_dir=str(enhancer_path),
            local_dir_use_symlinks=False,
        )
        
        print(f"✓ Prompt enhancer saved to {PROMPT_ENHANCER_PATH}")
        model, tokenizer = load(str(enhancer_path))
    
    _prompt_enhancer = (model, tokenizer)
    return _prompt_enhancer


def enhance_prompt(prompt, progress=gr.Progress()):
    """Use MLX-LM to enhance the user's prompt with a small local model"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt to enhance")
    
    progress(0.1, desc="Loading language model...")
    
    try:
        from mlx_lm import generate
    except ImportError:
        raise gr.Error("mlx-lm not installed. Run: pip install mlx-lm")
    
    progress(0.2, desc="Loading prompt enhancer...")
    
    model, tokenizer = load_prompt_enhancer()
    
    progress(0.4, desc="Enhancing prompt...")
    
    # System prompt for enhancement
    system_prompt = """You are an expert at writing detailed image generation prompts. 
When given a simple description, expand it into a detailed prompt that includes:
- Specific visual details (colors, textures, lighting)
- Composition and framing
- Artistic style or mood
- Background and environment details

Keep the enhanced prompt concise but descriptive (under 100 words).
Respond ONLY with the enhanced prompt, no explanations or preamble."""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Enhance this image prompt: {prompt}"}
    ]
    
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    progress(0.5, desc="Generating enhanced prompt...")
    
    # Create sampler with temperature
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.7)
    
    # Generate
    enhanced = generate(
        model,
        tokenizer,
        prompt=prompt_text,
        max_tokens=200,
        sampler=sampler,
        verbose=False,
    )
    
    progress(1.0, desc="Done!")
    
    # Clean up
    enhanced = enhanced.strip()
    
    return enhanced


def generate_mlx(prompt, width, height, steps, time_shift, seed, progress):
    """Generate image using MLX backend"""
    import mlx.core as mx
    
    models = load_mlx_models()
    
    model = models["model"]
    vae = models["vae"]
    vae_config = models["vae_config"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    
    # Update scheduler with time shift
    scheduler.config.shift = time_shift
    
    progress(0.1, desc="Encoding prompt...")
    
    # Apply chat template
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    text_inputs = tokenizer(
        prompt_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="np",
    )
    
    input_ids = mx.array(text_inputs["input_ids"])
    attention_mask = mx.array(text_inputs["attention_mask"])
    
    # Text encoder forward
    prompt_embeds = text_encoder(input_ids, attention_mask=attention_mask)
    prompt_embeds_list = [prompt_embeds[i] for i in range(len(prompt_embeds))]
    
    progress(0.2, desc="Preparing latents...")
    
    # Prepare latents
    num_channels_latents = 16
    batch_size = 1
    vae_scale_factor = 8
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))
    
    # Use PyTorch for reproducible random
    torch.manual_seed(seed)
    latents_pt = torch.randn(batch_size, num_channels_latents, 1, latent_height, latent_width)
    latents = mx.array(latents_pt.numpy())
    
    # Denoising loop
    scheduler.set_timesteps(steps)
    timesteps = scheduler.timesteps
    
    for i, t in enumerate(timesteps):
        progress(0.2 + 0.6 * (i / len(timesteps)), desc=f"Denoising step {i+1}/{steps}...")
        
        # Timestep
        t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
        
        # Forward pass
        noise_pred = model(latents, t_mx, prompt_embeds_list)
        
        # Negate noise prediction (as PyTorch pipeline does)
        noise_pred = -noise_pred
        mx.eval(noise_pred)
        
        # Scheduler step
        noise_pred_sq = noise_pred.squeeze(2)
        latents_sq = latents.squeeze(2)
        
        noise_pred_pt = torch.from_numpy(np.array(noise_pred_sq))
        latents_pt = torch.from_numpy(np.array(latents_sq))
        
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=True)
        latents = mx.array(step_output.prev_sample.numpy())
        latents = latents[:, :, None, :, :]
    
    progress(0.85, desc="Decoding image...")
    
    # Decode latents
    latents = latents.squeeze(2)
    latents = latents.transpose(0, 2, 3, 1)
    
    scaling_factor = vae_config.get("scaling_factor", 0.3611)
    shift_factor = vae_config.get("shift_factor", 0.1159)
    
    latents = (latents / scaling_factor) + shift_factor
    
    image = vae.decode(latents)
    mx.eval(image)
    
    # Convert to PIL
    image = (image / 2 + 0.5)
    image = mx.clip(image, 0, 1)
    image = np.array(image)[0]
    image = (image * 255).round().astype("uint8")
    
    return Image.fromarray(image)


def generate_pytorch(prompt, width, height, steps, time_shift, seed, progress):
    """Generate image using PyTorch backend"""
    pipe = load_pytorch_pipeline()
    
    # Update scheduler with time shift
    pipe.scheduler.config.shift = time_shift
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    progress(0.1, desc="Generating with PyTorch...")
    
    def callback_fn(pipe_obj, step, timestep, callback_kwargs):
        progress(0.1 + 0.8 * (step / steps), desc=f"Denoising step {step+1}/{steps}...")
        return callback_kwargs
    
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(seed),
        callback_on_step_end=callback_fn,
    ).images[0]
    
    return image


def update_aspect_ratios(base_resolution):
    """Update aspect ratio choices based on selected base resolution"""
    choices = get_aspect_ratio_choices(base_resolution)
    # Set value to first non-header choice (the 1:1 square)
    return gr.update(choices=choices, value=choices[0])


def add_to_gallery(image, prompt, seed, png_path, jpg_path, steps, time_shift, backend, width, height):
    """Add a generated image to the gallery"""
    global _image_gallery
    _image_gallery.append({
        "image": image,
        "prompt": prompt,
        "seed": seed,
        "png_path": png_path,
        "jpg_path": jpg_path,
        "steps": steps,
        "time_shift": time_shift,
        "backend": backend,
        "width": width,
        "height": height,
    })
    return [item["image"] for item in _image_gallery]


def get_gallery_images():
    """Get all images in the gallery"""
    return [item["image"] for item in _image_gallery]


def get_selected_image_info(evt: gr.SelectData):
    """Get info for the selected image from gallery"""
    if evt.index < len(_image_gallery):
        item = _image_gallery[evt.index]
        details = f"Seed: {item['seed']}\nSize: {item['width']}×{item['height']}\nSteps: {item['steps']} | Time Shift: {item['time_shift']}\nBackend: {item['backend']}\nIndex: {evt.index + 1} of {len(_image_gallery)}"
        return (
            details,
            item["prompt"],
            evt.index,
            item["png_path"],
            item["jpg_path"],
            item["seed"],
            item["prompt"],
        )
    return "", "", None, "", "", 0, ""


def clear_selection():
    """Clear the selected image info"""
    return None, "", "", "", "", 0, ""


def delete_from_gallery(selected_index):
    """Delete selected image from gallery"""
    global _image_gallery
    if selected_index is not None and 0 <= selected_index < len(_image_gallery):
        _image_gallery.pop(selected_index)
    return [item["image"] for item in _image_gallery], None, "", "", "", "", 0, ""


def clear_gallery():
    """Clear all images from gallery"""
    global _image_gallery
    _image_gallery = []
    return [], None, "", "", "", "", 0, ""


def create_metadata_string(prompt, seed, steps, time_shift, backend, width, height):
    """Create a metadata string for embedding in images"""
    return f"""{prompt}

---
Model: Z-Image-Turbo
Size: {width}×{height}
Seed: {seed}
Steps: {steps}
Time Shift: {time_shift}
Backend: {backend}"""


def save_image_with_metadata(pil_image, filepath, prompt, seed, steps, time_shift, backend, width, height):
    """Save image with embedded metadata"""
    metadata_str = create_metadata_string(prompt, seed, steps, time_shift, backend, width, height)
    
    if filepath.lower().endswith('.png'):
        # PNG metadata using PngInfo
        png_info = PngInfo()
        png_info.add_text("parameters", metadata_str)
        png_info.add_text("prompt", prompt)
        png_info.add_text("seed", str(seed))
        png_info.add_text("steps", str(steps))
        png_info.add_text("time_shift", str(time_shift))
        png_info.add_text("backend", backend)
        png_info.add_text("width", str(width))
        png_info.add_text("height", str(height))
        png_info.add_text("model", "Z-Image-Turbo")
        pil_image.save(filepath, "PNG", pnginfo=png_info)
    else:
        # JPEG metadata using EXIF UserComment
        from PIL.ExifTags import TAGS
        import piexif
        
        # Save image first
        pil_image.save(filepath, "JPEG", quality=95)
        
        # Add EXIF metadata
        try:
            exif_dict = piexif.load(filepath)
            # UserComment field (tag 0x9286)
            user_comment = piexif.helper.UserComment.dump(metadata_str, encoding="unicode")
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
            # ImageDescription
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = metadata_str.encode('utf-8')
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, filepath)
        except Exception as e:
            print(f"Warning: Could not add EXIF metadata to JPEG: {e}")
            # Fallback: just save without EXIF if piexif not available


def save_to_dataset(image_path, prompt, seed, dataset_location, format="png"):
    """Save image and prompt text file to dataset location"""
    if not dataset_location or not dataset_location.strip():
        raise gr.Error("Please specify a dataset save location")
    
    # Expand user path (handles ~)
    dataset_path = Path(dataset_location).expanduser()
    
    # Create directory if it doesn't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp-based filename with seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_base = f"{timestamp}_{seed}"
    
    # Determine extension
    ext = ".png" if format == "png" else ".jpg"
    
    # Copy image to dataset location
    image_dest = dataset_path / f"{filename_base}{ext}"
    shutil.copy2(image_path, image_dest)
    
    # Save prompt to text file
    prompt_dest = dataset_path / f"{filename_base}.txt"
    with open(prompt_dest, "w", encoding="utf-8") as f:
        f.write(prompt)
    
    return f"Saved to {dataset_path}:\n  - {filename_base}{ext}\n  - {filename_base}.txt"


def save_all_to_dataset(dataset_location, format="png"):
    """Save all images in gallery to dataset location"""
    global _image_gallery
    
    if not _image_gallery:
        raise gr.Error("No images in gallery to save")
    
    if not dataset_location or not dataset_location.strip():
        raise gr.Error("Please specify a dataset save location")
    
    saved_count = 0
    for item in _image_gallery:
        image_path = item["png_path"] if format == "png" else item["jpg_path"]
        try:
            save_to_dataset(image_path, item["prompt"], item["seed"], dataset_location, format)
            saved_count += 1
        except Exception as e:
            print(f"Error saving image: {e}")
            continue
    
    dataset_path = Path(dataset_location).expanduser()
    return f"Saved {saved_count} image(s) to {dataset_path}"


def save_selected_or_all(selected_index, png_path, jpg_path, prompt, seed, dataset_location, format="png"):
    """Save selected image or all images if none selected"""
    if selected_index is not None:
        # Save single selected image
        image_path = png_path if format == "png" else jpg_path
        result = save_to_dataset(image_path, prompt, seed, dataset_location, format)
        return f"SINGLE IMAGE SAVED:\n{result}"
    else:
        # Save all images
        result = save_all_to_dataset(dataset_location, format)
        return f"BATCH SAVE COMPLETE:\n{result}"


def generate_image(prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend, progress=gr.Progress()):
    """Generate an image using selected backend"""
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")
    
    # Get dimensions from preset
    width, height = DIMENSION_PRESETS[base_resolution][aspect_ratio]
    
    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    seed = int(seed)
    
    progress(0, desc=f"Loading {backend} models...")
    
    if backend == "MLX (Apple Silicon)":
        pil_image = generate_mlx(prompt, width, height, steps, time_shift, seed, progress)
    else:
        pil_image = generate_pytorch(prompt, width, height, steps, time_shift, seed, progress)
    
    progress(0.95, desc="Saving temporary files...")
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.gettempdir()
    
    png_path = os.path.join(temp_dir, f"{timestamp}.png")
    jpg_path = os.path.join(temp_dir, f"{timestamp}.jpg")
    
    # Save images with embedded metadata
    save_image_with_metadata(pil_image, png_path, prompt, seed, steps, time_shift, backend, width, height)
    save_image_with_metadata(pil_image, jpg_path, prompt, seed, steps, time_shift, backend, width, height)
    
    progress(1.0, desc="Done!")
    
    # Add to gallery with all generation parameters
    gallery_images = add_to_gallery(pil_image, prompt, seed, png_path, jpg_path, steps, time_shift, backend, width, height)
    
    return gallery_images, png_path, jpg_path, prompt, seed


# Create Gradio interface
with gr.Blocks(title="Z-Image-Turbo") as demo:
    gr.Markdown(
        """
        # 🎨 Z-Image-Turbo
        
        Generate high-quality images using Z-Image-Turbo with MLX or PyTorch backend.
        """
    )
    
    with gr.Tabs():
        with gr.Tab("🖼️ Generate"):
            with gr.Row():
                with gr.Column(scale=1):
                    prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="Describe the image you want to generate...",
                        lines=5,
                        max_lines=10,
                    )
                    
                    enhance_btn = gr.Button("✨ Enhance Prompt", variant="secondary", size="sm")
                    
                    with gr.Row():
                        base_resolution = gr.Dropdown(
                            choices=list(DIMENSION_PRESETS.keys()),
                            value=DEFAULT_BASE_RESOLUTION,
                            label="Resolution Category",
                        )
                        
                        aspect_ratio = gr.Dropdown(
                            choices=get_aspect_ratio_choices(DEFAULT_BASE_RESOLUTION),
                            value=DEFAULT_ASPECT_RATIO,
                            label="Width × Height (Ratio)",
                            interactive=True,
                        )
                    
                    with gr.Row():
                        backend = gr.Dropdown(
                            choices=["MLX (Apple Silicon)", "PyTorch"],
                            value="MLX (Apple Silicon)",
                            label="Backend",
                        )
                    
                    with gr.Row():
                        steps = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=9,
                            step=1,
                            label="Inference Steps",
                            info="More steps = better quality but slower (9 recommended)",
                        )
                        
                        time_shift = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=3.0,
                            step=0.1,
                            label="Time Shift",
                            info="Scheduler shift parameter (default: 3.0)",
                        )
                    
                    with gr.Row():
                        seed = gr.Slider(
                            minimum=-1,
                            maximum=2147483647,
                            value=-1,
                            step=1,
                            label="Seed",
                            info="-1 for random seed",
                        )
                        random_seed_checkbox = gr.Checkbox(
                            label="Random Seed",
                            value=True,
                        )
                    
                    generate_btn = gr.Button("🚀 Generate", variant="primary", size="lg")
                    
                    with gr.Column(visible=False) as dataset_section:
                        gr.Markdown("---\n### 💾 Save to Dataset")
                        
                        dataset_location = gr.Textbox(
                            label="Dataset Save Location",
                            placeholder="e.g., ~/Documents/datasets/z-image or /Users/you/datasets/z-image",
                            info="Images and prompts will be saved here for training datasets (you can drag a folder here)",
                        )
                        
                        with gr.Row():
                            save_png_btn = gr.Button("💾 Save to Dataset (PNG)", variant="secondary")
                            save_jpg_btn = gr.Button("💾 Save to Dataset (JPG)", variant="secondary")
                        
                        gr.Markdown("*Tip: Select an image to save just that one, or save all if none selected*")
                        
                        save_status = gr.Textbox(label="Save Status", interactive=False, lines=3, max_lines=10)
                
                with gr.Column(scale=1):
                    output_gallery = gr.Gallery(
                        label="Generated Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=2,
                        rows=2,
                        height="auto",
                        object_fit="contain",
                    )
                    
                    with gr.Row():
                        clear_selection_btn = gr.Button("↩️ Deselect (for Batch Save)", variant="secondary", size="sm")
                        delete_selected_btn = gr.Button("❌ Delete Selected", variant="stop", size="sm")
                        clear_gallery_btn = gr.Button("🗑️ Clear All", variant="stop", size="sm")
                    
                    with gr.Accordion("Selected Image Info", open=True):
                        selected_info_display = gr.Textbox(label="Generation Details", interactive=False, lines=5)
                        selected_prompt_display = gr.Textbox(label="Prompt", interactive=False, lines=4)
            
            # Example prompts
            gr.Examples(
                examples=[
                    ["A majestic lion with a flowing golden mane, photorealistic, dramatic lighting"],
                    ["A cozy coffee shop interior with warm lighting, watercolor style"],
                    ["A futuristic cityscape at sunset, cyberpunk aesthetic, neon lights"],
                    ["A serene Japanese garden with cherry blossoms, traditional ink painting style"],
                    ["An astronaut floating in space with Earth in the background, cinematic"],
                ],
                inputs=[prompt],
                label="Example Prompts",
            )
        
        with gr.Tab("⚙️ Model Settings"):
            gr.Markdown(
                """
                ### Z-Image-Turbo Model Management
                
                Load and manage Z-Image-Turbo models and compatible fine-tunes.
                
                **Supported formats:**
                - Fine-tuned models from Hugging Face (diffusers format)
                - Single-file checkpoints (.safetensors) - ComfyUI compatible
                - Models are converted to MLX format for Apple Silicon acceleration
                """
            )
            
            gr.Markdown("---")
            gr.Markdown("#### 🎯 Select Active Model")
            with gr.Row():
                model_selector = gr.Dropdown(
                    choices=get_available_mlx_models(),
                    value=get_current_model_name(),
                    label="Available MLX Models",
                    info="Models in ./models/mlx/ - select one for generation",
                    scale=3,
                )
                refresh_models_btn = gr.Button("🔄 Refresh", scale=1)
            select_model_status = gr.Textbox(label="Selection Status", interactive=False, lines=2)
            
            gr.Markdown("---")
            gr.Markdown("#### 📥 Import from Hugging Face")
            gr.Markdown(
                """
                <small>Download Z-Image-Turbo or compatible fine-tuned models from Hugging Face Hub.
                The model will be downloaded and converted to MLX format.</small>
                """
            )
            with gr.Row():
                hf_model_id = gr.Textbox(
                    label="Hugging Face Model ID",
                    value="Tongyi-MAI/Z-Image-Turbo",
                    info="e.g., Tongyi-MAI/Z-Image-Turbo or username/model-name",
                    scale=2,
                )
                hf_model_name = gr.Textbox(
                    label="Save As (optional)",
                    placeholder="Leave blank to use repo name",
                    info="Custom name for the imported model",
                    scale=1,
                )
            with gr.Row():
                download_hf_btn = gr.Button("⬇️ Download & Convert to MLX", variant="primary")
                download_pytorch_btn = gr.Button("⬇️ Download (PyTorch only)", variant="secondary")
            hf_status = gr.Textbox(label="Import Status", interactive=False, lines=4)
            
            gr.Markdown("---")
            gr.Markdown("#### 📁 Import Single-File Checkpoint")
            gr.Markdown(
                """
                <small>Import a single .safetensors checkpoint file (ComfyUI format).
                Only Z-Image-Turbo architecture checkpoints are supported.</small>
                """
            )
            with gr.Row():
                single_file_path = gr.Textbox(
                    label="Checkpoint File Path",
                    placeholder="e.g., /path/to/z-image-finetune.safetensors",
                    info="Full path to the .safetensors file",
                    scale=2,
                )
                validate_btn = gr.Button("🔍 Validate", variant="secondary", scale=0)
            validate_status = gr.Textbox(label="Validation Result", interactive=False, lines=3)
            
            with gr.Row():
                model_output_name = gr.Textbox(
                    label="Model Name",
                    placeholder="e.g., my_custom_model",
                    info="Name for the imported model",
                    scale=1,
                )
                convert_format = gr.Radio(
                    choices=["MLX (Apple Silicon)", "PyTorch"],
                    value="MLX (Apple Silicon)",
                    label="Target Format",
                    scale=1,
                )
            convert_single_btn = gr.Button("📦 Import Checkpoint", variant="primary")
            convert_status = gr.Textbox(label="Import Status", interactive=False, lines=4)
            
            gr.Markdown("---")
            gr.Markdown("#### 📊 Installed Models")
            model_status = gr.Textbox(label="Model Inventory", interactive=False, lines=8, value="Checking...")
            refresh_status_btn = gr.Button("🔄 Refresh Inventory", variant="secondary")
    
    # Hidden state to store temporary paths, prompt, seed, and selected index
    temp_png_path = gr.State()
    temp_jpg_path = gr.State()
    stored_prompt = gr.State()
    stored_seed = gr.State()
    selected_index = gr.State(value=None)
    
    # --- Event Handlers ---
    
    # Update aspect ratio choices when base resolution changes
    base_resolution.change(
        fn=update_aspect_ratios,
        inputs=[base_resolution],
        outputs=[aspect_ratio],
    )
    
    # Toggle seed slider based on random checkbox
    def toggle_seed(random_checked):
        return gr.update(value=-1 if random_checked else 42, interactive=not random_checked)
    
    random_seed_checkbox.change(
        fn=toggle_seed,
        inputs=[random_seed_checkbox],
        outputs=[seed],
    )
    
    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[prompt],
        outputs=[prompt],
    )
    
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend],
        outputs=[output_gallery, temp_png_path, temp_jpg_path, stored_prompt, stored_seed],
    ).then(
        fn=lambda: (gr.update(visible=True), None, "", "", "", "", 0, ""),
        inputs=None,
        outputs=[dataset_section, selected_index, selected_info_display, selected_prompt_display, temp_png_path, temp_jpg_path, stored_seed, stored_prompt],
    )
    
    # Handle gallery selection
    output_gallery.select(
        fn=get_selected_image_info,
        inputs=None,
        outputs=[selected_info_display, selected_prompt_display, selected_index, temp_png_path, temp_jpg_path, stored_seed, stored_prompt],
    )
    
    # Clear selection
    clear_selection_btn.click(
        fn=clear_selection,
        inputs=None,
        outputs=[selected_index, selected_info_display, selected_prompt_display, temp_png_path, temp_jpg_path, stored_seed, stored_prompt],
    )
    
    # Clear entire gallery
    clear_gallery_btn.click(
        fn=clear_gallery,
        inputs=None,
        outputs=[output_gallery, selected_index, selected_info_display, selected_prompt_display, temp_png_path, temp_jpg_path, stored_seed, stored_prompt],
    )
    
    # Delete selected image
    delete_selected_btn.click(
        fn=delete_from_gallery,
        inputs=[selected_index],
        outputs=[output_gallery, selected_index, selected_info_display, selected_prompt_display, temp_png_path, temp_jpg_path, stored_seed, stored_prompt],
    )
    
    # Save PNG to dataset (selected or all)
    save_png_btn.click(
        fn=save_selected_or_all,
        inputs=[selected_index, temp_png_path, temp_jpg_path, stored_prompt, stored_seed, dataset_location],
        outputs=[save_status],
    )
    
    # Save JPG to dataset (selected or all)
    save_jpg_btn.click(
        fn=lambda idx, png, jpg, p, s, loc: save_selected_or_all(idx, png, jpg, p, s, loc, "jpg"),
        inputs=[selected_index, temp_png_path, temp_jpg_path, stored_prompt, stored_seed, dataset_location],
        outputs=[save_status],
    )
    
    # --- Model Settings Event Handlers ---
    
    def check_model_status():
        """Check the status of installed models with details"""
        status_lines = []
        
        # Show currently selected model
        current_model = get_current_model_name()
        status_lines.append(f"🎯 Active Model: {current_model or 'None selected'}")
        status_lines.append("")
        
        # List MLX models
        mlx_models = get_available_mlx_models()
        status_lines.append(f"📦 MLX Models ({len(mlx_models)}):")
        if mlx_models:
            for model in mlx_models:
                marker = " ← active" if model == current_model else ""
                # Get validation details
                model_path = Path(MLX_MODELS_DIR) / model
                valid, details = validate_mlx_model(model_path)
                status_lines.append(f"  • {model}{marker}")
                status_lines.append(f"    {details}")
        else:
            status_lines.append("  (none)")
        
        # List PyTorch models
        pytorch_path = Path(PYTORCH_MODELS_DIR)
        status_lines.append("")
        if pytorch_path.exists():
            pytorch_models = [d.name for d in pytorch_path.iterdir() if d.is_dir() and (d / "transformer").exists()]
            status_lines.append(f"🔥 PyTorch Models ({len(pytorch_models)}):")
            if pytorch_models:
                for model in pytorch_models:
                    status_lines.append(f"  • {model}")
            else:
                status_lines.append("  (none)")
        
        return "\n".join(status_lines)
    
    def validate_checkpoint_file(file_path):
        """Validate a checkpoint file and return detailed info"""
        if not file_path or not file_path.strip():
            return "Enter a file path and click Validate"
        
        file_path = Path(file_path.strip()).expanduser()
        if not file_path.exists():
            return f"❌ File not found: {file_path}"
        
        if not str(file_path).endswith(".safetensors"):
            return "❌ File must be a .safetensors file"
        
        is_compatible, arch_name, details, components, has_all = validate_safetensors_file(file_path)
        
        lines = []
        if is_compatible:
            lines.append(f"✅ Compatible: {arch_name}")
            lines.append(f"   {details}")
            lines.append(f"   Components: {', '.join(components)}")
            if has_all:
                lines.append("   ✓ All components present (Transformer, VAE, Text Encoder)")
            else:
                lines.append(f"   ⚠️ Missing components - may need separate files")
        else:
            lines.append(f"❌ Incompatible: {arch_name}")
            lines.append(f"   {details}")
            lines.append("")
            lines.append("This tool only supports Z-Image-Turbo architecture models.")
        
        return "\n".join(lines)
    
    def download_from_hf(model_id, custom_name, progress=gr.Progress()):
        """Download model from Hugging Face and convert to MLX"""
        if not model_id or not model_id.strip():
            return "❌ Please enter a Hugging Face model ID"
        
        model_id = model_id.strip()
        
        # Determine model name
        if custom_name and custom_name.strip():
            model_name = custom_name.strip().replace(" ", "_")
        else:
            model_name = model_id.split("/")[-1].replace(" ", "_")
        
        try:
            from convert_to_mlx import ensure_model_downloaded, convert_weights
            import convert_to_mlx
            
            # Ensure directories exist
            Path(MLX_MODELS_DIR).mkdir(parents=True, exist_ok=True)
            Path(PYTORCH_MODELS_DIR).mkdir(parents=True, exist_ok=True)
            
            pytorch_save_path = str(Path(PYTORCH_MODELS_DIR) / model_name)
            mlx_save_path = str(Path(MLX_MODELS_DIR) / model_name)
            
            progress(0.1, desc=f"Downloading {model_id}...")
            
            # Download from HuggingFace
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=pytorch_save_path,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )
            
            progress(0.5, desc="Converting to MLX format...")
            
            # Convert to MLX
            class Args:
                pass
            Args.model_path = str(Path(pytorch_save_path) / "transformer")
            Args.output_path = mlx_save_path
            
            convert_to_mlx.args = Args()
            convert_weights(Args.model_path, Args.output_path)
            
            progress(1.0, desc="Complete!")
            return f"✅ Successfully imported: {model_name}\n\n• PyTorch: {pytorch_save_path}\n• MLX: {mlx_save_path}\n\nYou can now select '{model_name}' from the model dropdown."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def download_pytorch_only(model_id, custom_name, progress=gr.Progress()):
        """Download model from Hugging Face without MLX conversion"""
        if not model_id or not model_id.strip():
            return "❌ Please enter a Hugging Face model ID"
        
        model_id = model_id.strip()
        
        # Determine model name
        if custom_name and custom_name.strip():
            model_name = custom_name.strip().replace(" ", "_")
        else:
            model_name = model_id.split("/")[-1].replace(" ", "_")
        
        try:
            Path(PYTORCH_MODELS_DIR).mkdir(parents=True, exist_ok=True)
            pytorch_save_path = str(Path(PYTORCH_MODELS_DIR) / model_name)
            
            progress(0.1, desc=f"Downloading {model_id}...")
            
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=model_id,
                local_dir=pytorch_save_path,
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )
            
            progress(1.0, desc="Complete!")
            return f"✅ Downloaded: {model_name}\n\nSaved to: {pytorch_save_path}\n\n⚠️ Note: PyTorch models require conversion to MLX for use with the MLX backend."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def convert_single_file(file_path, output_name, target_format, progress=gr.Progress()):
        """Import single-file safetensors checkpoint"""
        if not file_path or not file_path.strip():
            return "❌ Please provide a path to the .safetensors file"
        
        file_path = Path(file_path.strip()).expanduser()
        if not file_path.exists():
            return f"❌ File not found: {file_path}"
        
        if not str(file_path).endswith(".safetensors"):
            return "❌ File must be a .safetensors file"
        
        # Determine output name
        if output_name and output_name.strip():
            model_name = output_name.strip().replace(" ", "_")
        else:
            model_name = file_path.stem.replace(" ", "_")
        
        # First, validate the model
        progress(0.05, desc="Validating model architecture...")
        is_compatible, arch_name, details, components, has_all = validate_safetensors_file(file_path)
        
        if not is_compatible:
            return f"❌ Incompatible model architecture!\n\nDetected: {arch_name}\n{details}\n\nOnly Z-Image-Turbo compatible models can be imported."
        
        if not has_all:
            return f"❌ Incomplete checkpoint!\n\nFound components: {', '.join(components)}\n\nAll-in-one checkpoints must contain Transformer, VAE, and Text Encoder."
        
        # Determine target path based on format
        is_mlx = "MLX" in target_format
        if is_mlx:
            output_path = str(Path(MLX_MODELS_DIR) / model_name)
        else:
            output_path = str(Path(PYTORCH_MODELS_DIR) / model_name)
        
        try:
            if is_mlx:
                convert_single_file_to_mlx(str(file_path), output_path, progress)
                return f"✅ Successfully imported: {model_name}\n\nSaved to: {output_path}\n\nYou can now select '{model_name}' from the model dropdown."
            else:
                # For PyTorch, we need to extract and save in diffusers format
                progress(0.5, desc="Extracting checkpoint...")
                # TODO: Implement PyTorch extraction
                return f"⚠️ PyTorch import from single-file is not yet implemented.\n\nPlease use 'MLX (Apple Silicon)' format for now, or download from Hugging Face."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def handle_model_select(model_name):
        """Handle model selection from dropdown"""
        if not model_name:
            return "⚠️ No model selected"
        
        success = select_mlx_model(model_name)
        if success:
            return f"✅ Selected: {model_name}\n\nThe model will be loaded on next generation."
        else:
            return f"❌ Could not select model: {model_name}\n\nModel may be invalid or missing files."
    
    def refresh_model_list():
        """Refresh the list of available models"""
        models = get_available_mlx_models()
        current = get_current_model_name()
        return gr.Dropdown(choices=models, value=current)
    
    # Wire up event handlers
    validate_btn.click(
        fn=validate_checkpoint_file,
        inputs=[single_file_path],
        outputs=[validate_status],
    )
    
    download_hf_btn.click(
        fn=download_from_hf,
        inputs=[hf_model_id, hf_model_name],
        outputs=[hf_status],
    )
    
    download_pytorch_btn.click(
        fn=download_pytorch_only,
        inputs=[hf_model_id, hf_model_name],
        outputs=[hf_status],
    )
    
    convert_single_btn.click(
        fn=convert_single_file,
        inputs=[single_file_path, model_output_name, convert_format],
        outputs=[convert_status],
    )
    
    # Model selector event handlers
    model_selector.change(
        fn=handle_model_select,
        inputs=[model_selector],
        outputs=[select_model_status],
    )
    
    refresh_models_btn.click(
        fn=refresh_model_list,
        inputs=None,
        outputs=[model_selector],
    )
    
    refresh_status_btn.click(
        fn=check_model_status,
        inputs=None,
        outputs=[model_status],
    )
    
    # Initialize model status on load
    demo.load(
        fn=check_model_status,
        inputs=None,
        outputs=[model_status],
    )


if __name__ == "__main__":
    # Check and setup models on startup
    print("\n" + "="*50)
    print("Z-Image-Turbo - Checking models...")
    print("="*50 + "\n")
    
    check_and_setup_models()
    
    print("\n" + "="*50)
    print("Starting Gradio interface...")
    print("="*50 + "\n")
    
    demo.launch()
