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
from diffusers import FlowMatchEulerDiscreteScheduler
try:
    from diffusers import ZImagePipeline
    PYTORCH_AVAILABLE = True
except ImportError:
    ZImagePipeline = None
    PYTORCH_AVAILABLE = False
    print("Warning: ZImagePipeline not available. PyTorch backend disabled.")
    print("To enable PyTorch, install diffusers >= 0.36.0 or the dev version.")
import json
import random
import sys
import os
from pathlib import Path
from datetime import datetime
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Upscaler imports
try:
    from upscaler import get_available_upscalers, load_upscaler, upscale_image as upscale_image_esrgan
    UPSCALER_AVAILABLE = True
except ImportError:
    UPSCALER_AVAILABLE = False
    print("Warning: Upscaler module not available. Install torch to enable upscaling.")

# Global model cache
_mlx_models = None
_pytorch_pipe = None
_upscaler_model = None  # Cached upscaler model
_current_upscaler_name = None  # Track which upscaler is loaded
_current_mlx_model_path = None  # Track which MLX model is currently loaded
_current_pytorch_model_path = None  # Track which PyTorch model is currently loaded
_current_applied_lora = None  # Track which LoRA is currently applied to the cached model

# LoRA state - list of (lora_name, scale) tuples for active LoRAs
_active_loras = []  # Format: [{"name": str, "path": str, "scale": float}, ...]

# Session image gallery storage
_image_gallery = []  # List of dicts: {"image": PIL.Image, "prompt": str, "seed": int, "png_path": str, "jpg_path": str}

# Temp directory for session images
TEMP_DIR = "./temp"

# Model paths - organized by platform
MODELS_DIR = "./models"  # Base directory for all models
MLX_MODELS_DIR = "./models/mlx"  # MLX models directory
PYTORCH_MODELS_DIR = "./models/pytorch"  # PyTorch models directory
PYTORCH_MODEL_PATH = "./models/pytorch/Z-Image-Turbo"  # Default PyTorch model
MLX_MODEL_PATH = "./models/mlx/Z-Image-Turbo-MLX"  # Default MLX model
SINGLE_FILE_MODEL_PATH = "./models/single_file"  # For single-file .safetensors models
LORAS_DIR = "./models/loras"  # LoRA models directory (supports subfolders)
UPSCALERS_DIR = "./models/upscalers"  # ESRGAN upscaler models directory

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
    "_class_name": "AutoencoderKL",
    "act_fn": "silu",
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": [128, 256, 512, 512],
    "down_block_types": [
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D"
    ],
    "up_block_types": [
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D"
    ],
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
    "force_upcast": True,
    "mid_block_add_attention": True,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
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


def map_transformer_key(key: str) -> str:
    """
    Map transformer weight keys from various checkpoint formats to MLX format.
    
    This is the SINGLE SOURCE OF TRUTH for all transformer key mappings.
    All conversion code should use this function.
    
    Handles:
    - ComfyUI all-in-one format (diffusion_* prefix)
    - HuggingFace diffusers format
    - Various naming conventions for the same weights
    """
    new_key = key
    
    # Step 1: Remove common prefixes
    prefixes_to_remove = [
        "model.diffusion_model.",
        "diffusion_model.",
        "diffusion_",
        "transformer.",
        "model.",
    ]
    for prefix in prefixes_to_remove:
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
            break  # Only remove one prefix
    
    # Step 2: Map layer/block naming conventions
    # all_final_layer.2-1.* -> final_layer.*
    if new_key.startswith("all_final_layer.2-1."):
        new_key = new_key.replace("all_final_layer.2-1.", "final_layer.")
    
    # all_x_embedder.2-1.* -> x_embedder.*
    if new_key.startswith("all_x_embedder.2-1."):
        new_key = new_key.replace("all_x_embedder.2-1.", "x_embedder.")
    
    # Step 3: Handle standalone final layer components
    # norm_final.weight should be SKIPPED - MLX uses affine=False LayerNorm
    if "norm_final.weight" in new_key or "final_layer.norm_final.weight" in new_key:
        return None  # Signal to skip this weight
    
    # norm_final.* -> final_layer.norm_final.* (for any other norm_final keys)
    if new_key.startswith("norm_final."):
        new_key = "final_layer." + new_key
    
    # proj_out.* -> final_layer.linear.*
    if new_key.startswith("proj_out."):
        new_key = new_key.replace("proj_out.", "final_layer.linear.")
    
    # Step 4: Handle MLP layer numbering
    # t_embedder.mlp.0.* -> t_embedder.mlp.layers.0.*
    if "t_embedder.mlp." in new_key and ".mlp.layers." not in new_key:
        new_key = new_key.replace("t_embedder.mlp.", "t_embedder.mlp.layers.")
    
    # Step 5: Handle adaLN_modulation (Sequential -> Linear)
    # adaLN_modulation.0.* or adaLN_modulation.1.* -> adaLN_modulation.*
    if "adaLN_modulation.0." in new_key:
        new_key = new_key.replace("adaLN_modulation.0.", "adaLN_modulation.")
    if "adaLN_modulation.1." in new_key:
        new_key = new_key.replace("adaLN_modulation.1.", "adaLN_modulation.")
    
    # Step 6: Handle to_out (Sequential -> Linear)
    # to_out.0.* -> to_out.*
    if "to_out.0." in new_key:
        new_key = new_key.replace("to_out.0.", "to_out.")
    
    # Step 7: Handle cap_embedder (Sequential -> layers)
    # cap_embedder.0.* -> cap_embedder.layers.0.*
    if "cap_embedder.0." in new_key:
        new_key = new_key.replace("cap_embedder.0.", "cap_embedder.layers.0.")
    if "cap_embedder.1." in new_key:
        new_key = new_key.replace("cap_embedder.1.", "cap_embedder.layers.1.")
    
    # Step 8: Handle attention norm naming
    # .attention.q_norm.* -> .attention.norm_q.*
    if ".attention.q_norm." in new_key:
        new_key = new_key.replace(".attention.q_norm.", ".attention.norm_q.")
    if ".attention.k_norm." in new_key:
        new_key = new_key.replace(".attention.k_norm.", ".attention.norm_k.")
    
    # Step 9: Handle attention output projection naming
    # .attention.out.* -> .attention.to_out.*
    if ".attention.out." in new_key:
        new_key = new_key.replace(".attention.out.", ".attention.to_out.")
    
    return new_key


def map_vae_key(key: str, layers_per_block: int = 2) -> str:
    """
    Map VAE weight keys from various checkpoint formats to MLX format.
    
    This is the SINGLE SOURCE OF TRUTH for all VAE key mappings.
    """
    new_key = key
    
    # Remove vae. prefix if present
    if new_key.startswith("vae."):
        new_key = new_key[4:]
    
    # Map down_blocks structure
    # down_blocks.X.resnets.Y.* -> down_blocks.X.layers.Y.*
    # down_blocks.X.downsamplers.0.* -> down_blocks.X.layers.{layers_per_block}.*
    if "down_blocks" in new_key:
        if "resnets" in new_key:
            parts = new_key.split(".")
            if "resnets" in parts:
                resnet_pos = parts.index("resnets")
                parts[resnet_pos] = "layers"
                new_key = ".".join(parts)
        elif "downsamplers" in new_key:
            parts = new_key.split(".")
            if "downsamplers" in parts:
                ds_pos = parts.index("downsamplers")
                parts[ds_pos] = "layers"
                parts[ds_pos + 1] = str(layers_per_block)
                new_key = ".".join(parts)
    
    # Map up_blocks structure
    # up_blocks.X.resnets.Y.* -> up_blocks.X.layers.Y.*
    # up_blocks.X.upsamplers.0.* -> up_blocks.X.layers.{layers_per_block+1}.*
    if "up_blocks" in new_key:
        if "resnets" in new_key:
            parts = new_key.split(".")
            if "resnets" in parts:
                resnet_pos = parts.index("resnets")
                parts[resnet_pos] = "layers"
                new_key = ".".join(parts)
        elif "upsamplers" in new_key:
            parts = new_key.split(".")
            if "upsamplers" in parts:
                us_pos = parts.index("upsamplers")
                parts[us_pos] = "layers"
                parts[us_pos + 1] = str(layers_per_block + 1)
                new_key = ".".join(parts)
    
    # Map mid_block structure
    # mid_block.resnets.0.* -> mid_block.layers.0.*
    # mid_block.attentions.0.* -> mid_block.layers.1.*
    # mid_block.resnets.1.* -> mid_block.layers.2.*
    if "mid_block" in new_key:
        if "resnets.0" in new_key:
            new_key = new_key.replace("resnets.0", "layers.0")
        elif "attentions.0" in new_key:
            new_key = new_key.replace("attentions.0", "layers.1")
        elif "resnets.1" in new_key:
            new_key = new_key.replace("resnets.1", "layers.2")
    
    # Handle attention to_out
    if "to_out.0" in new_key:
        new_key = new_key.replace("to_out.0", "to_out")
    
    return new_key


def map_text_encoder_key(key: str) -> str:
    """
    Map text encoder weight keys from various checkpoint formats to MLX format.
    
    This is the SINGLE SOURCE OF TRUTH for all text encoder key mappings.
    """
    new_key = key
    
    # Remove common prefixes
    prefixes_to_remove = [
        "text_encoders.qwen3_4b.transformer.",
        "text_encoders.qwen3_4b.",
        "text_encoder.transformer.",
        "text_encoder.",
        "transformer.",  # Some checkpoints have this prefix
    ]
    for prefix in prefixes_to_remove:
        if new_key.startswith(prefix):
            new_key = new_key[len(prefix):]
            break
    
    return new_key


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
            # Transformer: look for attention layers (with or without diffusion_ prefix)
            has_transformer = any("layers.0.attention" in k or "diffusion_layers.0.attention" in k for k in keys)
            
            # VAE: look for vae. prefix OR bare decoder./encoder. patterns
            has_vae = any(
                k.startswith("vae.") or 
                "vae.decoder." in k or 
                "vae.encoder." in k or
                (("decoder." in k or "encoder." in k) and not "text_encoder" in k.lower())
                for k in keys
            )
            
            # Text encoder: look for model.layers (Qwen) or text_encoders prefix
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


def get_available_pytorch_models():
    """Scan the PyTorch models directory for available models"""
    models_path = Path(PYTORCH_MODELS_DIR)
    available_models = []
    
    if not models_path.exists():
        models_path.mkdir(parents=True, exist_ok=True)
        return available_models
    
    # Check each subdirectory for PyTorch/Diffusers model files
    for item in models_path.iterdir():
        if item.is_dir():
            # Check for diffusers format (has transformer/, vae/, etc.)
            transformer_dir = item / "transformer"
            if transformer_dir.exists():
                # Check for safetensors files in transformer dir
                if list(transformer_dir.glob("*.safetensors")):
                    available_models.append(item.name)
    
    return sorted(available_models)


def get_available_models_for_backend(backend):
    """Get available models for the specified backend"""
    if backend == "MLX (Apple Silicon)":
        return get_available_mlx_models()
    else:
        return get_available_pytorch_models()


def select_mlx_model(model_name):
    """Select and load a specific MLX model"""
    global _mlx_models, _current_mlx_model_path, _current_applied_lora
    
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
    _current_applied_lora = None  # Reset LoRA when model changes
    
    return True


def select_pytorch_model(model_name):
    """Select a specific PyTorch model"""
    global _pytorch_pipe, _current_pytorch_model_path
    
    if not model_name:
        return False
    
    model_path = Path(PYTORCH_MODELS_DIR) / model_name
    
    if not model_path.exists():
        return False
    
    transformer_dir = model_path / "transformer"
    if not transformer_dir.exists():
        return False
    
    # Clear cached pipeline to force reload with new model
    _pytorch_pipe = None
    _current_pytorch_model_path = str(model_path)
    
    return True


def get_current_model_name(backend="MLX (Apple Silicon)"):
    """Get the name of the currently selected model for a backend"""
    global _current_mlx_model_path, _current_pytorch_model_path
    
    if backend == "MLX (Apple Silicon)":
        if _current_mlx_model_path:
            return Path(_current_mlx_model_path).name
        # Default to MLX_MODEL_PATH if it exists
        if Path(MLX_MODEL_PATH).exists():
            return Path(MLX_MODEL_PATH).name
    else:
        if _current_pytorch_model_path:
            return Path(_current_pytorch_model_path).name
        # Default to PYTORCH_MODEL_PATH if it exists
        if Path(PYTORCH_MODEL_PATH).exists():
            return Path(PYTORCH_MODEL_PATH).name
    
    return None


def convert_single_file_to_mlx(single_file_path, output_path=None, progress=None, precision="FP16", missing_components=None):
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
        precision: "Original", "FP16", or "FP8" - target precision for weights
        missing_components: List of missing components to download from HuggingFace (e.g., ["VAE", "Text Encoder"])
    """
    if missing_components is None:
        missing_components = []
    
    if output_path is None:
        # Use the source filename as the model name in the MLX directory
        model_name = Path(single_file_path).stem
        output_path = str(Path(MLX_MODELS_DIR) / model_name)
    import mlx.core as mx
    import numpy as np
    from safetensors.torch import load_file
    
    # Determine target dtype based on precision
    if precision == "Original":
        target_dtype = None  # Keep original
    elif precision == "FP8":
        target_dtype = "float16"  # We'll quantize after conversion
    else:  # FP16 is default
        target_dtype = "float16"
    
    if progress:
        progress(0.05, desc=f"Loading single-file model (target: {precision})...")
    
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
        elif key.startswith("text_encoder.") or key.startswith("text_encoders."):
            # Handle both text_encoder. and text_encoders.qwen3_4b. formats
            new_key = key.replace("text_encoder.", "").replace("text_encoders.qwen3_4b.", "")
            text_encoder_weights[new_key] = value
        elif key.startswith("diffusion_"):
            # ComfyUI all-in-one format: diffusion_* keys are transformer
            new_key = key.replace("diffusion_", "")
            transformer_weights[new_key] = value
        else:
            # Try to infer from key structure
            if "x_embedder" in key or "t_embedder" in key or "final_layer" in key or "transformer_blocks" in key:
                transformer_weights[key] = value
            elif ("decoder." in key or "quant_conv" in key or "post_quant_conv" in key) and "text" not in key.lower():
                # VAE decoder/encoder but NOT text encoder
                vae_weights[key] = value
            elif "embed_tokens" in key or ("self_attn" in key and "model.layers" in key) or ("mlp." in key and "model.layers" in key):
                text_encoder_weights[key] = value
            else:
                # Default to transformer
                transformer_weights[key] = value
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert transformer weights
    if progress:
        progress(0.2, desc=f"Converting {len(transformer_weights)} transformer weights...")
    
    if transformer_weights:
        mlx_transformer = {}
        total = len(transformer_weights)
        dim = DEFAULT_TRANSFORMER_CONFIG.get("dim", 3840)  # Hidden dimension for QKV split
        
        for i, (key, value) in enumerate(transformer_weights.items()):
            if progress and i % 50 == 0:
                progress(0.2 + 0.25 * (i / total), desc=f"Converting transformer weights ({i}/{total})...")
            
            # Use centralized key mapping
            new_key = map_transformer_key(key)
            
            # Skip weights that should be ignored (e.g., non-affine LayerNorm weights)
            if new_key is None:
                continue
            
            # Handle fused QKV weights - split into separate Q, K, V
            if ".attention.qkv.weight" in new_key:
                base_key = new_key.replace(".qkv.weight", "")
                # QKV is [3*dim, dim], split along first axis
                value_np = value.float().numpy()
                q, k, v = np.split(value_np, 3, axis=0)
                
                if target_dtype == "float16":
                    mlx_transformer[f"{base_key}.to_q.weight"] = mx.array(q.astype("float16"))
                    mlx_transformer[f"{base_key}.to_k.weight"] = mx.array(k.astype("float16"))
                    mlx_transformer[f"{base_key}.to_v.weight"] = mx.array(v.astype("float16"))
                else:
                    mlx_transformer[f"{base_key}.to_q.weight"] = mx.array(q)
                    mlx_transformer[f"{base_key}.to_k.weight"] = mx.array(k)
                    mlx_transformer[f"{base_key}.to_v.weight"] = mx.array(v)
                continue
            
            # Skip pad tokens (model initializes these)
            if "pad_token" in new_key:
                continue
            
            # Convert based on precision setting
            if target_dtype == "float16":
                mlx_transformer[new_key] = mx.array(value.float().numpy().astype("float16"))
            else:
                # Original precision - keep as float32 (safest for compatibility)
                mlx_transformer[new_key] = mx.array(value.float().numpy())
        
        mx.save_safetensors(str(output_dir / "weights.safetensors"), mlx_transformer)
        
        # Save config
        with open(output_dir / "config.json", "w") as f:
            json.dump(DEFAULT_TRANSFORMER_CONFIG, f, indent=2)
    
    # VAE: ComfyUI format uses completely different key names than Diffusers/MLX format
    # (e.g., "decoder.mid.block_1" vs "decoder.mid_block.resnets.0")
    # Rather than complex key mapping, copy from the known-good reference model
    if progress:
        progress(0.45, desc="Copying VAE from reference model...")
    
    source_model = Path("models/mlx/Z-Image-Turbo-MLX")
    vae_copied = False
    
    if source_model.exists():
        vae_src = source_model / "vae.safetensors"
        vae_config_src = source_model / "vae_config.json"
        
        if vae_src.exists():
            shutil.copy(vae_src, output_dir / "vae.safetensors")
            if vae_config_src.exists():
                shutil.copy(vae_config_src, output_dir / "vae_config.json")
            else:
                with open(output_dir / "vae_config.json", "w") as f:
                    json.dump(DEFAULT_VAE_CONFIG, f, indent=2)
            vae_copied = True
            print(f"✓ Copied VAE from {source_model}")
    
    if not vae_copied:
        # Add VAE to missing components so it gets downloaded later
        if "VAE" not in missing_components:
            missing_components.append("VAE")
        print("  Note: VAE will be downloaded from HuggingFace")
    
    # Convert text encoder weights
    if progress:
        progress(0.65, desc=f"Converting {len(text_encoder_weights)} text encoder weights...")
    
    if text_encoder_weights:
        mlx_te = {}
        total = len(text_encoder_weights)
        for i, (key, value) in enumerate(text_encoder_weights.items()):
            if progress and i % 50 == 0:
                progress(0.65 + 0.2 * (i / total), desc=f"Converting text encoder weights ({i}/{total})...")
            
            # Apply key mapping to strip prefixes
            new_key = map_text_encoder_key(key)
            
            # Convert based on precision setting
            if target_dtype == "float16":
                mlx_te[new_key] = mx.array(value.float().numpy().astype("float16"))
            else:
                mlx_te[new_key] = mx.array(value.float().numpy())
        
        mx.save_safetensors(str(output_dir / "text_encoder.safetensors"), mlx_te)
        
        # Copy text encoder config from reference model (has correct architecture)
        source_model = Path("models/mlx/Z-Image-Turbo-MLX")
        te_config_src = source_model / "text_encoder_config.json"
        if te_config_src.exists():
            shutil.copy(te_config_src, output_dir / "text_encoder_config.json")
        else:
            with open(output_dir / "text_encoder_config.json", "w") as f:
                json.dump(DEFAULT_TEXT_ENCODER_CONFIG, f, indent=2)
    
    # Copy missing components from the known-good Z-Image-Turbo-MLX model
    if missing_components:
        if progress:
            progress(0.85, desc=f"Copying missing components: {', '.join(missing_components)}...")
        _copy_missing_components(output_dir, missing_components, progress)
    
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
    
    if hf_scheduler.exists() and not scheduler_dir.exists():
        shutil.copytree(hf_scheduler, scheduler_dir)
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
    
    # Apply FP8 quantization if requested
    if precision == "FP8":
        if progress:
            progress(0.92, desc="Applying FP8 quantization...")
        _apply_fp8_quantization(output_dir, progress)
    
    # Update config to note the precision
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
        config["precision"] = precision
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    if progress:
        progress(1.0, desc="Conversion complete!")
    
    return True


def _apply_fp8_quantization(model_path, progress=None):
    """
    Apply true 8-bit quantization to model weights using MLX's affine quantization.
    
    This saves weights in MLX's native quantized format with:
    - weight: quantized values (uint32 packed)
    - scales: per-group scales (float16)
    - biases: per-group biases (float16)
    
    The model loading code will detect quantized weights and use nn.quantize()
    to convert model layers to QuantizedLinear for inference.
    
    Note: VAE is NOT quantized because its conv layers are incompatible with 
    MLX's quantization (which targets Linear layers). VAE remains FP16.
    
    Typical size reduction: ~50% (transformer quantized, text encoder and VAE stay FP16)
    """
    import mlx.core as mx
    
    model_path = Path(model_path)
    # Only quantize transformer weights - text encoder and VAE stay FP16 for stability
    # Text encoder quantization causes inference issues (zeros output)
    # VAE conv layers are incompatible with MLX quantization
    weight_files = ["weights.safetensors"]
    group_size = 64  # Must match the group_size used in load_mlx_models
    
    for weight_file in weight_files:
        file_path = model_path / weight_file
        if not file_path.exists():
            continue
        
        if progress:
            progress(desc=f"Quantizing {weight_file}...")
        
        orig_size = file_path.stat().st_size
        weights = mx.load(str(file_path))
        quantized_weights = {}
        q_count = 0
        
        for key, value in weights.items():
            # Only quantize weight matrices with compatible dimensions
            can_quantize = (
                len(value.shape) >= 2 and 
                value.shape[-1] >= group_size and 
                value.shape[-1] % group_size == 0 and
                "weight" in key
            )
            
            if can_quantize:
                try:
                    # Quantize with 8-bit affine mode
                    wq, scales, biases = mx.quantize(value, group_size=group_size, bits=8)
                    # Evaluate to force computation before saving
                    mx.eval(wq, scales, biases)
                    
                    # Store in MLX's quantized format
                    quantized_weights[key] = wq
                    quantized_weights[key.replace(".weight", ".scales")] = scales
                    quantized_weights[key.replace(".weight", ".biases")] = biases
                    q_count += 1
                except Exception:
                    # Fall back to keeping original
                    quantized_weights[key] = value
            else:
                # Keep non-weight tensors (biases, norms, etc.) as-is
                quantized_weights[key] = value
        
        mx.save_safetensors(str(file_path), quantized_weights)
        new_size = file_path.stat().st_size
        
        print(f"  Quantized {weight_file}: {q_count} tensors, {orig_size/1e9:.2f}GB -> {new_size/1e9:.2f}GB ({new_size/orig_size:.1%})")


def _copy_missing_components(output_dir, missing_components, progress=None):
    """
    Copy missing model components (VAE, Text Encoder) from the known-good local MLX model.
    
    This uses the pre-converted Z-Image-Turbo-MLX model which has correct mappings,
    avoiding the need to re-download and re-convert from HuggingFace.
    
    Args:
        output_dir: Path to the output MLX model directory
        missing_components: List of missing components ["VAE", "Text Encoder"]
        progress: Optional Gradio progress callback
    """
    output_dir = Path(output_dir)
    source_model = Path("models/mlx/Z-Image-Turbo-MLX")
    
    if not source_model.exists():
        print(f"Warning: Source model {source_model} not found, cannot copy missing components")
        return
    
    if "VAE" in missing_components:
        if progress:
            progress(desc="Copying VAE from Z-Image-Turbo-MLX...")
        
        vae_src = source_model / "vae.safetensors"
        vae_config_src = source_model / "vae_config.json"
        
        if vae_src.exists():
            shutil.copy(vae_src, output_dir / "vae.safetensors")
            if vae_config_src.exists():
                shutil.copy(vae_config_src, output_dir / "vae_config.json")
            print(f"✓ Copied VAE from {source_model}")
        else:
            print(f"Warning: VAE not found at {vae_src}")
    
    if "Text Encoder" in missing_components:
        if progress:
            progress(desc="Copying Text Encoder from Z-Image-Turbo-MLX...")
        
        te_src = source_model / "text_encoder.safetensors"
        te_config_src = source_model / "text_encoder_config.json"
        
        if te_src.exists():
            shutil.copy(te_src, output_dir / "text_encoder.safetensors")
            if te_config_src.exists():
                shutil.copy(te_config_src, output_dir / "text_encoder_config.json")
            print(f"✓ Copied Text Encoder from {source_model}")
        else:
            print(f"Warning: Text Encoder not found at {te_src}")
    
    # Also copy tokenizer, scheduler, and config if not present
    tokenizer_dir = output_dir / "tokenizer"
    scheduler_dir = output_dir / "scheduler"
    config_file = output_dir / "config.json"
    
    if not tokenizer_dir.exists():
        tokenizer_src = source_model / "tokenizer"
        if tokenizer_src.exists():
            shutil.copytree(tokenizer_src, tokenizer_dir)
            print(f"✓ Copied tokenizer from {source_model}")
    
    if not scheduler_dir.exists():
        scheduler_src = source_model / "scheduler"
        if scheduler_src.exists():
            shutil.copytree(scheduler_src, scheduler_dir)
            print(f"✓ Copied scheduler from {source_model}")
    
    if not config_file.exists():
        config_src = source_model / "config.json"
        if config_src.exists():
            shutil.copy(config_src, config_file)
            print(f"✓ Copied config.json from {source_model}")


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


def _is_quantized_weights(weights: dict) -> bool:
    """Check if weights are in quantized format (have .scales and .biases keys)."""
    for key in weights.keys():
        if key.endswith('.scales') or key.endswith('.biases'):
            return True
    return False


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
        global _current_applied_lora
        _current_applied_lora = None  # Reset LoRA when model changes
    
    import mlx.core as mx
    import mlx.nn as nn
    from z_image_mlx import ZImageTransformer2DModel
    from vae import AutoencoderKL
    from text_encoder import TextEncoder
    
    # Load Transformer
    with open(f"{model_path}/config.json", "r") as f:
        config = json.load(f)
    model = ZImageTransformer2DModel(config)
    weights = mx.load(f"{model_path}/weights.safetensors")
    
    # Check if weights are quantized and apply quantization to model if needed
    is_quantized = _is_quantized_weights(weights)
    if is_quantized:
        nn.quantize(model, group_size=64, bits=8)
    
    # Process weights: handle prefixes and key mappings
    # The weights may have:
    # - "diffusion_" prefix (from all-in-one format)
    # - Fused QKV weights (qkv.weight instead of to_q/to_k/to_v)
    # - Different norm names (q_norm vs norm_q)
    # - Different output names (out vs to_out)
    processed_weights = {}
    dim = config.get("hidden_size", 3840)
    
    for key, value in weights.items():
        # Use centralized key mapping (skip for quantized scale/bias keys)
        if key.endswith('.scales') or key.endswith('.biases'):
            processed_weights[key] = value
            continue
        
        new_key = map_transformer_key(key)
        
        # Skip weights that should be ignored (e.g., non-affine LayerNorm weights)
        if new_key is None:
            continue
        
        # Skip pad tokens
        if "pad_token" in new_key:
            continue
        
        # Handle fused QKV weights - split into separate Q, K, V (only for non-quantized)
        if ".attention.qkv.weight" in new_key and not is_quantized:
            base_key = new_key.replace(".qkv.weight", "")
            # QKV is [3*dim, dim], split along first axis
            q, k, v = mx.split(value, 3, axis=0)
            processed_weights[f"{base_key}.to_q.weight"] = q
            processed_weights[f"{base_key}.to_k.weight"] = k
            processed_weights[f"{base_key}.to_v.weight"] = v
            continue
        
        processed_weights[new_key] = value
    
    model.load_weights(list(processed_weights.items()), strict=False)
    model.eval()
    
    # Load VAE
    with open(f"{model_path}/vae_config.json", "r") as f:
        vae_config = json.load(f)
    vae = AutoencoderKL(vae_config)
    vae_weights = mx.load(f"{model_path}/vae.safetensors")
    
    # Check if VAE weights are quantized
    if _is_quantized_weights(vae_weights):
        nn.quantize(vae, group_size=64, bits=8)
    
    vae.load_weights(list(vae_weights.items()), strict=False)
    vae.eval()
    
    # Load Text Encoder (always FP16 - quantization causes inference issues)
    with open(f"{model_path}/text_encoder_config.json", "r") as f:
        te_config = json.load(f)
    text_encoder = TextEncoder(te_config)
    te_weights = mx.load(f"{model_path}/text_encoder.safetensors")
    
    # Note: We do NOT quantize text encoder even if weights look quantized
    # Text encoder quantization causes zeros in output. Keep it FP16.
    
    # Handle text encoder weights using centralized key mapping
    processed_te_weights = []
    for key, value in te_weights.items():
        # Skip keys that don't belong to the text encoder model
        if key == 'logit_scale':
            continue
        
        # Skip quantized scale/bias keys - we don't use them for text encoder
        if key.endswith('.scales') or key.endswith('.biases'):
            continue
        
        new_key = map_text_encoder_key(key)
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
    
    return _mlx_models


def load_pytorch_pipeline(model_path=None):
    """Load PyTorch pipeline (cached globally)"""
    global _pytorch_pipe, _current_pytorch_model_path
    
    if not PYTORCH_AVAILABLE:
        raise ImportError(
            "PyTorch backend not available. ZImagePipeline requires diffusers >= 0.36.0.\\n\"\n            \"Install with: pip install git+https://github.com/huggingface/diffusers.git"
        )
    
    # Use selected model path, or default
    if model_path is None:
        if _current_pytorch_model_path:
            model_path = _current_pytorch_model_path
        else:
            model_path = PYTORCH_MODEL_PATH
    
    # Check if we need to reload (different model selected)
    if _pytorch_pipe is not None:
        # If same model, return cached
        if _current_pytorch_model_path == model_path:
            return _pytorch_pipe
        # Different model, clear cache
        _pytorch_pipe = None
    
    _current_pytorch_model_path = model_path
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    _pytorch_pipe = ZImagePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    _pytorch_pipe.to(device)
    
    return _pytorch_pipe


# --- LoRA Management Functions ---

def get_available_loras():
    """Get list of available LoRA files (relative paths)"""
    from lora import get_available_loras as _get_loras
    return _get_loras(Path(LORAS_DIR))


def get_loras_with_folders():
    """Get list of (folder, filename) tuples for all LoRAs"""
    from lora import get_lora_with_folders
    return get_lora_with_folders(Path(LORAS_DIR))


def get_lora_info(lora_path):
    """Get detailed info about a LoRA (lora_path is relative to LORAS_DIR)"""
    from lora import get_lora_info as _get_info
    full_path = Path(LORAS_DIR) / lora_path
    return _get_info(full_path)


def get_lora_default_weight(lora_path):
    """Get default weight for a LoRA from metadata (lora_path is relative to LORAS_DIR)"""
    from lora import get_lora_default_weight as _get_default_weight
    full_path = Path(LORAS_DIR) / lora_path
    return _get_default_weight(full_path)


# Maximum number of LoRA slots in the UI
MAX_LORA_SLOTS = 20


def get_lora_choices():
    """Get list of LoRA choices for dropdown.
    
    Returns list of (display_name, value) tuples where:
    - display_name: "folder/filename" or just "filename" 
    - value: relative path to the LoRA
    """
    lora_entries = get_loras_with_folders()
    choices = []
    for folder, filename in lora_entries:
        if folder:
            display = f"{folder}/{filename}"
            value = f"{folder}/{filename}"
        else:
            display = filename
            value = filename
        choices.append((display, value))
    return choices


def get_lora_trigger_for_path(lora_path):
    """Get trigger words for a LoRA path"""
    if not lora_path:
        return ""
    try:
        info = get_lora_info(lora_path)
        return ", ".join(info.get('trigger_words', [])) or ""
    except:
        return ""


def build_lora_tags_from_components(lora_states):
    """Build LoRA tags string from component states.
    
    Args:
        lora_states: List of (enabled, lora_path, trigger, weight) tuples
        
    Returns:
        String like "<lora:name:1.0> trigger1, trigger2"
    """
    lora_tags = []
    triggers = []
    
    for enabled, lora_path, trigger, weight in lora_states:
        if enabled and lora_path:
            # Get clean name (without .safetensors extension and folder)
            lora_name = Path(lora_path).stem
            weight_val = float(weight) if weight else 1.0
            lora_tags.append(f"<lora:{lora_name}:{weight_val:.2f}>")
            
            # Use the user-edited trigger words
            if trigger and trigger.strip():
                triggers.append(trigger.strip())
    
    parts = []
    if lora_tags:
        parts.append(" ".join(lora_tags))
    if triggers:
        parts.append(", ".join(triggers))
    
    return " ".join(parts)


def get_enabled_loras_from_components(lora_states):
    """Extract enabled LoRA configs from component states.
    
    Args:
        lora_states: List of (enabled, lora_path, trigger, weight) tuples
        
    Returns:
        List of {"name": str, "scale": float, "trigger": str} dicts
    """
    enabled_loras = []
    for enabled, lora_path, trigger, weight in lora_states:
        if enabled and lora_path:
            try:
                scale = float(weight) if weight else 1.0
            except (ValueError, TypeError):
                scale = 1.0
            
            enabled_loras.append({
                "name": lora_path,
                "scale": scale,
                "trigger": trigger.strip() if trigger else ""
            })
    return enabled_loras


def get_lora_table_data():
    """Get LoRA data formatted for table display.
    
    Returns a list of [enabled, folder, name, trigger_words, weight, row_index] rows, 
    sorted alphabetically by folder then name.
    Row index is used for per-row weight adjustment.
    """
    lora_entries = get_loras_with_folders()
    rows = []
    for idx, (folder, filename) in enumerate(lora_entries):
        try:
            # Build relative path for get_lora_info
            if folder:
                rel_path = f"{folder}/{filename}"
            else:
                rel_path = filename
            info = get_lora_info(rel_path)
            trigger = ", ".join(info.get('trigger_words', [])) or ""
            default_weight = get_lora_default_weight(rel_path)
            # [Enabled, Folder, Name, Trigger Words, Weight]
            rows.append([False, folder, filename, trigger, default_weight])
        except Exception as e:
            print(f"Error loading LoRA info for {filename}: {e}")
            rows.append([False, folder, filename, "", 1.0])
    return rows


def adjust_single_lora_weight(table_data, row_index, delta):
    """Adjust weight of a single LoRA row by delta amount.
    
    Args:
        table_data: The table data
        row_index: Index of the row to adjust
        delta: Amount to add to weight (+0.05 or -0.05)
    
    Returns:
        Updated table data
    """
    if table_data is None:
        return table_data
    
    # Handle pandas DataFrame
    try:
        if hasattr(table_data, 'values'):
            table_data = table_data.values.tolist()
    except Exception:
        pass
    
    # Make a copy
    modified = [list(row) for row in table_data]
    
    # Adjust the specific row
    if 0 <= row_index < len(modified):
        row = modified[row_index]
        if len(row) >= 5:
            try:
                current_weight = float(row[4]) if row[4] else 1.0
                new_weight = round(current_weight + delta, 2)
                # Clamp between 0 and 2
                new_weight = max(0.0, min(2.0, new_weight))
                modified[row_index][4] = new_weight
            except (ValueError, TypeError):
                pass
    
    return modified


def get_lora_display_info(lora_name):
    """Get formatted display info for a LoRA"""
    try:
        info = get_lora_info(lora_name)
        lines = [f"**{info['name']}**"]
        lines.append(f"Rank: {info['rank']} | Weights: {info['num_weights']}")
        lines.append(f"Layers: {info['layer_range'][0]}-{info['layer_range'][1]}")
        if info['trigger_words']:
            lines.append(f"**Trigger words:** `{', '.join(info['trigger_words'])}`")
        return "\n".join(lines)
    except Exception as e:
        return f"Error loading LoRA info: {e}"


def get_enabled_loras_from_table(table_data):
    """Extract list of enabled LoRA configs from table data.
    
    Args:
        table_data: List/DataFrame of [enabled, folder, name, trigger, weight] rows
        
    Returns:
        List of {"name": str, "scale": float} dicts for enabled LoRAs
        (name includes folder path if in a subfolder)
    """
    if table_data is None:
        return []
    
    # Handle pandas DataFrame (Gradio may send this)
    try:
        if hasattr(table_data, 'values'):
            table_data = table_data.values.tolist()
    except Exception:
        pass
    
    enabled_loras = []
    for row in table_data:
        try:
            if len(row) >= 5 and row[0]:  # row[0] is the enabled checkbox
                folder = str(row[1]) if row[1] else ""
                filename = str(row[2]) if row[2] else ""
                trigger = str(row[3]) if row[3] else ""
                weight = row[4]
                
                # Build relative path
                if folder:
                    rel_path = f"{folder}/{filename}"
                else:
                    rel_path = filename
                
                # Parse weight - handle various formats
                try:
                    scale = float(weight) if weight else 1.0
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse weight '{weight}', using 1.0")
                    scale = 1.0
                
                enabled_loras.append({
                    "name": rel_path,
                    "scale": scale,
                    "trigger": trigger
                })
        except Exception as e:
            print(f"Error processing LoRA row {row}: {e}")
            continue
            
    return enabled_loras


def build_lora_tags_string(table_data):
    """Build a string of LoRA tags from table data.
    
    Format: <lora:name:weight> for each enabled LoRA
    Also includes trigger words if present.
    
    Args:
        table_data: List/DataFrame of [enabled, folder, name, trigger, weight] rows
        
    Returns:
        String like "<lora:style_lora:1.0> trigger1, trigger2"
    """
    if table_data is None:
        return ""
    
    # Handle pandas DataFrame (Gradio sends this)
    try:
        if hasattr(table_data, 'values'):
            table_data = table_data.values.tolist()
    except Exception:
        pass
    
    lora_tags = []
    triggers = []
    
    try:
        for row in table_data:
            if len(row) >= 5 and row[0]:  # row[0] is enabled
                folder = str(row[1]) if row[1] else ""
                filename = str(row[2]) if row[2] else ""
                trigger = str(row[3]) if row[3] else ""
                weight = row[4]
                
                # Get clean name (without .safetensors extension)
                # Use just the filename for the tag, not the folder path
                lora_name = filename.replace('.safetensors', '')
                
                # Format weight
                weight_val = float(weight) if weight else 1.0
                
                # Build tag (folder is for organization only, not in the tag)
                lora_tags.append(f"<lora:{lora_name}:{weight_val:.2f}>")
                
                # Collect trigger words
                if trigger and trigger.strip():
                    triggers.append(trigger.strip())
    except Exception as e:
        print(f"Error building LoRA tags: {e}")
        return ""
    
    # Combine: lora tags first, then triggers
    parts = []
    if lora_tags:
        parts.append(" ".join(lora_tags))
    if triggers:
        parts.append(", ".join(triggers))
    
    return " ".join(parts)


def apply_loras_to_model(model, lora_configs, progress=None):
    """
    Apply multiple LoRAs to a model.
    
    Args:
        model: The transformer model
        lora_configs: List of dicts with 'name' and 'scale' keys
        progress: Optional progress callback
        
    Returns:
        Number of LoRAs successfully applied
    """
    from lora import load_lora, apply_lora_to_model
    
    applied = 0
    for config in lora_configs:
        lora_path = Path(LORAS_DIR) / config['name']
        scale = config.get('scale', 1.0)
        
        if progress:
            progress(desc=f"Applying LoRA: {config['name']} (scale={scale})...")
        
        try:
            lora_weights = load_lora(lora_path)
            apply_lora_to_model(model, lora_weights, scale=scale, verbose=False)
            applied += 1
            print(f"Applied LoRA: {config['name']} with scale {scale}")
        except Exception as e:
            print(f"Error applying LoRA {config['name']}: {e}")
    
    return applied


def save_fused_model(model_name, backend, source_model, lora_configs, save_mlx=True, save_pytorch=False, save_comfyui=False, progress=gr.Progress()):
    """
    Save a new model with LoRAs fused into the weights.
    
    Args:
        model_name: Name for the new model
        backend: Current backend ("MLX (Apple Silicon)" or "PyTorch")
        source_model: Source model name/path
        lora_configs: List of dicts with 'name' and 'scale' keys
        save_mlx: Save in MLX format
        save_pytorch: Save in PyTorch/Diffusers format
        save_comfyui: Save in ComfyUI single-file format
        progress: Gradio progress callback
        
    Returns:
        Status message
    """
    import mlx.core as mx
    
    if not model_name or not model_name.strip():
        return "❌ Please enter a model name"
    
    model_name = model_name.strip()
    
    # Validate model name (alphanumeric, hyphens, underscores)
    import re
    if not re.match(r'^[\w\-]+$', model_name):
        return "❌ Model name can only contain letters, numbers, hyphens, and underscores"
    
    # Check if any format is selected
    if not save_mlx and not save_pytorch and not save_comfyui:
        return "❌ Please select at least one output format"
    
    # Check if any LoRAs are selected
    if not lora_configs:
        return "❌ No LoRAs selected. Enable at least one LoRA to fuse."
    
    # Filter to only enabled LoRAs with non-zero scale
    active_loras = [c for c in lora_configs if c.get('scale', 0) > 0]
    if not active_loras:
        return "❌ No active LoRAs. Enable at least one LoRA with weight > 0."
    
    results = []
    
    try:
        # Save MLX format
        if save_mlx:
            result = _save_fused_mlx_model(model_name, source_model, active_loras, progress, backend)
            results.append(result)
        
        # Save PyTorch format
        if save_pytorch:
            result = _save_fused_pytorch_model(model_name, source_model, active_loras, progress, backend)
            results.append(result)
        
        # Save ComfyUI format
        if save_comfyui:
            result = _save_fused_comfyui_model(model_name, source_model, active_loras, progress, backend)
            results.append(result)
        
        return "\n".join(results)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error saving model: {str(e)}"


def _save_fused_mlx_model(model_name, source_model, lora_configs, progress, backend="MLX (Apple Silicon)"):
    """Save an MLX model with fused LoRAs."""
    import mlx.core as mx
    from safetensors.numpy import save_file
    from lora import load_lora, apply_lora_to_model
    
    # Determine source model path based on backend
    if backend == "MLX (Apple Silicon)":
        if source_model and source_model != "Z-Image-Turbo-MLX":
            source_path = Path(MLX_MODELS_DIR) / source_model
        else:
            source_path = Path(MLX_MODEL_PATH)
    else:
        # For PyTorch backend, we need to convert from PyTorch to MLX first
        # For now, use the MLX model path as fallback
        source_path = Path(MLX_MODEL_PATH)
    
    if not source_path.exists():
        return f"❌ Source model not found: {source_path}"
    
    # Create output directory
    output_path = Path(MLX_MODELS_DIR) / model_name
    if output_path.exists():
        return f"❌ Model '{model_name}' already exists. Choose a different name."
    
    progress(0.1, desc="Loading source model...")
    
    # Load fresh model (don't use cached to avoid modifying it)
    from z_image_mlx import ZImageTransformer2DModel
    with open(f"{source_path}/config.json", "r") as f:
        config = json.load(f)
    
    model = ZImageTransformer2DModel(config)
    weights = mx.load(f"{source_path}/weights.safetensors")
    
    # Process weights
    processed_weights = {}
    for key, value in weights.items():
        new_key = map_transformer_key(key)
        if new_key is None or "pad_token" in (new_key or ""):
            continue
        processed_weights[new_key] = value
    
    model.load_weights(list(processed_weights.items()), strict=False)
    
    progress(0.3, desc="Applying LoRAs...")
    
    # Apply each LoRA
    lora_names = []
    for i, config_item in enumerate(lora_configs):
        lora_path = Path(LORAS_DIR) / config_item['name']
        scale = config_item.get('scale', 1.0)
        lora_names.append(f"{config_item['name']} ({scale})")
        
        progress(0.3 + (0.4 * i / len(lora_configs)), desc=f"Applying: {config_item['name']}...")
        
        lora_weights = load_lora(lora_path)
        apply_lora_to_model(model, lora_weights, scale=scale, verbose=False)
    
    progress(0.7, desc="Saving fused weights...")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract and save fused weights
    fused_weights = {}
    for name, param in model.named_modules():
        if hasattr(param, 'weight'):
            weight = param.weight
            fused_weights[f"{name}.weight"] = np.array(weight)
        if hasattr(param, 'bias') and param.bias is not None:
            bias = param.bias
            fused_weights[f"{name}.bias"] = np.array(bias)
    
    # Save weights
    save_file(fused_weights, str(output_path / "weights.safetensors"))
    
    progress(0.85, desc="Copying config files...")
    
    # Copy config files
    shutil.copy(source_path / "config.json", output_path / "config.json")
    shutil.copy(source_path / "vae_config.json", output_path / "vae_config.json")
    shutil.copy(source_path / "vae.safetensors", output_path / "vae.safetensors")
    shutil.copy(source_path / "text_encoder_config.json", output_path / "text_encoder_config.json")
    shutil.copy(source_path / "text_encoder.safetensors", output_path / "text_encoder.safetensors")
    
    # Copy directories
    shutil.copytree(source_path / "tokenizer", output_path / "tokenizer")
    shutil.copytree(source_path / "scheduler", output_path / "scheduler")
    
    # Create a metadata file noting the fused LoRAs
    metadata = {
        "base_model": str(source_path.name),
        "fused_loras": lora_names,
        "created": datetime.now().isoformat(),
    }
    with open(output_path / "lora_fusion_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    progress(1.0, desc="Done!")
    
    lora_list = ", ".join(lora_names)
    return f"✅ Saved MLX model '{model_name}' with fused LoRAs: {lora_list}"


def _save_fused_pytorch_model(model_name, source_model, lora_configs, progress, backend="PyTorch"):
    """Save a PyTorch model with fused LoRAs."""
    if not PYTORCH_AVAILABLE:
        return "❌ PyTorch backend not available"
    
    # Determine source model path  
    if source_model and source_model != "Z-Image-Turbo":
        source_path = Path(PYTORCH_MODELS_DIR) / source_model
    else:
        source_path = Path(PYTORCH_MODEL_PATH)
    
    if not source_path.exists():
        return f"❌ Source model not found: {source_path}"
    
    # Create output directory
    output_path = Path(PYTORCH_MODELS_DIR) / model_name
    if output_path.exists():
        return f"❌ Model '{model_name}' already exists. Choose a different name."
    
    progress(0.1, desc="Loading source model...")
    
    # Load pipeline
    pipe = ZImagePipeline.from_pretrained(
        str(source_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    
    progress(0.3, desc="Applying LoRAs...")
    
    # Apply LoRAs to the transformer
    from lora import load_lora
    import mlx.core as mx
    
    lora_names = []
    for i, config_item in enumerate(lora_configs):
        lora_path = Path(LORAS_DIR) / config_item['name']
        scale = config_item.get('scale', 1.0)
        lora_names.append(f"{config_item['name']} ({scale})")
        
        progress(0.3 + (0.4 * i / len(lora_configs)), desc=f"Applying: {config_item['name']}...")
        
        # Load LoRA weights (MLX format)
        lora_weights = load_lora(lora_path)
        
        # Apply to PyTorch model
        _apply_lora_to_pytorch_model(pipe.transformer, lora_weights, scale)
    
    progress(0.7, desc="Saving fused model...")
    
    # Save the entire pipeline
    pipe.save_pretrained(str(output_path))
    
    # Create metadata file
    metadata = {
        "base_model": str(source_path.name),
        "fused_loras": lora_names,
        "created": datetime.now().isoformat(),
    }
    with open(output_path / "lora_fusion_info.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    progress(1.0, desc="Done!")
    
    lora_list = ", ".join(lora_names)
    return f"✅ Saved PyTorch model '{model_name}' with fused LoRAs: {lora_list}"


def _apply_lora_to_pytorch_model(model, lora_weights, scale=1.0):
    """Apply MLX LoRA weights to a PyTorch model."""
    from lora import parse_lora_weights
    import mlx.core as mx
    
    lora_pairs = parse_lora_weights(lora_weights)
    
    for model_key, (lora_a, lora_b) in lora_pairs.items():
        layer_path = model_key.rsplit(".weight", 1)[0]
        
        try:
            # Navigate to the layer in PyTorch model
            parts = layer_path.split(".")
            layer = model
            for part in parts:
                if part.isdigit():
                    layer = layer[int(part)]
                else:
                    layer = getattr(layer, part)
            
            if not hasattr(layer, 'weight'):
                continue
            
            # Convert MLX arrays to numpy then to torch
            a_np = np.array(lora_a)
            b_np = np.array(lora_b)
            
            # Compute delta: B @ A (with proper dimension handling)
            if len(a_np.shape) == 2 and len(b_np.shape) == 2:
                delta_np = b_np @ a_np
            else:
                continue
            
            # Apply to weight
            with torch.no_grad():
                delta_torch = torch.from_numpy(delta_np * scale).to(layer.weight.dtype).to(layer.weight.device)
                if delta_torch.shape == layer.weight.shape:
                    layer.weight.add_(delta_torch)
                    
        except (AttributeError, IndexError, KeyError):
            continue


def _save_fused_comfyui_model(model_name, source_model, lora_configs, progress, backend="MLX (Apple Silicon)"):
    """Save a ComfyUI single-file model with fused LoRAs."""
    if not PYTORCH_AVAILABLE:
        return "❌ PyTorch required for ComfyUI format"
    
    from safetensors.torch import save_file as torch_save_safetensors
    from convert_pytorch_to_comfyui import (
        convert_transformer_key_to_comfyui,
        convert_text_encoder_key_to_comfyui,
        convert_vae_key_to_comfyui,
    )
    
    # Determine source model path - need PyTorch format for ComfyUI
    if source_model and source_model != "Z-Image-Turbo":
        source_path = Path(PYTORCH_MODELS_DIR) / source_model
    else:
        source_path = Path(PYTORCH_MODEL_PATH)
    
    if not source_path.exists():
        return f"❌ PyTorch source model not found: {source_path}"
    
    # Output to comfyui directory
    output_dir = Path(MODELS_DIR) / "comfyui"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.safetensors"
    
    if output_file.exists():
        return f"❌ ComfyUI model '{model_name}.safetensors' already exists"
    
    progress(0.05, desc="Loading PyTorch model for ComfyUI conversion...")
    
    # Load pipeline
    pipe = ZImagePipeline.from_pretrained(
        str(source_path),
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    
    progress(0.15, desc="Applying LoRAs...")
    
    # Apply LoRAs to the transformer
    from lora import load_lora
    
    lora_names = []
    for i, config_item in enumerate(lora_configs):
        lora_path = Path(LORAS_DIR) / config_item['name']
        scale = config_item.get('scale', 1.0)
        lora_names.append(f"{config_item['name']} ({scale})")
        
        progress(0.15 + (0.25 * i / len(lora_configs)), desc=f"Applying: {config_item['name']}...")
        
        lora_weights = load_lora(lora_path)
        _apply_lora_to_pytorch_model(pipe.transformer, lora_weights, scale)
    
    progress(0.4, desc="Converting transformer weights...")
    
    all_weights = {}
    
    # Convert transformer weights with QKV fusion
    transformer = pipe.transformer.state_dict()
    
    # Identify Q/K/V groups for fusion
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
    
    # Fuse Q, K, V weights
    for base_key, qkv_keys in qkv_groups.items():
        if "q" in qkv_keys and "k" in qkv_keys and "v" in qkv_keys:
            q = transformer[qkv_keys["q"]]
            k = transformer[qkv_keys["k"]]
            v = transformer[qkv_keys["v"]]
            fused = torch.cat([q, k, v], dim=0)
            comfyui_key = convert_transformer_key_to_comfyui(f"{base_key}.qkv.weight")
            all_weights[comfyui_key] = fused
    
    # Convert other transformer weights
    for key in other_keys:
        comfyui_key = convert_transformer_key_to_comfyui(key)
        all_weights[comfyui_key] = transformer[key]
    
    progress(0.6, desc="Converting text encoder weights...")
    
    # Convert text encoder weights
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        te_state = pipe.text_encoder.state_dict()
        for key, value in te_state.items():
            comfyui_key = convert_text_encoder_key_to_comfyui(key)
            all_weights[comfyui_key] = value
    
    progress(0.75, desc="Converting VAE weights...")
    
    # Convert VAE weights
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        vae_state = pipe.vae.state_dict()
        for key, value in vae_state.items():
            comfyui_key = convert_vae_key_to_comfyui(key)
            all_weights[comfyui_key] = value
    
    progress(0.9, desc="Saving ComfyUI checkpoint...")
    
    # Save combined checkpoint
    torch_save_safetensors(all_weights, str(output_file))
    
    # Calculate file size
    file_size = output_file.stat().st_size / (1024 * 1024 * 1024)
    
    progress(1.0, desc="Done!")
    
    lora_list = ", ".join(lora_names)
    return f"✅ Saved ComfyUI model '{model_name}.safetensors' ({file_size:.2f}GB) with fused LoRAs: {lora_list}"


# --- Upscaler Functions ---

def get_upscaler_choices():
    """Get list of available upscalers for dropdown.
    Returns ["None"] plus all supported (ESRGAN/RRDB) upscaler names.
    """
    if not UPSCALER_AVAILABLE:
        return ["None (upscaler not available)"]
    
    # Only show supported upscalers (ESRGAN/RRDB architecture)
    upscalers = get_available_upscalers(Path(UPSCALERS_DIR), filter_supported=True)
    if not upscalers:
        return ["None (no supported upscalers found)"]
    return ["None"] + upscalers


def load_cached_upscaler(upscaler_name):
    """Load and cache an upscaler model.
    Returns None if upscaler_name is "None" or invalid.
    """
    global _upscaler_model, _current_upscaler_name
    
    if not UPSCALER_AVAILABLE:
        return None
    
    if upscaler_name == "None" or not upscaler_name or upscaler_name.startswith("None"):
        _upscaler_model = None
        _current_upscaler_name = None
        return None
    
    # Check if already loaded
    if _upscaler_model is not None and _current_upscaler_name == upscaler_name:
        return _upscaler_model
    
    # Load new upscaler
    try:
        _upscaler_model = load_upscaler(upscaler_name, Path(UPSCALERS_DIR))
        _current_upscaler_name = upscaler_name
        return _upscaler_model
    except ValueError as e:
        # Unsupported architecture (e.g., SPAN models)
        print(f"Warning: {e}")
        _upscaler_model = None
        _current_upscaler_name = None
        return None
    except Exception as e:
        print(f"Error loading upscaler {upscaler_name}: {e}")
        _upscaler_model = None
        _current_upscaler_name = None
        return None


def apply_upscaler(image, upscaler_name, scale_factor=2.0, progress=None):
    """Apply upscaler to an image.
    
    Args:
        image: PIL Image to upscale
        upscaler_name: Name of the upscaler to use (or "None" to skip)
        scale_factor: Target scale factor (1.0-4.0). ESRGAN does 4x then resizes down.
        progress: Optional Gradio progress callback
        
    Returns:
        Upscaled PIL Image (or original if upscaler_name is "None" or scale_factor <= 1)
    """
    if not UPSCALER_AVAILABLE:
        return image
    
    if upscaler_name == "None" or not upscaler_name or upscaler_name.startswith("None"):
        return image
    
    if scale_factor <= 1.0:
        return image
    
    if progress:
        progress(desc=f"Loading upscaler: {upscaler_name}...")
    
    model = load_cached_upscaler(upscaler_name)
    if model is None:
        print(f"Warning: Could not load upscaler {upscaler_name}")
        return image
    
    if progress:
        progress(desc=f"Upscaling image ({scale_factor}×)...")
    
    try:
        # ESRGAN always does 4x, then we resize to target scale
        upscaled = upscale_image_esrgan(model, image)
        
        # If scale_factor < 4.0, resize down to target
        if scale_factor < 4.0:
            orig_w, orig_h = image.size
            target_w = int(orig_w * scale_factor)
            target_h = int(orig_h * scale_factor)
            upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        
        return upscaled
    except Exception as e:
        print(f"Error upscaling image: {e}")
        return image


# --- Latent Upscaling Functions ---

def get_available_memory_gb():
    """Estimate available GPU memory in GB for tile size calculation.
    
    Returns conservative estimate based on system memory and Metal limits.
    """
    import subprocess
    try:
        # Get total system memory (proxy for unified memory on Apple Silicon)
        result = subprocess.run(['sysctl', '-n', 'hw.memsize'], capture_output=True, text=True)
        total_bytes = int(result.stdout.strip())
        total_gb = total_bytes / (1024 ** 3)
        
        # Conservative estimate: assume 50% available for GPU, minus overhead
        # This accounts for system usage, other apps, and MLX overhead
        available_gb = (total_gb * 0.5) - 4.0  # Reserve 4GB for system
        return max(4.0, available_gb)  # Minimum 4GB
    except Exception:
        return 8.0  # Default fallback


def calculate_optimal_tile_size(latent_height, latent_width, scale_factor, available_memory_gb=None):
    """Calculate optimal tile size based on available memory.
    
    Args:
        latent_height: Height of latent tensor
        latent_width: Width of latent tensor
        scale_factor: Upscale multiplier (1.0-4.0)
        available_memory_gb: Override for available memory (auto-detect if None)
        
    Returns:
        Tuple of (tile_size, overlap) in latent pixels
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb()
    
    # Estimate memory per latent pixel (empirically determined)
    # Transformer forward pass is ~0.5MB per latent pixel at 1x scale
    # Scale quadratically with upscale factor
    memory_per_pixel_mb = 0.5 * (scale_factor ** 2)
    
    # Calculate max tile area that fits in memory
    # Reserve 30% for overhead (attention, intermediates, etc.)
    usable_memory_mb = available_memory_gb * 1024 * 0.7
    max_tile_area = usable_memory_mb / memory_per_pixel_mb
    
    # Calculate tile size (square tiles)
    max_tile_size = int(max_tile_area ** 0.5)
    
    # Clamp to reasonable range and ensure divisible by 8
    tile_size = max(32, min(256, max_tile_size))
    tile_size = (tile_size // 8) * 8
    
    # Overlap should be ~12.5% of tile size for smooth blending
    overlap = max(8, tile_size // 8)
    
    # If the full latent fits in memory, use no tiling
    total_area = latent_height * latent_width * (scale_factor ** 2)
    if total_area <= max_tile_area * 0.8:  # 80% threshold for safety
        return None, None  # No tiling needed
    
    return tile_size, overlap


def blend_tiles(output, tile, y_start, x_start, tile_h, tile_w, overlap, full_h, full_w):
    """Blend a tile into the output with linear gradient at overlapping edges.
    
    Args:
        output: Full output tensor [B, H, W, C]
        tile: Tile to blend [B, tile_h, tile_w, C]
        y_start, x_start: Position of tile in output
        tile_h, tile_w: Size of tile
        overlap: Overlap size in pixels
        full_h, full_w: Full output dimensions
        
    Returns:
        Updated output tensor
    """
    import mlx.core as mx
    
    # Create weight mask for blending
    weight = mx.ones((tile_h, tile_w))
    
    # Apply linear gradient at edges that overlap with other tiles
    # Top edge
    if y_start > 0:
        for i in range(min(overlap, tile_h)):
            weight = weight.at[i, :].set(weight[i, :] * (i / overlap))
    
    # Left edge
    if x_start > 0:
        for j in range(min(overlap, tile_w)):
            weight = weight.at[:, j].set(weight[:, j] * (j / overlap))
    
    # Bottom edge
    if y_start + tile_h < full_h:
        for i in range(min(overlap, tile_h)):
            idx = tile_h - 1 - i
            weight = weight.at[idx, :].set(weight[idx, :] * (i / overlap))
    
    # Right edge
    if x_start + tile_w < full_w:
        for j in range(min(overlap, tile_w)):
            idx = tile_w - 1 - j
            weight = weight.at[:, idx].set(weight[:, idx] * (j / overlap))
    
    # Expand weight to match tile shape [B, H, W, C]
    weight = weight[None, :, :, None]
    
    # Blend into output
    y_end = y_start + tile_h
    x_end = x_start + tile_w
    
    current = output[:, y_start:y_end, x_start:x_end, :]
    blended = current + (tile - current) * weight
    output = output.at[:, y_start:y_end, x_start:x_end, :].set(blended)
    
    return output


def latent_upscale_mlx(latents, prompt_embeds, model, vae, vae_config, scheduler, 
                       scale_factor=2.0, denoise_strength=0.55, hires_steps=None,
                       seed=None, interp_mode='cubic', progress=None, progress_start=0.8):
    """Upscale latents in latent space with refinement denoising.
    
    This adds detail at higher resolution by:
    1. Spatially upscaling the denoised latents
    2. Adding noise at specified strength
    3. Running partial denoising to generate new detail
    
    Uses tiled processing for memory efficiency on large upscales.
    
    Args:
        latents: Denoised latents [B, H, W, C] in NHWC format
        prompt_embeds: Text embeddings (list of tensors)
        model: Transformer model
        vae: VAE (not used directly, for future encode/decode cycle)
        vae_config: VAE configuration dict
        scheduler: FlowMatchEulerDiscreteScheduler
        scale_factor: Upscale multiplier (1.0-4.0)
        denoise_strength: How much to re-noise (0.0-1.0, higher = more change)
        hires_steps: Refinement steps (None = auto based on denoise_strength)
        seed: Random seed for noise
        interp_mode: Interpolation mode ('nearest', 'linear', 'cubic')
        progress: Optional Gradio progress callback
        progress_start: Progress percentage where latent upscale starts
        
    Returns:
        Upscaled and refined latents [B, H*scale, W*scale, C]
    """
    import mlx.core as mx
    import mlx.nn as nn
    
    if scale_factor <= 1.0 or denoise_strength <= 0.0:
        return latents
    
    batch_size, h, w, channels = latents.shape
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    if progress:
        progress(progress_start, desc=f"Latent upscale: {h}×{w} → {new_h}×{new_w}...")
    
    # Step 1: Spatially upscale latents
    # MLX Upsample expects NHWC format which we already have
    upsampler = nn.Upsample(scale_factor=scale_factor, mode=interp_mode, align_corners=True)
    upscaled = upsampler(latents)
    mx.eval(upscaled)
    
    if progress:
        progress(progress_start + 0.02, desc="Adding noise for refinement...")
    
    # Step 2: Determine number of refinement steps
    if hires_steps is None or hires_steps == 0:
        # Auto: scale steps based on denoise strength
        # At strength 1.0, use full steps; at 0.5, use half
        hires_steps = max(1, int(9 * denoise_strength))  # Base on typical 9 steps
    
    # Step 3: Calculate start step based on denoise strength
    # strength 1.0 = start from full noise (step 0)
    # strength 0.0 = no denoising (would skip entirely)
    scheduler.set_timesteps(hires_steps)
    timesteps = scheduler.timesteps
    
    # Start step: strength=1.0 → step 0; strength=0.5 → step halfway
    start_step = int((1.0 - denoise_strength) * len(timesteps))
    start_step = max(0, min(start_step, len(timesteps) - 1))
    
    # Get timestep for noise level
    t_start = timesteps[start_step]
    
    # Step 4: Generate noise and add to upscaled latents
    if seed is not None:
        torch.manual_seed(seed + 1000)  # Offset seed for variety
    
    # Generate noise in PyTorch for reproducibility, then convert
    noise_pt = torch.randn(batch_size, channels, 1, new_h, new_w)
    noise = mx.array(noise_pt.numpy())
    noise = noise.squeeze(2).transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    
    # Add noise at the calculated timestep
    # For flow matching: noisy = (1 - t) * clean + t * noise, where t is normalized
    t_normalized = t_start.item() / 1000.0  # Normalize to [0, 1]
    noisy_latents = (1.0 - t_normalized) * upscaled + t_normalized * noise
    mx.eval(noisy_latents)
    
    # Step 5: Check if tiling is needed
    tile_size, overlap = calculate_optimal_tile_size(new_h, new_w, 1.0)  # Already upscaled
    
    use_tiling = tile_size is not None and (new_h > tile_size or new_w > tile_size)
    
    if use_tiling:
        if progress:
            progress(progress_start + 0.05, desc=f"Using tiled processing ({tile_size}×{tile_size} tiles)...")
        return _latent_upscale_tiled(
            noisy_latents, prompt_embeds, model, scheduler, timesteps,
            start_step, new_h, new_w, tile_size, overlap, progress, progress_start
        )
    else:
        # Process full resolution
        return _latent_denoise_steps(
            noisy_latents, prompt_embeds, model, scheduler, timesteps,
            start_step, progress, progress_start
        )


def _latent_denoise_steps(latents, prompt_embeds, model, scheduler, timesteps,
                          start_step, progress=None, progress_start=0.8):
    """Run denoising steps on latents.
    
    Args:
        latents: Noisy latents [B, H, W, C] in NHWC format
        prompt_embeds: Text embeddings
        model: Transformer model
        scheduler: Scheduler with timesteps set
        timesteps: Full timestep schedule
        start_step: Index to start denoising from
        progress: Optional progress callback
        progress_start: Base progress percentage
        
    Returns:
        Denoised latents [B, H, W, C]
    """
    import mlx.core as mx
    
    remaining_steps = len(timesteps) - start_step
    
    for i, t in enumerate(timesteps[start_step:]):
        step_num = i + 1
        total_steps = remaining_steps
        
        if progress:
            step_progress = progress_start + 0.1 + (0.08 * (step_num / total_steps))
            progress(step_progress, desc=f"Hires step {step_num}/{total_steps}...")
        
        # Convert latents to model format: [B, H, W, C] -> [B, C, 1, H, W]
        latents_5d = latents.transpose(0, 3, 1, 2)[:, :, None, :, :]
        
        # Timestep in model format
        t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
        
        # Forward pass
        noise_pred = model(latents_5d, t_mx, prompt_embeds)
        noise_pred = -noise_pred  # Negate as in main pipeline
        mx.eval(noise_pred)
        
        # Scheduler step (via PyTorch)
        noise_pred_sq = noise_pred.squeeze(2)  # [B, C, H, W]
        latents_sq = latents.transpose(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        
        noise_pred_pt = torch.from_numpy(np.array(noise_pred_sq))
        latents_pt = torch.from_numpy(np.array(latents_sq))
        
        step_output = scheduler.step(noise_pred_pt, t, latents_pt, return_dict=True)
        
        # Convert back to MLX NHWC format
        latents = mx.array(step_output.prev_sample.numpy())
        latents = latents.transpose(0, 2, 3, 1)  # [B, C, H, W] -> [B, H, W, C]
    
    return latents


def _latent_upscale_tiled(latents, prompt_embeds, model, scheduler, timesteps,
                          start_step, full_h, full_w, tile_size, overlap,
                          progress=None, progress_start=0.8):
    """Process latent denoising in tiles for memory efficiency.
    
    Args:
        latents: Noisy latents [B, H, W, C]
        prompt_embeds: Text embeddings
        model: Transformer model
        scheduler: Scheduler
        timesteps: Timestep schedule
        start_step: Index to start from
        full_h, full_w: Full latent dimensions
        tile_size: Size of each tile
        overlap: Overlap between tiles
        progress: Optional progress callback
        progress_start: Base progress percentage
        
    Returns:
        Denoised latents [B, H, W, C]
    """
    import mlx.core as mx
    
    batch_size = latents.shape[0]
    channels = latents.shape[3]
    remaining_steps = len(timesteps) - start_step
    
    # Calculate tile positions
    stride = tile_size - overlap
    y_positions = list(range(0, full_h - overlap, stride))
    x_positions = list(range(0, full_w - overlap, stride))
    
    # Ensure we cover the full image
    if y_positions[-1] + tile_size < full_h:
        y_positions.append(full_h - tile_size)
    if x_positions[-1] + tile_size < full_w:
        x_positions.append(full_w - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    
    # Process each denoising step
    for step_idx, t in enumerate(timesteps[start_step:]):
        step_num = step_idx + 1
        
        if progress:
            step_progress = progress_start + 0.1 + (0.08 * (step_num / remaining_steps))
            progress(step_progress, desc=f"Hires step {step_num}/{remaining_steps} ({total_tiles} tiles)...")
        
        # Initialize output for this step
        output = mx.zeros((batch_size, full_h, full_w, channels))
        weight_sum = mx.zeros((1, full_h, full_w, 1))
        
        # Process each tile
        for y_idx, y_start in enumerate(y_positions):
            for x_idx, x_start in enumerate(x_positions):
                # Calculate actual tile bounds (may be smaller at edges)
                y_end = min(y_start + tile_size, full_h)
                x_end = min(x_start + tile_size, full_w)
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                
                # Extract tile
                tile = latents[:, y_start:y_end, x_start:x_end, :]
                
                # Convert to model format: [B, H, W, C] -> [B, C, 1, H, W]
                tile_5d = tile.transpose(0, 3, 1, 2)[:, :, None, :, :]
                
                # Timestep
                t_mx = mx.array([(1000.0 - t.item()) / 1000.0])
                
                # Forward pass on tile
                noise_pred = model(tile_5d, t_mx, prompt_embeds)
                noise_pred = -noise_pred
                mx.eval(noise_pred)
                
                # Scheduler step
                noise_pred_sq = noise_pred.squeeze(2)
                tile_sq = tile.transpose(0, 3, 1, 2)
                
                noise_pred_pt = torch.from_numpy(np.array(noise_pred_sq))
                tile_pt = torch.from_numpy(np.array(tile_sq))
                
                step_output = scheduler.step(noise_pred_pt, t, tile_pt, return_dict=True)
                
                # Convert back to NHWC
                denoised_tile = mx.array(step_output.prev_sample.numpy())
                denoised_tile = denoised_tile.transpose(0, 2, 3, 1)
                
                # Create blending weights
                weight = mx.ones((1, tile_h, tile_w, 1))
                
                # Apply gradient at overlapping edges
                if y_start > 0:
                    for i in range(min(overlap, tile_h)):
                        alpha = i / overlap
                        weight = weight.at[:, i, :, :].set(weight[:, i, :, :] * alpha)
                
                if x_start > 0:
                    for j in range(min(overlap, tile_w)):
                        alpha = j / overlap
                        weight = weight.at[:, :, j, :].set(weight[:, :, j, :] * alpha)
                
                if y_end < full_h:
                    for i in range(min(overlap, tile_h)):
                        idx = tile_h - 1 - i
                        alpha = i / overlap
                        weight = weight.at[:, idx, :, :].set(weight[:, idx, :, :] * alpha)
                
                if x_end < full_w:
                    for j in range(min(overlap, tile_w)):
                        idx = tile_w - 1 - j
                        alpha = j / overlap
                        weight = weight.at[:, :, idx, :].set(weight[:, :, idx, :] * alpha)
                
                # Accumulate weighted tile
                output = output.at[:, y_start:y_end, x_start:x_end, :].add(denoised_tile * weight)
                weight_sum = weight_sum.at[:, y_start:y_end, x_start:x_end, :].add(weight)
        
        # Normalize by weight sum
        latents = output / (weight_sum + 1e-8)
        mx.eval(latents)
    
    return latents


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
        model, tokenizer = load(str(enhancer_path))
    else:
        # Download and save locally
        from huggingface_hub import snapshot_download
        
        # Download to local path
        enhancer_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=PROMPT_ENHANCER_MODEL,
            local_dir=str(enhancer_path),
            local_dir_use_symlinks=False,
        )
        
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


def generate_mlx(prompt, width, height, steps, time_shift, seed, progress, lora_configs=None,
                 latent_scale=1.0, latent_steps=0, latent_denoise=0.55, latent_interp='cubic',
                 cache_mode=None):
    """Generate image using MLX backend
    
    Args:
        lora_configs: Optional list of dicts with 'name' and 'scale' keys for LoRAs to apply
        latent_scale: Latent space upscale factor (1.0-4.0, 1.0 = no upscale)
        latent_steps: Hires refinement steps (0 = auto based on denoise strength)
        latent_denoise: Denoising strength for latent upscale (0.0-1.0)
        latent_interp: Interpolation mode ('nearest', 'linear', 'cubic')
        cache_mode: LeMiCa cache mode ('slow', 'medium', 'fast') or None to disable
    """
    import mlx.core as mx
    global _mlx_models, _current_applied_lora
    
    # Determine if latent upscaling is enabled
    do_latent_upscale = latent_scale > 1.0 and latent_denoise > 0.0
    
    # Normalize lora_configs to empty list if None
    if lora_configs is None:
        lora_configs = []
    
    # Create a key to identify the current LoRA configuration (sorted for consistency)
    if lora_configs:
        lora_key = "|".join(sorted([f"{c['name']}@{c['scale']}" for c in lora_configs]))
    else:
        lora_key = None
    
    # Check if we need to reload the model due to LoRA change
    need_lora_apply = False
    if _current_applied_lora != lora_key:
        # LoRA config changed - need to reload base model before applying new LoRAs
        if _mlx_models is not None:
            progress(0.02, desc="LoRA configuration changed - reloading base model...")
            _mlx_models = None  # Force reload
        need_lora_apply = len(lora_configs) > 0
    
    models = load_mlx_models()
    
    model = models["model"]
    vae = models["vae"]
    vae_config = models["vae_config"]
    text_encoder = models["text_encoder"]
    tokenizer = models["tokenizer"]
    scheduler = models["scheduler"]
    
    # Apply LoRAs if we just reloaded the model and have LoRA configs
    if need_lora_apply and lora_configs:
        lora_names = [c['name'] for c in lora_configs]
        progress(0.05, desc=f"Applying {len(lora_configs)} LoRA(s): {', '.join(lora_names)}...")
        try:
            applied = apply_loras_to_model(model, lora_configs, progress=None)
            if applied > 0:
                mx.eval(model.parameters())  # Ensure LoRA deltas are computed
                _current_applied_lora = lora_key  # Mark LoRAs as successfully applied
                print(f"Successfully applied {applied} LoRA(s)")
        except Exception as e:
            print(f"Warning: Failed to apply LoRAs: {e}")
            _current_applied_lora = None  # Reset on failure
    elif not lora_configs:
        _current_applied_lora = None
    
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
    
    # Configure LeMiCa caching if enabled
    if cache_mode and cache_mode.lower() != "none":
        model.configure_lemica(cache_mode, steps)
    else:
        model.configure_lemica(None)
    
    # Reset LeMiCa state for this generation (ensures counter starts at 0)
    model.reset_lemica_state()
    
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
    
    # Convert latents to NHWC format for potential latent upscaling and VAE decode
    latents = latents.squeeze(2)  # [B, C, H, W]
    latents = latents.transpose(0, 2, 3, 1)  # [B, H, W, C]
    
    # Apply latent upscaling if enabled
    if do_latent_upscale:
        # Disable LeMiCa during hires refinement - the cached residuals from base generation
        # are not valid for the upscaled latent space
        model.configure_lemica(None)
        
        progress(0.80, desc=f"Latent upscale {latent_scale}× ({latent_interp})...")
        latents = latent_upscale_mlx(
            latents=latents,
            prompt_embeds=prompt_embeds_list,
            model=model,
            vae=vae,
            vae_config=vae_config,
            scheduler=scheduler,
            scale_factor=latent_scale,
            denoise_strength=latent_denoise,
            hires_steps=latent_steps if latent_steps > 0 else None,
            seed=seed,
            interp_mode=latent_interp,
            progress=progress,
            progress_start=0.80,
        )
        progress(0.92, desc="Decoding upscaled image...")
    else:
        progress(0.85, desc="Decoding image...")
    
    # Decode latents (already in NHWC format)
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


def add_to_gallery(image, prompt, seed, png_path, jpg_path, steps, time_shift, backend, width, height, model_name, lora_configs=None, upscaler_name=None, upscale_factor=None, latent_scale=None, latent_denoise=None, latent_interp=None):
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
        "model_name": model_name,
        "lora_configs": lora_configs or [],
        "upscaler": upscaler_name,
        "upscale_factor": upscale_factor,
        "latent_scale": latent_scale,
        "latent_denoise": latent_denoise,
        "latent_interp": latent_interp,
    })
    return [item["image"] for item in _image_gallery]


def get_gallery_images():
    """Get all images in the gallery"""
    return [item["image"] for item in _image_gallery]


def get_selected_image_info(evt: gr.SelectData):
    """Get info for the selected image from gallery"""
    if evt.index < len(_image_gallery):
        item = _image_gallery[evt.index]
        model_name = item.get('model_name', 'Unknown')
        
        # Build details in logical order
        details_lines = [
            f"Backend: {item['backend']}",
            f"Model: {model_name}",
            f"Seed: {item['seed']}",
            f"Size: {item['width']}×{item['height']}",
            f"Steps: {item['steps']}",
            f"Time Shift: {item['time_shift']}",
        ]
        
        # Add LoRA info if present
        lora_configs = item.get('lora_configs', [])
        if lora_configs:
            lora_strs = [f"{c['name']}: {c['scale']}" for c in lora_configs]
            details_lines.append(f"LoRAs: {', '.join(lora_strs)}")
        
        # Add latent upscale info if used
        latent_scale = item.get('latent_scale')
        if latent_scale and latent_scale > 1.0:
            latent_denoise = item.get('latent_denoise', 0.55)
            latent_interp = item.get('latent_interp', 'cubic')
            details_lines.append(f"Latent Upscale: {latent_scale}× (denoise: {latent_denoise}, {latent_interp})")
        
        # Add ESRGAN upscaler info if used
        upscaler = item.get('upscaler')
        if upscaler and upscaler != "None":
            upscale_factor = item.get('upscale_factor')
            scale_str = f"{upscale_factor}×" if upscale_factor else "4×"
            details_lines.append(f"ESRGAN: {upscaler} ({scale_str})")
        
        details_lines.append(f"Index: {evt.index + 1} of {len(_image_gallery)}")
        details = "\n".join(details_lines)
        
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


def create_metadata_string(prompt, seed, steps, time_shift, backend, width, height, model_name, lora_configs=None, upscaler_name=None, upscale_factor=None, latent_scale=None, latent_denoise=None, latent_interp=None):
    """Create a metadata string for embedding in images"""
    lines = [
        prompt,
        "",
        "---",
        f"Backend: {backend}",
        f"Model: {model_name}",
        f"Seed: {seed}",
        f"Size: {width}×{height}",
        f"Steps: {steps}",
        f"Time Shift: {time_shift}",
    ]
    
    # Add LoRA info if present
    if lora_configs:
        lora_strs = [f"{c['name']}: {c['scale']}" for c in lora_configs]
        lines.append(f"LoRAs: {', '.join(lora_strs)}")
    
    # Add latent upscale info if used
    if latent_scale and latent_scale > 1.0:
        lines.append(f"Latent Upscale: {latent_scale}× (denoise: {latent_denoise}, interp: {latent_interp})")
    
    # Add ESRGAN upscaler info if used
    if upscaler_name and upscaler_name != "None":
        scale_str = f"{upscale_factor}×" if upscale_factor else "4×"
        lines.append(f"ESRGAN: {upscaler_name} ({scale_str})")
    
    lines.extend([
        "",
        "Generated with Z-Image-Turbo-MLX",
        "https://github.com/FiditeNemini/z-image-turbo-mlx"
    ])
    
    return "\n".join(lines)


def save_image_with_metadata(pil_image, filepath, prompt, seed, steps, time_shift, backend, width, height, model_name, lora_configs=None, upscaler_name=None, upscale_factor=None, latent_scale=None, latent_denoise=None, latent_interp=None):
    """Save image with embedded metadata"""
    metadata_str = create_metadata_string(prompt, seed, steps, time_shift, backend, width, height, model_name, lora_configs, upscaler_name, upscale_factor, latent_scale, latent_denoise, latent_interp)
    
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
        png_info.add_text("model", model_name)
        if lora_configs:
            lora_strs = [f"{c['name']}: {c['scale']}" for c in lora_configs]
            png_info.add_text("loras", ", ".join(lora_strs))
        if latent_scale and latent_scale > 1.0:
            png_info.add_text("latent_scale", str(latent_scale))
            png_info.add_text("latent_denoise", str(latent_denoise))
            png_info.add_text("latent_interp", latent_interp)
        if upscaler_name and upscaler_name != "None":
            png_info.add_text("upscaler", upscaler_name)
            png_info.add_text("upscale_factor", str(upscale_factor))
        png_info.add_text("generator", "Z-Image-Turbo-MLX")
        png_info.add_text("generator_url", "https://github.com/FiditeNemini/z-image-turbo-mlx")
        pil_image.save(filepath, "PNG", pnginfo=png_info)
    else:
        # JPEG metadata using EXIF UserComment
        import piexif
        
        # Save image first
        pil_image.save(filepath, "JPEG", quality=95)
        
        # Add EXIF metadata
        try:
            exif_dict = piexif.load(filepath)
            # UserComment field - encode as ASCII with proper header
            # Format: charset code (8 bytes) + comment
            user_comment = b"UNICODE\x00" + metadata_str.encode('utf-16')
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


def generate_image(prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend, model_name, 
                   lora_configs=None, lora_tags="", 
                   latent_scale=1.0, latent_steps=0, latent_denoise=0.55, latent_interp='cubic',
                   upscaler_name="None", upscale_factor=2.0, cache_mode=None, progress=gr.Progress()):
    """Generate an image using selected backend and model
    
    Args:
        lora_configs: List of {"name": str, "scale": float, "trigger": str} dicts
        lora_tags: Pre-built string of LoRA tags to append to prompt
        latent_scale: Latent upscale factor (1.0-4.0, 1.0 = disabled)
        latent_steps: Hires refinement steps (0 = auto)
        latent_denoise: Denoising strength for latent upscale (0.0-1.0)
        latent_interp: Interpolation mode ('nearest', 'linear', 'cubic')
        upscaler_name: Name of ESRGAN upscaler to use (or "None" to skip)
        upscale_factor: ESRGAN scale factor (1.0-4.0)
        cache_mode: LeMiCa cache mode ('slow', 'medium', 'fast') or None for no caching
    """
    
    if not prompt.strip():
        raise gr.Error("Please enter a prompt")
    
    if not model_name:
        raise gr.Error(f"No model selected for {backend}")
    
    # Get dimensions from preset
    width, height = DIMENSION_PRESETS[base_resolution][aspect_ratio]
    
    # Handle random seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    seed = int(seed)
    
    # Ensure lora_configs is a list
    if lora_configs is None:
        lora_configs = []
    
    # Append LoRA tags to prompt if present
    if lora_tags:
        full_prompt = f"{prompt.strip()}, {lora_tags}"
        print(f"Full prompt with LoRA tags: {full_prompt}")
    else:
        full_prompt = prompt.strip()
    
    # Determine if ESRGAN upscaling is enabled
    will_upscale = (upscaler_name and upscaler_name != "None" and 
                    not upscaler_name.startswith("None") and upscale_factor > 1.0)
    
    # Determine if latent upscaling is enabled
    will_latent_upscale = latent_scale > 1.0 and latent_denoise > 0.0
    
    progress(0, desc=f"Loading {model_name}...")
    
    if backend == "MLX (Apple Silicon)":
        # Select the MLX model
        select_mlx_model(model_name)
        pil_image = generate_mlx(
            full_prompt, width, height, steps, time_shift, seed, progress, 
            lora_configs=lora_configs,
            latent_scale=latent_scale if will_latent_upscale else 1.0,
            latent_steps=latent_steps,
            latent_denoise=latent_denoise,
            latent_interp=latent_interp,
            cache_mode=cache_mode
        )
    else:
        # Select the PyTorch model
        select_pytorch_model(model_name)
        if lora_configs:
            print(f"Warning: LoRA support for PyTorch backend not yet implemented")
        if will_latent_upscale:
            print(f"Warning: Latent upscaling for PyTorch backend not yet implemented")
        pil_image = generate_pytorch(full_prompt, width, height, steps, time_shift, seed, progress)
    
    # Apply ESRGAN upscaling if enabled
    if will_upscale:
        progress(0.93, desc=f"ESRGAN {upscale_factor}× with {upscaler_name}...")
        original_size = pil_image.size
        pil_image = apply_upscaler(pil_image, upscaler_name, upscale_factor, progress)
        new_size = pil_image.size
        print(f"ESRGAN: {original_size[0]}×{original_size[1]} → {new_size[0]}×{new_size[1]}")
    
    progress(0.97, desc="Saving temporary files...")
    
    # Generate timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Ensure temp directory exists
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    png_path = os.path.join(TEMP_DIR, f"{timestamp}.png")
    jpg_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
    
    # Save images with embedded metadata (use final image size after upscaling)
    final_width, final_height = pil_image.size
    actual_upscale_factor = upscale_factor if will_upscale else None
    actual_latent_scale = latent_scale if will_latent_upscale else None
    actual_latent_denoise = latent_denoise if will_latent_upscale else None
    actual_latent_interp = latent_interp if will_latent_upscale else None
    save_image_with_metadata(pil_image, png_path, prompt, seed, steps, time_shift, backend, final_width, final_height, model_name, lora_configs, upscaler_name, actual_upscale_factor, actual_latent_scale, actual_latent_denoise, actual_latent_interp)
    save_image_with_metadata(pil_image, jpg_path, prompt, seed, steps, time_shift, backend, final_width, final_height, model_name, lora_configs, upscaler_name, actual_upscale_factor, actual_latent_scale, actual_latent_denoise, actual_latent_interp)
    
    progress(1.0, desc="Done!")
    
    # Add to gallery with all generation parameters
    gallery_images = add_to_gallery(pil_image, prompt, seed, png_path, jpg_path, steps, time_shift, backend, final_width, final_height, model_name, lora_configs, upscaler_name, actual_upscale_factor, actual_latent_scale, actual_latent_denoise, actual_latent_interp)
    
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
                        active_model_dropdown = gr.Dropdown(
                            choices=get_available_mlx_models(),
                            value=get_current_model_name("MLX (Apple Silicon)"),
                            label="Model",
                            info="Available models for selected backend",
                        )
                        refresh_active_model_btn = gr.Button("🔄", scale=0, min_width=40)
                    
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
                        cache_mode = gr.Dropdown(
                            choices=["None", "slow", "medium", "fast"],
                            value="None",
                            label="⚡ LeMiCa Speed",
                            info="Cache mode: slow ~14%, medium ~22%, fast ~30% faster",
                        )
                    
                    # LoRA Section - Individual rows with spinners
                    with gr.Accordion("🎨 LoRA Settings", open=False) as lora_accordion:
                        gr.Markdown("*Enable LoRAs and adjust weights using the spinners (0.05 increments). Edit trigger words as needed.*")
                        
                        # Get initial LoRA data
                        initial_lora_choices = get_lora_choices()
                        
                        # Create individual LoRA slot rows
                        lora_rows = []  # Will store (checkbox, lora_path, trigger_textbox, weight) tuples
                        
                        if not initial_lora_choices:
                            gr.Markdown("*No LoRAs found. Place `.safetensors` files in `models/loras/`*")
                        
                        for i, (display_name, lora_path) in enumerate(initial_lora_choices):
                            # Get trigger words for this LoRA
                            trigger = get_lora_trigger_for_path(lora_path)
                            
                            with gr.Row():
                                enabled = gr.Checkbox(
                                    label="",
                                    show_label=False,
                                    value=False,
                                    min_width=30,
                                    scale=0,
                                )
                                # Hidden state to store the lora path
                                lora_path_state = gr.State(value=lora_path)
                                # Display name as static text
                                gr.Textbox(
                                    value=display_name,
                                    label="",
                                    show_label=False,
                                    interactive=False,
                                    scale=3,
                                )
                                # Editable trigger words textbox
                                trigger_textbox = gr.Textbox(
                                    value=trigger,
                                    label="",
                                    show_label=False,
                                    placeholder="trigger words",
                                    interactive=True,
                                    scale=2,
                                )
                                weight = gr.Number(
                                    label="",
                                    show_label=False,
                                    value=1.0,
                                    minimum=0.0,
                                    maximum=2.0,
                                    step=0.05,
                                    scale=1,
                                    min_width=80,
                                )
                                lora_rows.append((enabled, lora_path_state, trigger_textbox, weight))
                        
                        lora_tags_display = gr.Textbox(
                            label="Applied LoRAs (auto-appended to prompt)",
                            value="",
                            interactive=False,
                            placeholder="Enable LoRAs above to see tags here...",
                            info="These tags and triggers will be automatically added to your prompt",
                        )
                        
                        refresh_loras_btn = gr.Button("🔄 Refresh LoRA List", size="sm")
                        
                        gr.Markdown("---")
                        gr.Markdown("**💾 Save Model with Fused LoRAs**")
                        gr.Markdown("*Create a new model with selected LoRAs permanently baked in.*")
                        
                        with gr.Row():
                            fused_model_name = gr.Textbox(
                                label="New Model Name",
                                placeholder="e.g., my-custom-model",
                                info="Alphanumeric, hyphens, underscores only",
                                scale=3,
                            )
                        
                        with gr.Row():
                            save_mlx_checkbox = gr.Checkbox(
                                label="MLX",
                                value=True,
                                info="Apple Silicon",
                            )
                            save_pytorch_checkbox = gr.Checkbox(
                                label="PyTorch",
                                value=False,
                                info="Diffusers format",
                            )
                            save_comfyui_checkbox = gr.Checkbox(
                                label="ComfyUI",
                                value=False,
                                info="Single .safetensors",
                            )
                            save_fused_btn = gr.Button("💾 Save Fused Model", variant="secondary", scale=1)
                        
                        fused_model_status = gr.Textbox(
                            label="Status",
                            value="",
                            interactive=False,
                            visible=True,
                        )
                    
                    # Upscaler Section
                    with gr.Accordion("🔍 Upscaling", open=False) as upscaler_accordion:
                        gr.Markdown("**Latent Upscale** - Adds detail by upscaling in latent space before decoding")
                        
                        with gr.Row():
                            latent_scale = gr.Slider(
                                minimum=1.0,
                                maximum=4.0,
                                value=1.0,
                                step=0.25,
                                label="Latent Scale",
                                info="1.0 = disabled, higher = more detail",
                            )
                            latent_interp = gr.Dropdown(
                                choices=["nearest", "linear", "cubic"],
                                value="cubic",
                                label="Interpolation",
                                info="Upscale method (cubic = smoothest)",
                            )
                        
                        with gr.Row():
                            latent_steps = gr.Slider(
                                minimum=0,
                                maximum=20,
                                value=0,
                                step=1,
                                label="Hires Steps",
                                info="0 = auto (based on denoise)",
                            )
                            latent_denoise = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.55,
                                step=0.05,
                                label="Denoise Strength",
                                info="Higher = more change/detail",
                            )
                        
                        gr.Markdown("---")
                        gr.Markdown("**ESRGAN Upscale** - Pixel-space sharpening after generation")
                        
                        upscaler_dropdown = gr.Dropdown(
                            choices=get_upscaler_choices(),
                            value="None",
                            label="ESRGAN Model",
                            info="Select an upscaler or 'None' to skip",
                        )
                        
                        with gr.Row():
                            upscale_factor = gr.Slider(
                                minimum=1.0,
                                maximum=4.0,
                                value=2.0,
                                step=0.25,
                                label="ESRGAN Scale",
                                info="Scale factor (1× = no upscale)",
                            )
                        
                        if not UPSCALER_AVAILABLE:
                            gr.Markdown("*⚠️ ESRGAN not available. Install PyTorch: `pip install torch`*")
                        else:
                            gr.Markdown("*💡 Combine latent upscale + ESRGAN for maximum detail.*")
                    
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
                hf_precision = gr.Radio(
                    choices=["Original", "FP16", "FP8"],
                    value="FP16",
                    label="Model Precision",
                    info="Original: keep source precision, FP16: half precision (default), FP8: 8-bit float (smaller, may affect quality)",
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
                model_output_name = gr.Textbox(
                    label="Save As (optional)",
                    placeholder="Leave blank to use filename",
                    info="Custom name for the imported model",
                    scale=1,
                )
            with gr.Row():
                single_file_precision = gr.Radio(
                    choices=["Original", "FP16", "FP8"],
                    value="FP16",
                    label="Model Precision",
                    info="Original: keep source precision, FP16: half precision (default), FP8: 8-bit float (smaller, may affect quality)",
                )
            with gr.Row():
                convert_mlx_btn = gr.Button("📦 Import to MLX", variant="primary")
                convert_pytorch_btn = gr.Button("📦 Import to PyTorch", variant="secondary")
            convert_status = gr.Textbox(label="Import Status", interactive=False, lines=4)
            
            gr.Markdown("---")
            gr.Markdown("#### 📊 Installed Models")
            model_status = gr.Textbox(label="Model Inventory", interactive=False, lines=12, max_lines=30, autoscroll=False, value="Checking...")
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
    
    # Update model selector when backend changes
    def update_model_selector(backend_choice):
        models = get_available_models_for_backend(backend_choice)
        current = get_current_model_name(backend_choice)
        # If current model not in list, use first available or None
        if current not in models:
            current = models[0] if models else None
        return gr.update(choices=models, value=current)
    
    backend.change(
        fn=update_model_selector,
        inputs=[backend],
        outputs=[active_model_dropdown],
    )
    
    # Refresh model list for current backend
    def refresh_model_list_for_backend(backend_choice):
        models = get_available_models_for_backend(backend_choice)
        current = get_current_model_name(backend_choice)
        if current not in models:
            current = models[0] if models else None
        return gr.update(choices=models, value=current)
    
    refresh_active_model_btn.click(
        fn=refresh_model_list_for_backend,
        inputs=[backend],
        outputs=[active_model_dropdown],
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
    
    # LoRA event handlers for component-based UI
    def update_lora_tags_from_rows(*args):
        """Update LoRA tags from individual row components.
        
        Args are in order: enabled1, lora1, weight1, enabled2, lora2, weight2, ...
        """
        lora_states = []
        for i in range(0, len(args), 4):
            if i + 3 < len(args):
                enabled = args[i]
                lora_path = args[i + 1]
                trigger = args[i + 2]
                weight = args[i + 3]
                lora_states.append((enabled, lora_path, trigger, weight))
        
        return build_lora_tags_from_components(lora_states)
    
    # Flatten lora_rows into individual components for event binding
    all_lora_components = []
    for enabled, lora_path_state, trigger_textbox, weight in lora_rows:
        all_lora_components.extend([enabled, lora_path_state, trigger_textbox, weight])
    
    # Update tags when checkbox, trigger, or weight changes
    for enabled, lora_path_state, trigger_textbox, weight in lora_rows:
        enabled.change(
            fn=update_lora_tags_from_rows,
            inputs=all_lora_components,
            outputs=[lora_tags_display],
        )
        trigger_textbox.change(
            fn=update_lora_tags_from_rows,
            inputs=all_lora_components,
            outputs=[lora_tags_display],
        )
        weight.change(
            fn=update_lora_tags_from_rows,
            inputs=all_lora_components,
            outputs=[lora_tags_display],
        )
    
    # Refresh button - just show a message since LoRAs are fixed at load time
    refresh_loras_btn.click(
        fn=lambda: "Refresh the page to see new LoRAs",
        inputs=None,
        outputs=[lora_tags_display],
    )
    
    # Save fused model button
    def save_fused_with_loras(model_name, backend, source_model, save_mlx, save_pytorch, save_comfyui, *lora_args):
        """Wrapper that collects LoRA states and saves fused model"""
        # lora_args are: enabled1, lora1, trigger1, weight1, enabled2, lora2, trigger2, weight2, ...
        lora_states = []
        for i in range(0, len(lora_args), 4):
            if i + 3 < len(lora_args):
                enabled = lora_args[i]
                lora_path = lora_args[i + 1]
                trigger = lora_args[i + 2]
                weight = lora_args[i + 3]
                lora_states.append((enabled, lora_path, trigger, weight))
        
        # Get enabled LoRA configs
        lora_configs = get_enabled_loras_from_components(lora_states)
        
        return save_fused_model(model_name, backend, source_model, lora_configs, save_mlx, save_pytorch, save_comfyui)
    
    save_fused_btn.click(
        fn=save_fused_with_loras,
        inputs=[fused_model_name, backend, active_model_dropdown, save_mlx_checkbox, save_pytorch_checkbox, save_comfyui_checkbox] + all_lora_components,
        outputs=[fused_model_status],
    ).then(
        # Refresh model dropdown after saving
        fn=lambda be: gr.update(choices=get_available_mlx_models()) if be == "MLX (Apple Silicon)" else gr.update(choices=get_available_pytorch_models()),
        inputs=[backend],
        outputs=[active_model_dropdown],
    )
    
    # Wrapper function to collect LoRA states and generate
    def generate_with_loras(prompt, base_res, aspect, steps, time_shift, seed, backend, model, 
                           lat_scale, lat_interp, lat_steps, lat_denoise,
                           upscaler, upscale_by, lemica_cache, *lora_args):
        """Wrapper that collects LoRA component states and calls generate_image"""
        # lora_args are: enabled1, lora1, trigger1, weight1, enabled2, lora2, trigger2, weight2, ...
        lora_states = []
        for i in range(0, len(lora_args), 4):
            if i + 3 < len(lora_args):
                enabled = lora_args[i]
                lora_path = lora_args[i + 1]
                trigger = lora_args[i + 2]
                weight = lora_args[i + 3]
                lora_states.append((enabled, lora_path, trigger, weight))
        
        # Get enabled LoRA configs
        lora_configs = get_enabled_loras_from_components(lora_states)
        
        # Build tags string
        lora_tags = build_lora_tags_from_components(lora_states)
        
        # Process cache mode
        cache = lemica_cache if lemica_cache and lemica_cache.lower() != "none" else None
        
        return generate_image(prompt, base_res, aspect, steps, time_shift, seed, backend, model, 
                            lora_configs, lora_tags, 
                            lat_scale, lat_steps, lat_denoise, lat_interp,
                            upscaler, upscale_by, cache)
    
    generate_btn.click(
        fn=generate_with_loras,
        inputs=[prompt, base_resolution, aspect_ratio, steps, time_shift, seed, backend, active_model_dropdown, 
                latent_scale, latent_interp, latent_steps, latent_denoise,
                upscaler_dropdown, upscale_factor, cache_mode] + all_lora_components,
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
                # Check for precision info in config
                config_path = model_path / "config.json"
                precision_info = ""
                if config_path.exists():
                    try:
                        with open(config_path, "r") as f:
                            config = json.load(f)
                        if "precision" in config:
                            precision_info = f" [{config['precision']}]"
                    except Exception:
                        pass
                status_lines.append(f"  • {model}{precision_info}{marker}")
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
    
    def apply_precision_to_mlx_model(model_path, precision, progress=None):
        """
        Apply precision conversion to an MLX model.
        
        Args:
            model_path: Path to the MLX model directory
            precision: "Original", "FP16", or "FP8"
            progress: Optional Gradio progress callback
        """
        import mlx.core as mx
        
        model_path = Path(model_path)
        
        # Define which files to process
        weight_files = [
            "weights.safetensors",
            "vae.safetensors", 
            "text_encoder.safetensors"
        ]
        
        for weight_file in weight_files:
            file_path = model_path / weight_file
            if not file_path.exists():
                continue
                
            if progress:
                progress(desc=f"Converting {weight_file} to {precision}...")
            
            # Load weights
            weights = mx.load(str(file_path))
            
            if precision == "FP16":
                # Convert to float16
                converted_weights = {}
                for key, value in weights.items():
                    if value.dtype in [mx.float32, mx.bfloat16]:
                        converted_weights[key] = value.astype(mx.float16)
                    else:
                        converted_weights[key] = value
                mx.save_safetensors(str(file_path), converted_weights)
                
            elif precision == "FP8":
                # Use MLX's mxfp8 quantization mode
                # FP8 quantization in MLX requires specific handling
                # For transformer weights, we quantize to 8-bit using the mxfp8 mode
                converted_weights = {}
                for key, value in weights.items():
                    # Only quantize 2D+ tensors (weights), not biases or 1D params
                    if len(value.shape) >= 2 and value.shape[-1] % 32 == 0:
                        try:
                            # mxfp8 mode quantizes to 4 bits but stores as fp8 scales
                            # For actual fp8 storage, we convert to float16 first then apply
                            # the quantization at load time
                            # Since MLX's mxfp8 is actually 4-bit, we use 8-bit affine quantization
                            wq, scales, biases = mx.quantize(value, group_size=32, bits=8, mode="affine")
                            # Store quantized format - but for simplicity, we'll store dequantized fp16
                            # as true fp8 storage requires model architecture changes
                            dequantized = mx.dequantize(wq, scales, biases, group_size=32, bits=8)
                            converted_weights[key] = dequantized.astype(mx.float16)
                        except Exception:
                            # Fall back to float16 for tensors that can't be quantized
                            if value.dtype in [mx.float32, mx.bfloat16]:
                                converted_weights[key] = value.astype(mx.float16)
                            else:
                                converted_weights[key] = value
                    else:
                        # Keep small tensors and biases as-is but ensure float16
                        if value.dtype in [mx.float32, mx.bfloat16]:
                            converted_weights[key] = value.astype(mx.float16)
                        else:
                            converted_weights[key] = value
                mx.save_safetensors(str(file_path), converted_weights)
        
        # Update config to note the precision
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
            config["precision"] = precision
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
    
    def download_from_hf(model_id, custom_name, precision="FP16", progress=gr.Progress()):
        """Download model from Hugging Face and convert to MLX with specified precision"""
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
            
            progress(0.5, desc=f"Converting to MLX format ({precision})...")
            
            # Convert to MLX with specified precision
            class Args:
                pass
            Args.model_path = str(Path(pytorch_save_path) / "transformer")
            Args.output_path = mlx_save_path
            
            convert_to_mlx.args = Args()
            convert_weights(Args.model_path, Args.output_path)
            
            # Apply precision conversion if needed
            if precision != "Original":
                progress(0.8, desc=f"Applying {precision} precision...")
                apply_precision_to_mlx_model(mlx_save_path, precision, progress)
            
            precision_note = f" ({precision})" if precision != "Original" else ""
            progress(1.0, desc="Complete!")
            return f"✅ Successfully imported: {model_name}{precision_note}\n\n• PyTorch: {pytorch_save_path}\n• MLX: {mlx_save_path}\n\nYou can now select '{model_name}' from the model dropdown."
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
    
    def convert_single_file_to_format(file_path, output_name, target_format, precision="FP16", progress=gr.Progress()):
        """Import single-file safetensors checkpoint with specified precision"""
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
        
        # Check what's missing - we require at least transformer
        has_transformer = "Transformer" in components
        has_vae = "VAE" in components
        has_text_encoder = "Text Encoder" in components
        
        if not has_transformer:
            return f"❌ Incomplete checkpoint!\n\nFound components: {', '.join(components)}\n\nCheckpoint must contain at least a Transformer."
        
        missing_components = []
        if not has_vae:
            missing_components.append("VAE")
        if not has_text_encoder:
            missing_components.append("Text Encoder")
        
        # Determine target path based on format
        is_mlx = target_format == "mlx"
        if is_mlx:
            output_path = str(Path(MLX_MODELS_DIR) / model_name)
        else:
            output_path = str(Path(PYTORCH_MODELS_DIR) / model_name)
        
        try:
            if is_mlx:
                convert_single_file_to_mlx(str(file_path), output_path, progress, precision, missing_components)
                precision_note = f" ({precision})" if precision != "Original" else ""
                missing_note = f"\n\n📥 Downloaded missing: {', '.join(missing_components)}" if missing_components else ""
                return f"✅ Successfully imported: {model_name}{precision_note}{missing_note}\n\nSaved to: {output_path}\n\nYou can now select '{model_name}' from the model dropdown."
            else:
                # For PyTorch, use the ComfyUI to PyTorch converter
                progress(0.1, desc="Converting to PyTorch format...")
                from src.convert_comfyui_to_pytorch import convert_comfyui_to_pytorch
                convert_comfyui_to_pytorch(str(file_path), output_path, precision)
                progress(1.0, desc="Complete!")
                precision_note = f" ({precision})" if precision != "Original" else ""
                return f"✅ Successfully imported: {model_name}{precision_note}\n\nSaved to: {output_path}\n\nYou can now select '{model_name}' from the model dropdown (PyTorch backend)."
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def convert_to_mlx(file_path, output_name, precision="FP16", progress=gr.Progress()):
        """Import single-file checkpoint to MLX format"""
        return convert_single_file_to_format(file_path, output_name, "mlx", precision, progress)
    
    def convert_to_pytorch(file_path, output_name, precision="FP16", progress=gr.Progress()):
        """Import single-file checkpoint to PyTorch format"""
        return convert_single_file_to_format(file_path, output_name, "pytorch", precision, progress)
    
    # Wire up event handlers
    download_hf_btn.click(
        fn=download_from_hf,
        inputs=[hf_model_id, hf_model_name, hf_precision],
        outputs=[hf_status],
    )
    
    download_pytorch_btn.click(
        fn=download_pytorch_only,
        inputs=[hf_model_id, hf_model_name],
        outputs=[hf_status],
    )
    
    convert_mlx_btn.click(
        fn=convert_to_mlx,
        inputs=[single_file_path, model_output_name, single_file_precision],
        outputs=[convert_status],
    )
    
    convert_pytorch_btn.click(
        fn=convert_to_pytorch,
        inputs=[single_file_path, model_output_name, single_file_precision],
        outputs=[convert_status],
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


def cleanup_temp_files():
    """Clean up temporary files on exit"""
    global _image_gallery
    if os.path.exists(TEMP_DIR):
        try:
            shutil.rmtree(TEMP_DIR)
            print(f"Cleaned up temp directory: {TEMP_DIR}")
        except Exception as e:
            print(f"Warning: Could not clean up temp directory: {e}")
    _image_gallery = []


if __name__ == "__main__":
    import atexit
    atexit.register(cleanup_temp_files)
    
    # Check and setup models on startup
    print("\n" + "="*50)
    print("Z-Image-Turbo - Checking models...")
    print("="*50 + "\n")
    
    check_and_setup_models()
    
    print("\n" + "="*50)
    print("Starting Gradio interface...")
    print("="*50 + "\n")
    
    demo.launch()
