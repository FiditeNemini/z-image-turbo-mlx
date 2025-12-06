"""
ESRGAN/RRDB and SPAN Upscaler for MLX

Implements:
- Real-ESRGAN style upscaling using RRDB (Residual-in-Residual Dense Block) architecture
- SPAN (Swift Parameter-free Attention Network) architecture

Supports loading PyTorch .pth model files and running inference with MLX.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

# Default upscaler directory
UPSCALER_DIR = Path(__file__).parent.parent / "models" / "upscalers"


def get_available_upscalers(upscaler_dir: Optional[Path] = None, filter_supported: bool = False) -> List[str]:
    """
    Get list of available upscaler models.
    
    Args:
        upscaler_dir: Directory containing upscaler models. Defaults to models/upscalers/
        filter_supported: If True, only return upscalers with supported architecture (ESRGAN/RRDB)
        
    Returns:
        List of upscaler model names (without .pth extension)
    """
    if upscaler_dir is None:
        upscaler_dir = UPSCALER_DIR
    
    upscaler_dir = Path(upscaler_dir)
    if not upscaler_dir.exists():
        return []
    
    upscalers = []
    for f in upscaler_dir.glob("*.pth"):
        if filter_supported:
            # Check if this model is supported
            if is_supported_upscaler(f):
                upscalers.append(f.stem)
        else:
            upscalers.append(f.stem)
    
    return sorted(upscalers)


def is_supported_upscaler(model_path: Path) -> bool:
    """
    Check if an upscaler model has a supported architecture.
    
    Args:
        model_path: Path to the .pth model file
        
    Returns:
        True if the model uses a supported architecture (ESRGAN/RRDB only)
    """
    try:
        import torch
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # Handle different checkpoint formats
        if 'params' in state_dict:
            state_dict = state_dict['params']
        elif 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        
        architecture = detect_architecture(state_dict)
        # Only ESRGAN/RRDB is currently supported
        return architecture in ('esrgan', 'esrgan_new')
    except Exception:
        return False


def detect_architecture(state_dict: Dict[str, Any]) -> str:
    """
    Detect the upscaler architecture from its state dict keys.
    
    Returns:
        One of: 'esrgan', 'esrgan_new', 'span', 'unknown'
    """
    keys = list(state_dict.keys())
    
    # Check for ESRGAN/RRDB (old sequential format: model.0, model.1.sub.N.RDBX)
    if any('model.1.sub.' in k and 'RDB' in k for k in keys):
        return 'esrgan'
    
    # Check for ESRGAN/RRDB (new format: body.N.rdbX, conv_first, etc.)
    if any('body.' in k and 'rdb' in k for k in keys):
        return 'esrgan_new'
    
    # Check for SPAN (block_N.cX_r or conv_1.sk)
    if any('block_' in k and ('c1_r' in k or 'c2_r' in k) for k in keys) or any('conv_1.sk' in k for k in keys):
        return 'span'
    
    return 'unknown'


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) used in RRDB.
    
    5 conv layers with dense connections and LeakyReLU activation.
    """
    
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, kernel_size=3, padding=1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        self.scale = 0.2  # Residual scaling
    
    def __call__(self, x: mx.array) -> mx.array:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(mx.concatenate([x, x1], axis=-1)))
        x3 = self.lrelu(self.conv3(mx.concatenate([x, x1, x2], axis=-1)))
        x4 = self.lrelu(self.conv4(mx.concatenate([x, x1, x2, x3], axis=-1)))
        x5 = self.conv5(mx.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * self.scale + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    
    Combines 3 RDBs with residual connection.
    """
    
    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.scale = 0.2
    
    def __call__(self, x: mx.array) -> mx.array:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * self.scale + x


class RRDBNet(nn.Module):
    """
    RRDB Network for image super-resolution.
    
    Architecture:
    - First conv (3 -> num_feat)
    - N x RRDB blocks
    - Trunk conv
    - Upsampling (2x per upsample layer, using nearest neighbor + conv)
    - Final conv layers (num_feat -> 3)
    """
    
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        self.num_feat = num_feat
        
        # First conv
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1)
        
        # RRDB blocks
        self.body = [RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        
        # Trunk conv (after RRDB blocks)
        self.conv_body = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        
        # Upsampling layers (nearest neighbor upsample + conv)
        # For 4x: 2 upsampling layers (each does 2x)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        
        # High-resolution conv
        self.conv_hr = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
        
        # Final output conv
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
    
    def upsample_nearest(self, x: mx.array, scale: int) -> mx.array:
        """
        Nearest neighbor upsampling.
        """
        b, h, w, c = x.shape
        # Repeat along height and width
        x = mx.repeat(x, scale, axis=1)  # (b, h*scale, w, c)
        x = mx.repeat(x, scale, axis=2)  # (b, h*scale, w*scale, c)
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        # First conv
        feat = self.conv_first(x)
        
        # RRDB blocks
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        
        # Trunk conv with residual
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # Upsampling (nearest neighbor + conv)
        feat = self.lrelu(self.conv_up1(self.upsample_nearest(feat, 2)))
        feat = self.lrelu(self.conv_up2(self.upsample_nearest(feat, 2)))
        
        # HR conv and output
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        
        return out


# ============================================================================
# SPAN (Swift Parameter-free Attention Network) Architecture
# ============================================================================

class Conv3XC(nn.Module):
    """
    Conv3XC block used in SPAN - a re-parameterizable convolution block.
    
    During inference, uses a single fused 3x3 conv (eval_conv).
    The model weights come pre-fused for inference.
    """
    
    def __init__(self, in_channels: int, out_channels: int, gain: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # For inference, we use the fused eval_conv directly
        # The sk (skip) is a 1x1 conv for residual connection
        self.eval_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.sk = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        
        # Note: conv.0, conv.1, conv.2 are training-time weights that get fused into eval_conv
        # We don't need them for inference as the model is already fused
    
    def __call__(self, x: mx.array) -> mx.array:
        # Use the fused evaluation conv + skip connection
        return self.eval_conv(x) + self.sk(x)


class SPAB(nn.Module):
    """
    SPAN Attention Block (SPAB).
    
    Contains 3 Conv3XC layers with attention mechanism.
    Returns (out, out1, attention) for feature concatenation.
    """
    
    def __init__(self, in_channels: int, mid_channels: int = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        
        self.c1_r = Conv3XC(in_channels, mid_channels)
        self.c2_r = Conv3XC(mid_channels, mid_channels)
        self.c3_r = Conv3XC(mid_channels, in_channels)
    
    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        # Forward through the 3 conv blocks with activations
        out1 = self.c1_r(x)
        out1_act = nn.silu(out1)
        
        out2 = self.c2_r(out1_act)
        out2_act = nn.silu(out2)
        
        out3 = self.c3_r(out2_act)
        
        # SPAN attention: sigmoid(out3) - 0.5, then multiply with (out3 + x)
        sim_att = mx.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att
        
        return out, out1, sim_att


class SPANNet(nn.Module):
    """
    SPAN Network for image super-resolution.
    
    Architecture:
    - Initial conv (conv_1): 3 -> num_feat
    - N x SPAB blocks
    - Final conv (conv_2)
    - Concatenation conv (conv_cat): combines features from all blocks
    - Upsampler: pixel shuffle for 4x upscaling
    """
    
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 48,
        num_block: int = 6,
    ):
        super().__init__()
        self.scale = scale
        self.num_feat = num_feat
        self.num_block = num_block
        
        # Initial convolution
        self.conv_1 = Conv3XC(num_in_ch, num_feat)
        
        # SPAB blocks
        self.blocks = [SPAB(num_feat) for _ in range(num_block)]
        
        # Final conv before concat
        self.conv_2 = Conv3XC(num_feat, num_feat)
        
        # Concatenation conv: combines block_1 output, block_N output, and conv_2 output
        # That's 3 * num_feat channels -> num_feat
        # Actually looking at the weights, it's 4 * num_feat (192) -> num_feat (48)
        self.conv_cat = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1, padding=0)
        
        # Upsampler (pixel shuffle for 4x)
        # For 4x scale with num_feat=48, need to output 48 * 16 = 768 channels before shuffle
        # But the model has upsampler.0 with shape [48, 48, 3, 3]
        # This suggests it uses a different upsampling approach
        self.upsampler = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1)
    
    def pixel_shuffle(self, x: mx.array, scale: int) -> mx.array:
        """
        Pixel shuffle operation for upscaling.
        Rearranges (B, H, W, C*r*r) -> (B, H*r, W*r, C)
        """
        b, h, w, c = x.shape
        out_c = c // (scale * scale)
        
        # Reshape to (B, H, W, r, r, C_out)
        x = x.reshape(b, h, w, scale, scale, out_c)
        # Transpose to (B, H, r, W, r, C_out)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Reshape to (B, H*r, W*r, C_out)
        x = x.reshape(b, h * scale, w * scale, out_c)
        
        return x
    
    def upsample_nearest(self, x: mx.array, scale: int) -> mx.array:
        """Nearest neighbor upsampling."""
        x = mx.repeat(x, scale, axis=1)
        x = mx.repeat(x, scale, axis=2)
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        # Initial conv
        out_feature = self.conv_1(x)
        
        # Store intermediate features for concatenation
        # SPAN concatenates: initial, mid-block, last-block, conv_2
        block_outputs = []
        
        out = out_feature
        for i, block in enumerate(self.blocks):
            out = block(out)
            # Store first and last block outputs
            if i == 0:
                block_outputs.append(out)
            elif i == self.num_block - 1:
                block_outputs.append(out)
        
        # Final conv
        out = self.conv_2(out)
        block_outputs.append(out)
        
        # Also include the initial features
        block_outputs.insert(0, out_feature)
        
        # Concatenate all features
        out = mx.concatenate(block_outputs, axis=-1)
        
        # 1x1 conv to reduce channels
        out = self.conv_cat(out)
        
        # Residual with initial features
        out = out + out_feature
        
        # Upsample using nearest neighbor + conv (simpler than pixel shuffle)
        # The model seems to use a single conv for upsampling prep
        out = self.upsampler(out)
        
        # Use nearest neighbor for the actual 4x upscale
        out = self.upsample_nearest(out, self.scale)
        
        # Final projection to RGB is implicit in the upsampler
        # Actually, we need to output 3 channels
        # Looking at the architecture more carefully...
        
        return out


class SPANNetWithPixelShuffle(nn.Module):
    """
    SPAN Network with proper pixel shuffle upsampling.
    
    This version properly handles the upsampling as in the original SPAN.
    Includes input normalization with ImageNet mean values.
    """
    
    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 48,
        num_block: int = 6,
    ):
        super().__init__()
        self.scale = scale
        self.num_feat = num_feat
        self.num_block = num_block
        self.img_range = 255.0
        # RGB mean values (ImageNet-ish) - not a parameter, just a constant
        self._mean = mx.array([0.4488, 0.4371, 0.4040]).reshape(1, 1, 1, 3)
        
        # Initial convolution
        self.conv_1 = Conv3XC(num_in_ch, num_feat)
        
        # SPAB blocks (6 blocks for standard SPAN)
        self.blocks = [SPAB(num_feat) for _ in range(num_block)]
        
        # Final conv before concat
        self.conv_2 = Conv3XC(num_feat, num_feat)
        
        # Concatenation conv: out_feature + out_b6 + out_b1 + out_b5_2 = 4 * num_feat
        self.conv_cat = nn.Conv2d(num_feat * 4, num_feat, kernel_size=1, padding=0)
        
        # Upsampler conv - prepares for pixel shuffle
        # For 4x with pixel shuffle: num_feat -> num_out_ch * scale^2
        # 48 -> 3 * 16 = 48
        self.upsampler_conv = nn.Conv2d(num_feat, num_out_ch * scale * scale, kernel_size=3, padding=1)
    
    def pixel_shuffle(self, x: mx.array, scale: int) -> mx.array:
        """
        Pixel shuffle operation for upscaling.
        Rearranges (B, H, W, C*r*r) -> (B, H*r, W*r, C)
        """
        b, h, w, c = x.shape
        out_c = c // (scale * scale)
        
        # Reshape to (B, H, W, scale, scale, C_out)
        x = x.reshape(b, h, w, scale, scale, out_c)
        # Transpose to (B, H, scale, W, scale, C_out)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Reshape to (B, H*scale, W*scale, C_out)
        x = x.reshape(b, h * scale, w * scale, out_c)
        
        return x
    
    def __call__(self, x: mx.array) -> mx.array:
        # Input normalization: (x - mean) * img_range
        # Input x is in [0, 1], mean is ~0.44
        x = (x - self._mean) * self.img_range
        
        # Initial conv (no activation here - it's applied inside Conv3XC if needed)
        out_feature = self.conv_1(x)
        
        # Run through blocks, collecting intermediate outputs
        # SPAN concatenates: out_feature, out_b6, out_b1, out_b5_2 (from 2nd-to-last block)
        out = out_feature
        out_b1 = None  # First block intermediate
        out_b5_2 = None  # Second-to-last block intermediate (out1)
        
        for i, block in enumerate(self.blocks):
            out, out1, _ = block(out)
            if i == 0:
                out_b1 = out1  # First block's out1
            if i == self.num_block - 2:  # Second to last block (block_5 in 6-block config)
                out_b5_2 = out1
        
        out_b6 = out  # Last block output
        
        # Apply conv_2 to last block output
        out_b6 = self.conv_2(out_b6)
        
        # Concatenate: [out_feature, out_b6, out_b1, out_b5_2]
        out = mx.concatenate([out_feature, out_b6, out_b1, out_b5_2], axis=-1)
        
        # 1x1 conv to reduce channels
        out = self.conv_cat(out)
        
        # Upsample: conv then pixel shuffle
        out = self.upsampler_conv(out)
        out = self.pixel_shuffle(out, self.scale)
        
        return out
        
        return out


def load_upscaler(
    model_name: str,
    upscaler_dir: Optional[Path] = None,
) -> Union[RRDBNet, SPANNetWithPixelShuffle]:
    """
    Load an upscaler model from a .pth file.
    
    Supports:
    - ESRGAN/RRDB architecture
    - SPAN architecture
    
    Args:
        model_name: Name of the upscaler (without .pth extension)
        upscaler_dir: Directory containing upscaler models
        
    Returns:
        Loaded model (RRDBNet or SPANNetWithPixelShuffle)
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model architecture is not supported
    """
    if upscaler_dir is None:
        upscaler_dir = UPSCALER_DIR
    
    model_path = Path(upscaler_dir) / f"{model_name}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Upscaler model not found: {model_path}")
    
    # Load PyTorch state dict
    import torch
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    
    # Handle different checkpoint formats
    if 'params' in state_dict:
        state_dict = state_dict['params']
    elif 'params_ema' in state_dict:
        state_dict = state_dict['params_ema']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    # Detect architecture
    architecture = detect_architecture(state_dict)
    
    if architecture == 'span':
        return _load_span_upscaler(model_name, state_dict)
    elif architecture in ('esrgan', 'esrgan_new'):
        return _load_esrgan_upscaler(model_name, state_dict)
    else:
        raise ValueError(
            f"Unsupported upscaler architecture: {architecture}. "
            f"Model '{model_name}' uses an unknown architecture."
        )


def _load_span_upscaler(model_name: str, state_dict: Dict[str, Any]) -> SPANNetWithPixelShuffle:
    """Load a SPAN architecture upscaler."""
    # Detect number of blocks
    num_block = 0
    for key in state_dict.keys():
        if key.startswith('block_'):
            parts = key.split('.')
            block_num = int(parts[0].replace('block_', ''))
            num_block = max(num_block, block_num)
    
    # Get num_feat from conv_cat weights: [num_feat, num_feat*4, 1, 1]
    if 'conv_cat.weight' in state_dict:
        num_feat = state_dict['conv_cat.weight'].shape[0]
    else:
        num_feat = 48
    
    print(f"Loading upscaler: {model_name} (SPAN)")
    print(f"  Blocks: {num_block}, Features: {num_feat}")
    
    # Create model
    model = SPANNetWithPixelShuffle(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=num_feat,
        num_block=num_block,
    )
    
    # Convert and load weights
    mlx_weights = convert_span_to_mlx(state_dict, num_block)
    model.load_weights(list(mlx_weights.items()))
    
    return model


def _load_esrgan_upscaler(model_name: str, state_dict: Dict[str, Any]) -> RRDBNet:
    """Load an ESRGAN/RRDB architecture upscaler."""
    
    # Detect model parameters from state dict
    num_block = 0
    for key in state_dict.keys():
        if 'sub.' in key and 'RDB1' in key:
            parts = key.split('.')
            for i, p in enumerate(parts):
                if p == 'sub' and i + 1 < len(parts):
                    block_idx = int(parts[i + 1])
                    num_block = max(num_block, block_idx + 1)
    
    # Default to 23 blocks if not detected (standard ESRGAN)
    if num_block == 0:
        num_block = 23
    
    # Get num_feat from first conv
    if 'model.0.weight' in state_dict:
        first_weight = state_dict['model.0.weight']
        num_feat = first_weight.shape[0]
    elif 'conv_first.weight' in state_dict:
        first_weight = state_dict['conv_first.weight']
        num_feat = first_weight.shape[0]
    else:
        num_feat = 64
    
    # Get num_grow_ch from RDB conv
    num_grow_ch = 32
    for key in state_dict.keys():
        if 'RDB1.conv1.0.weight' in key or 'rdb1.conv1.weight' in key:
            num_grow_ch = state_dict[key].shape[0]
            break
    
    print(f"Loading upscaler: {model_name}")
    print(f"  Blocks: {num_block}, Features: {num_feat}, Grow channels: {num_grow_ch}")
    
    # Create model
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=num_feat,
        num_block=num_block,
        num_grow_ch=num_grow_ch,
    )
    
    # Convert and load weights
    mlx_weights = convert_pytorch_to_mlx(state_dict, num_block)
    model.load_weights(list(mlx_weights.items()))
    
    return model


def convert_pytorch_to_mlx(state_dict: Dict[str, Any], num_block: int) -> Dict[str, mx.array]:
    """
    Convert PyTorch ESRGAN state dict to MLX format.
    
    Handles the key mapping and tensor transposition.
    """
    mlx_weights = {}
    
    # Key mapping from PyTorch sequential format to our named format
    # PyTorch: model.0.weight -> conv_first.weight
    # PyTorch: model.1.sub.N.RDB1.conv1.0.weight -> body.N.rdb1.conv1.weight
    # etc.
    
    for pt_key, pt_value in state_dict.items():
        # Convert to numpy then MLX
        np_value = pt_value.numpy()
        
        # Transpose conv weights from PyTorch (out, in, H, W) to MLX (out, H, W, in)
        if 'weight' in pt_key and len(np_value.shape) == 4:
            np_value = np.transpose(np_value, (0, 2, 3, 1))
        
        # Map keys
        mlx_key = map_pytorch_key_to_mlx(pt_key, num_block)
        if mlx_key:
            mlx_weights[mlx_key] = mx.array(np_value)
    
    return mlx_weights


def map_pytorch_key_to_mlx(pt_key: str, num_block: int) -> Optional[str]:
    """
    Map a PyTorch state dict key to MLX model key.
    """
    # model.0 -> conv_first
    if pt_key == 'model.0.weight':
        return 'conv_first.weight'
    if pt_key == 'model.0.bias':
        return 'conv_first.bias'
    
    # model.1.sub.N.RDBX.convY.0 -> body.N.rdbX.convY
    # model.1.sub.{num_block} -> conv_body (trunk conv after RRDB blocks)
    if pt_key.startswith('model.1.sub.'):
        parts = pt_key.split('.')
        if len(parts) >= 4 and parts[3].isdigit():
            block_idx = int(parts[3])
            
            # Check if this is the final trunk conv (model.1.sub.{num_block})
            # It has just weight/bias, not RDB structure
            if block_idx == num_block and len(parts) == 5:
                # model.1.sub.23.weight -> conv_body.weight
                suffix = parts[4]  # weight or bias
                return f'conv_body.{suffix}'
            
            # Regular RRDB block (has RDB1/RDB2/RDB3 structure)
            if len(parts) >= 6:
                rdb_name = parts[4].lower()  # RDB1 -> rdb1
                conv_name = parts[5]  # conv1, conv2, etc.
                suffix = parts[-1]  # weight or bias
                return f'body.{block_idx}.{rdb_name}.{conv_name}.{suffix}'
    
    # model.3 -> conv_up1
    if pt_key.startswith('model.3.'):
        suffix = pt_key.split('.')[-1]
        return f'conv_up1.{suffix}'
    
    # model.6 -> conv_up2
    if pt_key.startswith('model.6.'):
        suffix = pt_key.split('.')[-1]
        return f'conv_up2.{suffix}'
    
    # model.8 -> conv_hr
    if pt_key.startswith('model.8.'):
        suffix = pt_key.split('.')[-1]
        return f'conv_hr.{suffix}'
    
    # model.10 -> conv_last
    if pt_key.startswith('model.10.'):
        suffix = pt_key.split('.')[-1]
        return f'conv_last.{suffix}'
    
    return None


def convert_span_to_mlx(state_dict: Dict[str, Any], num_block: int) -> Dict[str, mx.array]:
    """
    Convert PyTorch SPAN state dict to MLX format.
    
    Handles the key mapping and tensor transposition.
    """
    mlx_weights = {}
    
    for pt_key, pt_value in state_dict.items():
        # Convert to numpy then MLX
        np_value = pt_value.numpy()
        
        # Transpose conv weights from PyTorch (out, in, H, W) to MLX (out, H, W, in)
        if 'weight' in pt_key and len(np_value.shape) == 4:
            np_value = np.transpose(np_value, (0, 2, 3, 1))
        
        # Map keys
        mlx_key = map_span_key_to_mlx(pt_key, num_block)
        if mlx_key:
            mlx_weights[mlx_key] = mx.array(np_value)
    
    return mlx_weights


def map_span_key_to_mlx(pt_key: str, num_block: int) -> Optional[str]:
    """
    Map a PyTorch SPAN state dict key to MLX model key.
    
    PyTorch format:
    - conv_1.eval_conv.weight/bias -> conv_1.eval_conv.weight/bias
    - conv_1.sk.weight/bias -> conv_1.sk.weight/bias
    - block_N.cX_r.eval_conv.weight/bias -> blocks.N-1.cX_r.eval_conv.weight/bias
    - block_N.cX_r.sk.weight/bias -> blocks.N-1.cX_r.sk.weight/bias
    - conv_2.eval_conv/sk -> conv_2.eval_conv/sk
    - conv_cat.weight/bias -> conv_cat.weight/bias
    - upsampler.0.weight/bias -> upsampler_conv.weight/bias
    """
    # conv_1 (initial conv)
    if pt_key.startswith('conv_1.'):
        # Only use eval_conv and sk, skip training conv.0/1/2
        if 'conv.0' in pt_key or 'conv.1' in pt_key or 'conv.2' in pt_key:
            return None  # Skip training weights
        return pt_key  # Keep as-is for eval_conv and sk
    
    # conv_2 (final conv before concat)
    if pt_key.startswith('conv_2.'):
        if 'conv.0' in pt_key or 'conv.1' in pt_key or 'conv.2' in pt_key:
            return None  # Skip training weights
        return pt_key  # Keep as-is
    
    # block_N.cX_r (SPAB blocks)
    if pt_key.startswith('block_'):
        parts = pt_key.split('.')
        block_num = int(parts[0].replace('block_', ''))
        # Convert 1-indexed to 0-indexed for blocks list
        rest = '.'.join(parts[1:])
        
        # Skip training weights
        if 'conv.0' in rest or 'conv.1' in rest or 'conv.2' in rest:
            return None
        
        # Map block_1 -> blocks.0, block_2 -> blocks.1, etc.
        return f'blocks.{block_num - 1}.{rest}'
    
    # conv_cat (concatenation conv)
    if pt_key.startswith('conv_cat.'):
        return pt_key
    
    # upsampler.0 -> upsampler_conv
    if pt_key.startswith('upsampler.0.'):
        suffix = pt_key.replace('upsampler.0.', '')
        return f'upsampler_conv.{suffix}'
    
    return None


def upscale_image(
    model: Union[RRDBNet, SPANNetWithPixelShuffle],
    image: Image.Image,
    tile_size: int = 512,
    tile_overlap: int = 32,
) -> Image.Image:
    """
    Upscale an image using the ESRGAN model.
    
    Args:
        model: Loaded RRDBNet model
        image: Input PIL Image
        tile_size: Size of tiles for processing (to manage memory)
        tile_overlap: Overlap between tiles for seamless blending
        
    Returns:
        Upscaled PIL Image
    """
    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(image).astype(np.float32) / 255.0
    
    # Check if image is small enough to process in one go
    h, w = img_np.shape[:2]
    
    if h <= tile_size and w <= tile_size:
        # Process entire image at once
        output = _upscale_tile(model, img_np)
    else:
        # Process in tiles
        output = _upscale_tiled(model, img_np, tile_size, tile_overlap)
    
    # Convert back to PIL Image
    output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(output)


def _upscale_tile(model: RRDBNet, img_np: np.ndarray) -> np.ndarray:
    """
    Upscale a single tile/image.
    """
    # Add batch dimension: (H, W, C) -> (1, H, W, C)
    x = mx.array(img_np)[None, ...]
    
    # Run model
    output = model(x)
    mx.eval(output)
    
    # Remove batch dimension and convert to numpy
    return np.array(output[0])


def _upscale_tiled(
    model: RRDBNet,
    img_np: np.ndarray,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Upscale an image using tiled processing for memory efficiency.
    """
    scale = model.scale
    h, w, c = img_np.shape
    
    # Output dimensions
    out_h, out_w = h * scale, w * scale
    output = np.zeros((out_h, out_w, c), dtype=np.float32)
    weight_map = np.zeros((out_h, out_w, 1), dtype=np.float32)
    
    # Create blending weights (linear ramp at edges)
    def create_blend_weights(tile_h: int, tile_w: int, overlap: int) -> np.ndarray:
        weights = np.ones((tile_h, tile_w, 1), dtype=np.float32)
        
        if overlap > 0:
            # Fade in from left
            for i in range(overlap):
                weights[:, i, 0] *= i / overlap
            # Fade in from top
            for i in range(overlap):
                weights[i, :, 0] *= i / overlap
            # Fade out to right
            for i in range(overlap):
                weights[:, -(i+1), 0] *= i / overlap
            # Fade out to bottom
            for i in range(overlap):
                weights[-(i+1), :, 0] *= i / overlap
        
        return weights
    
    # Calculate tile positions
    step = tile_size - overlap
    y_positions = list(range(0, h - tile_size + 1, step))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    
    x_positions = list(range(0, w - tile_size + 1, step))
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    
    total_tiles = len(y_positions) * len(x_positions)
    tile_idx = 0
    
    for y in y_positions:
        for x in x_positions:
            tile_idx += 1
            print(f"  Processing tile {tile_idx}/{total_tiles}...")
            
            # Extract tile
            tile = img_np[y:y+tile_size, x:x+tile_size, :]
            
            # Upscale tile
            upscaled_tile = _upscale_tile(model, tile)
            tile_h, tile_w = upscaled_tile.shape[:2]
            
            # Create blend weights for this tile
            blend_weights = create_blend_weights(tile_h, tile_w, overlap * scale)
            
            # Output position
            out_y, out_x = y * scale, x * scale
            
            # Add to output with blending
            output[out_y:out_y+tile_h, out_x:out_x+tile_w, :] += upscaled_tile * blend_weights
            weight_map[out_y:out_y+tile_h, out_x:out_x+tile_w, :] += blend_weights
    
    # Normalize by weights
    output = output / np.maximum(weight_map, 1e-8)
    
    return output


# Convenience function for quick upscaling
def quick_upscale(
    image: Image.Image,
    model_name: str = "4x-UltraSharp",
    upscaler_dir: Optional[Path] = None,
) -> Image.Image:
    """
    Quick convenience function to upscale an image.
    
    Args:
        image: Input PIL Image
        model_name: Name of upscaler model to use
        upscaler_dir: Directory containing upscaler models
        
    Returns:
        Upscaled PIL Image (4x resolution)
    """
    model = load_upscaler(model_name, upscaler_dir)
    return upscale_image(model, image)


if __name__ == "__main__":
    # Test upscaler
    import sys
    
    print("Available upscalers:", get_available_upscalers())
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else "upscaled.png"
        model_name = sys.argv[3] if len(sys.argv) > 3 else "4x-UltraSharp"
        
        print(f"\nUpscaling {input_path} with {model_name}...")
        
        img = Image.open(input_path).convert("RGB")
        print(f"Input size: {img.size}")
        
        result = quick_upscale(img, model_name)
        print(f"Output size: {result.size}")
        
        result.save(output_path)
        print(f"Saved to {output_path}")
    else:
        print("\nUsage: python upscaler.py <input_image> [output_image] [model_name]")
