import mlx.core as mx
import mlx.nn as nn
import numpy as np

class Upsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, padding=1)

    def __call__(self, x):
        B, H, W, C = x.shape
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample2D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(self.channels, self.out_channels, kernel_size=3, stride=2, padding=padding)

    def __call__(self, x):
        if self.use_conv:
            x = self.conv(x)
        else:
            x = nn.AvgPool2d(kernel_size=2, stride=2)(x)
        return x

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=512, groups=32, eps=1e-6):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, pytorch_compatible=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        if temb_channels is not None:
            self.time_emb_proj = nn.Linear(temb_channels, out_channels)
        else:
            self.time_emb_proj = None
            
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, pytorch_compatible=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if self.in_channels != self.out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def __call__(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = nn.silu(h)
        h = self.conv1(h)
        
        if temb is not None and self.time_emb_proj is not None:
            temb = nn.silu(temb)
            h = h + self.time_emb_proj(temb)[:, None, None, :]
            
        h = self.norm2(h)
        h = nn.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
            
        return x + h

class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, norm_num_groups=32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.group_norm = nn.GroupNorm(norm_num_groups, dim, eps=1e-6, pytorch_compatible=True)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

    def __call__(self, x):
        B, H, W, C = x.shape
        x_in = x
        
        x = self.group_norm(x)
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, H * W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, H * W, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, H, W, C)
        out = self.to_out(out)
        
        return x_in + out

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers_per_block = config["layers_per_block"]
        
        # Initial convolution
        self.conv_in = nn.Conv2d(config["in_channels"], config["block_out_channels"][0], kernel_size=3, padding=1)
        
        self.down_blocks = []
        
        # Down blocks
        output_channel = config["block_out_channels"][0]
        for i, down_block_type in enumerate(config["down_block_types"]):
            input_channel = output_channel
            output_channel = config["block_out_channels"][i]
            is_final_block = i == len(config["block_out_channels"]) - 1
            
            block = []
            for _ in range(self.layers_per_block):
                block.append(ResnetBlock2D(in_channels=input_channel, out_channels=output_channel, temb_channels=None))
                input_channel = output_channel
                
            if not is_final_block:
                block.append(Downsample2D(output_channel, use_conv=True, out_channels=output_channel))
                
            self.down_blocks.append(nn.Sequential(*block))
            
        # MLX doesn't have ModuleList, so we just use a list and iterate manually or register carefully
        # But for weight loading, we need to match structure.
        # We can use a list of Sequentials.
        self.down_blocks = [block for block in self.down_blocks] # Just a list of Sequentials

        # Mid block
        self.mid_block = nn.Sequential(
            ResnetBlock2D(output_channel, output_channel, temb_channels=None),
            Attention(output_channel), # Assuming mid_block_add_attention=True
            ResnetBlock2D(output_channel, output_channel, temb_channels=None),
        )
        
        self.conv_norm_out = nn.GroupNorm(32, output_channel, eps=1e-6, pytorch_compatible=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(output_channel, 2 * config["latent_channels"], kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        
        for block in self.down_blocks:
            x = block(x)
            
        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers_per_block = config["layers_per_block"]
        num_out_channels = config["out_channels"]
        
        block_out_channels = config["block_out_channels"]
        reversed_block_out_channels = list(reversed(block_out_channels))
        
        # Initial convolution
        self.conv_in = nn.Conv2d(config["latent_channels"], reversed_block_out_channels[0], kernel_size=3, padding=1)
        
        # Mid block
        self.mid_block = nn.Sequential(
            ResnetBlock2D(reversed_block_out_channels[0], reversed_block_out_channels[0], temb_channels=None),
            Attention(reversed_block_out_channels[0]),
            ResnetBlock2D(reversed_block_out_channels[0], reversed_block_out_channels[0], temb_channels=None),
        )
        
        self.up_blocks = []
        
        # Up blocks
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(config["up_block_types"]):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = prev_output_channel
            
            is_final_block = i == len(config["block_out_channels"]) - 1
            
            block = []
            for _ in range(self.layers_per_block + 1):
                block.append(ResnetBlock2D(in_channels=input_channel, out_channels=output_channel, temb_channels=None))
                input_channel = output_channel
                
            if not is_final_block:
                block.append(Upsample2D(output_channel, use_conv=True, out_channels=output_channel))
                
            self.up_blocks.append(nn.Sequential(*block))
            
        self.up_blocks = [block for block in self.up_blocks]

        self.conv_norm_out = nn.GroupNorm(32, output_channel, eps=1e-6, pytorch_compatible=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(output_channel, num_out_channels, kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.conv_in(x)
        x = self.mid_block(x)
        
        for block in self.up_blocks:
            x = block(x)
            
        x = self.conv_norm_out(x)
        x = nn.silu(x)
        x = self.conv_out(x)
        return x

class AutoencoderKL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        
        if config.get("use_quant_conv", True):
            self.quant_conv = nn.Conv2d(2 * config["latent_channels"], 2 * config["latent_channels"], kernel_size=1)
        else:
            self.quant_conv = None
            
        if config.get("use_post_quant_conv", True):
            self.post_quant_conv = nn.Conv2d(config["latent_channels"], config["latent_channels"], kernel_size=1)
        else:
            self.post_quant_conv = None

    def encode(self, x):
        h = self.encoder(x)
        if self.quant_conv is not None:
            moments = self.quant_conv(h)
        else:
            moments = h
        mean, logvar = mx.split(moments, 2, axis=-1)
        return mean, logvar

    def decode(self, z):
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
