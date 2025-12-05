import mlx.core as mx
import mlx.nn as nn
import math
import numpy as np

# Constants from PyTorch source
ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

# LeMiCa cache schedules for 9-step inference
# Based on https://github.com/UnicomAI/LeMiCa/tree/main/LeMiCa4Z-Image
LEMICA_SCHEDULES = {
    "slow": [0, 1, 2, 3, 5, 7, 8],      # 7/9 steps computed (highest quality)
    "medium": [0, 1, 2, 4, 6, 8],       # 6/9 steps computed (balanced)
    "fast": [0, 1, 2, 5, 8],            # 5/9 steps computed (fastest)
}

def get_lemica_bool_list(cache_mode, num_steps=9):
    """Get boolean list for which steps to fully compute vs use cache.
    
    Args:
        cache_mode: 'slow', 'medium', 'fast', or None for disabled
        num_steps: Total number of inference steps
    
    Returns:
        List of booleans, True = compute, False = use cache
        Returns None if cache_mode is None/disabled
    """
    if cache_mode is None or cache_mode.lower() == "none":
        return None
    
    cache_key = cache_mode.lower()
    if cache_key not in LEMICA_SCHEDULES:
        print(f"Warning: Unknown LeMiCa mode '{cache_mode}', using 'medium'")
        cache_key = "medium"
    
    calc_steps = LEMICA_SCHEDULES[cache_key]
    bool_list = [i in calc_steps for i in range(num_steps)]
    return bool_list

class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, mid_size, bias=True),
            nn.SiLU(),
            nn.Linear(mid_size, out_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, dim, max_period=10000):
        # t: [B]
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None] * freqs[None]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x):
        # SwiGLU: w2(silu(w1(x)) * w3(x))
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))

class MRoPE(nn.Module):
    def __init__(self, dim, sections=[32, 48, 48], theta=256.0):
        super().__init__()
        self.dim = dim
        self.sections = sections
        self.theta = theta
            
    def compute_freqs_cis(self, position_ids):
        """
        position_ids: [B, L, 3] or [B, L, len(sections)]
        Returns: cos_emb, sin_emb [B, L, head_dim]
        """
        # position_ids: [B, L, 3]
        # We need to compute freqs for each section and concatenate
        
        cos_parts = []
        sin_parts = []
        
        for i, s in enumerate(self.sections):
            # Get position IDs for this section
            ids = position_ids[..., i] # [B, L]
            
            # Compute inverse frequencies
            inv_freq = 1.0 / (self.theta ** (mx.arange(0, s, 2).astype(mx.float32) / s))
            
            # Compute theta
            theta = ids[..., None] * inv_freq[None, None, :] # [B, L, s/2]
            
            # Compute cos/sin
            cos_theta = mx.cos(theta)
            sin_theta = mx.sin(theta)
            
            # Repeat for real/imag parts in RoPE
            cos_full = mx.repeat(cos_theta, 2, axis=-1) # [B, L, s]
            sin_full = mx.repeat(sin_theta, 2, axis=-1)
            
            cos_parts.append(cos_full)
            sin_parts.append(sin_full)
            
        cos_emb = mx.concatenate(cos_parts, axis=-1)
        sin_emb = mx.concatenate(sin_parts, axis=-1)
        
        return cos_emb, sin_emb

    def apply_rotary_emb(self, x, cos, sin):
        # x: [B, L, num_heads, head_dim]
        # cos, sin: [B, L, head_dim]
        
        # Expand cos/sin for heads
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        
        # Reshape x to [..., D//2, 2] to handle rotation
        *rest, D = x.shape
        x_reshaped = x.reshape(*rest, D//2, 2)
        
        x_real = x_reshaped[..., 0]
        x_imag = x_reshaped[..., 1]
        
        # Reshape cos/sin similarly (they are repeated, so we can just take every other)
        # Or just reshape: [B, L, 1, D//2, 2]
        cos_reshaped = cos.reshape(*cos.shape[:-1], D//2, 2)
        sin_reshaped = sin.reshape(*sin.shape[:-1], D//2, 2)
        
        cos_val = cos_reshaped[..., 0]
        sin_val = sin_reshaped[..., 0]
        
        # Rotate
        # x_new_real = x_real * cos - x_imag * sin
        # x_new_imag = x_real * sin + x_imag * cos
        
        out_real = x_real * cos_val - x_imag * sin_val
        out_imag = x_real * sin_val + x_imag * cos_val
        
        out = mx.stack([out_real, out_imag], axis=-1).flatten(-2)
        return out

class Attention(nn.Module):
    def __init__(self, dim, num_heads, head_dim, qk_norm=True, norm_eps=1e-5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        
        self.norm_q = nn.RMSNorm(head_dim, eps=norm_eps) if qk_norm else None
        self.norm_k = nn.RMSNorm(head_dim, eps=norm_eps) if qk_norm else None
        
        # RoPE helper
        self.mrope = MRoPE(head_dim) # Helper for application

    def __call__(self, x, cos_emb, sin_emb, mask=None):
        B, L, D = x.shape
        
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_heads, self.head_dim)
        v = v.reshape(B, L, self.num_heads, self.head_dim)
        
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.norm_k is not None:
            k = self.norm_k(k)
            
        # Apply RoPE
        q = self.mrope.apply_rotary_emb(q, cos_emb, sin_emb)
        k = self.mrope.apply_rotary_emb(k, cos_emb, sin_emb)
        
        # Attention
        q = q.transpose(0, 2, 1, 3) # [B, H, L, D]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        dots = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        
        if mask is not None:
            # mask: [B, L] or [B, 1, 1, L]
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            dots = dots + mask
            
        attn = mx.softmax(dots, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.to_out(out)
        
        return out

class ZImageTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, head_dim, mlp_dim, norm_eps=1e-5, qk_norm=True, modulation=True):
        super().__init__()
        self.dim = dim
        self.modulation = modulation
        
        self.attention = Attention(dim, num_heads, head_dim, qk_norm, norm_eps)
        self.feed_forward = FeedForward(dim, mlp_dim)
        
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        
        if modulation:
            self.adaLN_modulation = nn.Linear(ADALN_EMBED_DIM, 4 * dim, bias=True)

    def __call__(self, x, cos_emb, sin_emb, mask=None, adaln_input=None):
        if self.modulation:
            assert adaln_input is not None
            # PyTorch: scale_msa, gate_msa, scale_mlp, gate_mlp = chunk(4)
            chunks = self.adaLN_modulation(adaln_input).split(4, axis=-1)
            scale_msa, gate_msa, scale_mlp, gate_mlp = chunks
            
            gate_msa = mx.tanh(gate_msa)
            gate_mlp = mx.tanh(gate_mlp)
            scale_msa = 1.0 + scale_msa
            scale_mlp = 1.0 + scale_mlp
            
            # Expand for broadcasting
            scale_msa = scale_msa[:, None, :]
            gate_msa = gate_msa[:, None, :]
            scale_mlp = scale_mlp[:, None, :]
            gate_mlp = gate_mlp[:, None, :]
            
            # Attention Block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                cos_emb, sin_emb, mask
            )
            x = x + gate_msa * self.attention_norm2(attn_out)
            
            # FFN Block
            ffn_out = self.feed_forward(
                self.ffn_norm1(x) * scale_mlp
            )
            x = x + gate_mlp * self.ffn_norm2(ffn_out)
            
        else:
            # No modulation
            attn_out = self.attention(
                self.attention_norm1(x),
                cos_emb, sin_emb, mask
            )
            x = x + self.attention_norm2(attn_out)
            
            ffn_out = self.feed_forward(self.ffn_norm1(x))
            x = x + self.ffn_norm2(ffn_out)
            
        return x

class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels, bias=True)
        self.adaLN_modulation = nn.Linear(ADALN_EMBED_DIM, dim, bias=True)

    def __call__(self, x, c):
        # PyTorch: Sequential(SiLU, Linear)
        # MLX: silu then linear
        scale = self.adaLN_modulation(nn.silu(c))
        scale = 1.0 + scale[:, None, :]
        
        x = self.norm_final(x) * scale
        x = self.linear(x)
        return x

class ZImageTransformer2DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = self.dim // self.num_heads
        self.mlp_dim = config["intermediate_size"]
        self.in_channels = config["in_channels"]
        self.out_channels = config["in_channels"]
        
        # LeMiCa caching state
        self.enable_lemica = False
        self.lemica_bool_list = None
        self.lemica_step_counter = 0
        self.lemica_previous_residual = None
        
        # Embedders
        # all_x_embedder is a dict in PyTorch. We'll implement the one we need.
        # Assuming patch_size=2, f_patch_size=1 for now as per logs/defaults
        patch_size = 2
        f_patch_size = 1
        self.x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * self.in_channels, self.dim, bias=True)
        
        self.t_embedder = TimestepEmbedder(min(self.dim, ADALN_EMBED_DIM), mid_size=1024)
        
        # cap_embedder: RMSNorm -> Linear
        norm_eps = config.get("norm_eps", 1e-6)
        self.cap_embedder = nn.Sequential(
            nn.RMSNorm(config["text_embed_dim"], eps=norm_eps),
            nn.Linear(config["text_embed_dim"], self.dim, bias=True)
        )
        
        self.rope_embedder = MRoPE(self.head_dim)
        
        # Blocks
        self.noise_refiner = [
            ZImageTransformerBlock(self.dim, self.num_heads, self.head_dim, self.mlp_dim, modulation=True)
            for _ in range(config["n_refiner_layers"]) # 2
        ]
        
        self.context_refiner = [
            ZImageTransformerBlock(self.dim, self.num_heads, self.head_dim, self.mlp_dim, modulation=False)
            for _ in range(2) # 2
        ]
        
        self.layers = [
            ZImageTransformerBlock(self.dim, self.num_heads, self.head_dim, self.mlp_dim, modulation=True)
            for _ in range(config["num_hidden_layers"])
        ]
        
        self.final_layer = FinalLayer(self.dim, patch_size * patch_size * f_patch_size * self.out_channels)
        
        self.t_scale = 1000.0
        self.patch_size = patch_size
        self.f_patch_size = f_patch_size
    
    def reset_lemica_state(self):
        """Reset LeMiCa caching state for a new generation."""
        self.lemica_step_counter = 0
        self.lemica_previous_residual = None
    
    def configure_lemica(self, cache_mode, num_steps=9):
        """Configure LeMiCa caching for acceleration.
        
        Args:
            cache_mode: 'slow', 'medium', 'fast', or None to disable
            num_steps: Number of inference steps
        """
        self.reset_lemica_state()
        if cache_mode is None or cache_mode.lower() == "none":
            self.enable_lemica = False
            self.lemica_bool_list = None
        else:
            self.enable_lemica = True
            self.lemica_bool_list = get_lemica_bool_list(cache_mode, num_steps)
            computed_steps = sum(self.lemica_bool_list) if self.lemica_bool_list else num_steps
            print(f"LeMiCa enabled: {cache_mode} mode ({computed_steps}/{num_steps} steps computed)")

    def create_coordinate_grid(self, size):
        # size: (F, H, W)
        F, H, W = size
        # In PyTorch: flatten(0, 2) -> [F*H*W, 3]
        
        # MLX:
        f_ids = mx.arange(F)
        h_ids = mx.arange(H)
        w_ids = mx.arange(W)
        
        # Meshgrid
        # Note: MLX meshgrid indexing might differ.
        # We want [f, h, w] for each point.
        
        # Using broadcasting
        f_grid = mx.repeat(f_ids[:, None, None], H*W, axis=1).reshape(-1)
        h_grid = mx.repeat(mx.repeat(h_ids[None, :, None], F, axis=0), W, axis=2).reshape(-1)
        w_grid = mx.repeat(w_ids[None, None, :], F*H, axis=0).reshape(-1)
        
        # But wait, the order matters.
        # PyTorch: (F, H, W)
        # flatten(0, 2) means F is slowest, W is fastest.
        
        f_grid = mx.repeat(f_ids, H*W)
        h_grid = mx.tile(mx.repeat(h_ids, W), F)
        w_grid = mx.tile(w_ids, F*H)
        
        return mx.stack([f_grid, h_grid, w_grid], axis=-1)

    def patchify_and_embed(self, x, cap_feats):
        # x: [B, C, F, H, W]
        # cap_feats: [B, L_cap, D_cap]
        
        B = x.shape[0]
        
        # Process Caption
        # Assuming cap_feats is already padded or we handle it.
        # PyTorch adds padding to SEQ_MULTI_OF.
        # For simplicity in this port, assuming lengths are fine or we just use them.
        
        all_cap_pos_ids = []
        all_cap_feats_out = []
        
        for i in range(B):
            cap_feat = cap_feats[i]
            L_cap = cap_feat.shape[0]
            
            # Coordinate grid for caption: (L_cap, 1, 1) start=(1, 0, 0)
            # t=1+idx, h=0, w=0
            t_ids = mx.arange(L_cap) + 1
            h_ids = mx.zeros((L_cap,))
            w_ids = mx.zeros((L_cap,))
            
            pos_ids = mx.stack([t_ids, h_ids, w_ids], axis=-1)
            all_cap_pos_ids.append(pos_ids)
            all_cap_feats_out.append(cap_feat)
            
        # Process Image
        # x: [B, C, F, H, W]
        # Patchify
        pF = self.f_patch_size
        pH = self.patch_size
        pW = self.patch_size
        
        C, F, H, W = x.shape[1:]
        F_tokens = F // pF
        H_tokens = H // pH
        W_tokens = W // pW
        
        # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
        # MLX reshape/transpose
        x = x.reshape(B, C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
        x = x.transpose(0, 2, 4, 6, 3, 5, 7, 1) # [B, Ft, Ht, Wt, pF, pH, pW, C]
        x = x.reshape(B, F_tokens * H_tokens * W_tokens, pF * pH * pW * C)
        
        all_x_pos_ids = []
        
        for i in range(B):
            # Coordinate grid for image
            # start=(cap_len + 1, 0, 0)
            cap_len = all_cap_feats_out[i].shape[0]
            start_t = cap_len + 1
            
            # Grid (Ft, Ht, Wt)
            # t = start_t + f_idx
            f_ids = mx.arange(F_tokens) + start_t
            h_ids = mx.arange(H_tokens)
            w_ids = mx.arange(W_tokens)
            
            # Meshgrid
            # F is slowest
            f_grid = mx.repeat(f_ids, H_tokens * W_tokens)
            h_grid = mx.tile(mx.repeat(h_ids, W_tokens), F_tokens)
            w_grid = mx.tile(w_ids, F_tokens * H_tokens)
            
            pos_ids = mx.stack([f_grid, h_grid, w_grid], axis=-1)
            all_x_pos_ids.append(pos_ids)
            
        return x, all_cap_feats_out, all_x_pos_ids, all_cap_pos_ids

    def unpatchify(self, x, original_shape):
        # x: [B, L, D]
        # original_shape: (F, H, W)
        B = x.shape[0]
        F, H, W = original_shape
        C = self.out_channels
        
        pF = self.f_patch_size
        pH = self.patch_size
        pW = self.patch_size
        
        F_tokens = F // pF
        H_tokens = H // pH
        W_tokens = W // pW
        
        # Reverse: (f h w) (pf ph pw c) -> c f pf h ph w pw
        x = x.reshape(B, F_tokens, H_tokens, W_tokens, pF, pH, pW, C)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6) # [B, C, Ft, pF, Ht, pH, Wt, pW]
        x = x.reshape(B, C, F, H, W)
        
        return x

    def __call__(self, x, t, cap_feats):
        # x: [B, C, F, H, W]
        # t: [B]
        # cap_feats: [B, L_cap, D_cap]
        
        B = x.shape[0]
        
        # Store original shape
        original_shape = (x.shape[2], x.shape[3], x.shape[4])
        
        # Timestep embedding
        t = t * self.t_scale
        t_emb = self.t_embedder(t) # [B, D]
        
        # Patchify and Embed
        x_patches, cap_feats_list, x_pos_ids_list, cap_pos_ids_list = self.patchify_and_embed(x, cap_feats)
        
        # x_patches: [B, L_img, patch_dim]
        # Embed x
        x = self.x_embedder(x_patches)
        
        # Embed caption
        # Assuming batch size 1 or equal lengths for now, or we need to handle padding/masking
        # MLX doesn't like list of tensors for batch ops unless we stack/pad.
        # quick_pytorch_test.py uses batch size 1.
        
        cap_feats = mx.stack(cap_feats_list, axis=0) # [B, L_cap, D_cap]
        cap_feats = self.cap_embedder(cap_feats)
        
        # RoPE
        x_pos_ids = mx.stack(x_pos_ids_list, axis=0) # [B, L_img, 3]
        cap_pos_ids = mx.stack(cap_pos_ids_list, axis=0) # [B, L_cap, 3]
        
        x_cos, x_sin = self.rope_embedder.compute_freqs_cis(x_pos_ids)
        cap_cos, cap_sin = self.rope_embedder.compute_freqs_cis(cap_pos_ids)
        
        # Noise Refiner (x only)
        for i, layer in enumerate(self.noise_refiner):
            x = layer(x, x_cos, x_sin, mask=None, adaln_input=t_emb)
            
        # Context Refiner (cap only)
        for i, layer in enumerate(self.context_refiner):
            cap_feats = layer(cap_feats, cap_cos, cap_sin, mask=None)
            
        # Unified
        # Concatenate x and cap
        # PyTorch: unified = [x, cap]
        unified = mx.concatenate([x, cap_feats], axis=1) # [B, L_total, D]
        unified_cos = mx.concatenate([x_cos, cap_cos], axis=1)
        unified_sin = mx.concatenate([x_sin, cap_sin], axis=1)
        
        # Main Layers - with LeMiCa caching
        if self.enable_lemica and self.lemica_bool_list is not None:
            # Determine if we should compute or use cache
            step_idx = self.lemica_step_counter
            should_compute = self.lemica_bool_list[step_idx] if step_idx < len(self.lemica_bool_list) else True
            
            if should_compute:
                # Full computation - store residual for future cache reuse
                unified_input = unified
                for i, layer in enumerate(self.layers):
                    unified = layer(unified, unified_cos, unified_sin, mask=None, adaln_input=t_emb)
                # Store residual
                self.lemica_previous_residual = unified - unified_input
            else:
                # Use cached residual instead of full computation
                if self.lemica_previous_residual is not None:
                    unified = unified + self.lemica_previous_residual
                else:
                    # Fallback to full computation if no cache available
                    for i, layer in enumerate(self.layers):
                        unified = layer(unified, unified_cos, unified_sin, mask=None, adaln_input=t_emb)
            
            # Increment step counter
            self.lemica_step_counter += 1
        else:
            # Standard computation without caching
            for i, layer in enumerate(self.layers):
                unified = layer(unified, unified_cos, unified_sin, mask=None, adaln_input=t_emb)
            
        # Final Layer
        unified = self.final_layer(unified, t_emb)
        
        # Unpatchify
        # Extract x part
        L_img = x.shape[1]
        x_out = unified[:, :L_img]
        
        x_out = self.unpatchify(x_out, original_shape) # Pass original F, H, W
        
        return x_out
