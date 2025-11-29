import mlx.core as mx
import mlx.nn as nn
import math

class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight.astype(input_dtype) * hidden_states.astype(input_dtype)

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute cos/sin
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        self.inv_freq = inv_freq

    def __call__(self, x, seq_len):
        # x: [B, H, L, D]
        t = mx.arange(seq_len, dtype=mx.float32)
        freqs = mx.outer(t, self.inv_freq)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb)[None, None, :, :]
        sin = mx.sin(emb)[None, None, :, :]
        return cos, sin

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate((-x2, x1), axis=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # q, k: [B, H, L, D]
    # cos, sin: [1, 1, L, D]
    
    # Crop cos/sin to seq_len
    seq_len = q.shape[2]
    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

class Qwen2Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_attention_heads"]
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config["num_key_value_heads"]
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config["max_position_embeddings"]
        self.rope_theta = config["rope_theta"]
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # QK Norm (Qwen3 specific?)
        # Weights are (head_dim,), so applied per head
        self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config["rms_norm_eps"])
        
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        B, L, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_key_value_heads, self.head_dim)
        v = v.reshape(B, L, self.num_key_value_heads, self.head_dim)
        
        # Apply QK Norm (per head)
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        # Transpose for attention
        q = q.transpose(0, 2, 1, 3) # [B, H, L, D]
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # RoPE
        cos, sin = self.rotary_emb(v, L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        # GQA repeat
        if self.num_key_value_groups > 1:
            k = mx.repeat(k, self.num_key_value_groups, axis=1)
            v = mx.repeat(v, self.num_key_value_groups, axis=1)
            
        # Attention
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        
        if attention_mask is not None:
            scores = scores + attention_mask
            
        attn = mx.softmax(scores, axis=-1)
        out = attn @ v
        
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        out = self.o_proj(out)
        
        return out

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = Qwen2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, hidden_states, attention_mask=None, position_ids=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class Qwen2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [Qwen2DecoderLayer(config) for _ in range(config["num_hidden_layers"])]
        self.norm = Qwen2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def __call__(self, input_ids, attention_mask=None):
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Create causal mask
        # attention_mask is [B, L] (1 for keep, 0 for mask)
        B, L = input_ids.shape
        
        # Causal mask: 0 for attending, -inf for masked
        # Lower triangular is 0, upper is -inf
        mask = mx.triu(mx.full((L, L), -math.inf), k=1)
        mask = mask[None, None, :, :] # [1, 1, L, L]
        
        if attention_mask is not None:
            # Expand padding mask: [B, 1, 1, L]
            # attention_mask is 1 for keep, 0 for mask
            # Use mx.where to avoid 0 * inf = NaN
            padding_mask = mx.where(attention_mask == 1, 0, -math.inf)
            padding_mask = padding_mask[:, None, None, :]
            
            # Combine
            mask = mask + padding_mask
            
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=mask)
            
        hidden_states = self.norm(hidden_states)
        return hidden_states

class TextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Qwen2Model(config)
        
    def __call__(self, input_ids, attention_mask=None):
        return self.model(input_ids, attention_mask)
