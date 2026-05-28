"""
Latent-Control Head (LatCH) — lightweight bidirectional Transformer that
predicts MIR control features from (noisy) VAE latents.

Moved from scripts/latch_model.py into the package so the Gradio inference
pipeline can import and use it.
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# Restrict SDPA to fast backends. MATH is ~19x slower at this head's shape and we
# never want a silent fallback. Through torch.compile, Inductor preserves SDPA as
# an external aten call so this priority still applies.
_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]


# ── Rotary Position Embedding ──────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = torch.cos(emb), torch.sin(emb)
        # Cast to inv_freq dtype for half-precision compat
        return cos.to(self.inv_freq.dtype), sin.to(self.inv_freq.dtype)


def apply_rotary_emb(x, cos, sin):
    b, s, h, d = x.shape
    d_half = d // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (x_rotated * sin)


# ── Transformer blocks ────────────────────────────────────────────────────

class LatCHAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with sdpa_kernel(_SDPA_BACKENDS):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class LatCHBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LatCHAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LatCHBlockAdaLN(nn.Module):
    """DiT-style adaLN-zero block. Each block receives t_emb and produces 6
    modulators (γ1,β1,α1 for attention path; γ2,β2,α2 for MLP path). The final
    linear is zero-initialized so all modulators are 0 at step 0 → residual
    contributions vanish (block is identity), and the model warms up safely.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LatCHAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
        )
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, t_emb, cos, sin):
        g1, b1, a1, g2, b2, a2 = self.adaLN_mod(t_emb).chunk(6, dim=-1)
        n1 = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        x = x + a1.unsqueeze(1) * self.attn(n1, cos, sin)
        n2 = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.mlp(n2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        emb = emb.to(self.mlp[0].weight.dtype)  # sinusoidal embeds are fp32; cast to model dtype
        return self.mlp(emb)


# ── Main model ────────────────────────────────────────────────────────────

class LatCH(nn.Module):
    """
    Latent-Control Head (LatCH).
    Bidirectional Transformer operating on VAE latents, predicting control features.
    ~5–7 M parameters with default hypers.
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=1,
        dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        t_injection="concat",
    ):
        super().__init__()
        assert t_injection in ("concat", "film", "adaln_zero"), f"unknown t_injection={t_injection!r}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth
        self.t_injection = t_injection

        self.latent_proj = nn.Linear(in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        # FiLM head: outputs (scale, shift) applied once after latent_proj.
        # Eliminates the prepend-token trick so T stays at 256 (FA-aligned: 17% faster
        # attention than T=257). Only present in 'film' mode; absent in legacy 'concat'.
        if t_injection == "film":
            self.t_film = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.t_film.weight)  # identity init: x untouched at step 0
            nn.init.zeros_(self.t_film.bias)
        # adaLN-zero: each block carries its own modulation MLP; T=256.
        block_cls = LatCHBlockAdaLN if t_injection == "adaln_zero" else LatCHBlock
        self.blocks = nn.ModuleList(
            [block_cls(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_channels)
        self.rotary_emb = RotaryEmbedding(dim // num_heads)

    def forward(self, x, t):
        """
        x : [B, in_channels, T]  (noisy latents)
        t : [B]                  (timesteps 0..1)
        Returns : [B, out_channels, T]
        """
        B, C, T_seq = x.shape
        x = x.transpose(1, 2)                       # → [B, T, C]
        x = self.latent_proj(x)
        t_emb = self.t_embedder(t)                   # [B, dim]

        if self.t_injection == "film":
            scale, shift = self.t_film(t_emb).chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            pos = torch.arange(T_seq, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.norm_final(x)
        elif self.t_injection == "adaln_zero":
            pos = torch.arange(T_seq, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for block in self.blocks:
                x = block(x, t_emb, cos, sin)
            x = self.norm_final(x)
        else:  # concat (legacy)
            x = torch.cat([t_emb.unsqueeze(1), x], dim=1)
            pos = torch.arange(T_seq + 1, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.norm_final(x)
            x = x[:, 1:, :]                          # strip t-token

        out = self.out_proj(x)                        # [B, T, out_channels]
        return out.transpose(1, 2)                    # [B, out_channels, T]


# ── Factory ────────────────────────────────────────────────────────────────

def load_latch_from_checkpoint(path: str, device="cpu") -> LatCH:
    """Load a LatCH model from a checkpoint, supporting three formats:

    Legacy: a bare state_dict (flat dict of param-name -> tensor).
    New:    {"state_dict": <state_dict>, "feature_name": str,
             "feature_stats": dict, "target_kind_default": str, ...}
    Fusion: same as New, plus "averaged_state_dict" — the Schedule-Free averaged
            iterate x_t (the deployable model per the SF contract). Prefer it
            over the live z_t in "state_dict" for inference.

    The returned model carries a ``model.metadata`` attribute with everything
    in the checkpoint other than the state-dict tensors (empty for legacy files).
    """
    raw = torch.load(path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        # FusionOpt heads: prefer averaged x_t over the live z_t for inference
        if "averaged_state_dict" in raw and isinstance(raw["averaged_state_dict"], dict):
            state = raw["averaged_state_dict"]
        else:
            state = raw["state_dict"]
        metadata = {
            k: v for k, v in raw.items()
            if k not in ("state_dict", "averaged_state_dict")
        }
    else:
        state = raw
        metadata = {}

    in_channels  = state["latent_proj.weight"].shape[1]
    dim          = state["latent_proj.weight"].shape[0]
    out_channels = state["out_proj.weight"].shape[0]
    depth        = sum(1 for k in state if k.endswith(".attn.qkv.weight"))
    num_heads_dim = state["rotary_emb.inv_freq"].shape[0] * 2
    num_heads    = dim // num_heads_dim
    # Infer t-injection mode from state-dict: 'adaln_zero' has per-block adaLN_mod tensors,
    # 'film' has a top-level t_film.* tensor, 'concat' has neither. Metadata may override.
    if metadata.get("t_injection"):
        t_injection = metadata["t_injection"]
    elif any(k.endswith(".adaLN_mod.1.weight") for k in state):
        t_injection = "adaln_zero"
    elif "t_film.weight" in state:
        t_injection = "film"
    else:
        t_injection = "concat"

    model = LatCH(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        t_injection=t_injection,
    )
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    model.metadata = metadata
    return model
