# Copyright (c) 2024 Alibaba Inc
# Self-contained neural network modules for Flow model
# No external CosyVoice/Matcha-TTS dependencies

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from utils import make_pad_mask, add_optional_chunk_mask, mask_to_bias


# =============================================================================
# Basic Building Blocks
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        """
        Args:
            x: timestep tensor of shape (batch_size,)
            scale: scaling factor for timestep, default 1000 (MUST match Matcha-TTS!)
        """
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        # CRITICAL: scale factor must be 1000 to match pretrained weights!
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepEmbedding(nn.Module):
    """MLP for timestep embedding"""
    def __init__(self, in_channels, time_embed_dim, act_fn="silu"):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = nn.SiLU() if act_fn == "silu" else nn.Mish()
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class Block1D(nn.Module):
    """Basic 1D convolutional block"""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x, mask):
        # Match Matcha-TTS: apply mask before and after block
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    """ResNet block with timestep embedding - matches Matcha-TTS structure"""
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out),
        )
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        # IMPORTANT: Always use Conv1d, never Identity - must match pretrained weights
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(self, x, mask, t):
        h = self.block1(x, mask)
        h = h + self.mlp(t).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask if mask is not None else x)
        return output


class Downsample1D(nn.Module):
    """1D downsampling"""
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1D(nn.Module):
    """1D upsampling"""
    def __init__(self, dim, use_conv_transpose=True):
        super().__init__()
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
        else:
            self.conv = nn.Conv1d(dim, dim, 3, 1, 1)
            self.use_conv_transpose = False

    def forward(self, x):
        if hasattr(self, 'use_conv_transpose') and not self.use_conv_transpose:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


# =============================================================================
# Attention Modules
# =============================================================================

class GELU(nn.Module):
    """
    GELU activation with linear projection.
    Matches diffusers GELU class structure with 'proj' attribute.
    """
    def __init__(self, dim_in, dim_out, approximate="tanh"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, x):
        x = self.proj(x)
        return F.gelu(x, approximate=self.approximate)


class GEGLU(nn.Module):
    """GEGLU activation with linear projection. Has 'proj' attribute."""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class SnakeBeta(nn.Module):
    """
    SnakeBeta activation function for audio synthesis.
    Uses separate trainable parameters for frequency (alpha) and magnitude (beta).
    Formula: x + 1/β * sin²(αx)

    This is crucial for audio waveform modeling as it provides periodic activation.
    """
    def __init__(self, dim_in, dim_out, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super().__init__()
        self.dim_out = dim_out if isinstance(dim_out, list) else [dim_out]
        self.proj = nn.Linear(dim_in, dim_out)

        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(self.dim_out) * alpha)
            self.beta = nn.Parameter(torch.zeros(self.dim_out) * alpha)
        else:
            self.alpha = nn.Parameter(torch.ones(self.dim_out) * alpha)
            self.beta = nn.Parameter(torch.ones(self.dim_out) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x):
        x = self.proj(x)
        if self.alpha_logscale:
            alpha = torch.exp(self.alpha)
            beta = torch.exp(self.beta)
        else:
            alpha = self.alpha
            beta = self.beta

        x = x + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(torch.sin(x * alpha), 2)
        return x


class FeedForward(nn.Module):
    """
    Feed-forward network with configurable activation.
    Matches diffusers FeedForward structure: net.0 is activation (with proj), net.2 is output Linear.
    """
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, activation_fn="geglu"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim

        # Select activation function - all must have 'proj' attribute for weight loading
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh")
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim)
        elif activation_fn == "snakebeta" or activation_fn == "snake":
            act_fn = SnakeBeta(dim, inner_dim)
        else:
            # Default to GELU to match CosyVoice-300M config
            act_fn = GELU(dim, inner_dim)

        self.net = nn.ModuleList([
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
        ])

    def forward(self, x):
        for module in self.net:
            x = module(x)
        return x


class Attention(nn.Module):
    """
    Multi-head attention compatible with diffusers.models.attention_processor.Attention
    Key naming: to_q, to_k, to_v, to_out.0 (matches pretrained weights)
    """
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.0, bias=False,
                 cross_attention_dim=None, upcast_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.upcast_attention = upcast_attention

        # Use cross_attention_dim if provided, otherwise use query_dim
        context_dim = cross_attention_dim if cross_attention_dim is not None else query_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=bias)

        # to_out is a ModuleList with [Linear, Dropout] to match diffusers naming: to_out.0.weight
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        ])

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        """
        Args:
            hidden_states: (batch, seq_len, dim)
            encoder_hidden_states: optional context for cross-attention
            attention_mask: attention bias mask
        """
        b, n, _ = hidden_states.shape
        h = self.heads

        # Use encoder_hidden_states for cross-attention, otherwise self-attention
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states

        q = self.to_q(hidden_states)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        if self.upcast_attention:
            q, k = q.float(), k.float()

        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            sim = sim + attention_mask

        attn = sim.softmax(dim=-1)

        if self.upcast_attention:
            attn = attn.to(v.dtype)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply to_out layers
        for module in self.to_out:
            out = module(out)
        return out


class BasicTransformerBlock(nn.Module):
    """
    Transformer block matching diffusers BasicTransformerBlock structure.
    Uses attn1 (not attn) to match pretrained weight naming.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        dropout=0.0,
        activation_fn="snakebeta",
        cross_attention_dim=None,
        attention_bias=False,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # 1. Self-Attention (named attn1 to match pretrained weights)
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )

        # 2. Cross-Attention (optional, named attn2)
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = nn.LayerNorm(dim)
            self.attn2 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                upcast_attention=upcast_attention,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward (norm3 to match pretrained weights)
        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None):
        # 1. Self-Attention
        norm_hidden = self.norm1(hidden_states)
        attn_output = self.attn1(
            norm_hidden,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=encoder_attention_mask if self.only_cross_attention else attention_mask,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention (if exists)
        if self.attn2 is not None:
            norm_hidden = self.norm2(hidden_states)
            attn_output = self.attn2(
                norm_hidden,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden)
        hidden_states = hidden_states + ff_output

        return hidden_states


# =============================================================================
# Positional Encoding
# =============================================================================

class RelPositionalEncoding(nn.Module):
    """Relative positional encoding"""
    def __init__(self, d_model, dropout_rate=0.0, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        self.extend_pe(x)
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1: self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self, offset, size):
        self.extend_pe(torch.tensor(0.0).expand(1, offset + size))
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - offset - size + 1: self.pe.size(1) // 2 - offset + size,
        ]
        return pos_emb


class LinearNoSubsampling(nn.Module):
    """Linear input layer (no subsampling)"""
    def __init__(self, idim, odim, dropout_rate, pos_enc):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(idim, odim),
            nn.LayerNorm(odim, eps=1e-5),
            nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(self, x, x_mask, offset=0):
        x = self.out(x)
        x, pos_emb = self.pos_enc(x)
        return x, pos_emb, x_mask


# =============================================================================
# Conformer Modules
# =============================================================================

class ConvolutionModule(nn.Module):
    """Conformer convolution module"""
    def __init__(self, channels, kernel_size, activation, norm="batch_norm", causal=False):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels, kernel_size=1, stride=1, padding=0)

        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0

        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size,
            stride=1, padding=padding, groups=channels
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation

    def forward(self, x, mask_pad=torch.ones((0, 0, 0), dtype=torch.bool), cache=torch.zeros((0, 0, 0))):
        """Compute convolution module.
        Args:
            x: Input tensor (#batch, time, channels).
            mask_pad: used for batch padding (#batch, 1, time)
            cache: left context cache (#batch, channels, cache_t)
        Returns:
            Output tensor (#batch, time, channels).
            New cache tensor.
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:
                x = F.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)
                assert cache.size(1) == x.size(1)
                x = torch.cat((cache, x), dim=2)
            assert x.size(2) > self.lorder
            new_cache = x[:, :, -self.lorder:]
        else:
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)

        # mask batch padding
        if mask_pad.size(2) > 0:
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache


class RelPositionMultiHeadedAttention(nn.Module):
    """Multi-head attention with relative positional encoding"""
    def __init__(self, n_head, n_feat, dropout_rate, key_bias=True):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.h = n_head

        self.linear_q = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_k = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_v = nn.Linear(n_feat, n_feat, bias=key_bias)
        self.linear_out = nn.Linear(n_feat, n_feat, bias=key_bias)

        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)

        self.dropout = nn.Dropout(p=dropout_rate)

    def rel_shift(self, x):
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, :x.size(-1) // 2 + 1]
        return x

    def forward_qkv(self, query, key, value):
        """Transform query, key and value."""
        batch_size = query.size(0)
        q = self.linear_q(query).view(batch_size, -1, self.h, self.d_k)
        k = self.linear_k(key).view(batch_size, -1, self.h, self.d_k)
        v = self.linear_v(value).view(batch_size, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector."""
        batch_size = value.size(0)
        if mask.size(2) > 0:  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            mask = mask[:, :, :, :scores.size(-1)]
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb, cache=torch.zeros((0, 0, 0, 0))):
        """Compute scaled dot product attention with relative positional encoding.

        Args:
            query: (batch, time1, size)
            key: (batch, time2, size)
            value: (batch, time2, size)
            mask: (batch, 1, time2) or (batch, time1, time2)
            pos_emb: (batch, time2, size) - positional embedding
            cache: (1, head, cache_t1, d_k * 2)
        """
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # Handle cache
        if cache.size(0) > 0:
            key_cache, value_cache = torch.split(cache, cache.size(-1) // 2, dim=-1)
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)
        new_cache = torch.cat((k, v), dim=-1)

        # Process positional embedding
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, time1, head, d_k) + (head, d_k) -> (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # Compute attention scores
        # matrix_ac: (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        # matrix_bd: (batch, head, time1, time2)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))

        # Apply rel_shift if shapes don't match (for espnet rel_pos_emb)
        if matrix_ac.shape != matrix_bd.shape:
            matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask), new_cache


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward layer"""
    def __init__(self, idim, hidden_units, dropout_rate, activation):
        super().__init__()
        self.w_1 = nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.w_2 = nn.Linear(hidden_units, idim)

    def forward(self, xs):
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


class ConformerEncoderLayer(nn.Module):
    """Conformer encoder layer"""
    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module

        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)

        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(
        self,
        x,
        mask,
        pos_emb,
        mask_pad=torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache=torch.zeros((0, 0, 0, 0)),
        cnn_cache=torch.zeros((0, 0, 0, 0)),
    ):
        # Macaron FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

        # Multi-head attention
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, cache=att_cache)
        x = residual + self.dropout(x_att)

        # Conv module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

        # FFN
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))

        # Final norm
        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


# =============================================================================
# Conformer Encoder
# =============================================================================

class ConformerEncoder(nn.Module):
    """Conformer encoder"""
    def __init__(
        self,
        input_size: int,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        cnn_module_kernel: int = 15,
        use_cnn_module: bool = True,
        macaron_style: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self._output_size = output_size

        # Embedding layer
        pos_enc = RelPositionalEncoding(output_size, positional_dropout_rate)
        self.embed = LinearNoSubsampling(input_size, output_size, dropout_rate, pos_enc)

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)

        activation = nn.SiLU()

        # Build encoder layers
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation),
                PositionwiseFeedForward(output_size, linear_units, dropout_rate, activation) if macaron_style else None,
                ConvolutionModule(output_size, cnn_module_kernel, activation, "layer_norm", causal) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            )
            for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs, xs_lens, decoding_chunk_size=0, num_decoding_left_chunks=-1):
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        xs, pos_emb, masks = self.embed(xs, masks)

        chunk_masks = add_optional_chunk_mask(
            xs, masks, False, False, decoding_chunk_size, 0, num_decoding_left_chunks
        )

        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, masks)

        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks


# =============================================================================
# Length Regulator
# =============================================================================

class InterpolateRegulator(nn.Module):
    """Length regulator using interpolation"""
    def __init__(self, channels: int, sampling_ratios: Tuple, out_channels: int = None, groups: int = 1):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels

        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(nn.Conv1d(channels, out_channels, 1, 1))
        self.model = nn.Sequential(*model)

    def forward(self, x, ylens=None):
        mask = (~make_pad_mask(ylens)).to(x).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='linear')
        out = self.model(x).transpose(1, 2).contiguous()
        return out * mask, ylens

    def inference(self, x1, x2, mel_len1, mel_len2, input_frame_rate=50):
        if x2.shape[1] > 40:
            x2_head = F.interpolate(x2[:, :20].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode='linear')
            x2_mid = F.interpolate(x2[:, 20:-20].transpose(1, 2).contiguous(), size=mel_len2 - int(20 / input_frame_rate * 22050 / 256) * 2, mode='linear')
            x2_tail = F.interpolate(x2[:, -20:].transpose(1, 2).contiguous(), size=int(20 / input_frame_rate * 22050 / 256), mode='linear')
            x2 = torch.concat([x2_head, x2_mid, x2_tail], dim=2)
        else:
            x2 = F.interpolate(x2.transpose(1, 2).contiguous(), size=mel_len2, mode='linear')
        if x1.shape[1] != 0:
            x1 = F.interpolate(x1.transpose(1, 2).contiguous(), size=mel_len1, mode='linear')
            x = torch.concat([x1, x2], dim=2)
        else:
            x = x2
        out = self.model(x).transpose(1, 2).contiguous()
        return out, mel_len1 + mel_len2


# =============================================================================
# Prompt Isolation Mask - 防止 Attention 泄漏
# =============================================================================

def create_prompt_isolation_mask(
    seq_len: int,
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    创建 Prompt Isolation Mask，阻止 target 看到 prompt

    遮罩设计:
    - prompt 区域 (query) 看 prompt 区域 (key) → 允许 (0)
    - prompt 区域 (query) 看 target 区域 (key) → 阻止 (-inf)
    - target 区域 (query) 看 prompt 区域 (key) → 阻止 (-inf)  ← 关键!
    - target 区域 (query) 看 target 区域 (key) → 允许 (0)

    Args:
        seq_len: 总序列长度 (prompt_len + target_len)
        prompt_len: prompt 区域的长度
        device: 设备
        dtype: 数据类型

    Returns:
        attention_bias: (1, 1, seq_len, seq_len) 的遮罩
    """
    mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=dtype)

    if prompt_len <= 0 or prompt_len >= seq_len:
        return mask

    # 阻止 target 看到 prompt (关键!)
    mask[:, :, prompt_len:, :prompt_len] = float('-inf')

    # 阻止 prompt 看到 target
    mask[:, :, :prompt_len, prompt_len:] = float('-inf')

    return mask


# =============================================================================
# Conditional Decoder (U-Net style) with Prompt Isolation
# =============================================================================

class ConditionalDecoder(nn.Module):
    """
    Conditional decoder for flow matching with Prompt Isolation support.

    【重要】集成了 Prompt Isolation Mask，在 Attention 计算时阻止 target 看到 prompt，
    从而减少语义泄漏。
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        channels=(256, 256),
        dropout=0.05,
        attention_head_dim=64,
        n_blocks=1,
        num_mid_blocks=2,
        num_heads=4,
        act_fn="snakebeta",
    ):
        super().__init__()
        channels = tuple(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Prompt Isolation 配置
        self.prompt_isolation_enabled = True
        self.prompt_isolation_len = 0  # 在 forward 之前设置

        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))

        for _ in range(num_mid_blocks):
            resnet = ResnetBlock1D(dim=channels[-1], dim_out=channels[-1], time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=channels[-1],
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2

            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=num_heads,
                    attention_head_dim=attention_head_dim,
                    dropout=dropout,
                    activation_fn=act_fn,
                )
                for _ in range(n_blocks)
            ])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))

        self.final_block = Block1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None):
        """
        Forward pass with Prompt Isolation.

        如果 self.prompt_isolation_len > 0，会在 Attention 计算时添加隔离遮罩，
        阻止 target 位置看到 prompt 位置的信息。
        """
        t = self.time_embeddings(t).to(t.dtype)
        t = self.time_mlp(t)

        x = pack([x, mu], "b * t")[0]

        if spks is not None:
            spks = repeat(spks, "b c -> b c t", t=x.shape[-1])
            x = pack([x, spks], "b * t")[0]
        if cond is not None:
            x = pack([x, cond], "b * t")[0]

        hiddens = []
        masks = [mask]

        # 获取 prompt isolation 配置
        isolation_enabled = getattr(self, 'prompt_isolation_enabled', False)
        prompt_len = getattr(self, 'prompt_isolation_len', 0)

        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = rearrange(x, "b c t -> b t c").contiguous()

            seq_len = x.size(1)
            attn_mask = mask_down.bool()
            attn_mask = attn_mask.expand(-1, seq_len, -1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            # 添加 Prompt Isolation Mask
            if isolation_enabled and prompt_len > 0:
                # 根据 downsampling 缩放 prompt_len
                scale = seq_len / mask.shape[-1]
                scaled_prompt_len = max(1, int(prompt_len * scale))
                if scaled_prompt_len < seq_len:
                    isolation_mask = create_prompt_isolation_mask(
                        seq_len, scaled_prompt_len, x.device, x.dtype
                    )
                    attn_mask = attn_mask + isolation_mask.squeeze(1)

            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = rearrange(x, "b c t -> b t c").contiguous()

            seq_len = x.size(1)
            attn_mask = mask_mid.bool()
            attn_mask = attn_mask.expand(-1, seq_len, -1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            # 添加 Prompt Isolation Mask
            if isolation_enabled and prompt_len > 0:
                scale = seq_len / mask.shape[-1]
                scaled_prompt_len = max(1, int(prompt_len * scale))
                if scaled_prompt_len < seq_len:
                    isolation_mask = create_prompt_isolation_mask(
                        seq_len, scaled_prompt_len, x.device, x.dtype
                    )
                    attn_mask = attn_mask + isolation_mask.squeeze(1)

            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()

        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = pack([x[:, :, :skip.shape[-1]], skip], "b * t")[0]
            x = resnet(x, mask_up, t)
            x = rearrange(x, "b c t -> b t c").contiguous()

            seq_len = x.size(1)
            attn_mask = mask_up.bool()
            attn_mask = attn_mask.expand(-1, seq_len, -1)
            attn_mask = mask_to_bias(attn_mask, x.dtype)

            # 添加 Prompt Isolation Mask
            if isolation_enabled and prompt_len > 0:
                scale = seq_len / mask.shape[-1]
                scaled_prompt_len = max(1, int(prompt_len * scale))
                if scaled_prompt_len < seq_len:
                    isolation_mask = create_prompt_isolation_mask(
                        seq_len, scaled_prompt_len, x.device, x.dtype
                    )
                    attn_mask = attn_mask + isolation_mask.squeeze(1)

            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t)
            x = rearrange(x, "b t c -> b c t").contiguous()
            x = upsample(x * mask_up)

        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask
