import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
import math
from typing import Optional

def precompute_freqs_cis_3d(dim: int, max_d: int, max_h: int, max_w: int, theta: float = 10000.0):
    # 将dim平均分成三部分
    dim_d = dim_h = dim_w = dim // 3
    
    # 为每个维度计算频率（每个维度使用相同的频率范围）
    max_dim = dim_d
    freqs_base = 1.0 / (theta ** (torch.arange(0, max_dim, 2, dtype=torch.float32) / max_dim))
    
    # 创建3D位置网格
    d_grid, h_grid, w_grid = torch.meshgrid(
        torch.arange(max_d), torch.arange(max_h), torch.arange(max_w), indexing='ij'
    )
    d_flat = d_grid.flatten().float()  # [N]
    h_flat = h_grid.flatten().float()  # [N]
    w_flat = w_grid.flatten().float()  # [N]
    
    # 为d维度计算频率（独立编码）
    num_freqs_d = dim_d // 2
    freqs_d = d_flat.unsqueeze(1) * freqs_base[:num_freqs_d].unsqueeze(0)  # [N, num_freqs_d]
    cos_d = torch.cos(freqs_d)  # [N, num_freqs_d]
    sin_d = torch.sin(freqs_d)  # [N, num_freqs_d]
    
    # 为h维度计算频率（独立编码）
    num_freqs_h = dim_h // 2
    freqs_h = h_flat.unsqueeze(1) * freqs_base[:num_freqs_h].unsqueeze(0)  # [N, num_freqs_h]
    cos_h = torch.cos(freqs_h)  # [N, num_freqs_h]
    sin_h = torch.sin(freqs_h)  # [N, num_freqs_h]
    
    # 为w维度计算频率（独立编码）
    num_freqs_w = dim_w // 2
    freqs_w = w_flat.unsqueeze(1) * freqs_base[:num_freqs_w].unsqueeze(0)  # [N, num_freqs_w]
    cos_w = torch.cos(freqs_w)  # [N, num_freqs_w]
    sin_w = torch.sin(freqs_w)  # [N, num_freqs_w]

    cos_combined = torch.cat([cos_d, cos_h, cos_w], dim=-1)  # [N, dim/2]
    sin_combined = torch.cat([sin_d, sin_h, sin_w], dim=-1)  # [N, dim/2]
    
    return torch.cat([cos_combined, sin_combined], dim=-1)  # [N, dim]


def apply_rotary_pos_emb_3d(x: torch.Tensor, freqs_cis: torch.Tensor):
    total_tokens, num_heads, head_dim = x.shape
    N_window = freqs_cis.shape[0]
    half_dim = head_dim // 2
    
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    token_idx = torch.arange(total_tokens, device=x.device) % N_window
    
    cos_freq = freqs_cis[token_idx, :half_dim].unsqueeze(1)  # [total_tokens, 1, half_dim]
    sin_freq = freqs_cis[token_idx, half_dim:2 * half_dim].unsqueeze(1)  # [total_tokens, 1, half_dim]

    return torch.cat([
        x1 * cos_freq - x2 * sin_freq,
        x1 * sin_freq + x2 * cos_freq
    ], dim=-1)


class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor_d=1, upscale_factor_h=1, upscale_factor_w=1):
        super().__init__()
        self.r_d = upscale_factor_d
        self.r_h = upscale_factor_h
        self.r_w = upscale_factor_w

    def forward(self, x):
        r_d, r_h, r_w = self.r_d, self.r_h, self.r_w
        B, C, D, H, W = x.shape
        assert C % (r_d*r_h*r_w) == 0, f"Channels {C} not divisible by {r_d*r_h*r_w}"
        c = C // (r_d*r_h*r_w)
        x = x.view(B, c, r_d, r_h, r_w, D, H, W)
        x = x.permute(0,1,5,2,6,3,7,4).contiguous()
        x = x.view(B, c, D*r_d, H*r_h, W*r_w)
        return x

class HorizonFiLM(nn.Module):
    def __init__(self, channels, num_steps):
        super().__init__()
        self.embed = nn.Embedding(num_steps, channels*2)

    def forward(self, x, step_id):
        # x: [B,C,1,H,W]
        gamma_beta = self.embed(step_id)   # [B,2C]
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta

class SpatialGate(nn.Module):
    def __init__(self, channels, num_experts):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_experts)
        hidden = max(channels * 2, 4)
        self.local_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, num_experts, 1)
        )
        self.local_alpha = nn.Parameter(torch.tensor(0.1))
        nn.init.zeros_(self.local_gate[-1].weight)
        nn.init.zeros_(self.local_gate[-1].bias)

    def forward(self, x):
        # x: [B,C,H,W]
        g = self.pool(x).flatten(1)
        global_logits = self.fc(g).unsqueeze(-1).unsqueeze(-1)
        local_logits = self.local_gate(x)
        w = torch.softmax(global_logits + self.local_alpha * local_logits, dim=1)  # [B,num_experts,H,W]
        return w

class MoEExperts(nn.Module):
    def __init__(self, channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(channels, 1, 1)
            )
            for _ in range(num_experts)
        ])
        self.gate = SpatialGate(channels, num_experts)

    def forward(self, x):
        weights = self.gate(x)  # [B,E,H,W]
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # [B,E,1,H,W]
        weights = weights[:, :, None, :, :]
        out = (outs * weights).sum(1)
        return out

class Decoder3D(nn.Module):
    def __init__(self, in_channels, num_timesteps=6, num_experts=4):
        super().__init__()
        self.num_timesteps = num_timesteps

        # --------- 共享部分 ----------
        self.stage1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GELU(),
            PixelShuffle3D(1,1,2)
        )

        self.stage2 = nn.Sequential(
            nn.Conv3d(in_channels//2, in_channels//2, 3, padding=1),
            nn.GELU(),
            PixelShuffle3D(2,3,3)
        )

        self.film = HorizonFiLM(in_channels//36, num_timesteps)

        self.stage3_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_channels//36, in_channels//36, 3, padding=1),
                nn.GELU(),
                PixelShuffle3D(1,2,2)
            )
            for _ in range(num_timesteps)
        ])

        self.experts = nn.ModuleList([
            MoEExperts(in_channels//144, num_experts=num_experts)
            for _ in range(num_timesteps)
        ])

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        x = self.stage1(x)
        x = self.stage2(x)

        steps = x.unbind(dim=2)

        outputs = []
        for i in range(self.num_timesteps):
            step = steps[i].unsqueeze(2)  # [B,C,1,H,W]

            step_ids = torch.full((B,), i, device=device, dtype=torch.long)
            step = self.film(step, step_ids)

            step = self.stage3_branches[i](step)
            step = step.squeeze(2)

            out = self.experts[i](step)
            outputs.append(out) #{[B,1,H,W],[B,1,H,W],...,[B,1,H,W]}

        out = torch.stack(outputs, dim=2)  # [B,1,T,H,W]
        return torch.sigmoid(out)


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def _to_tuple_3d(size):
    return size if isinstance(size, tuple) else (size, size, size)


def window_partition(x, window_size):
    Wd, Wh, Ww = _to_tuple_3d(window_size)
    B, D, H, W, C = x.shape
    x = x.view(B, D // Wd, Wd, H // Wh, Wh, W // Ww, Ww, C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, Wd, Wh, Ww, C)


def window_reverse(windows, window_size, D: int, H: int, W: int):
    Wd, Wh, Ww = _to_tuple_3d(window_size)
    B = windows.shape[0] // ((D // Wd) * (H // Wh) * (W // Ww))
    x = windows.view(B, D // Wd, H // Wh, W // Ww, Wd, Wh, Ww, -1)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), stride=(4, 4, 4), in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size  # (D, H, W)
        self.stride = stride          # (D, H, W)
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch_size
        
        # 计算padding
        pad_d = (pd - D % pd) % pd
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        
        x = self.proj(x)
        B, _, D, H, W = x.shape
        x = self.norm(x.flatten(2).transpose(1, 2))
        return x, D, H, W


class PatchMerging(nn.Module):
    def __init__(self, dim, merge_size=(2, 2, 2), norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.merge_size = merge_size
        Ms, Mt, Mu = merge_size
        self.reduction = nn.Linear(int(dim * Ms * Mt * Mu), 2 * dim, bias=False)
        self.norm = norm_layer(int(dim * Ms * Mt * Mu))

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        assert L == D * H * W, f"Expected L={D*H*W}, got L={L}"
        x = x.view(B, D, H, W, C)
        Ms, Mt, Mu = self.merge_size
        
        # Padding
        pad_d = (Ms - D % Ms) % Ms
        pad_h = (Mt - H % Mt) % Mt
        pad_w = (Mu - W % Mu) % Mu
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            D, H, W = D + pad_d, H + pad_h, W + pad_w
        
        # 3D patch merging
        x_list = [
            x[:, d::Ms, h::Mt, w::Mu, :]
            for d in range(Ms) for h in range(Mt) for w in range(Mu)
        ]
        x = torch.cat(x_list, -1).view(B, -1, Ms * Mt * Mu * C)
        return self.reduction(self.norm(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = _to_tuple_3d(window_size)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _apply_rope(self, q, k, spatial_shape):
        """应用RoPE到Q和K"""
        Wd, Wh, Ww = self.window_size
        D, H, W = spatial_shape
        B_, num_heads, N, head_dim = q.shape
        
        nW = (D // Wd) * (H // Wh) * (W // Ww)
        B = B_ // nW
        
        # 重塑并应用RoPE
        q = q.view(B, nW, num_heads, N, head_dim)
        k = k.view(B, nW, num_heads, N, head_dim)
        rope_freqs = precompute_freqs_cis_3d(head_dim, Wd, Wh, Ww).to(q.device)
        
        q_flat = q.permute(0, 1, 3, 2, 4).reshape(B * nW * N, num_heads, head_dim)
        k_flat = k.permute(0, 1, 3, 2, 4).reshape(B * nW * N, num_heads, head_dim)
        
        q_rope = apply_rotary_pos_emb_3d(q_flat, rope_freqs)
        k_rope = apply_rotary_pos_emb_3d(k_flat, rope_freqs)
        
        # 恢复形状
        q = q_rope.view(B, nW, N, num_heads, head_dim).permute(0, 1, 3, 2, 4).reshape(B_, num_heads, N, head_dim)
        k = k_rope.view(B, nW, N, num_heads, head_dim).permute(0, 1, 3, 2, 4).reshape(B_, num_heads, N, head_dim)
        return q, k

    def forward(self, x, mask: Optional[torch.Tensor] = None, spatial_shape: Optional[tuple] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        
        if spatial_shape is not None:
            q, k = self._apply_rope(q, k, spatial_shape)
        
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class WindowTTT(nn.Module):
    """3D window Test-Time Training block adapted from ViT3.

    The inner loop uses one full-batch dot-product update per window. Most heads
    use the simplified SwiGLU inner model, while an extra branch uses a 3D
    depthwise convolution so local temporal-spatial structure can be encoded into
    the test-time-updated kernel.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, proj_drop=0.):
        super().__init__()
        self.window_size = _to_tuple_3d(window_size)
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3 + self.head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, self.head_dim))
        self.w2 = nn.Parameter(torch.empty(1, num_heads, self.head_dim, self.head_dim))
        self.w3 = nn.Parameter(torch.empty(self.head_dim, 1, 3, 3, 3))
        nn.init.trunc_normal_(self.w1, std=.02)
        nn.init.trunc_normal_(self.w2, std=.02)
        nn.init.trunc_normal_(self.w3, std=.02)

        self.proj = nn.Linear(dim + self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.scale = 27 ** -0.5

    def _apply_rope(self, q, k, spatial_shape):
        Wd, Wh, Ww = self.window_size
        D, H, W = spatial_shape
        B_, num_heads, N, head_dim = q.shape

        nW = (D // Wd) * (H // Wh) * (W // Ww)
        B = B_ // nW

        q = q.view(B, nW, num_heads, N, head_dim)
        k = k.view(B, nW, num_heads, N, head_dim)
        rope_freqs = precompute_freqs_cis_3d(head_dim, Wd, Wh, Ww).to(q.device)

        q_flat = q.permute(0, 1, 3, 2, 4).reshape(B * nW * N, num_heads, head_dim)
        k_flat = k.permute(0, 1, 3, 2, 4).reshape(B * nW * N, num_heads, head_dim)

        q_rope = apply_rotary_pos_emb_3d(q_flat, rope_freqs)
        k_rope = apply_rotary_pos_emb_3d(k_flat, rope_freqs)

        q = q_rope.view(B, nW, N, num_heads, head_dim).permute(0, 1, 3, 2, 4).reshape(B_, num_heads, N, head_dim)
        k = k_rope.view(B, nW, N, num_heads, head_dim).permute(0, 1, 3, 2, 4).reshape(B_, num_heads, N, head_dim)
        return q, k

    def inner_train_swiglu(self, k, v, w1, w2, lr=1.0):
        z1 = k @ w1
        z2 = k @ w2
        sig = torch.sigmoid(z2)
        gate = z2 * sig

        e = -v / float(v.shape[2]) * self.scale
        g1 = k.transpose(-2, -1) @ (e * gate)
        g2 = k.transpose(-2, -1) @ (e * z1 * (sig * (1.0 + z2 * (1.0 - sig))))

        g1 = g1 / (g1.norm(dim=-2, keepdim=True) + 1.0)
        g2 = g2 / (g2.norm(dim=-2, keepdim=True) + 1.0)
        return w1 - lr * g1, w2 - lr * g2

    def inner_train_3d_dwc(self, k, v, w, lr=1.0):
        B, C, D, H, W = k.shape
        e = -v / float(D * H * W) * self.scale
        k_pad = F.pad(k, (1, 1, 1, 1, 1, 1))

        outs = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    zs, ys, xs = 1 + dz, 1 + dy, 1 + dx
                    dot = (k_pad[:, :, zs:zs + D, ys:ys + H, xs:xs + W] * e).sum(dim=(-3, -2, -1))
                    outs.append(dot)

        g = torch.stack(outs, dim=-1).reshape(B * C, 1, 3, 3, 3)
        g_norm = torch.linalg.vector_norm(g.flatten(2), dim=-1).view(B * C, 1, 1, 1, 1)
        g = g / (g_norm + 1.0)
        return w.repeat(B, 1, 1, 1, 1) - lr * g

    def forward(self, x, mask: Optional[torch.Tensor] = None, spatial_shape: Optional[tuple] = None):
        B_, N, C = x.shape
        Wd, Wh, Ww = self.window_size
        assert N == Wd * Wh * Ww, f"Expected window tokens {Wd * Wh * Ww}, got {N}"

        d = self.head_dim
        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [C, C, C, d, d, d], dim=-1)
        q1 = q1.reshape(B_, N, self.num_heads, d).transpose(1, 2)
        k1 = k1.reshape(B_, N, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(B_, N, self.num_heads, d).transpose(1, 2)

        if spatial_shape is not None:
            q1, k1 = self._apply_rope(q1, k1, spatial_shape)

        q2 = q2.reshape(B_, Wd, Wh, Ww, d).permute(0, 4, 1, 2, 3)
        k2 = k2.reshape(B_, Wd, Wh, Ww, d).permute(0, 4, 1, 2, 3)
        v2 = v2.reshape(B_, Wd, Wh, Ww, d).permute(0, 4, 1, 2, 3)

        w1, w2 = self.inner_train_swiglu(k1, v1, self.w1, self.w2)
        w3 = self.inner_train_3d_dwc(k2, v2, self.w3)

        x1 = (q1 @ w1) * F.silu(q1 @ w2)
        x1 = x1.transpose(1, 2).reshape(B_, N, C)

        x2 = F.conv3d(q2.reshape(1, B_ * d, Wd, Wh, Ww), w3, padding=1, groups=B_ * d)
        x2 = x2.reshape(B_, d, N).transpose(1, 2)

        x = torch.cat([x1, x2], dim=-1)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.window_size = _to_tuple_3d(window_size)
        self.shift_size = _to_tuple_3d(shift_size)
        for i, (s, w) in enumerate(zip(self.shift_size, self.window_size)):
            assert 0 <= s < w, f"shift_size[{i}]={s} must be in [0, window_size[{i}]={w})"

        self.norm1 = norm_layer(dim)
        self.attn = WindowTTT(dim, self.window_size, num_heads, qkv_bias, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        D, H, W = self.D, self.H, self.W
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)
        
        # Padding
        Wd, Wh, Ww = self.window_size
        pad_d = (Wd - D % Wd) % Wd
        pad_h = (Wh - H % Wh) % Wh
        pad_w = (Ww - W % Ww) % Ww
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

        # Shift
        has_shift = any(s > 0 for s in self.shift_size)
        if has_shift:
            x = torch.roll(x, shifts=tuple(-s for s in self.shift_size), dims=(1, 2, 3))
        else:
            attn_mask = None

        # Window attention
        x_windows = window_partition(x, self.window_size)
        N = Wd * Wh * Ww
        attn_windows = self.attn(
            x_windows.view(-1, N, C), 
            mask=attn_mask, 
            spatial_shape=(Dp, Hp, Wp)
        )
        
        # Merge windows
        x = window_reverse(attn_windows.view(-1, Wd, Wh, Ww, C), self.window_size, Dp, Hp, Wp)
        
        # Reverse shift
        if has_shift:
            x = torch.roll(x, shifts=self.shift_size, dims=(1, 2, 3))
        
        # Remove padding
        if pad_d or pad_h or pad_w:
            x = x[:, :D, :H, :W, :].contiguous()
        
        x = x.view(B, L, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 merge_size=(2, 2, 2), shift_size_tuple=None):
        super().__init__()
        self.window_size = _to_tuple_3d(window_size)
        self.shift_size = _to_tuple_3d(shift_size_tuple) if shift_size_tuple else (0, 0, 0)
        self.use_checkpoint = use_checkpoint
        self.merge_size = merge_size
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim, num_heads, window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            ) for i in range(depth)
        ])
        
        self.downsample = downsample(dim, merge_size, norm_layer) if downsample else None

    def create_mask(self, x, D, H, W):
        Wd, Wh, Ww = self.window_size
        sD, sH, sW = self.shift_size
        Dp, Hp, Wp = int(np.ceil(D / Wd)) * Wd, int(np.ceil(H / Wh)) * Wh, int(np.ceil(W / Ww)) * Ww
        
        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=x.device)
        # 为每个维度定义三个区域：主区域、中间区域、shift区域
        d_slices = (slice(0, -Wd), slice(-Wd, -sD), slice(-sD, None))
        h_slices = (slice(0, -Wh), slice(-Wh, -sH), slice(-sH, None))
        w_slices = (slice(0, -Ww), slice(-Ww, -sW), slice(-sW, None))
        
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        
        N = Wd * Wh * Ww
        mask_windows = window_partition(img_mask, self.window_size).view(-1, N)
        attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x, D, H, W):
        attn_mask = self.create_mask(x, D, H, W)
        
        for i, blk in enumerate(self.blocks):
            blk.D, blk.H, blk.W = D, H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask)
        
        if self.downsample is not None:
            x = self.downsample(x, D, H, W)
            Ms, Mt, Mu = self.merge_size
            D, H, W = D // Ms, H // Mt, W // Mu
        
        return x, D, H, W


class SwinTransformer(nn.Module):
    def __init__(self, patch_size=(4, 4, 4), stride=(4, 4, 4), in_chans=3,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_sizes=None, merge_sizes=None, shift_size_tuples=None,
                 mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, eps=1e-6):
        super().__init__()

        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            patch_size, stride, in_chans, embed_dim,
            norm_layer if patch_norm else None
        )
        self.pos_drop = nn.Dropout(drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=int(embed_dim * 2 ** i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                merge_size=merge_sizes[i] if i < self.num_layers - 1 else None,
                shift_size_tuple=shift_size_tuples[i] if shift_size_tuples else None
            ) for i in range(self.num_layers)
        ])

        self.norm = norm_layer(self.num_features)
        self.decoder = Decoder3D(self.num_features)
        self.apply(self._init_weights)
        self.eps = eps

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, C, D_in, H_in, W_in]
        x, Dp, Hp, Wp = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            x, Dp, Hp, Wp = layer(x, Dp, Hp, Wp)

        x = self.norm(x)  # [B, L, C]
        
        B, _, C = x.shape
        x = x.view(B, Dp, Hp, Wp, C)
        x = x.permute(0, 4, 1, 2, 3)  # [B, Dp, Hp, Wp, C] -> [B, C, Dp, Hp, Wp]
        
        x = self.decoder(x)
        
        return x

def model(**kwargs):
    in_chans = kwargs.pop('in_chans', 9)
    model = SwinTransformer(in_chans=in_chans,
                           patch_size=(1, 2, 2),
                           stride=(1, 2, 2),
                           embed_dim=108,
                           depths=(2, 6, 4),
                           num_heads=(3, 6, 12),
                           window_sizes=[(3, 5, 5), (3, 5, 5), (3, 5, 5)],
                           shift_size_tuples=[(1, 2, 2), (1, 2, 2), (1, 2, 2)],
                           merge_sizes=[(2, 3, 3), (1, 1, 2)])
    return model
