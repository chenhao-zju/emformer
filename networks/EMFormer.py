import math
import time
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from flash_attn import flash_attn_qkvpacked_func, flash_attn_func, flash_attn_with_kvcache
import xformers.ops as xops
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager

# from .l2_loss import L2_LOSS, weighted_rmse_torch 
# from .test_triple_conv_cuda import TripleConvLayer
from l2_loss import L2_LOSS, weighted_rmse_torch 
from test_triple_conv_cuda import TripleConvLayer

COMPAT = False


class TripleConv2(nn.Module):
    def __init__(self, in_channels, out_channels, groups):
        super().__init__()
        
        self.conv_layers = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, 1, padding=0, groups=groups, bias=False),
                nn.Conv2d(in_channels, out_channels, 3, padding=1, groups=groups, bias=False), 
                nn.Conv2d(in_channels, out_channels, 5, padding=2, groups=groups, bias=False),
            ])

    def forward(self, x):
        outputs = [conv(x) for conv in self.conv_layers]
        output = sum(outputs)
        return output




class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x



def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pad_to_multiple(tensor, multiple, dim = -1, value = 0):
    seq_len = tensor.shape[dim]
    m = seq_len / multiple
    if m.is_integer():
    # if (m == m.round()).all().item():
        return tensor
    remainder = math.ceil(m) * multiple - seq_len
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(tensor, (*pad_offset, 0, remainder), value = value)

def cast_tuple(val, depth = 1):
    return val if isinstance(val, tuple) else ((val,) * depth)

# factory

def get_emtransformer(
    dim,
    *,
    depth,
    t_in,
    shorten_factor,
    attn_resampling,
    updown_sample_type,
    H,
    W,
    ff_mult,
    add_kv = False,
    **kwargs
):
    assert not (isinstance(depth, int) and shorten_factor), 'there does not need to be a shortening factor when only a single transformer block is indicated (depth of one integer value)'

    if isinstance(depth, int):
        return MidTransformer(dim = dim, H=H, W=W, depth = depth, t_in = t_in, ff_mult = ff_mult, add_kv = add_kv, **kwargs)

    return EMTransformer(dim = dim, depth = depth, t_in = t_in, shorten_factor = shorten_factor, attn_resampling = attn_resampling, updown_sample_type = updown_sample_type, H=H, W=W, ff_mult = ff_mult, add_kv = add_kv, **kwargs)
    

# up and down sample classes

class NaiveDownsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return reduce(x, 'b (n s) d -> b n d', 'mean', s = self.shorten_factor)

class NaiveUpsample(nn.Module):
    def __init__(self, shorten_factor):
        super().__init__()
        self.shorten_factor = shorten_factor

    def forward(self, x):
        return repeat(x, 'b n d -> b (n s) d', s = self.shorten_factor)

class LinearDownsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim * shorten_factor, dim)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = rearrange(x, 'b (n s) d -> b n (s d)', s = self.shorten_factor)
        return self.proj(x)

class LinearUpsample(nn.Module):
    def __init__(self, dim, shorten_factor):
        super().__init__()
        self.proj = nn.Linear(dim, dim * shorten_factor)
        self.shorten_factor = shorten_factor

    def forward(self, x):
        x = self.proj(x)
        return rearrange(x, 'b n (s d) -> b (n s) d', s = self.shorten_factor)

# classes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, add_kv=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.add_kv = add_kv

    def forward(self, x, **kwargs):
        if self.add_kv:
            output, kv = self.fn(self.norm(x), **kwargs)
            return output + x, kv
        else:
            return self.fn(self.norm(x), **kwargs) + x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiConvMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = TripleConvLayer(in_features, in_features)
        # self.fc1 = TripleConv(in_features, in_features, groups=in_features)
        # self.fc1 = TripleConv2(in_features, in_features, groups=in_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h = H, w = W)
        x = self.fc1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0., causal=False, qkv_bias=False, window_size=None, rel_pos_spatial=False, add_kv = False,):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        dim_head = int(dim//num_heads)
        self.scale = dim_head ** -0.5
        inner_dim = num_heads * dim_head
        self.p = dropout
        self.add_kv = add_kv

        self.to_q = nn.Linear(dim, inner_dim, bias = qkv_bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, H=None, W=None, mask=None, kv_cache=None):
        B, N, C = x.shape

        h, device = self.num_heads, x.device
        kv_input = default(context, x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, k, v))

        out = xops.memory_efficient_attention(
            q, k, v, 
            attn_bias=mask,  
            p=self.p,  
            scale=self.scale 
        ).reshape(B, N, C)


        return self.to_out(out)




class MultiConvAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0., causal=False, qkv_bias=False, window_size=None, rel_pos_spatial=False, add_kv = False,):
        super().__init__()
        self.num_heads = num_heads
        self.causal = causal
        dim_head = int(dim//num_heads)
        self.scale = dim_head ** -0.5
        inner_dim = num_heads * dim_head
        self.p = dropout
        self.add_kv = add_kv

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)
        self.aggreg = TripleConvLayer(inner_dim * 3, inner_dim * 3)
        # self.aggreg = TripleConv(inner_dim * 3, inner_dim * 3, groups=inner_dim * 3)
        # self.aggreg = TripleConv2(inner_dim * 3, inner_dim * 3, groups=inner_dim * 3)

        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context=None, H=None, W=None, mask=None, kv_cache=None):
        B, N, C = x.shape

        h, device = self.num_heads, x.device

        qkv = self.to_qkv(x)
        input_type = qkv.dtype

        qkv = qkv.to(torch.float32)
        qkv = rearrange(qkv, 'b (h w) c -> b c h w', h = H, w = W)
        qkv = self.aggreg(qkv)
        qkv = rearrange(qkv, 'b c h w -> b (h w) c')
        qkv = qkv.to(input_type).contiguous()

        q, k, v = qkv.chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h = h), (q, k, v))

        out = xops.memory_efficient_attention(
            q, k, v, 
            attn_bias=mask,  
            p=self.p,  
            scale=self.scale  
        ).reshape(B, N, C)


        return self.to_out(out)




def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
        attn,
        q,
        q_shape,
        k_shape,
        rel_pos_h,
        rel_pos_w,
):
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio)
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio)
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, :, None]
            + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, rel_pos_spatial=False, add_kv = False,):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rel_pos_spatial=rel_pos_spatial
        self.add_kv = add_kv

        if COMPAT:
            q_size = window_size[0]
            kv_size = window_size[1]
            rel_sp_dim = 2 * q_size - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.aggreg = TripleConvLayer(dim * 3, dim * 3)
        # self.aggreg = TripleConv(inner_dim * 3, inner_dim * 3, groups=inner_dim * 3)
        # self.aggreg = TripleConv2(inner_dim * 3, inner_dim * 3, groups=inner_dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context=None, H=None, W=None, mask=None, kv_cache=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # print(f'processing windows attention: {self.window_size}, {H}, {W}')
        B_, N, C = x.shape
        prompt_dim=0
        if H*W != N:
            prompt_dim = N-H*W
            prompt_token, x = x.split([prompt_dim, H*W], dim=1)

        x = x.reshape(B_, H, W, C)

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size)  # num_Windows*B, window_size, window_size, C
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # num_Windows*B, window_size*window_size, C

            
        B_w = x.shape[0]
        N_w = x.shape[1]

        qkv = self.qkv(x)
        input_type = qkv.dtype

        qkv = qkv.to(torch.float32)
        qkv = rearrange(qkv, 'b (h w) c -> b c h w', h = Hp, w = Wp)
        qkv = self.aggreg(qkv)
        qkv = rearrange(qkv, 'b c h w -> b (h w) c')
        qkv = qkv.to(input_type).reshape(B_w, N_w,  3, self.num_heads, C // self.num_heads).contiguous()

        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        x = xops.memory_efficient_attention(
            q, k, v, 
            attn_bias=None,  
            p=0.0, 
            scale=self.scale  
        ).reshape(B_w, N_w, C)

                   
        x = self.proj(x)
        
        if prompt_dim>0:  # split the prompt embedding from each window 
            prompt_token, x = x.split([prompt_dim, self.window_size[1] * self.window_size[0]], dim=1)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size, Hp, Wp)  # B H' W' C

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)

        
        if prompt_dim>0:  # split the prompt embedding from each window 

            prompt_token =prompt_token.narrow(0,0,B_) # (dimension, start, length) 
            x = torch.cat([prompt_token, x], dim=1)

        return x


# transformer classes
class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        causal = False,
        heads = 8,
        dim_head = 64,
        H = 32,
        W = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        norm_out = False,
        add_kv = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.H = round(H)
        self.W = round(W)
        self.add_kv = add_kv

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNormResidual(dim, Attention(dim, num_heads=heads, qkv_bias=True, window_size=None)),
                PreNormResidual(dim, Mlp(dim, hidden_features=ff_mult*dim, out_features=dim, drop=ff_dropout))
            ]))

        self.norm = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, x, context = None, mask = None):
        for attn, ff in self.layers:
            # x = attn(x, context = context, mask = mask)
            x = attn(x, context=context, H=self.H, W=self.W)
            x = ff(x)
        return self.norm(x)


# transformer classes
class MidTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        t_in,
        causal = False,
        heads = 8,
        dim_head = 64,
        H = 32,
        W = 64,
        attn_dropout = 0.,
        ff_mult = 4,
        ff_dropout = 0.,
        norm_out = False,
        add_kv = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.full_layers = nn.ModuleList([])
        self.H = round(H)
        self.W = round(W)
        self.add_kv = add_kv

        for i in range(depth):
            if depth > 10 and i%2 == 0:
                j = i // 2
                if j % 3 == 0:      window_size = (8, 8)
                elif j % 3 == 1:    window_size = (16, 4)
                else:               window_size = (4, 16)
                self.layers.append(nn.ModuleList([
                    PreNormResidual(dim, WindowAttention(dim, num_heads=heads, qkv_bias=True, window_size=window_size)),
                    PreNormResidual(dim, Mlp(dim, hidden_features=ff_mult*dim, out_features=dim, drop=ff_dropout))
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNormResidual(dim, MultiConvAttention(dim, num_heads=heads, qkv_bias=True, window_size=None, add_kv=self.add_kv), add_kv=self.add_kv),
                    PreNormResidual(dim, Mlp(dim, hidden_features=ff_mult*dim, out_features=dim, drop=ff_dropout))
                ]))

        self.norm2 = nn.LayerNorm(dim) if norm_out else nn.Identity()

    def forward(self, cur_state, context = None, mask = None, kv_caches = None):
        for i, (attn, ff) in enumerate( self.layers ):            
            cur_state = attn(cur_state, context=context, H=self.H, W=self.W)
            cur_state = ff(cur_state)

        return self.norm2(cur_state)



class EMTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        t_in,
        H = 128,
        W = 256,
        drop_rate = 0.0,
        shorten_factor = 2,
        attn_resampling = True,
        updown_sample_type = 'linear',
        heads = 8,
        dim_head = 64,
        causal = False,
        norm_out = True,
        ff_mult = 4,
        add_kv = False,
    ):
        super().__init__()
        assert len(depth) == 3, 'depth should be a tuple of length 3'
        assert updown_sample_type in {'naive', 'linear'}, 'downsample / upsample type must be either naive (average pool and repeat) or linear (linear projection and reshape)'

        pre_layers_depth, valley_depth, post_layers_depth = depth

        self.dim = dim
        self.t_in = t_in
        self.H = H
        self.W = W
        h = self.H / (shorten_factor ** 0.5)
        w = self.W / (shorten_factor ** 0.5)
        self.add_kv = add_kv

        # shorten_factor = shorten_factor*shorten_factor

        if isinstance(shorten_factor, (tuple, list)):
            shorten_factor, *rest_shorten_factor = shorten_factor
        elif isinstance(valley_depth, int):
            shorten_factor, rest_shorten_factor = shorten_factor, None
        else:
            shorten_factor, rest_shorten_factor = shorten_factor, shorten_factor

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
        )

        self.causal = causal
        self.shorten_factor = shorten_factor

        if updown_sample_type == 'naive':
            self.downsample = NaiveDownsample(shorten_factor)
            self.upsample   = NaiveUpsample(shorten_factor)
        elif updown_sample_type == 'linear':
            self.downsample = LinearDownsample(dim, shorten_factor)
            self.upsample   = LinearUpsample(dim, shorten_factor)
        else:
            raise ValueError(f'unknown updown_sample_type keyword value - must be either naive or linear for now')

        self.valley_transformer = get_emtransformer(
            shorten_factor = rest_shorten_factor,
            depth = valley_depth,
            t_in = self.t_in,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            causal = causal,
            H=h, 
            W=w,
            ff_mult=ff_mult,
            add_kv = self.add_kv,
            **transformer_kwargs
        )

        self.attn_resampling_pre_valley = Transformer(depth = 1, H=h, W=w, ff_mult=ff_mult, **transformer_kwargs) if attn_resampling else None
        self.attn_resampling_post_valley = Transformer(depth = 1, H=h, W=w, ff_mult=ff_mult, **transformer_kwargs) if attn_resampling else None

        self.pre_transformer = Transformer(depth = pre_layers_depth, causal = causal, H=self.H, W=self.W, ff_mult=ff_mult, **transformer_kwargs)
        self.post_transformer = Transformer(depth = post_layers_depth, causal = causal, H=self.H, W=self.W, ff_mult=ff_mult, **transformer_kwargs)
        self.norm_out = nn.LayerNorm(dim) if norm_out else nn.Identity()

    
    def shorten_layers(self, x, mask):
        # b : batch, n : sequence length, d : feature dimension, s : shortening factor
        self.s, self.b, self.n = self.shorten_factor, *x.shape[:2]

        # top half of emformer, pre-transformer layers
        x = self.pre_transformer(x, mask = mask)

        # pad to multiple of shortening factor, in preparation for pooling
        x = pad_to_multiple(x, self.s, dim = -2)
        if exists(mask):
            padded_mask = pad_to_multiple(mask, self.s, dim = -1, value = False)

        # save the residual, and for "attention resampling" at downsample and upsample
        x_residual = x.clone()

        # if autoregressive, do the shift by shortening factor minus one
        if self.causal:
            shift = self.s - 1
            x = F.pad(x, (0, 0, shift, -shift), value = 0.)

            if exists(mask):
                padded_mask = F.pad(padded_mask, (shift, -shift), value = False)

        # naive average pool
        downsampled = self.downsample(x)
        if exists(mask):
            downsampled_mask = reduce(padded_mask, 'b (n s) -> b n', 'sum', s = self.s) > 0
        else:
            downsampled_mask = None

        # pre-valley "attention resampling" - they have the pooled token in each bucket attend to the tokens pre-pooled
        if exists(self.attn_resampling_pre_valley):
            if exists(mask):
                attn_resampling_mask = rearrange(padded_mask, 'b (n s) -> (b n) s', s = self.s)
            else:
                attn_resampling_mask = None

            downsampled = self.attn_resampling_pre_valley(
                rearrange(downsampled, 'b n d -> (b n) () d'),
                rearrange(x, 'b (n s) d -> (b n) s d', s = self.s),
                mask = attn_resampling_mask
            )

            downsampled = rearrange(downsampled, '(b n) () d -> b n d', b = self.b)

        return downsampled, x_residual, downsampled_mask


    def largen_layers(self, x, x_residual, mask):
        valley_out = x.clone()

        # naive repeat upsample
        x = self.upsample(x)

        # add the residual
        x = x + x_residual

        # post-valley "attention resampling"
        if exists(self.attn_resampling_post_valley):
            x = self.attn_resampling_post_valley(
                rearrange(x, 'b (n s) d -> (b n) s d', s = self.s),
                rearrange(valley_out, 'b n d -> (b n) () d')
            )
            x = rearrange(x, '(b n) s d -> b (n s) d', b = 1)

        # bring sequence back to original length, if it were padded for pooling
        x = x[:, :self.n]

        # post-valley transformers
        x = self.post_transformer(x, mask = mask)
        x = self.norm_out(x)

        return x


    def forward(self, x, kv_caches = None, mask = None):
        # his_downsampled, _, _ = self.shorten_layers(his_state, mask)
        cur_downsampled, cur_residual, downsampled_mask = self.shorten_layers(x, mask)

        # the "valley" - either a regular transformer or another emformer
        cur_state = self.valley_transformer(cur_downsampled, kv_caches = kv_caches, mask = downsampled_mask)
        # cur_state = self.valley_transformer(cur_downsampled, mask = downsampled_mask)
       
        # print(x.shape, x_residual.shape)
        output = self.largen_layers(cur_state, cur_residual, downsampled_mask)

        return output


# main class

class EMTransformerCast(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        t_in = 1,
        in_chans = 70,
        out_chans = 70,
        H = 128,
        W = 256,
        patch_size = 2,
        drop_rate = 0.0,
        shorten_factor = 2,
        heads = 8,
        dim_head = 64,
        attn_resampling = True,
        updown_sample_type = 'naive',
        causal = True,
        ff_mult = 4,
        add_kv = False,
    ):
        super().__init__()

        self.dim = dim
        self.img_size = (H, W)
        self.patch_size = (patch_size, patch_size)
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.t_in = t_in
        self.H = H // self.patch_size[0]
        self.W = W // self.patch_size[1]
        self.add_kv = add_kv

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head = nn.Linear(self.dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        self.transformer = get_emtransformer(
            dim = dim,
            depth = depth,
            t_in = self.t_in,
            shorten_factor = shorten_factor,
            attn_resampling = attn_resampling,
            updown_sample_type = updown_sample_type,
            dim_head = dim_head,
            heads = heads,
            causal = causal,
            norm_out = True,
            H = self.H,
            W = self.W,
            ff_mult = ff_mult,
            add_kv = self.add_kv,
        )

        # self.to_logits = nn.Linear(dim, num_tokens)
        learn_log_variance=dict(flag=True, channels=self.out_chans, logvar_init=0., requires_grad=True)
        self.loss_gen = L2_LOSS(learn_log_variance=learn_log_variance)

        self.loss_weight = nn.Parameter(torch.tensor([-1.5708]))
        # self.loss_weight = nn.Parameter(torch.tensor([ 0.0 ]))


    def encoder(self, x):
        self.B = x.shape[0]

        x = self.patch_embed(x)
        # print('after embedding: ', x.shape)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # x = x.reshape(self.B, self.h, self.w, self.dim)
        return x
    
    def decoder(self, x):
        # x = rearrange(x, '(k b) n c -> k n (b c)', k = 1)

        x = self.head(x)
        x = rearrange(
            x,
            "b (h w) (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

    def forward(self, x, labels, kv_caches = None, mask = None):
        # his_inp, cur_inp = x[:, 0], x[:, -1]
        # his_state = self.encoder(his_inp)
        cur_state = self.encoder(x)

        out = self.transformer(cur_state, kv_caches = kv_caches, mask = mask)
        # out = self.transformer( cur_state, mask=mask)

        output = self.decoder(out)
        
        generated_loss = self.loss_gen(output, labels)
        rmse_loss = weighted_rmse_torch(output, labels).mean()

        # real_weight = 1 / (1 + torch.exp(self.loss_weight))
        real_weight = (torch.sin(self.loss_weight) + 1) / 2

        return output, (1-real_weight) * generated_loss + real_weight * rmse_loss




if __name__ == '__main__':
    seed = 1024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    device = 'cuda:0'

    model = EMTransformerCast(
        dim = 768,    # 768
        t_in = 1,
        shorten_factor = 2,
        depth = (2, (2, (2, (2, 24, 2), 2), 2), 2),
        ff_mult = 4,
        updown_sample_type = 'linear',
        in_chans = 70,
        out_chans = 70,
        H = 128,
        W = 256,
        patch_size = 2,
        attn_resampling = True,
        heads = 8,
        add_kv = False,
    ).to(device)


    img_tokens = torch.randn(1, 70, 128, 256).to(device)
    labels = torch.randn(1, 70, 128, 256).to(device)
    kv_caches = None    

    outputs, loss = model(img_tokens, labels, kv_caches = kv_caches) # (1, 1024, 512)
    img_tokens = outputs

    print(outputs.shape, loss)

