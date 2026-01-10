import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

# Helpers

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# Modules

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * self.scale

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = x.unsqueeze(1)
        freqs = x * self.weights.unsqueeze(0) * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, dropout = 0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, dropout = 0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.shape[0], time_emb.shape[1], 1, 1)
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, num_mem_kv = 4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        heads = self.heads
        
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        
        # rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads)
        def rearrange_qkv(t):
            # t: b, (h c), x, y
            return t.view(b, heads, -1, h * w)

        q, k, v = map(rearrange_qkv, qkv)

        # mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        mk, mv = self.mem_kv[0], self.mem_kv[1] # h c n
        mk = mk.unsqueeze(0).expand(b, -1, -1, -1)
        mv = mv.unsqueeze(0).expand(b, -1, -1, -1)
        
        k = torch.cat((mk, k), dim = -1)
        v = torch.cat((mv, v), dim = -1)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        
        # rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        out = out.view(b, -1, h, w)
        
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, num_mem_kv = 4):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        heads = self.heads

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        
        # rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads)
        def rearrange_qkv(t):
             # t: b, (h c), x, y
             return t.view(b, heads, -1, h * w).permute(0, 1, 3, 2)

        q, k, v = map(rearrange_qkv, qkv)
        
        # mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        mk, mv = self.mem_kv[0], self.mem_kv[1] # h n d
        mk = mk.unsqueeze(0).expand(b, -1, -1, -1)
        mv = mv.unsqueeze(0).expand(b, -1, -1, -1)

        k = torch.cat((mk, k), dim = -2)
        v = torch.cat((mv, v), dim = -2)

        out = F.scaled_dot_product_attention(q, k, v)

        # rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        out = out.permute(0, 1, 3, 2).reshape(b, -1, h, w)
        return self.to_out(out)

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

class Downsample(nn.Module):
    def __init__(self, dim, dim_out = None):
        super().__init__()
        self.conv = nn.Conv2d(dim * 4, default(dim_out, dim), 1)

    def forward(self, x):
        # b c (h p1) (w p2) -> b (c p1 p2) h w
        b, c, h, w = x.shape
        # We assume h and w are even
        x = x.view(b, c, h // 2, 2, w // 2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 4, h // 2, w // 2)
        return self.conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        dropout = 0.,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = None,
        flash_attn = False
    ):
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = Attention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, dropout = dropout),
                ResnetBlock(dim_in, dim_in, time_emb_dim = time_dim, dropout = dropout),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, dropout = dropout)
        self.mid_attn = Attention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim = time_dim, dropout = dropout)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = Attention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, dropout = dropout),
                ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim = time_dim, dropout = dropout),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = ResnetBlock(init_dim * 2, init_dim, time_emb_dim = time_dim, dropout = dropout)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

class DDPM(nn.Module):

  def __init__(self, timesteps, beta_scheduler, noise_predictor):
    super().__init__()
    self.T = timesteps 
    self.register_buffer('betas', self._set_schedule(beta_scheduler))
    
    # precalculate different terms 
    alphas = 1. - self.betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    self.register_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
    self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
    self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

    self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
    self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
    self.register_buffer('posterior_variance', self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)) 
    self.register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min = 1e-20)))
    self.register_buffer('posterior_mean_coef1', self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
    self.register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    down_channels = noise_predictor.get("down_channels", (64, 128, 256, 512, 1024))
    dim = down_channels[0]
    dim_mults = tuple([ch // dim for ch in down_channels[1:]])

    self.noise_predictor = Unet(
        dim=dim,
        out_dim=noise_predictor.get("out_dim", 3),
        channels=noise_predictor.get("image_channels", 3),
        dim_mults=dim_mults,
        dropout=noise_predictor.get("dropout", 0.0)
    )
    self.device = None

  def _set_schedule(self, beta_scheduler):
    beta_schedule = beta_scheduler["name"]
    if beta_schedule == "linear": 
      start = beta_scheduler.get("start", 0.0001)
      end = beta_scheduler.get("end", 0.02)
      return torch.linspace(start, end, self.T)
    elif beta_schedule == "cosine":
      steps = self.T + 1
      s = 0.008
      x = torch.linspace(0, self.T, steps, dtype=torch.float64)
      alphas_cumprod = torch.cos(((x / self.T) + s) / (1 + s) * math.pi * 0.5) ** 2
      alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
      betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
      return torch.clip(betas, 0.0001, 0.9999).to(torch.float32)
    else: 
      raise NotImplementedError

  def _get_index_from_list(self, vals, t, x_shape):
    """ 
    vals, t are 1D Tensors. 
    returns the values of vals from index t. 
    Then adds dimensions for shape consistency with images. 
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

  def forward(self):
    raise NotImplementedError

  def forward_diffusion_sample(self, x_0, t):
    r""" 
    Given image x_0, samples the noisy image from distribution q(x_t | x_0), 
    Based off of formula: 
      x_t = \sqrt{\bar{\alpha}} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon
    """
    # sample of completely noisy distribution
    noise = torch.randn_like(x_0, device=self.device)
    sqrt_alphas_cumprod_t = self._get_index_from_list(self.sqrt_alphas_cumprod, t, x_0.shape).to(self.device)
    sqrt_one_minus_alphas_cumprod_t = self._get_index_from_list(
      self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
    ).to(self.device)
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    return x_t, noise

  def predict_start_from_noise(self, x_t, t, noise):
    return (
        self._get_index_from_list(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._get_index_from_list(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
    )

  def q_posterior(self, x_start, x_t, t):
    posterior_mean = (
        self._get_index_from_list(self.posterior_mean_coef1, t, x_t.shape) * x_start +
        self._get_index_from_list(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_log_variance_clipped = self._get_index_from_list(self.posterior_log_variance_clipped, t, x_t.shape)
    return posterior_mean, posterior_log_variance_clipped

  def sample(self, x, t): 
    """
    Calls the model to predict the noise in the image and returns the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    pred_noise = self.noise_predictor(x, t)
    x_start = self.predict_start_from_noise(x, t, pred_noise)
    x_start.clamp_(-1., 1.)

    model_mean, model_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
    
    noise = torch.randn_like(x) if (t > 0).any() else 0. 

    return model_mean + (0.5 * model_log_variance).exp() * noise