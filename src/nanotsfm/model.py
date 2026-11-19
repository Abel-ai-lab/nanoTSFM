import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nanotsfm.data import MAX_VARIATES

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SCALE = 1e-5


@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 512
    prediction_length: int = 256
    patch_length: int = 32
    width: int = 256
    layers: int = 3  # causal time layers; one variate layer follows them
    heads: int = 4
    dropout: float = 0.0

    def __post_init__(self):
        for key in (
            "context_length",
            "prediction_length",
            "patch_length",
            "width",
            "layers",
            "heads",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.context_length % self.patch_length:
            raise ValueError("context_length must divide into patches")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must split into heads of even size")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


def causal_scale(values: torch.Tensor, patch: int):
    observed = torch.isfinite(values)
    x = torch.where(observed, values, 0).double()
    count = observed.cumsum(-1).clamp_min(1)
    loc = x.cumsum(-1) / count
    previous = F.pad(loc[..., :-1], (1, 0))
    # Welford update: sum of (x - old mean) * (x - new mean) over observed values.
    m2 = ((x - previous) * (x - loc) * observed).cumsum(-1)
    std = (m2 / (count - 1).clamp_min(1)).clamp_min(MIN_SCALE**2).sqrt()
    end = slice(patch - 1, None, patch)
    return loc[..., end], std[..., end]


def rotate(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    frequency = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, causal: bool):
        super().__init__()
        self.heads, self.causal, self.dropout = config.heads, causal, config.dropout
        self.attention_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.mlp_norm = nn.RMSNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n, s, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).view(n, s, 3, self.heads, -1).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.causal:
            q, k = rotate(q), rotate(k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        x = x + self.out(attended.transpose(1, 2).reshape(n, s, width))
        return x + self.mlp(self.mlp_norm(x))


class NanoTSFM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        patch = config.patch_length
        self.embed = nn.Linear(2 * patch, config.width)
        self.embed_mlp = nn.Sequential(
            nn.Linear(2 * patch, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        self.time_blocks = nn.ModuleList(Block(config, causal=True) for _ in range(config.layers))
        self.variate_block = Block(config, causal=False)
        self.norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, patch * len(QUANTILES))

    def next_patch(self, values: torch.Tensor, series_ids: torch.Tensor):
        """Predict the next patch from [B,V,T] values; equal IDs share variate attention."""
        batch, variates, length = values.shape
        patch = self.config.patch_length
        if length % patch:
            raise ValueError("Sequence length must divide into patches")
        patches = length // patch
        loc, scale = causal_scale(values, patch)
        observed = torch.isfinite(values)
        # Zero-fill before arithmetic so NaN never reaches the gradient.
        clean = torch.where(observed, values, 0).double()
        scaled = (clean - loc.repeat_interleave(patch, -1)) / scale.repeat_interleave(patch, -1)
        scaled = torch.where(observed, scaled, 0).asinh().float()
        tokens = torch.cat(
            [scaled.unflatten(-1, (patches, patch)), observed.float().unflatten(-1, (-1, patch))],
            dim=-1,
        )
        x = self.embed(tokens) + self.embed_mlp(tokens)

        x = x.reshape(batch * variates, patches, -1)
        for block in self.time_blocks:
            x = block(x)
        x = (
            x.reshape(batch, variates, patches, -1)
            .transpose(1, 2)
            .reshape(batch * patches, variates, -1)
        )
        related = series_ids[:, :, None] == series_ids[:, None, :]
        related = related[:, None, None].expand(batch, patches, 1, variates, variates)
        x = self.variate_block(x, related.reshape(batch * patches, 1, variates, variates))
        x = x.reshape(batch, patches, variates, -1).transpose(1, 2)

        quantiles = self.head(self.norm(x)).float().unflatten(-1, (patch, len(QUANTILES)))
        return quantiles.sort(dim=-1).values, loc, scale

    def forward(self, history: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        patch, horizon = self.config.patch_length, self.config.prediction_length
        future = math.ceil(horizon / patch)
        values = F.pad(history, (0, future * patch), value=float("nan"))
        quantiles, loc, scale = self.next_patch(values, series_ids)
        # Position i predicts patch i + 1; the last history patch predicts the first future one.
        first = history.shape[-1] // patch - 1
        window = slice(first, first + future)
        location, spread = loc[..., window, None, None], scale[..., window, None, None]
        forecast = quantiles[:, :, window].double().sinh() * spread + location
        return forecast.flatten(2, 3)[:, :, :horizon].float()


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, cuda, or auto")
    return device


def load_checkpoint(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 3:
        raise ValueError("Unsupported checkpoint version")
    model = NanoTSFM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device_for(str(device))).eval()
    return model, checkpoint


@torch.inference_mode()
def forecast(model, history: torch.Tensor, horizon: int, series_ids=None) -> torch.Tensor:
    """Forecast [B,T] or [B,V,T] history, rolling out from median predictions if needed."""
    if history.ndim not in (2, 3) or history.shape[-1] == 0 or horizon <= 0:
        raise ValueError("Provide nonempty [B,T] or [B,V,T] history and a positive horizon")
    univariate = history.ndim == 2
    if univariate:
        history = history[:, None]
    if history.shape[1] > MAX_VARIATES:
        raise ValueError("Split related variates into consecutive groups of at most 32")
    device = next(model.parameters()).device
    history = history.to(device=device, dtype=torch.float32)
    if series_ids is None:
        series_ids = torch.zeros(history.shape[:2], dtype=torch.long, device=device)
    else:
        series_ids = series_ids.to(device)
    if series_ids.shape != history.shape[:2]:
        raise ValueError("series_ids must match batch and variate dimensions")
    context = model.config.context_length
    history = F.pad(history, (max(0, context - history.shape[-1]), 0), value=float("nan"))
    was_training = model.training
    model.eval()
    try:
        blocks = []
        remaining = horizon
        while remaining:
            prediction = model(history[..., -context:], series_ids)
            take = min(remaining, prediction.shape[-2])
            blocks.append(prediction[..., :take, :])
            history = torch.cat([history, prediction[..., 4]], dim=-1)[..., -context:]
            remaining -= take
        result = torch.cat(blocks, dim=-2)
        return result[:, 0] if univariate else result
    finally:
        model.train(was_training)
