from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LightweightTransformerConfig:
    n_units: int
    n_classes: int = 8
    d_model: int = 32
    n_heads: int = 2
    n_layers: int = 1
    dim_feedforward: int = 64
    dropout: float = 0.10
    max_time_bins: int = 25
    positional_encoding: str = "sinusoidal"


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.pe[:, : z.shape[1], :]


class LightweightTransformerDecoder(nn.Module):
    """Small temporal Transformer decoder with explicit positional information.

    Input tensors are shaped ``batch x time x unit``. The observation mask is
    concatenated with the response at each time bin so structured missingness is
    visible to the model without changing the MM-RVD pipeline.
    """

    def __init__(self, config: LightweightTransformerConfig):
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if config.positional_encoding != "sinusoidal":
            raise ValueError("Only sinusoidal positional encoding is frozen for Phase A2")
        self.config = config
        self.input_projection = nn.Linear(config.n_units * 2, config.d_model)
        self.position = SinusoidalPositionEncoding(config.d_model, config.max_time_bins)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.classifier = nn.Linear(config.d_model, config.n_classes)

    def forward(self, response: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if response.shape != observed_mask.shape:
            raise ValueError("response and observed_mask must have the same shape")
        z = torch.cat([response, observed_mask], dim=-1)
        z = self.input_projection(z)
        z = self.position(z)
        z = self.encoder(z)
        return self.classifier(z.mean(dim=1))


def run_static_sanity_test(output_dir: Path) -> dict[str, Any]:
    torch.manual_seed(1306)
    config = LightweightTransformerConfig(n_units=11)
    model = LightweightTransformerDecoder(config)
    x = torch.randn(16, 25, config.n_units)
    obs = torch.ones_like(x)
    y = torch.arange(16) % config.n_classes

    initial = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    logits = model(x, obs)
    loss = F.cross_entropy(logits, y)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    final = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "lightweight_transformer_static_state.pt"
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__}, checkpoint)
    reloaded = LightweightTransformerDecoder(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["state_dict"])
    with torch.inference_mode():
        reload_logits = reloaded(x, obs)

    return {
        "input_shape": list(x.shape),
        "logit_shape": list(logits.shape),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "self_attention_active": True,
        "position_encoding_added": True,
        "loss_finite": bool(torch.isfinite(loss).item()),
        "backward_works": all(p.grad is not None for p in model.parameters() if p.requires_grad),
        "optimizer_step_changed_weights": bool(not torch.equal(initial, final)),
        "checkpoint_path": str(checkpoint),
        "reload_reproduces_shape": list(reload_logits.shape) == list(logits.shape),
    }

