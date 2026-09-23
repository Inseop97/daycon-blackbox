from __future__ import annotations

import torch
from torch import nn
from torchvision.transforms import v2


BACKBONE_NAME = "vit_small_patch14_dinov2.lvd142m"
IMAGE_SIZE = 224
FRAME_TRANSFORM = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(IMAGE_SIZE),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def build_backbone(pretrained: bool) -> nn.Module:
    import timm

    return timm.create_model(
        BACKBONE_NAME,
        pretrained=pretrained,
        num_classes=0,
        img_size=IMAGE_SIZE,
    )


class ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Stage2TemporalModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.tcn = nn.Sequential(
            ResidualTCNBlock(hidden_dim, 1, dropout),
            ResidualTCNBlock(hidden_dim, 2, dropout),
            ResidualTCNBlock(hidden_dim, 4, dropout),
            ResidualTCNBlock(hidden_dim, 8, dropout),
        )
        self.collision_head = nn.Conv1d(hidden_dim, 1, 1)
        self.entry_head = nn.Conv1d(hidden_dim, 1, 1)
        self.side_head = nn.Linear(hidden_dim, 2)
        self.evasion_head = nn.Linear(hidden_dim, 2)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        return self.tcn(self.input_projection(features).transpose(1, 2)).transpose(1, 2)

    @staticmethod
    def _gather(hidden: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch, indices]

    def forward(
        self,
        features: torch.Tensor,
        entry_indices: torch.Tensor | None = None,
        collision_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        hidden = self.encode(features)
        collision_logits = self.collision_head(hidden.transpose(1, 2)).squeeze(1)
        entry_logits = self.entry_head(hidden.transpose(1, 2)).squeeze(1)
        if collision_indices is None:
            collision_indices = collision_logits.argmax(1)
        if entry_indices is None:
            entry_indices = entry_logits.argmax(1)
        return {
            "collision_logits": collision_logits,
            "entry_logits": entry_logits,
            "side_logits": self.side_head(self._gather(hidden, entry_indices)),
            "evasion_logits": self.evasion_head(self._gather(hidden, collision_indices)),
        }
