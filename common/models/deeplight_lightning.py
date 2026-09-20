"""
DeepLight-style lightning nowcaster adapted for SEVIR.

Architecture (adapted from Nasif et al., DeepLight):
  - Lightning stem: parallel multi-scale conv (k=3,5,7) summed → 48×48 features
  - VIL stem: same structure on bilinearly-resized radar (384×384 → 48×48)
  - Learned sigmoid gate for modality fusion
  - Single-layer ConvLSTM over lag_time past frames → last hidden state
  - Readout head → single binary logit per pixel

Input:
    x_vil_past  : (B, T, H_vil, W_vil, 1)  past VIL frames
    lgt_past    : (B, T, 48, 48, 1)         past lightning frames (log-normalised counts)

Output:
    logits      : (B, 1, 48, 48)            single-frame binary logit

Target:
    1-hour aggregated binary map: (lgt_future_raw_counts.max(T) > 0).float()
    shape: (B, 1, 48, 48)

Loss (DeepLight original):
    BCEWithLogitsLoss(pos_weight=20) + HazyLoss
    — HazyLoss: confidence-weighted BCE (penalises high-confidence errors)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------

class MultiScaleStem(nn.Module):
    """
    Three parallel convolutions (k=3,5,7) whose outputs are summed, then
    normalised and activated.  Kernels are scaled from DeepLight's (5,11,21,31)
    to match SEVIR's coarser 8 km/px grid vs. DeepLight's ~2 km/px grid.

    Sum-fusion (rather than concat+project) keeps parameter count low and
    lets each branch learn complementary spatial scales independently.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.branch5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.branch7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, bias=False)
        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=True)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.branch3(x) + self.branch5(x) + self.branch7(x)))


# ---------------------------------------------------------------------------
# ConvLSTM
# ---------------------------------------------------------------------------

class ConvLSTMCell(nn.Module):
    """Standard ConvLSTM cell (Shi et al. 2015), no peephole connections."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(
        self,
        batch_size: int,
        hw: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch_size, self.hidden_channels, hw[0], hw[1], device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_channels, hw[0], hw[1], device=device, dtype=dtype)
        return h, c


class ConvLSTM(nn.Module):
    """Single-layer ConvLSTM; returns the last hidden state (sequence-to-one)."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W)
        Returns:
            h: (B, hidden_channels, H, W)
        """
        B, T, C, H, W = x.shape
        h, c = self.cell.init_hidden(B, (H, W), x.device, x.dtype)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class HazyLoss(nn.Module):
    """
    Adapted from DeepLight (model.py) — made fp16-safe by replacing BCELoss
    with binary_cross_entropy_with_logits (reduction='none') so it is safe
    under torch.amp.autocast.

    weight = (1 - transformed_truth) * pred + transformed_truth * (1 - pred)

    This is confidence-weighted BCE: pixels where the model is confidently
    wrong receive a higher weight.  When transformed_truth == truth (binary),
    weight == 1 - p_t, equivalent to focal loss with gamma=1 and no alpha.
    """

    def forward(
        self,
        logits: torch.Tensor,
        truth: torch.Tensor,
        transformed_truth: torch.Tensor,
    ) -> torch.Tensor:
        pred = torch.sigmoid(logits)
        weight = (1.0 - transformed_truth) * pred + transformed_truth * (1.0 - pred)
        bce = F.binary_cross_entropy_with_logits(logits, truth, reduction="none")
        return (bce * weight).mean()


class DeepLightLoss(nn.Module):
    """
    DeepLight combined loss:
        w1 * BCEWithLogitsLoss(pos_weight=20) + w3 * HazyLoss

    Uses the logits-safe BCEWithLogitsLoss (the commented-out variant in the
    original train.py) which is numerically stable and more appropriate than
    BCELoss on raw logits.

    transformed_truth is set equal to truth (binary) since DeepLight's
    make_blur Gaussian smoothing is absent from their training loop.
    """

    def __init__(
        self,
        pos_weight: float = 20.0,
        w1: float = 1.0,
        w3: float = 1.0,
    ):
        super().__init__()
        self.w1 = w1
        self.w3 = w3
        self.register_buffer("_pos_weight", torch.tensor(pos_weight))
        self.hazy = HazyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : (B, 1, H, W)
            target : (B, 1, H, W) binary float
        """
        bce = F.binary_cross_entropy_with_logits(
            logits,
            target,
            pos_weight=self._pos_weight.to(logits.dtype),
        )
        hazy = self.hazy(logits, target, target)
        return self.w1 * bce + self.w3 * hazy


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def prepare_deeplight_target(lgt_future: torch.Tensor) -> torch.Tensor:
    """
    Aggregate future binary lightning frames into a single 1-hour binary map.

    Args:
        lgt_future: (B, T, H, W, 1) binary float, as returned by DynamicPairedVilLightningDataset
    Returns:
        target: (B, 1, H, W) binary float  (1 if any frame is positive)
    """
    # (B, T, H, W, 1) → (B, 1, H, W)
    return lgt_future.permute(0, 4, 1, 2, 3).amax(dim=2)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class LightningDeepLightSystem(nn.Module):
    """
    DeepLight-adapted nowcaster for SEVIR lightning.

    Predicts a single 1-hour aggregated binary lightning map from
    lag_time past VIL + lightning frames.
    """

    def __init__(
        self,
        lag_time: int,
        target_hw: tuple[int, int] = (48, 48),
        vil_input_hw: tuple[int, int] = (384, 384),
        stem_channels: int = 32,
        lstm_hidden: int = 64,
    ):
        super().__init__()
        self.lag_time = int(lag_time)
        self.target_hw = tuple(int(v) for v in target_hw)
        self.vil_input_hw = tuple(int(v) for v in vil_input_hw)
        self.stem_channels = int(stem_channels)
        self.lstm_hidden = int(lstm_hidden)

        self.lightning_stem = MultiScaleStem(1, stem_channels)
        self.vil_stem = MultiScaleStem(1, stem_channels)

        # Learned gate: fuses VIL features into the lightning representation
        self.gate_head = nn.Sequential(
            nn.Conv2d(2 * stem_channels, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, stem_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_channels, stem_channels, kernel_size=1),
        )

        self.convlstm = ConvLSTM(stem_channels, lstm_hidden, kernel_size=3)

        self.readout = nn.Sequential(
            nn.Conv2d(lstm_hidden, lstm_hidden // 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, lstm_hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(lstm_hidden // 2, 1, kernel_size=1),
        )

    def forward(self, x_vil_past: torch.Tensor, lgt_past: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_vil_past : (B, T, H_vil, W_vil, 1)
            lgt_past   : (B, T, 48, 48, 1)
        Returns:
            logits     : (B, 1, 48, 48)
        """
        B, T = x_vil_past.shape[:2]
        vil_h, vil_w = x_vil_past.shape[2], x_vil_past.shape[3]
        lgt_h, lgt_w = lgt_past.shape[2], lgt_past.shape[3]

        if T != self.lag_time:
            raise ValueError(f"Expected lag_time={self.lag_time}, got T={T}")
        if (vil_h, vil_w) != self.vil_input_hw and (vil_h, vil_w) != self.target_hw:
            raise ValueError(f"Expected VIL spatial size {self.vil_input_hw} or {self.target_hw}, got ({vil_h},{vil_w})")
        if (lgt_h, lgt_w) != self.target_hw:
            raise ValueError(f"Expected lightning spatial size {self.target_hw}, got ({lgt_h},{lgt_w})")

        # → (B*T, 1, H, W)
        vil_flat = x_vil_past.permute(0, 1, 4, 2, 3).reshape(B * T, 1, vil_h, vil_w)
        lgt_flat = lgt_past.permute(0, 1, 4, 2, 3).reshape(B * T, 1, lgt_h, lgt_w)

        if (vil_h, vil_w) != self.target_hw:
            vil_flat = F.interpolate(vil_flat, size=self.target_hw, mode="bilinear", align_corners=False)

        vil_feat = self.vil_stem(vil_flat)           # (B*T, C, H, W)
        lgt_feat = self.lightning_stem(lgt_flat)     # (B*T, C, H, W)

        gate = torch.sigmoid(self.gate_head(torch.cat([vil_feat, lgt_feat], dim=1)))
        fused = lgt_feat + gate * vil_feat           # (B*T, C, H, W)

        C, H, W = fused.shape[1:]
        fused_seq = fused.reshape(B, T, C, H, W)    # (B, T, C, H, W)

        h_last = self.convlstm(fused_seq)            # (B, lstm_hidden, H, W)
        return self.readout(h_last)                  # (B, 1, H, W)

    def predict_proba(self, x_vil_past: torch.Tensor, lgt_past: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x_vil_past, lgt_past))
