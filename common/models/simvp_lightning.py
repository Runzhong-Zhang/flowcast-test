"""
Lightweight SimVP-style nowcaster for SEVIR lightning prediction.

This module intentionally implements only the dependency-light IncepU path of SimVP,
so it can live inside this repository without requiring `timm`.

Main use case in this repo:

- input:
    - past VIL `(B, 13, 384, 384, 1)`
    - past lightning `(B, 13, 48, 48, 1)`
- target:
    - future lightning logits `(B, 12, 48, 48, 1)`

The raw VIL input is bilinearly resized to the lightning grid, then processed
by a separate radar stem. Past lightning is processed by a separate lightning
stem at its native 48x48 grid. The two streams are fused with a learned gate
before entering the SimVP temporal-spatial backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _shape_str(x: torch.Tensor) -> str:
    return "x".join(str(v) for v in x.shape)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        upsampling: bool = False,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        self.act_norm = act_norm
        if upsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.PixelShuffle(2),
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
        act_norm: bool = True,
        act_inplace: bool = True,
    ):
        super().__init__()
        stride = 2 if downsampling else 1
        padding = (kernel_size - stride + 1) // 2
        self.conv = BasicConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            upsampling=upsampling,
            act_norm=act_norm,
            act_inplace=act_inplace,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GroupConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act_norm: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        if in_channels % groups != 0:
            groups = 1
        self.act_norm = act_norm
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )
        self.norm = nn.GroupNorm(groups, out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class GInceptionST(nn.Module):
    """IncepU block used by the dependency-light SimVP path."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        inception_kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.layers = nn.ModuleList(
            [
                GroupConv2d(
                    hidden_channels,
                    out_channels,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel // 2,
                    groups=groups,
                    act_norm=True,
                )
                for kernel in inception_kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y = y + layer(x)
        return y


def _sampling_generator(
    num_stages: int,
    reverse: bool = False,
    enabled: bool = True,
) -> list[bool]:
    if not enabled:
        samplings = [False] * num_stages
    else:
        samplings = [False, True] * (num_stages // 2)
        samplings = samplings[:num_stages]
    return list(reversed(samplings)) if reverse else samplings


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_stages: int,
        spatio_kernel: int,
        use_spatial_sampling: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        samplings = _sampling_generator(num_stages, enabled=use_spatial_sampling)
        layers = [
            ConvSC(
                in_channels,
                hidden_channels,
                spatio_kernel,
                downsampling=samplings[0],
                act_inplace=act_inplace,
            )
        ]
        for sampling in samplings[1:]:
            layers.append(
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    downsampling=sampling,
                    act_inplace=act_inplace,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc1 = self.layers[0](x)
        latent = enc1
        for layer in self.layers[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        num_stages: int,
        spatio_kernel: int,
        use_spatial_sampling: bool = False,
        act_inplace: bool = True,
    ):
        super().__init__()
        samplings = _sampling_generator(
            num_stages,
            reverse=True,
            enabled=use_spatial_sampling,
        )
        layers = []
        for sampling in samplings[:-1]:
            layers.append(
                ConvSC(
                    hidden_channels,
                    hidden_channels,
                    spatio_kernel,
                    upsampling=sampling,
                    act_inplace=act_inplace,
                )
            )
        layers.append(
            ConvSC(
                hidden_channels,
                hidden_channels,
                spatio_kernel,
                upsampling=samplings[-1],
                act_inplace=act_inplace,
            )
        )
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, hidden: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            hidden = layer(hidden)
        y = self.layers[-1](hidden + skip)
        return self.readout(y)


class MidIncepNet(nn.Module):
    def __init__(
        self,
        channel_in: int,
        channel_hid: int,
        num_blocks: int,
        inception_kernels: tuple[int, ...] = (3, 5, 7, 11),
        groups: int = 8,
    ):
        super().__init__()
        if num_blocks < 2:
            raise ValueError("SimVP translator needs at least 2 blocks.")
        self.num_blocks = num_blocks

        enc_layers = [
            GInceptionST(
                channel_in,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        ]
        for _ in range(1, num_blocks - 1):
            enc_layers.append(
                GInceptionST(
                    channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    inception_kernels=inception_kernels,
                    groups=groups,
                )
            )
        enc_layers.append(
            GInceptionST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        )

        dec_layers = [
            GInceptionST(
                channel_hid,
                channel_hid // 2,
                channel_hid,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        ]
        for _ in range(1, num_blocks - 1):
            dec_layers.append(
                GInceptionST(
                    2 * channel_hid,
                    channel_hid // 2,
                    channel_hid,
                    inception_kernels=inception_kernels,
                    groups=groups,
                )
            )
        dec_layers.append(
            GInceptionST(
                2 * channel_hid,
                channel_hid // 2,
                channel_in,
                inception_kernels=inception_kernels,
                groups=groups,
            )
        )

        self.enc = nn.ModuleList(enc_layers)
        self.dec = nn.ModuleList(dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, channels, height, width = x.shape
        z = x.reshape(batch_size, time_steps * channels, height, width)

        skips = []
        for block_idx, block in enumerate(self.enc):
            z = block(z)
            if block_idx < self.num_blocks - 1:
                skips.append(z)

        z = self.dec[0](z)
        for block_idx in range(1, self.num_blocks):
            z = self.dec[block_idx](torch.cat([z, skips[-block_idx]], dim=1))

        return z.reshape(batch_size, time_steps, channels, height, width)


class SimVPBackbone(nn.Module):
    """
    Minimal SimVP backbone with a learnable temporal projection for
    `input_steps != output_steps`.
    """

    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        in_channels: int,
        out_channels: int,
        input_hw: tuple[int, int],
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        use_spatial_sampling: bool = False,
        act_inplace: bool = False,
    ):
        super().__init__()
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        if use_spatial_sampling:
            reduced_h = int(input_hw[0] / 2 ** (N_S / 2))
            reduced_w = int(input_hw[1] / 2 ** (N_S / 2))
        else:
            reduced_h, reduced_w = int(input_hw[0]), int(input_hw[1])
        self.encoder = Encoder(
            in_channels,
            hid_S,
            N_S,
            spatio_kernel_enc,
            use_spatial_sampling=use_spatial_sampling,
            act_inplace=act_inplace,
        )
        self.decoder = Decoder(
            hid_S,
            out_channels,
            N_S,
            spatio_kernel_dec,
            use_spatial_sampling=use_spatial_sampling,
            act_inplace=act_inplace,
        )
        self.translator = MidIncepNet(
            channel_in=self.input_steps * hid_S,
            channel_hid=hid_T,
            num_blocks=N_T,
        )
        self.temporal_proj = None
        if self.output_steps != self.input_steps:
            self.temporal_proj = nn.Linear(self.input_steps, self.output_steps)

        self.reduced_hw = (reduced_h, reduced_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "SimVP backbone expects channel-first video input with shape "
                f"(B, T, C, H, W), got {_shape_str(x)}"
            )
        batch_size, time_steps, channels, height, width = x.shape
        if time_steps != self.input_steps:
            raise ValueError(
                f"Expected {self.input_steps} input steps, got {time_steps}"
            )
        if channels != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {channels}"
            )

        x_flat = x.reshape(batch_size * time_steps, channels, height, width)
        embed, skip = self.encoder(x_flat)
        _, hidden_channels, hidden_h, hidden_w = embed.shape

        latent = embed.reshape(batch_size, time_steps, hidden_channels, hidden_h, hidden_w)
        latent = self.translator(latent)

        if self.temporal_proj is not None:
            latent = latent.permute(0, 3, 4, 2, 1).contiguous()
            latent = latent.reshape(-1, self.input_steps)
            latent = self.temporal_proj(latent)
            latent = latent.reshape(
                batch_size,
                hidden_h,
                hidden_w,
                hidden_channels,
                self.output_steps,
            )
            latent = latent.permute(0, 4, 3, 1, 2).contiguous()

        out_steps = latent.shape[1]
        latent_flat = latent.reshape(batch_size * out_steps, hidden_channels, hidden_h, hidden_w)

        if out_steps == time_steps:
            skip_used = skip
        else:
            skip_5d = skip.reshape(batch_size, time_steps, hidden_channels, height, width)
            skip_5d = skip_5d.permute(0, 3, 4, 2, 1).contiguous()
            skip_5d = skip_5d.reshape(-1, self.input_steps)
            skip_5d = self.temporal_proj(skip_5d)
            skip_5d = skip_5d.reshape(batch_size, height, width, hidden_channels, out_steps)
            skip_5d = skip_5d.permute(0, 4, 3, 1, 2).contiguous()
            skip_used = skip_5d.reshape(batch_size * out_steps, hidden_channels, height, width)

        decoded = self.decoder(latent_flat, skip_used)
        decoded = decoded.reshape(batch_size, out_steps, self.out_channels, height, width)
        expected_shape = (batch_size, self.output_steps, self.out_channels, height, width)
        if tuple(decoded.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected SimVP decoded shape {tuple(decoded.shape)}, "
                f"expected {expected_shape}"
            )
        return decoded


class SimVPTranslatorHead(nn.Module):
    """
    Fixed-resolution SimVP head:

    - no extra spatial encoder after modality fusion
    - Incep temporal-spatial translator on fused features
    - optional temporal projection from T_in -> T_out
    - lightweight per-frame readout head
    """

    def __init__(
        self,
        input_steps: int,
        output_steps: int,
        hidden_channels: int,
        out_channels: int,
        hid_T: int = 256,
        N_T: int = 4,
    ):
        super().__init__()
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)

        self.translator = MidIncepNet(
            channel_in=self.input_steps * self.hidden_channels,
            channel_hid=hid_T,
            num_blocks=N_T,
        )
        self.temporal_proj = None
        if self.output_steps != self.input_steps:
            self.temporal_proj = nn.Linear(self.input_steps, self.output_steps)

        self.readout = nn.Sequential(
            BasicConv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
                act_norm=True,
            ),
            nn.Conv2d(self.hidden_channels, self.out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "SimVP translator head expects shape (B, T, C, H, W), "
                f"got {_shape_str(x)}"
            )
        batch_size, time_steps, channels, height, width = x.shape
        if time_steps != self.input_steps:
            raise ValueError(f"Expected {self.input_steps} input steps, got {time_steps}")
        if channels != self.hidden_channels:
            raise ValueError(
                f"Expected hidden_channels={self.hidden_channels}, got {channels}"
            )

        latent = self.translator(x)
        if self.temporal_proj is not None:
            latent = latent.permute(0, 3, 4, 2, 1).contiguous()
            latent = latent.reshape(-1, self.input_steps)
            latent = self.temporal_proj(latent)
            latent = latent.reshape(
                batch_size,
                height,
                width,
                channels,
                self.output_steps,
            )
            latent = latent.permute(0, 4, 3, 1, 2).contiguous()

        out_steps = latent.shape[1]
        latent_flat = latent.reshape(batch_size * out_steps, channels, height, width)
        logits = self.readout(latent_flat)
        logits = logits.reshape(batch_size, out_steps, self.out_channels, height, width)
        expected_shape = (
            batch_size,
            self.output_steps,
            self.out_channels,
            height,
            width,
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected translator-head output shape {tuple(logits.shape)}, expected {expected_shape}"
            )
        return logits


class LightningSimVPSystem(nn.Module):
    """
    SimVP-based lightning nowcaster.

    Input fusion:
    - past VIL: resized from 384x384 to target_hw, then encoded by a SimVP-style
      radar encoder
    - past lightning: native 48x48 transformed-count condition encoded by a
      SimVP-style lightning encoder
    - learned gate computed from both feature streams
    - fused features are passed directly to the Incep temporal core and output head
    - no spatial downsampling / upsampling in the lightning nowcast path
    """

    def __init__(
        self,
        lag_time: int,
        lead_time: int,
        target_hw: tuple[int, int] = (48, 48),
        vil_input_hw: tuple[int, int] = (384, 384),
        input_channels: int = 2,
        output_channels: int = 1,
        hid_S: int = 32,
        hid_T: int = 256,
        N_S: int = 4,
        N_T: int = 4,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
    ):
        super().__init__()
        self.lag_time = int(lag_time)
        self.lead_time = int(lead_time)
        self.target_hw = tuple(int(v) for v in target_hw)
        self.vil_input_hw = tuple(int(v) for v in vil_input_hw)
        if int(input_channels) != 2:
            raise ValueError(
                f"LightningSimVPSystem expects 2 raw modalities (VIL + lightning), got input_channels={input_channels}"
            )
        self.stem_channels = int(hid_S)
        self.radar_encoder = Encoder(
            1,
            self.stem_channels,
            N_S,
            spatio_kernel_enc,
            use_spatial_sampling=False,
            act_inplace=False,
        )
        self.lightning_encoder = Encoder(
            1,
            self.stem_channels,
            N_S,
            spatio_kernel_enc,
            use_spatial_sampling=False,
            act_inplace=False,
        )
        self.gate_head = nn.Sequential(
            BasicConv2d(
                2 * self.stem_channels,
                self.stem_channels,
                kernel_size=3,
                padding=1,
                act_norm=True,
            ),
            nn.Conv2d(self.stem_channels, self.stem_channels, kernel_size=1),
        )
        self.fusion_refine = nn.Sequential(
            BasicConv2d(
                self.stem_channels,
                self.stem_channels,
                kernel_size=3,
                padding=1,
                act_norm=True,
            ),
            BasicConv2d(
                self.stem_channels,
                self.stem_channels,
                kernel_size=3,
                padding=1,
                act_norm=True,
            ),
        )
        self.backbone = SimVPTranslatorHead(
            input_steps=self.lag_time,
            output_steps=self.lead_time,
            hidden_channels=self.stem_channels,
            out_channels=output_channels,
            hid_T=hid_T,
            N_T=N_T,
        )

    def prepare_inputs(self, x_vil_past: torch.Tensor, lgt_past: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build aligned channel-first modality tensors with shape `(B, T, 1, H, W)`.
        """
        if x_vil_past.ndim != 5:
            raise ValueError(
                "Expected x_vil_past with shape (B, T, H, W, C), "
                f"got {_shape_str(x_vil_past)}"
            )
        if lgt_past.ndim != 5:
            raise ValueError(
                "Expected lgt_past with shape (B, T, H, W, C), "
                f"got {_shape_str(lgt_past)}"
            )

        batch_size, time_steps, vil_h, vil_w, vil_channels = x_vil_past.shape
        lgt_batch, lgt_steps, lgt_h, lgt_w, lgt_channels = lgt_past.shape
        if batch_size != lgt_batch:
            raise ValueError(
                f"Batch mismatch between VIL ({batch_size}) and lightning ({lgt_batch})"
            )
        if time_steps != self.lag_time:
            raise ValueError(f"Expected lag_time={self.lag_time}, got {time_steps}")
        if lgt_steps != self.lag_time:
            raise ValueError(f"Expected lgt_past lag_time={self.lag_time}, got {lgt_steps}")
        if vil_channels != 1 or lgt_channels != 1:
            raise ValueError("Expected single-channel VIL and lightning inputs.")
        if (vil_h, vil_w) != self.vil_input_hw:
            raise ValueError(
                f"Expected VIL spatial size {self.vil_input_hw}, got {(vil_h, vil_w)}"
            )
        if (lgt_h, lgt_w) != self.target_hw:
            raise ValueError(
                f"Expected lightning spatial size {self.target_hw}, got {(lgt_h, lgt_w)}"
            )

        vil = x_vil_past.permute(0, 1, 4, 2, 3).reshape(
            batch_size * time_steps,
            1,
            vil_h,
            vil_w,
        )
        vil = F.interpolate(
            vil,
            size=self.target_hw,
            mode="bilinear",
            align_corners=False,
        )
        vil = vil.reshape(batch_size, time_steps, 1, self.target_hw[0], self.target_hw[1])

        lightning = lgt_past.permute(0, 1, 4, 2, 3).contiguous()
        expected_vil_shape = (batch_size, self.lag_time, 1, self.target_hw[0], self.target_hw[1])
        expected_lgt_shape = (batch_size, self.lag_time, 1, self.target_hw[0], self.target_hw[1])
        if tuple(vil.shape) != expected_vil_shape:
            raise RuntimeError(
                f"Unexpected resized VIL shape {tuple(vil.shape)}, expected {expected_vil_shape}"
            )
        if tuple(lightning.shape) != expected_lgt_shape:
            raise RuntimeError(
                f"Unexpected lightning branch shape {tuple(lightning.shape)}, expected {expected_lgt_shape}"
            )
        return vil, lightning

    def fuse_inputs(self, vil: torch.Tensor, lightning: torch.Tensor) -> torch.Tensor:
        if vil.shape != lightning.shape:
            raise ValueError(
                f"Aligned modality tensors must share shape, got vil={tuple(vil.shape)} vs lightning={tuple(lightning.shape)}"
            )
        batch_size, time_steps, _, height, width = vil.shape
        vil_flat = vil.reshape(batch_size * time_steps, 1, height, width)
        lightning_flat = lightning.reshape(batch_size * time_steps, 1, height, width)

        radar_features, _ = self.radar_encoder(vil_flat)
        lightning_features, _ = self.lightning_encoder(lightning_flat)
        gate_logits = self.gate_head(torch.cat([radar_features, lightning_features], dim=1))
        gate = torch.sigmoid(gate_logits)
        fused_flat = radar_features + gate * lightning_features
        fused_flat = self.fusion_refine(fused_flat)
        fused = fused_flat.reshape(
            batch_size,
            time_steps,
            self.stem_channels,
            height,
            width,
        )
        expected_shape = (
            batch_size,
            self.lag_time,
            self.stem_channels,
            self.target_hw[0],
            self.target_hw[1],
        )
        if tuple(fused.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected gated-fusion input shape {tuple(fused.shape)}, expected {expected_shape}"
            )
        return fused

    def forward(self, x_vil_past: torch.Tensor, lgt_past: torch.Tensor) -> torch.Tensor:
        vil, lightning = self.prepare_inputs(x_vil_past, lgt_past)
        inputs = self.fuse_inputs(vil, lightning)
        logits = self.backbone(inputs)
        expected_shape = (
            x_vil_past.shape[0],
            self.lead_time,
            1,
            self.target_hw[0],
            self.target_hw[1],
        )
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected SimVP logits shape {tuple(logits.shape)}, expected {expected_shape}"
            )
        return logits

    def predict_proba(self, x_vil_past: torch.Tensor, lgt_past: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(self.forward(x_vil_past, lgt_past))
        expected_shape = (
            x_vil_past.shape[0],
            self.lead_time,
            1,
            self.target_hw[0],
            self.target_hw[1],
        )
        if tuple(probs.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected SimVP probability shape {tuple(probs.shape)}, expected {expected_shape}"
            )
        return probs
