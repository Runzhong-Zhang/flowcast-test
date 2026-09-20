"""
Model components for conditional lightning diffusion.

This module provides:

- a pretrained VIL encoder wrapper initialized from the existing AutoencoderKL
- an optional frozen pretrained VIL FlowCast wrapper that predicts future VIL latents
- a compact conditional denoiser for native-grid lightning diffusion
- a combined system module used by both training and inference scripts
"""

from numbers import Integral
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.autoencoders import AutoencoderKL
from omegaconf import OmegaConf
from torchdiffeq import odeint_adjoint as odeint

from .cuboid_transformer_unet import CuboidTransformerUNet
from .time_embed import TimeEmbedLayer, TimeEmbedResBlock
from .utils import timestep_embedding


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Remove DDP/DataParallel prefixes from checkpoint keys."""
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "")] = value
    return cleaned


def resolve_ae_chunk_size(
    chunk_size: Optional[Union[int, str]],
    batch_size: int,
    time_steps: int,
) -> int:
    """
    Resolve the VIL encoder chunk size from config/CLI input.

    `chunk_size` controls how many flattened `(batch, time)` VIL frames are sent
    through the encoder at once on each GPU. It is resolved per process:

    - `None`, `0`, negative values, and `"auto"` mean "use all local frames"
    - positive values are clamped to the local `batch_size * time_steps`
    """
    total_local_frames = int(batch_size) * int(time_steps)
    if total_local_frames <= 0:
        raise ValueError(
            f"Expected positive local frame count, got batch_size={batch_size}, "
            f"time_steps={time_steps}"
        )

    if chunk_size is None:
        return total_local_frames

    if isinstance(chunk_size, str):
        normalized = chunk_size.strip().lower()
        if normalized in {"", "auto", "max", "full"}:
            return total_local_frames
        try:
            chunk_size = int(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported ae_chunk_size value {chunk_size!r}. "
                "Use an integer or 'auto'."
            ) from exc
    elif not isinstance(chunk_size, Integral):
        raise TypeError(
            f"Unsupported ae_chunk_size type {type(chunk_size)!r}. "
            "Use an integer, None, or 'auto'."
        )

    resolved = int(chunk_size)
    if resolved <= 0:
        return total_local_frames
    return min(resolved, total_local_frames)


class PretrainedVILEncoder(nn.Module):
    """Wrapper around the pretrained VIL AutoencoderKL encoder."""

    def __init__(self, config_path: str, checkpoint_path: str):
        super().__init__()
        config = OmegaConf.load(config_path)
        model_params = config.model_params

        autoencoder = AutoencoderKL(
            in_channels=1,
            out_channels=1,
            down_block_types=list(model_params.down_block_types),
            up_block_types=list(model_params.up_block_types),
            block_out_channels=list(model_params.block_out_channels),
            act_fn=model_params.act_fn,
            latent_channels=model_params.latent_channels,
            norm_num_groups=model_params.norm_num_groups,
            layers_per_block=model_params.layers_per_block,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = strip_module_prefix(checkpoint["model_state_dict"])
        autoencoder.load_state_dict(state_dict, strict=True)

        self.autoencoder = autoencoder
        self.latent_channels = int(model_params.latent_channels)

    def freeze_all(self):
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def enable_finetune(self, full_finetune: bool = False):
        """Unfreeze encoder-side modules for task-specific adaptation."""
        for param in self.autoencoder.parameters():
            param.requires_grad = False

        if full_finetune:
            for module_name in ("encoder", "quant_conv"):
                module = getattr(self.autoencoder, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        else:
            module = getattr(self.autoencoder, "quant_conv", None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = True

    def encode_sequence(
        self,
        x_vil_past: torch.Tensor,
        target_hw: Tuple[int, int] = (48, 48),
        chunk_size: Optional[Union[int, str]] = None,
    ) -> torch.Tensor:
        """
        Encode a VIL sequence into latent condition features.

        Parameters
        ----------
        x_vil_past:
            Tensor with shape (B, T_in, H, W, 1) normalized to [0, 1].
        target_hw:
            Spatial size expected by the lightning model. Latents are resized if needed.
        chunk_size:
            Optional per-GPU frame chunk size for VAE encoding.
            `None`, `0`, negative values, or `"auto"` use all local `(B * T_in)` frames.
        """
        batch_size, time_steps, height, width, channels = x_vil_past.shape
        if channels != 1:
            raise ValueError(f"Expected VIL input channel size 1, got {channels}")

        frames = x_vil_past.permute(0, 1, 4, 2, 3).reshape(
            batch_size * time_steps, 1, height, width
        )

        encoded_chunks = []
        bs = resolve_ae_chunk_size(chunk_size, batch_size=batch_size, time_steps=time_steps)
        for start in range(0, frames.shape[0], bs):
            chunk = frames[start : start + bs]
            latent_obj = self.autoencoder.encode(chunk)
            latent = latent_obj.latent_dist.mode()
            if latent.shape[-2:] != target_hw:
                latent = F.interpolate(
                    latent,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            encoded_chunks.append(latent)

        latents = torch.cat(encoded_chunks, dim=0)
        _, latent_channels, latent_h, latent_w = latents.shape
        latents = latents.reshape(
            batch_size, time_steps, latent_channels, latent_h, latent_w
        )
        latents = latents.permute(0, 1, 3, 4, 2).contiguous()
        return latents

    def decode_latent_sequence(
        self,
        z_vil: torch.Tensor,
        chunk_size: Optional[Union[int, str]] = None,
    ) -> torch.Tensor:
        """
        Decode a latent VIL sequence back to VIL image space.

        Parameters
        ----------
        z_vil:
            Tensor with shape (B, T, H_latent, W_latent, C_latent).
        chunk_size:
            Optional per-GPU frame chunk size for VAE decoding.
        """
        if z_vil.ndim != 5:
            raise ValueError(
                f"Expected latent VIL tensor with shape (B, T, H, W, C), got {z_vil.shape}"
            )

        batch_size, time_steps, height, width, channels = z_vil.shape
        if channels != self.latent_channels:
            raise ValueError(
                f"Expected latent channel size {self.latent_channels}, got {channels}"
            )

        latents = z_vil.permute(0, 1, 4, 2, 3).reshape(
            batch_size * time_steps, channels, height, width
        )
        decoded_chunks = []
        bs = resolve_ae_chunk_size(
            chunk_size,
            batch_size=batch_size,
            time_steps=time_steps,
        )
        for start in range(0, latents.shape[0], bs):
            chunk = latents[start : start + bs]
            decoded_obj = self.autoencoder.decode(chunk)
            decoded_chunks.append(decoded_obj.sample)

        decoded = torch.cat(decoded_chunks, dim=0)
        _, out_channels, out_h, out_w = decoded.shape
        decoded = decoded.reshape(
            batch_size, time_steps, out_channels, out_h, out_w
        )
        decoded = decoded.permute(0, 1, 3, 4, 2).contiguous()
        return decoded


class FrozenVILFlowCastForecaster(nn.Module):
    """Frozen latent-space VIL FlowCast used as an auxiliary lightning condition."""

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        input_shape: Tuple[int, int, int, int],
        target_shape: Tuple[int, int, int, int],
    ):
        super().__init__()
        config = OmegaConf.load(config_path)
        model_cfg = OmegaConf.to_container(config.latent_model, resolve=True)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = strip_module_prefix(checkpoint["model_state_dict"])
        mean = checkpoint["mean"]
        std = checkpoint["std"]
        if isinstance(mean, torch.Tensor):
            mean = float(mean.item())
        if isinstance(std, torch.Tensor):
            std = float(std.item())

        depth = list(model_cfg["depth"])
        block_attn_patterns = [model_cfg["self_pattern"]] * len(depth)
        down_up_linear_init_mode = model_cfg["down_up_linear_init_mode"]

        self.model = CuboidTransformerUNet(
            input_shape=input_shape,
            target_shape=target_shape,
            base_units=model_cfg["base_units"],
            block_units=None,
            scale_alpha=model_cfg["scale_alpha"],
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            upsample_type=model_cfg["upsample_type"],
            upsample_kernel_size=model_cfg["upsample_kernel_size"],
            depth=depth,
            block_attn_patterns=block_attn_patterns,
            num_global_vectors=model_cfg["num_global_vectors"],
            use_global_vector_ffn=model_cfg["use_global_vector_ffn"],
            use_global_self_attn=model_cfg["use_global_self_attn"],
            separate_global_qkv=model_cfg["separate_global_qkv"],
            global_dim_ratio=model_cfg["global_dim_ratio"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            padding_type=model_cfg["padding_type"],
            checkpoint_level=model_cfg["checkpoint_level"],
            pos_embed_type=model_cfg["pos_embed_type"],
            use_relative_pos=model_cfg["use_relative_pos"],
            self_attn_use_final_proj=model_cfg["self_attn_use_final_proj"],
            attn_linear_init_mode=model_cfg["attn_linear_init_mode"],
            ffn_linear_init_mode=model_cfg["ffn_linear_init_mode"],
            ffn2_linear_init_mode=model_cfg["ffn2_linear_init_mode"],
            attn_proj_linear_init_mode=model_cfg["attn_proj_linear_init_mode"],
            conv_init_mode=model_cfg["conv_init_mode"],
            down_linear_init_mode=down_up_linear_init_mode,
            up_linear_init_mode=down_up_linear_init_mode,
            global_proj_linear_init_mode=model_cfg["global_proj_linear_init_mode"],
            norm_init_mode=model_cfg["norm_init_mode"],
            time_embed_channels_mult=model_cfg["time_embed_channels_mult"],
            time_embed_use_scale_shift_norm=model_cfg["time_embed_use_scale_shift_norm"],
            time_embed_dropout=model_cfg["time_embed_dropout"],
            unet_res_connect=model_cfg["unet_res_connect"],
            mean=mean,
            std=std,
        )
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.input_shape = tuple(input_shape)
        self.target_shape = tuple(target_shape)

    @torch.no_grad()
    def forecast(
        self,
        z_vil_past: torch.Tensor,
        probabilistic_samples: int = 1,
        euler_steps: int = 10,
    ) -> torch.Tensor:
        """Forecast future VIL latents from encoded past latents."""
        if z_vil_past.ndim != 5:
            raise ValueError(
                f"Expected z_vil_past with shape (B, T, H, W, C), got {z_vil_past.shape}"
            )

        batch_size, time_steps, height, width, channels = z_vil_past.shape
        expected = self.input_shape
        if (time_steps, height, width, channels) != expected:
            raise ValueError(
                "Encoded VIL latent shape does not match the frozen FlowCast forecaster: "
                f"got {(time_steps, height, width, channels)}, expected {expected}"
            )

        future_steps = self.target_shape[0]
        with torch.amp.autocast(device_type=z_vil_past.device.type, enabled=False):
            z_vil_past_fp32 = z_vil_past.float()
            x_cond = self.model.normalize(z_vil_past_fp32)
            latent_members = []

            for _ in range(max(1, int(probabilistic_samples))):
                x0_noise = torch.randn(
                    batch_size,
                    future_steps,
                    height,
                    width,
                    channels,
                    device=z_vil_past.device,
                    dtype=torch.float32,
                )
                x0_flat = x0_noise.view(
                    batch_size * future_steps, height, width, channels
                )

                def flow_dynamics(t_scalar, x_flat):
                    x_flow = x_flat.view(
                        batch_size, future_steps, height, width, channels
                    )
                    t_batched = t_scalar * torch.ones(batch_size, device=x_flow.device)
                    with torch.no_grad():
                        v_t = self.model(t_batched, x_flow, x_cond)
                    return v_t.contiguous().view(
                        batch_size * future_steps, height, width, channels
                    )

                t_span = torch.tensor(
                    [0.0, 1.0], device=z_vil_past.device, dtype=torch.float32
                )
                if euler_steps <= 0:
                    solution = odeint(
                        flow_dynamics,
                        x0_flat,
                        t_span,
                        method="adaptive_heun",
                        rtol=1e-2,
                        atol=1e-3,
                        adjoint_params=tuple(self.model.parameters()),
                    )
                else:
                    step_size = 1.0 / float(euler_steps)
                    solution = odeint(
                        flow_dynamics,
                        x0_flat,
                        t_span,
                        method="euler",
                        options=dict(step_size=step_size),
                        rtol=1e-2,
                        atol=1e-3,
                        adjoint_params=tuple(self.model.parameters()),
                    )
                latent_members.append(
                    solution[-1].view(batch_size, future_steps, height, width, channels)
                )

            forecast_norm = torch.stack(latent_members, dim=1).mean(dim=1)
            forecast = self.model.denormalize(forecast_norm)
        return forecast.to(dtype=z_vil_past.dtype)


class LightningConditionedDenoiser(nn.Module):
    """
    Compact conditional denoiser for native-grid lightning diffusion.

    Inputs
    ------
    - x_t: noisy future lightning `(B, T_out, H, W, 1)`
    - z_vil_past: encoded past VIL condition `(B, T_in, H, W, C_z)`
    - z_vil_future_pred: predicted future VIL condition `(B, T_out, H, W, C_z)` (optional)
    - lgt_past: transformed-count past lightning `(B, T_in, H, W, 1)`
    """

    def __init__(
        self,
        z_channels: int,
        hidden_channels: int = 96,
        num_blocks: int = 4,
        dropout: float = 0.0,
        use_future_vil_condition: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_future_vil_condition = bool(use_future_vil_condition)
        self.time_embed = TimeEmbedLayer(
            base_channels=hidden_channels,
            time_embed_channels=hidden_channels * 4,
        )

        self.xt_stem = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        self.vil_stem = nn.Conv3d(z_channels, hidden_channels, kernel_size=3, padding=1)
        if self.use_future_vil_condition:
            self.future_vil_stem = nn.Conv3d(
                z_channels, hidden_channels, kernel_size=3, padding=1
            )
        self.lgt_past_stem = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        fuse_inputs = 4 if self.use_future_vil_condition else 3
        self.fuse = nn.Conv3d(
            hidden_channels * fuse_inputs, hidden_channels, kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList(
            [
                TimeEmbedResBlock(
                    channels=hidden_channels,
                    emb_channels=hidden_channels * 4,
                    dropout=dropout,
                    out_channels=hidden_channels,
                    use_embed=True,
                    dims=3,
                )
                for _ in range(num_blocks)
            ]
        )

        self.flow_head = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)

    @staticmethod
    def _resize_condition_time(cond: torch.Tensor, target_t: int) -> torch.Tensor:
        """Resize the condition tensor along the temporal axis to match T_out."""
        if cond.shape[2] == target_t:
            return cond
        return F.interpolate(
            cond,
            size=(target_t, cond.shape[-2], cond.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        z_vil_past: torch.Tensor,
        lgt_past: torch.Tensor,
        z_vil_future_pred: Optional[torch.Tensor] = None,
    ):
        x_feat = self.xt_stem(x_t.permute(0, 4, 1, 2, 3))
        z_feat = self.vil_stem(z_vil_past.permute(0, 4, 1, 2, 3))
        p_feat = self.lgt_past_stem(lgt_past.permute(0, 4, 1, 2, 3))

        target_t = x_feat.shape[2]
        z_feat = self._resize_condition_time(z_feat, target_t)
        p_feat = self._resize_condition_time(p_feat, target_t)

        features = [x_feat, z_feat, p_feat]
        if self.use_future_vil_condition:
            if z_vil_future_pred is None:
                raise ValueError(
                    "z_vil_future_pred must be provided when use_future_vil_condition=True"
                )
            f_feat = self.future_vil_stem(z_vil_future_pred.permute(0, 4, 1, 2, 3))
            f_feat = self._resize_condition_time(f_feat, target_t)
            features.insert(2, f_feat)

        h = self.fuse(torch.cat(features, dim=1))
        t_emb = self.time_embed(timestep_embedding(t, self.hidden_channels))

        for block in self.blocks:
            h = block(h, t_emb)

        flow = self.flow_head(h).permute(0, 2, 3, 4, 1).contiguous()
        return flow


class LightningConditionedDenoiserCuboid(nn.Module):
    """
    CuboidTransformerUNet-based conditional denoiser for native-grid lightning diffusion.

    Follows the same temporal-concatenation + obs-indicator design as FlowCast, but
    with heterogeneous past/future channels:

        cond = cat([z_vil_past, lgt_past], dim=-1)   # (B, T_in, H, W, C_z+1)
        x    = cat([x_t, z_vil_future_pred], dim=-1)  # (B, T_out, H, W, C_z+1)

    When ``use_future_vil_condition=False``, the VIL future slots in ``x`` are
    filled with zeros so the channel count matches ``cond``.

    Normalization is set to identity (mean=0, std=1) because each modality is
    normalized externally by the training script before reaching this module.
    The model outputs only the lightning velocity channel ``out[..., :1]``.

    Inputs
    ------
    - x_t: noisy future lightning ``(B, T_out, H, W, 1)``
    - z_vil_past: encoded past VIL condition ``(B, T_in, H, W, C_z)``
    - lgt_past: transformed-count past lightning ``(B, T_in, H, W, 1)``
    - z_vil_future_pred: predicted future VIL ``(B, T_out, H, W, C_z)`` (optional)
    """

    def __init__(
        self,
        z_channels: int,
        lag_time: int,
        lead_time: int,
        target_hw: Tuple[int, int],
        use_future_vil_condition: bool = True,
        # CuboidTransformerUNet hyper-parameters
        base_units: int = 128,
        scale_alpha: float = 1.0,
        depth: Tuple[int, ...] = (4, 4),
        num_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        ffn_drop: float = 0.0,
        downsample: int = 2,
        downsample_type: str = "patch_merge",
        upsample_type: str = "upsample",
        upsample_kernel_size: int = 3,
        num_global_vectors: int = 8,
        use_global_vector_ffn: bool = True,
        use_global_self_attn: bool = False,
        separate_global_qkv: bool = False,
        global_dim_ratio: int = 1,
        self_pattern: str = "axial",
        ffn_activation: str = "gelu",
        gated_ffn: bool = False,
        norm_layer: str = "layer_norm",
        padding_type: str = "zeros",
        checkpoint_level: int = 0,
        pos_embed_type: str = "t+hw",
        use_relative_pos: bool = True,
        self_attn_use_final_proj: bool = True,
        attn_linear_init_mode: str = "0",
        ffn_linear_init_mode: str = "0",
        ffn2_linear_init_mode: str = "2",
        attn_proj_linear_init_mode: str = "2",
        conv_init_mode: str = "0",
        down_linear_init_mode: str = "0",
        up_linear_init_mode: str = "0",
        global_proj_linear_init_mode: str = "2",
        norm_init_mode: str = "0",
        time_embed_channels_mult: int = 4,
        time_embed_use_scale_shift_norm: bool = False,
        time_embed_dropout: float = 0.0,
        unet_res_connect: bool = True,
    ):
        super().__init__()
        self.z_channels = int(z_channels)
        self.use_future_vil_condition = bool(use_future_vil_condition)

        # cond = [z_vil_past, lgt_past]  ->  C_z + 1 channels
        # x    = [x_t, z_vil_future_pred] ->  1 + C_z channels  (same count)
        # When future VIL is unavailable x_t is zero-padded to C_z + 1.
        cond_channels = self.z_channels + 1
        H, W = tuple(target_hw)
        input_shape = (int(lag_time), H, W, cond_channels)
        target_shape = (int(lead_time), H, W, cond_channels)

        depth_list = list(depth)
        block_attn_patterns = [self_pattern] * len(depth_list)

        self.model = CuboidTransformerUNet(
            input_shape=input_shape,
            target_shape=target_shape,
            base_units=base_units,
            block_units=None,
            scale_alpha=scale_alpha,
            depth=depth_list,
            block_attn_patterns=block_attn_patterns,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            ffn_drop=ffn_drop,
            downsample=downsample,
            downsample_type=downsample_type,
            upsample_type=upsample_type,
            upsample_kernel_size=upsample_kernel_size,
            num_global_vectors=num_global_vectors,
            use_global_vector_ffn=use_global_vector_ffn,
            use_global_self_attn=use_global_self_attn,
            separate_global_qkv=separate_global_qkv,
            global_dim_ratio=global_dim_ratio,
            ffn_activation=ffn_activation,
            gated_ffn=gated_ffn,
            norm_layer=norm_layer,
            padding_type=padding_type,
            checkpoint_level=checkpoint_level,
            pos_embed_type=pos_embed_type,
            use_relative_pos=use_relative_pos,
            self_attn_use_final_proj=self_attn_use_final_proj,
            attn_linear_init_mode=attn_linear_init_mode,
            ffn_linear_init_mode=ffn_linear_init_mode,
            ffn2_linear_init_mode=ffn2_linear_init_mode,
            attn_proj_linear_init_mode=attn_proj_linear_init_mode,
            conv_init_mode=conv_init_mode,
            down_linear_init_mode=down_linear_init_mode,
            up_linear_init_mode=up_linear_init_mode,
            global_proj_linear_init_mode=global_proj_linear_init_mode,
            norm_init_mode=norm_init_mode,
            time_embed_channels_mult=time_embed_channels_mult,
            time_embed_use_scale_shift_norm=time_embed_use_scale_shift_norm,
            time_embed_dropout=time_embed_dropout,
            unet_res_connect=unet_res_connect,
            # Identity normalization: inputs are pre-normalized externally.
            mean=0.0,
            std=1.0,
        )

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        z_vil_past: torch.Tensor,
        lgt_past: torch.Tensor,
        z_vil_future_pred: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # cond: (B, T_in, H, W, C_z+1)
        cond = torch.cat([z_vil_past, lgt_past], dim=-1)

        # x: (B, T_out, H, W, C_z+1)
        if self.use_future_vil_condition:
            if z_vil_future_pred is None:
                raise ValueError(
                    "z_vil_future_pred must be provided when use_future_vil_condition=True"
                )
            x = torch.cat([x_t, z_vil_future_pred], dim=-1)
        else:
            # Zero-pad VIL future slots so channel count matches cond.
            vil_zeros = torch.zeros(
                x_t.shape[0], x_t.shape[1], x_t.shape[2], x_t.shape[3],
                self.z_channels,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            x = torch.cat([x_t, vil_zeros], dim=-1)

        # CuboidTransformerUNet: temporal cat + obs_indicator handled internally.
        # Output shape: (B, T_out, H, W, C_z+1)
        out = self.model(t, x, cond)

        # Return only the lightning velocity (channel 0).
        return out[..., :1].contiguous()


class LightningDiffusionSystem(nn.Module):
    """Combined VIL encoder + lightning denoiser system."""

    def __init__(
        self,
        ae_config_path: str,
        ae_checkpoint_path: str,
        lag_time: int = 13,
        lead_time: int = 12,
        target_hw: Tuple[int, int] = (48, 48),
        # ResBlock denoiser hyper-parameters (denoiser_backbone="resblock")
        hidden_channels: int = 96,
        num_blocks: int = 4,
        dropout: float = 0.0,
        # CuboidTransformerUNet denoiser hyper-parameters (denoiser_backbone="cuboid")
        cuboid_base_units: int = 128,
        cuboid_scale_alpha: float = 1.0,
        cuboid_depth: Tuple[int, ...] = (4, 4),
        cuboid_num_heads: int = 4,
        cuboid_attn_drop: float = 0.0,
        cuboid_proj_drop: float = 0.0,
        cuboid_ffn_drop: float = 0.0,
        cuboid_downsample: int = 2,
        cuboid_downsample_type: str = "patch_merge",
        cuboid_upsample_type: str = "upsample",
        cuboid_upsample_kernel_size: int = 3,
        cuboid_num_global_vectors: int = 8,
        cuboid_use_global_vector_ffn: bool = True,
        cuboid_use_global_self_attn: bool = False,
        cuboid_separate_global_qkv: bool = False,
        cuboid_global_dim_ratio: int = 1,
        cuboid_self_pattern: str = "axial",
        cuboid_ffn_activation: str = "gelu",
        cuboid_gated_ffn: bool = False,
        cuboid_norm_layer: str = "layer_norm",
        cuboid_padding_type: str = "zeros",
        cuboid_checkpoint_level: int = 0,
        cuboid_pos_embed_type: str = "t+hw",
        cuboid_use_relative_pos: bool = True,
        cuboid_self_attn_use_final_proj: bool = True,
        cuboid_time_embed_channels_mult: int = 4,
        cuboid_time_embed_use_scale_shift_norm: bool = False,
        cuboid_time_embed_dropout: float = 0.0,
        cuboid_unet_res_connect: bool = True,
        # Shared options
        use_future_vil_condition: bool = False,
        future_vil_online_forecaster: bool = True,
        future_vil_flowcast_config_path: Optional[str] = None,
        future_vil_flowcast_checkpoint_path: Optional[str] = None,
        future_vil_euler_steps: int = 10,
        future_vil_probabilistic_samples: int = 1,
        denoiser_backbone: str = "resblock",
    ):
        super().__init__()
        self.target_hw = tuple(target_hw)
        self.vil_encoder = PretrainedVILEncoder(
            config_path=ae_config_path,
            checkpoint_path=ae_checkpoint_path,
        )
        self.use_future_vil_condition = bool(use_future_vil_condition)
        self.future_vil_online_forecaster = bool(future_vil_online_forecaster)
        self.future_vil_euler_steps = int(future_vil_euler_steps)
        self.future_vil_probabilistic_samples = int(future_vil_probabilistic_samples)
        if self.use_future_vil_condition and self.future_vil_online_forecaster:
            if not future_vil_flowcast_config_path or not future_vil_flowcast_checkpoint_path:
                raise ValueError(
                    "Future VIL conditioning requires both a FlowCast config path and checkpoint path."
                )
            latent_channels = self.vil_encoder.latent_channels
            input_shape = (int(lag_time), self.target_hw[0], self.target_hw[1], latent_channels)
            target_shape = (
                int(lead_time),
                self.target_hw[0],
                self.target_hw[1],
                latent_channels,
            )
            self.future_vil_forecaster = FrozenVILFlowCastForecaster(
                config_path=future_vil_flowcast_config_path,
                checkpoint_path=future_vil_flowcast_checkpoint_path,
                input_shape=input_shape,
                target_shape=target_shape,
            )
        else:
            self.future_vil_forecaster = None

        denoiser_backbone = str(denoiser_backbone).lower()
        if denoiser_backbone == "resblock":
            self.denoiser = LightningConditionedDenoiser(
                z_channels=self.vil_encoder.latent_channels,
                hidden_channels=hidden_channels,
                num_blocks=num_blocks,
                dropout=dropout,
                use_future_vil_condition=self.use_future_vil_condition,
            )
        elif denoiser_backbone == "cuboid":
            self.denoiser = LightningConditionedDenoiserCuboid(
                z_channels=self.vil_encoder.latent_channels,
                lag_time=int(lag_time),
                lead_time=int(lead_time),
                target_hw=self.target_hw,
                use_future_vil_condition=self.use_future_vil_condition,
                base_units=cuboid_base_units,
                scale_alpha=cuboid_scale_alpha,
                depth=cuboid_depth,
                num_heads=cuboid_num_heads,
                attn_drop=cuboid_attn_drop,
                proj_drop=cuboid_proj_drop,
                ffn_drop=cuboid_ffn_drop,
                downsample=cuboid_downsample,
                downsample_type=cuboid_downsample_type,
                upsample_type=cuboid_upsample_type,
                upsample_kernel_size=cuboid_upsample_kernel_size,
                num_global_vectors=cuboid_num_global_vectors,
                use_global_vector_ffn=cuboid_use_global_vector_ffn,
                use_global_self_attn=cuboid_use_global_self_attn,
                separate_global_qkv=cuboid_separate_global_qkv,
                global_dim_ratio=cuboid_global_dim_ratio,
                self_pattern=cuboid_self_pattern,
                ffn_activation=cuboid_ffn_activation,
                gated_ffn=cuboid_gated_ffn,
                norm_layer=cuboid_norm_layer,
                padding_type=cuboid_padding_type,
                checkpoint_level=cuboid_checkpoint_level,
                pos_embed_type=cuboid_pos_embed_type,
                use_relative_pos=cuboid_use_relative_pos,
                self_attn_use_final_proj=cuboid_self_attn_use_final_proj,
                time_embed_channels_mult=cuboid_time_embed_channels_mult,
                time_embed_use_scale_shift_norm=cuboid_time_embed_use_scale_shift_norm,
                time_embed_dropout=cuboid_time_embed_dropout,
                unet_res_connect=cuboid_unet_res_connect,
            )
        else:
            raise ValueError(
                f"Unknown denoiser_backbone {denoiser_backbone!r}. "
                "Choose 'resblock' or 'cuboid'."
            )

    def freeze_vil_encoder(self):
        self.vil_encoder.freeze_all()

    def enable_vil_encoder_finetune(self, full_finetune: bool = False):
        self.vil_encoder.enable_finetune(full_finetune=full_finetune)

    def encode_vil(
        self,
        x_vil_past: torch.Tensor,
        chunk_size: Optional[Union[int, str]] = None,
    ) -> torch.Tensor:
        return self.vil_encoder.encode_sequence(
            x_vil_past,
            target_hw=self.target_hw,
            chunk_size=chunk_size,
        )

    def predict_future_vil_from_encoded(self, z_vil_past: torch.Tensor) -> Optional[torch.Tensor]:
        if self.future_vil_forecaster is None:
            return None
        return self.future_vil_forecaster.forecast(
            z_vil_past,
            probabilistic_samples=self.future_vil_probabilistic_samples,
            euler_steps=self.future_vil_euler_steps,
        )

    def prepare_vil_conditions(
        self,
        x_vil_past: torch.Tensor,
        ae_chunk_size: Optional[Union[int, str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_vil_past = self.encode_vil(x_vil_past, chunk_size=ae_chunk_size)
        z_vil_future_pred = None
        if self.future_vil_forecaster is not None:
            # The frozen FlowCast forecaster already returns denormalized VAE
            # latents, so we can condition on them directly while the VIL
            # encoder stays frozen. This avoids an unnecessary decode/re-encode
            # round trip at train and test time.
            z_vil_future_pred = self.predict_future_vil_from_encoded(z_vil_past)
        return z_vil_past, z_vil_future_pred

    def decode_vil_latents(
        self,
        z_vil: torch.Tensor,
        ae_chunk_size: Optional[Union[int, str]] = None,
    ) -> torch.Tensor:
        return self.vil_encoder.decode_latent_sequence(
            z_vil,
            chunk_size=ae_chunk_size,
        )

    def forward_from_encoded(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        z_vil_past: torch.Tensor,
        lgt_past: torch.Tensor,
        z_vil_future_pred: Optional[torch.Tensor] = None,
    ):
        return self.denoiser(
            t,
            x_t,
            z_vil_past,
            lgt_past,
            z_vil_future_pred=z_vil_future_pred,
        )

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        x_vil_past: Optional[torch.Tensor],
        lgt_past: torch.Tensor,
        ae_chunk_size: Optional[Union[int, str]] = None,
        z_vil_past: Optional[torch.Tensor] = None,
        z_vil_future_pred: Optional[torch.Tensor] = None,
    ):
        if z_vil_past is None:
            if x_vil_past is None:
                raise ValueError(
                    "Expected either raw x_vil_past or a precomputed z_vil_past latent."
                )
            z_vil_past, predicted_future_latent = self.prepare_vil_conditions(
                x_vil_past,
                ae_chunk_size=ae_chunk_size,
            )
            if z_vil_future_pred is None:
                z_vil_future_pred = predicted_future_latent
        return self.forward_from_encoded(
            t,
            x_t,
            z_vil_past,
            lgt_past,
            z_vil_future_pred=z_vil_future_pred,
        )
