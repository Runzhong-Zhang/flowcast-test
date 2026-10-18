# Isolated copy of dist_train_lightning_diffusion_moe.py for greedy MCR.
# train_total reports CFM + load balance; the targeted game has no scalar total.
"""
Distributed training script for FC Cuboid Sparse MoE lightning diffusion.

Mirrors dist_train_lightning_diffusion.py exactly for the diffusion/cuboid
path, with the following additions:

  * Uses LightningDiffusionMoESystem instead of LightningDiffusionSystem.
  * Reads an extra ``moe_params`` block from the YAML config.
  * Adds the load-balance auxiliary loss to the CFM training objective:
        total_loss = cfm_loss + load_balance_weight * aux_loss
  * Logs per-epoch mean auxiliary loss alongside train/val losses.

Data pipeline
-------------
Identical to FC Cuboid: supports both the online VIL-encoding path and the
precomputed-latents path (``precomputed_vil_condition_params``).

Example YAML additions
----------------------
    moe_params:
      num_experts: 8
      top_k: 2
      load_balance_weight: 0.01
"""

import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from common.cfm.cfm import ConditionalFlowMatcher
from common.models.flowcast.lightning_condition_diffusion_moe import (
    LightningDiffusionMoESystem,
)
from common.models.flowcast.moe_ffn import get_moe_layer_stats
from common.models.flowcast.mcr_routing_experiment import GreedyRoutingMCR
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
    PrecomputedPairedVilLightningDataset,
    precomputed_paired_vil_lightning_collate,
)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        dist.barrier(device_ids=[local_rank])
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return tensor
    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced /= world_size
    return reduced


def strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Loss / batch utilities  (identical to dist_train_lightning_diffusion.py)
# ---------------------------------------------------------------------------

def compute_cfm_loss(
    flow_pred: torch.Tensor,
    flow_target: torch.Tensor,
    lgt_future: Optional[torch.Tensor] = None,
    event_gamma: float = 0.0,
    multiscale_weight: float = 0.0,
    lead_alpha: float = 1.0,
) -> torch.Tensor:
    """
    Enhanced CFM loss for sparse lightning prediction.

    Three orthogonal improvements controlled by independent parameters.
    All default to 0 / 1.0, producing identical behavior to standard MSE.

    Parameters
    ----------
    flow_pred, flow_target : (B, T, H, W, 1)
        Predicted and target velocity fields.
    lgt_future : (B, T, H, W, 1), optional
        Binary lightning target. Required only when event_gamma > 0.
    event_gamma : float
        Event weighting strength. Lightning pixels get weight (1 + gamma),
        background pixels get weight 1. Default 0 = standard MSE.
        Recommended: 10 (raises effective lightning fraction from 0.76% to ~8%).
    multiscale_weight : float
        Weight for 2x-downsampled loss term. Default 0 = disabled.
        Recommended: 0.5. At 24x24, lightning covers ~3% of pixels (4x denser),
        giving the model a stronger "regional lightning" learning signal.
    lead_alpha : float
        Lead-time weighting ratio (last_lead / first_lead). Default 1.0 = equal.
        Recommended: 1.5. Allocates more gradient to harder long-range prediction.
    """
    pixel_err = (flow_pred - flow_target) ** 2    # (B, T, H, W, 1)

    # ── Event weighting ─────────────────────────────────────────────────
    if event_gamma > 0 and lgt_future is not None:
        event_weight = 1.0 + event_gamma * (lgt_future > 0.5).float()
    else:
        event_weight = 1.0

    # ── Lead-time weighting ─────────────────────────────────────────────
    if lead_alpha != 1.0:
        T = flow_pred.shape[1]
        lead_w = torch.linspace(1.0, lead_alpha, T, device=flow_pred.device)
        lead_w = lead_w.view(1, T, 1, 1, 1)
    else:
        lead_w = 1.0

    loss_full = (event_weight * lead_w * pixel_err).mean()

    # ── Multi-scale loss ────────────────────────────────────────────────
    if multiscale_weight > 0:
        B, T, H, W, C = flow_pred.shape
        # Reshape to (B*T, C, H, W) for avg_pool2d
        pred_2d = flow_pred.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        tgt_2d = flow_target.reshape(B * T, H, W, C).permute(0, 3, 1, 2)
        pred_ds = F.avg_pool2d(pred_2d, kernel_size=2)
        tgt_ds = F.avg_pool2d(tgt_2d, kernel_size=2)
        loss_ds = ((pred_ds - tgt_ds) ** 2).mean()
        return loss_full + multiscale_weight * loss_ds

    return loss_full


def unpack_condition_batch(batch, device: torch.device, use_precomputed_vil: bool):
    if use_precomputed_vil:
        z_vil_past, z_vil_future_pred, lgt_past, lgt_future, meta = batch
        return (
            None,
            z_vil_past.to(device, non_blocking=True),
            z_vil_future_pred.to(device, non_blocking=True),
            lgt_past.to(device, non_blocking=True),
            lgt_future.to(device, non_blocking=True),
            meta,
        )
    x_vil_past, lgt_past, lgt_future, meta = batch
    return (
        x_vil_past.to(device, non_blocking=True),
        None,
        None,
        lgt_past.to(device, non_blocking=True),
        lgt_future.to(device, non_blocking=True),
        meta,
    )


def quick_brier_estimate(
    system: torch.nn.Module,
    loader,
    device: torch.device,
    ae_chunk_size,
    use_precomputed_vil: bool = False,
    max_batches: int = 20,
) -> float:
    """
    Cheap single-Euler-step Brier score proxy for checkpoint selection.
    Matches the implementation in dist_train_lightning_diffusion.py.
    """
    system.eval()
    brier_sum = 0.0
    count     = 0
    with torch.no_grad():
        base = system.module if hasattr(system, "module") else system
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            (x_vil_past, z_vil_past, z_vil_future_pred, lgt_past, lgt_future, _) = \
                unpack_condition_batch(batch, device, use_precomputed_vil)

            if not use_precomputed_vil:
                z_vil_past, z_vil_future_pred = base.prepare_vil_conditions(
                    x_vil_past, ae_chunk_size=ae_chunk_size
                )

            x0    = torch.randn_like(lgt_future)
            t_z   = torch.zeros(lgt_future.shape[0], device=device)
            flow  = base.forward_from_encoded(
                t_z, x0, z_vil_past, lgt_past, z_vil_future_pred=z_vil_future_pred
            )
            x_hat  = x0 + flow
            probs  = (x_hat > 0.5).float()
            truth  = (lgt_future >= 0.5).float()
            bs     = lgt_future.shape[0]
            brier_sum += float(torch.mean((probs - truth) ** 2).item()) * bs
            count     += bs

    return brier_sum / max(count, 1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="DDP training for FC Cuboid Sparse MoE lightning diffusion."
)
parser.add_argument(
    "--config", type=str,
    default="experiments/sevir/runner/flowcast/lightning_diffusion_moe_config.yaml",
)
parser.add_argument("--train_vil_file",       type=str, required=True)
parser.add_argument("--train_lightning_file", type=str, required=True)
parser.add_argument("--train_meta",           type=str, required=True)
parser.add_argument("--val_vil_file",         type=str, required=True)
parser.add_argument("--val_lightning_file",   type=str, required=True)
parser.add_argument("--val_meta",             type=str, required=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    mcr_cfg = config.get("mcr_params", {})
    if not mcr_cfg:
        raise ValueError("Dedicated MCR runner requires mcr_params")
    if float(config.get("moe_params", {}).get("cmi_weight", 0)) != 0:
        raise ValueError("Disable the old CMI loss for the MCR experiment")
    if not config.precomputed_vil_condition_params.enable:
        raise ValueError("This isolated MCR experiment requires precomputed VIL latents")

    rank, world_size, local_rank, device = setup_ddp()
    is_main = rank == 0

    # ── run / artifact paths ────────────────────────────────────────────────
    debug_mode  = config.run_params.debug_mode
    run_string  = config.run_params.run_string
    artifacts_dir = os.path.join(
        "artifacts", "sevir", "lightning_diffusion_moe",
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + run_string,
    )
    model_dir = os.path.join(artifacts_dir, "models")
    if is_main:
        os.makedirs(model_dir, exist_ok=True)

    # ── training hyper-params ───────────────────────────────────────────────
    batch_size            = config.training_params.micro_batch_size
    num_workers           = config.training_params.num_workers
    num_epochs            = config.training_params.num_epochs
    fp16                  = config.training_params.fp16
    grad_clip             = config.training_params.gradient_clip_val
    init_checkpoint       = config.training_params.get("init_checkpoint", None)
    resume_optimizer      = bool(config.training_params.get("resume_optimizer", False))
    resume_scheduler      = bool(config.training_params.get("resume_scheduler", False))
    resume_scaler         = bool(config.training_params.get("resume_scaler", False))
    learning_rate         = config.optimizer_params.learning_rate
    weight_decay          = config.optimizer_params.weight_decay
    scheduler_min_lr_ratio = config.scheduler_params.lr_cosine_min_lr_ratio

    # ── data params ─────────────────────────────────────────────────────────
    lag_time     = config.data_params.lag_time
    lead_time    = config.data_params.lead_time
    time_spacing = config.data_params.time_spacing
    raw_seq_len  = config.data_params.raw_seq_len
    stride       = config.data_params.stride

    sigma = config.flow_matching_params.sigma
    c_cap = config.loss_params.c_cap
    event_gamma      = float(config.loss_params.get("event_loss_gamma", 0.0))
    multiscale_weight = float(config.loss_params.get("multiscale_weight", 0.0))
    lead_alpha       = float(config.loss_params.get("lead_time_alpha", 1.0))

    # ── VIL autoencoder ─────────────────────────────────────────────────────
    ae_config      = config.autoencoder_params.autoencoder_config
    ae_checkpoint  = config.autoencoder_params.autoencoder_checkpoint
    ae_chunk_size  = config.autoencoder_params.ae_chunk_size
    finetune_vil   = config.autoencoder_params.finetune_vil_encoder
    full_finetune  = config.autoencoder_params.full_encoder_finetune
    encoder_lr_ratio = config.autoencoder_params.encoder_lr_ratio

    # ── dataset keys ────────────────────────────────────────────────────────
    vil_dataset       = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key       = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    target_hw         = tuple(config.model_params.target_hw)

    # ── future VIL conditioning (optional, same as FC Plus / FC Cuboid) ─────
    future_vil_cfg         = config.get("future_vil_condition_params", {})
    use_future_vil         = bool(future_vil_cfg.get("enable", False))
    future_vil_fc_config   = future_vil_cfg.get("flowcast_config", None)
    future_vil_fc_ckpt     = future_vil_cfg.get("flowcast_checkpoint", None)
    future_vil_euler       = int(future_vil_cfg.get("euler_steps", 10))
    future_vil_prob        = int(future_vil_cfg.get("probabilistic_samples", 1))

    # ── precomputed latents (same as FC Cuboid training) ────────────────────
    pre_cfg            = config.get("precomputed_vil_condition_params", {})
    use_precomputed    = bool(pre_cfg.get("enable", False))
    train_latent_file  = pre_cfg.get("train_latent_file", None)
    val_latent_file    = pre_cfg.get("val_latent_file", None)
    latent_past_key    = pre_cfg.get("latent_past_dataset", "z_vil_past")
    latent_future_key  = pre_cfg.get("latent_future_pred_dataset", "z_vil_future_pred")

    if use_precomputed and finetune_vil:
        raise ValueError(
            "Precomputed VIL conditions are incompatible with VIL encoder fine-tuning."
        )

    # ── Cuboid backbone params (identical to FC Cuboid) ─────────────────────
    cuboid_cfg = config.model_params.get("cuboid_params", {})

    # ── MoE params (new) ─────────────────────────────────────────────────────
    moe_cfg              = config.get("moe_params", {})
    moe_num_experts      = int(moe_cfg.get("num_experts", 8))
    moe_top_k            = int(moe_cfg.get("top_k", 2))
    moe_lb_weight        = float(moe_cfg.get("load_balance_weight", 0.01))
    moe_strategy         = str(moe_cfg.get("strategy", "up_l0_t"))
    moe_cross_modal      = bool(moe_cfg.get("cross_modal_routing", False))
    moe_vil_cond_dim     = int(moe_cfg.get("vil_cond_dim", 128))
    moe_router_type      = str(moe_cfg.get("cross_modal_router_type", "concat"))
    moe_shared_expert    = bool(moe_cfg.get("shared_expert", False))

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    if use_precomputed:
        collate_fn = precomputed_paired_vil_lightning_collate
        train_dataset = PrecomputedPairedVilLightningDataset(
            latent_file=train_latent_file,
            lightning_file=args.train_lightning_file,
            meta_csv=args.train_meta,
            lightning_dataset=lightning_dataset,
            lightning_row_key=lightning_row_key,
            latent_past_dataset=latent_past_key,
            latent_future_pred_dataset=latent_future_key,
            lag_time=lag_time,
            lead_time=lead_time,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )
        val_dataset = PrecomputedPairedVilLightningDataset(
            latent_file=val_latent_file,
            lightning_file=args.val_lightning_file,
            meta_csv=args.val_meta,
            lightning_dataset=lightning_dataset,
            lightning_row_key=lightning_row_key,
            latent_past_dataset=latent_past_key,
            latent_future_pred_dataset=latent_future_key,
            lag_time=lag_time,
            lead_time=lead_time,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )
    else:
        collate_fn = dynamic_paired_vil_lightning_collate
        train_dataset = DynamicPairedVilLightningDataset(
            meta_csv=args.train_meta,
            vil_file=args.train_vil_file,
            lightning_file=args.train_lightning_file,
            vil_dataset=vil_dataset,
            lightning_dataset=lightning_dataset,
            vil_row_key=vil_row_key,
            lightning_row_key=lightning_row_key,
            raw_seq_len=raw_seq_len,
            lag_time=lag_time,
            lead_time=lead_time,
            time_spacing=time_spacing,
            stride=stride,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )
        val_dataset = DynamicPairedVilLightningDataset(
            meta_csv=args.val_meta,
            vil_file=args.val_vil_file,
            lightning_file=args.val_lightning_file,
            vil_dataset=vil_dataset,
            lightning_dataset=lightning_dataset,
            vil_row_key=vil_row_key,
            lightning_row_key=lightning_row_key,
            raw_seq_len=raw_seq_len,
            lag_time=lag_time,
            lead_time=lead_time,
            time_spacing=time_spacing,
            stride=stride,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    nw = num_workers if not debug_mode else 0
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=collate_fn, num_workers=nw,
        pin_memory=not debug_mode, persistent_workers=not debug_mode and nw > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        collate_fn=collate_fn, num_workers=nw,
        pin_memory=not debug_mode, persistent_workers=not debug_mode and nw > 0,
        drop_last=False,
    )

    # ── Model creation ────────────────────────────────────────────────────────
    system = LightningDiffusionMoESystem(
        ae_config_path=ae_config,
        ae_checkpoint_path=ae_checkpoint,
        lag_time=lag_time,
        lead_time=lead_time,
        target_hw=target_hw,
        use_future_vil_condition=use_future_vil,
        future_vil_online_forecaster=not use_precomputed,
        future_vil_flowcast_config_path=future_vil_fc_config,
        future_vil_flowcast_checkpoint_path=future_vil_fc_ckpt,
        future_vil_euler_steps=future_vil_euler,
        future_vil_probabilistic_samples=future_vil_prob,
        # ── cuboid backbone params ──────────────────────────────────────────
        cuboid_base_units=int(cuboid_cfg.get("base_units", 128)),
        cuboid_scale_alpha=float(cuboid_cfg.get("scale_alpha", 1.0)),
        cuboid_depth=tuple(cuboid_cfg.get("depth", [4, 4])),
        cuboid_num_heads=int(cuboid_cfg.get("num_heads", 4)),
        cuboid_attn_drop=float(cuboid_cfg.get("attn_drop", 0.0)),
        cuboid_proj_drop=float(cuboid_cfg.get("proj_drop", 0.0)),
        cuboid_ffn_drop=float(cuboid_cfg.get("ffn_drop", 0.0)),
        cuboid_downsample=int(cuboid_cfg.get("downsample", 2)),
        cuboid_downsample_type=str(cuboid_cfg.get("downsample_type", "patch_merge")),
        cuboid_upsample_type=str(cuboid_cfg.get("upsample_type", "upsample")),
        cuboid_upsample_kernel_size=int(cuboid_cfg.get("upsample_kernel_size", 3)),
        cuboid_num_global_vectors=int(cuboid_cfg.get("num_global_vectors", 8)),
        cuboid_use_global_vector_ffn=bool(cuboid_cfg.get("use_global_vector_ffn", True)),
        cuboid_use_global_self_attn=bool(cuboid_cfg.get("use_global_self_attn", False)),
        cuboid_separate_global_qkv=bool(cuboid_cfg.get("separate_global_qkv", False)),
        cuboid_global_dim_ratio=int(cuboid_cfg.get("global_dim_ratio", 1)),
        cuboid_self_pattern=str(cuboid_cfg.get("self_pattern", "axial")),
        cuboid_ffn_activation=str(cuboid_cfg.get("ffn_activation", "gelu")),
        cuboid_gated_ffn=bool(cuboid_cfg.get("gated_ffn", False)),
        cuboid_norm_layer=str(cuboid_cfg.get("norm_layer", "layer_norm")),
        cuboid_padding_type=str(cuboid_cfg.get("padding_type", "zeros")),
        cuboid_checkpoint_level=int(cuboid_cfg.get("checkpoint_level", 0)),
        cuboid_pos_embed_type=str(cuboid_cfg.get("pos_embed_type", "t+hw")),
        cuboid_use_relative_pos=bool(cuboid_cfg.get("use_relative_pos", True)),
        cuboid_self_attn_use_final_proj=bool(cuboid_cfg.get("self_attn_use_final_proj", True)),
        cuboid_time_embed_channels_mult=int(cuboid_cfg.get("time_embed_channels_mult", 4)),
        cuboid_time_embed_use_scale_shift_norm=bool(cuboid_cfg.get("time_embed_use_scale_shift_norm", False)),
        cuboid_time_embed_dropout=float(cuboid_cfg.get("time_embed_dropout", 0.0)),
        cuboid_unet_res_connect=bool(cuboid_cfg.get("unet_res_connect", True)),
        # ── MoE params ─────────────────────────────────────────────────────
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_strategy=moe_strategy,
        cross_modal_routing=moe_cross_modal,
        vil_cond_dim=moe_vil_cond_dim,
        cross_modal_router_type=moe_router_type,
        moe_shared_expert=moe_shared_expert,
    ).to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in system.parameters())
        denoiser_params = sum(p.numel() for p in system.denoiser.parameters())
        print(f"  MoE strategy: {moe_strategy} | "
              f"cross_modal: {moe_cross_modal} | "
              f"router_type: {moe_router_type} | "
              f"shared_expert: {moe_shared_expert} | "
              f"replaced {system.moe_n_replaced} FFN layers | "
              f"E={moe_num_experts} K={moe_top_k} | "
              f"total params: {total_params/1e6:.1f}M | "
              f"denoiser params: {denoiser_params/1e6:.1f}M")

    if finetune_vil:
        system.enable_vil_encoder_finetune(full_finetune=full_finetune)
    else:
        system.freeze_vil_encoder()

    # ── Optimiser ────────────────────────────────────────────────────────────
    model_params   = [p for p in system.denoiser.parameters()    if p.requires_grad]
    encoder_params = [p for p in system.vil_encoder.parameters() if p.requires_grad]
    param_groups   = [{"params": model_params, "lr": learning_rate}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": learning_rate * encoder_lr_ratio})

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)

    start_epoch   = 0
    best_val_brier = float("inf")
    checkpoint    = None

    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        state_dict = strip_module_prefix(checkpoint["model_state_dict"])
        # Handle legacy key rename: _vil_proj -> vil_proj
        state_dict = {k.replace("._vil_proj.", ".vil_proj."): v for k, v in state_dict.items()}
        system.load_state_dict(state_dict, strict=True)
        start_epoch    = int(checkpoint.get("epoch", -1)) + 1
        best_val_brier = float(checkpoint.get("best_val_brier", float("inf")))
        if resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if is_main:
            print(f"Warm-start from {init_checkpoint} | start_epoch={start_epoch}")

    # ── LR scheduler ─────────────────────────────────────────────────────────
    sched_last = -1
    if start_epoch > 0 and not resume_scheduler:
        for g in optimizer.param_groups:
            g.setdefault("initial_lr", g["lr"])
        sched_last = start_epoch - 1
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=learning_rate * scheduler_min_lr_ratio,
        last_epoch=sched_last,
    )

    mcr = GreedyRoutingMCR(system, mcr_cfg)
    if is_main:
        print(f"Greedy routing MCR: {dict(mcr_cfg)}", flush=True)

    # ── DDP wrap ─────────────────────────────────────────────────────────────
    if world_size > 1:
        system = DDP(
            system,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    flow_matcher = ConditionalFlowMatcher(sigma=sigma)
    scaler       = torch.amp.GradScaler(enabled=fp16 and device.type == "cuda")

    if checkpoint is not None:
        if resume_scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if resume_scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_ckpt = os.path.join(model_dir, "best_lightning_diffusion_moe.pt")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        system.train()
        train_cfm_meter  = 0.0
        train_aux_meter  = 0.0
        train_total_meter = 0.0
        train_count      = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch} (Rank {rank})",
            disable=not is_main,
            leave=False,
        )

        for batch_index, batch in enumerate(train_bar):
            (x_vil_past, z_vil_past, z_vil_future_pred,
             lgt_past, lgt_future, _) = unpack_condition_batch(
                batch, device, use_precomputed
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=fp16):
                x0_noise = torch.randn_like(lgt_future, device=device)
                t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                    x0_noise, lgt_future
                )
                flow_pred = system(
                    t, x_t, x_vil_past, lgt_past,
                    ae_chunk_size=ae_chunk_size,
                    z_vil_past=z_vil_past,
                    z_vil_future_pred=z_vil_future_pred,
                )
                cfm_loss = compute_cfm_loss(
                    flow_pred, u_t,
                    lgt_future=lgt_future,
                    event_gamma=event_gamma,
                    multiscale_weight=multiscale_weight,
                    lead_alpha=lead_alpha,
                )

                # MoE load-balance auxiliary loss
                base_sys = system.module if hasattr(system, "module") else system
                aux_loss = base_sys.collect_moe_aux_loss()
                loss     = cfm_loss + moe_lb_weight * aux_loss

            scaler.scale(loss).backward()
            if mcr.due(epoch, batch_index):
                # Main DDP backward has finished. Auxiliary gradients are reduced
                # explicitly by GreedyRoutingMCR before combining with task grads.
                with torch.amp.autocast(device_type=device.type, enabled=fp16):
                    mcr_logs = mcr.add_gradients(
                        z_vil_past, lgt_past, z_vil_future_pred,
                        t, x_t, epoch, scale=scaler.get_scale(),
                    )
                if is_main:
                    print(f"MCR epoch={epoch} batch={batch_index} " +
                          " ".join(f"{k}={v:.8g}" for k, v in mcr_logs.items()), flush=True)
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
            if mcr.due(epoch, batch_index) and is_main:
                print(f"MCR preclip_grad_norm={float(grad_norm):.8g} clipped={float(grad_norm) > grad_clip}", flush=True)
            scaler.step(optimizer)
            scaler.update()

            bs = lgt_future.size(0)
            train_cfm_meter   += float(cfm_loss.item())  * bs
            train_aux_meter   += float(aux_loss.item())  * bs
            train_total_meter += float(loss.item())      * bs
            train_count       += bs

            if is_main:
                train_bar.set_postfix({
                    "cfm": f"{cfm_loss.item():.4f}",
                    "aux": f"{aux_loss.item():.4f}",
                })

        scheduler.step()

        # Aggregate training metrics across ranks
        for tensor_sum, count_t in [
            (torch.tensor(train_cfm_meter,   device=device), None),
            (torch.tensor(train_aux_meter,   device=device), None),
            (torch.tensor(train_total_meter, device=device), None),
            (torch.tensor(float(train_count), device=device), None),
        ]:
            if world_size > 1:
                dist.all_reduce(tensor_sum, op=dist.ReduceOp.SUM)

        train_count_t      = torch.tensor(float(train_count), device=device)
        train_cfm_sum_t    = torch.tensor(train_cfm_meter,    device=device)
        train_aux_sum_t    = torch.tensor(train_aux_meter,    device=device)
        train_total_sum_t  = torch.tensor(train_total_meter,  device=device)
        if world_size > 1:
            for t in [train_count_t, train_cfm_sum_t, train_aux_sum_t, train_total_sum_t]:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
        denom = train_count_t.clamp_min(1.0)
        train_cfm   = (train_cfm_sum_t   / denom).item()
        train_aux   = (train_aux_sum_t   / denom).item()
        train_total = (train_total_sum_t / denom).item()

        # ── Validation ────────────────────────────────────────────────────────
        system.eval()
        val_loss_meter = 0.0
        val_count      = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Val Epoch {epoch} (Rank {rank})",
                disable=not is_main,
                leave=False,
            )
            for batch in val_bar:
                (x_vil_past, z_vil_past, z_vil_future_pred,
                 lgt_past, lgt_future, _) = unpack_condition_batch(
                    batch, device, use_precomputed
                )
                with torch.amp.autocast(device_type=device.type, enabled=fp16):
                    x0_noise = torch.randn_like(lgt_future, device=device)
                    t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                        x0_noise, lgt_future
                    )
                    flow_pred = system(
                        t, x_t, x_vil_past, lgt_past,
                        ae_chunk_size=ae_chunk_size,
                        z_vil_past=z_vil_past,
                        z_vil_future_pred=z_vil_future_pred,
                    )
                    val_loss = compute_cfm_loss(flow_pred, u_t)

                bs = lgt_future.size(0)
                val_loss_meter += float(val_loss.item()) * bs
                val_count      += bs

        val_loss_t  = torch.tensor(val_loss_meter, device=device)
        val_count_t = torch.tensor(float(val_count), device=device)
        if world_size > 1:
            dist.all_reduce(val_loss_t,  op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_t, op=dist.ReduceOp.SUM)
        val_loss_avg = (val_loss_t / val_count_t.clamp_min(1.0)).item()

        # Quick Brier proxy on rank 0 for checkpoint selection
        val_brier = float("inf")
        if is_main:
            brier_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=collate_fn, num_workers=0,
            )
            val_brier = quick_brier_estimate(
                system, brier_loader, device, ae_chunk_size,
                use_precomputed_vil=use_precomputed, max_batches=20,
            )

            # Log MoE layer stats (mean aux loss across all layers)
            base_sys = system.module if hasattr(system, "module") else system
            moe_stats = get_moe_layer_stats(base_sys.denoiser)
            mean_layer_aux = (
                sum(moe_stats.values()) / len(moe_stats) if moe_stats else 0.0
            )

            print(
                f"Epoch {epoch:03d} | "
                f"train_cfm={train_cfm:.4f} | train_aux={train_aux:.4f} | "
                f"train_total={train_total:.4f} | val_cfm={val_loss_avg:.4f} | "
                f"val_brier(proxy)={val_brier:.4f} | "
                f"mean_layer_aux={mean_layer_aux:.4f} | "
                f"n_moe_layers={len(moe_stats)}"
            )

            save_sys = system.module if hasattr(system, "module") else system
            save_payload = {
                "epoch": epoch,
                "best_val_brier": best_val_brier,
                "model_state_dict": save_sys.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": OmegaConf.to_container(config, resolve=True),
                "moe_num_experts": moe_num_experts,
                "moe_top_k": moe_top_k,
                "moe_strategy": moe_strategy,
                "cross_modal_routing": moe_cross_modal,
                "cross_modal_router_type": moe_router_type,
                "shared_expert": moe_shared_expert,
            }

            if val_brier < best_val_brier:
                best_val_brier = val_brier
                save_payload["best_val_brier"] = best_val_brier
                torch.save(save_payload, best_ckpt)
                print(f"  -> Saved best checkpoint: {best_ckpt}")

            # Periodic checkpoint every 5 epochs (epoch 4,9,14,...,49)
            if (epoch + 1) % 5 == 0:
                periodic_ckpt = os.path.join(
                    model_dir, f"checkpoint_epoch{epoch + 1:03d}.pt"
                )
                torch.save(save_payload, periodic_ckpt)
                print(f"  -> Saved periodic checkpoint: {periodic_ckpt}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
