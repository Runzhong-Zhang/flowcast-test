"""
Distributed training script for conditional lightning diffusion on SEVIR.

This script mirrors the general workflow of `dist_train_flowcast.py`:

- `torchrun` / DDP friendly
- paired SEVIR VIL + lightning dataloading
- mixed precision support
- checkpoint saving on rank 0

Main model equation:

    v_t = f_theta(t, x_t, z_vil_past, z_vil_future_pred, lgt_past)
"""

import argparse
import datetime
import os
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from common.cfm.cfm import ConditionalFlowMatcher
from common.models.flowcast.lightning_condition_diffusion import (
    LightningDiffusionSystem,
)
from common.models.simvp_lightning import LightningSimVPSystem
from common.models.deeplight_lightning import (
    LightningDeepLightSystem,
    DeepLightLoss,
    prepare_deeplight_target,
    HazyLoss,
)
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
    PrecomputedPairedVilLightningDataset,
    precomputed_paired_vil_lightning_collate,
)


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Initializing DDP: Rank {rank}/{world_size}, Local Rank {local_rank}")
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        dist.barrier(device_ids=[local_rank])
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    print("Not running in distributed mode. Using single device.")
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


def strip_module_prefix_local(state_dict):
    """Remove an optional DistributedDataParallel module prefix."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}



def compute_loss(flow_pred, flow_target):
    return torch.mean((flow_pred - flow_target) ** 2)


def compute_soft_dice_loss_from_probs(
    probs: torch.Tensor,
    target_cf: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    if probs.shape != target_cf.shape:
        raise ValueError(
            f"Dice shape mismatch: probs={tuple(probs.shape)} vs target_cf={tuple(target_cf.shape)}"
        )
    probs_flat = probs.reshape(probs.shape[0], -1)
    target_flat = target_cf.reshape(target_cf.shape[0], -1)
    intersection = torch.sum(probs_flat * target_flat, dim=1)
    denominator = torch.sum(probs_flat, dim=1) + torch.sum(target_flat, dim=1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def compute_weighted_focal_bce_from_logits(
    logits: torch.Tensor,
    target_cf: torch.Tensor,
    pos_weight: float | None,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> torch.Tensor:
    if logits.shape != target_cf.shape:
        raise ValueError(
            f"Focal BCE shape mismatch: logits={tuple(logits.shape)} vs target_cf={tuple(target_cf.shape)}"
        )

    pos_weight_tensor = None
    if pos_weight is not None and pos_weight > 0:
        pos_weight_tensor = torch.tensor(
            pos_weight,
            device=logits.device,
            dtype=logits.dtype,
        )

    bce_per_elem = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target_cf,
        pos_weight=pos_weight_tensor,
        reduction="none",
    )
    probs = torch.sigmoid(logits)
    p_t = probs * target_cf + (1.0 - probs) * (1.0 - target_cf)
    focal_factor = (1.0 - p_t).clamp_min(0.0) ** float(gamma)

    if alpha is not None:
        alpha = float(alpha)
        alpha_factor = alpha * target_cf + (1.0 - alpha) * (1.0 - target_cf)
        focal_factor = focal_factor * alpha_factor

    return (focal_factor * bce_per_elem).mean()


_hazy_loss_fn = HazyLoss()


def compute_simvp_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float | None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    dice_weight: float = 0.1,
    use_hazy_loss: bool = False,
    hazy_weight: float = 1.0,
):
    if logits.ndim != 5:
        raise ValueError(
            f"SimVP logits must have shape (B, T, C, H, W), got {tuple(logits.shape)}"
        )
    if target.ndim != 5:
        raise ValueError(
            f"Lightning target must have shape (B, T, H, W, C), got {tuple(target.shape)}"
        )
    target_cf = target.permute(0, 1, 4, 2, 3).contiguous()
    if logits.shape != target_cf.shape:
        raise ValueError(
            "SimVP logits/target shape mismatch after target permutation: "
            f"logits={tuple(logits.shape)} vs target_cf={tuple(target_cf.shape)}"
        )
    focal_bce_loss = compute_weighted_focal_bce_from_logits(
        logits,
        target_cf,
        pos_weight=pos_weight,
        gamma=focal_gamma,
        alpha=focal_alpha,
    )
    probs = torch.sigmoid(logits)
    dice_loss = compute_soft_dice_loss_from_probs(probs, target_cf)
    loss = focal_bce_loss + float(dice_weight) * dice_loss
    if use_hazy_loss:
        loss = loss + float(hazy_weight) * _hazy_loss_fn(logits, target_cf, target_cf)
    return loss


def compute_brier_from_logits(logits: torch.Tensor, target: torch.Tensor) -> float:
    probs = torch.sigmoid(logits)
    target_cf = target.permute(0, 1, 4, 2, 3).contiguous()
    if probs.shape != target_cf.shape:
        raise ValueError(
            f"Brier shape mismatch: probs={tuple(probs.shape)} vs target_cf={tuple(target_cf.shape)}"
        )
    return float(torch.mean((probs - target_cf) ** 2).item())


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
    Cheap single-Euler-step Brier score estimate for checkpoint selection.

    Approximates the full ODE with one step: x_final = x0 + flow(t=0, x0, cond).
    This is a fast proxy — use the test script for the definitive evaluation.
    """
    system.eval()
    brier_sum = 0.0
    count = 0
    with torch.no_grad():
        base_system = system.module if hasattr(system, "module") else system
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            (
                x_vil_past,
                z_vil_past,
                z_vil_future_pred,
                lgt_past,
                lgt_future,
                _,
            ) = unpack_condition_batch(batch, device, use_precomputed_vil)

            if not use_precomputed_vil:
                z_vil_past, z_vil_future_pred = base_system.prepare_vil_conditions(
                    x_vil_past,
                    ae_chunk_size=ae_chunk_size,
                )

            x0_noise = torch.randn_like(lgt_future)
            t_zero = torch.zeros(lgt_future.shape[0], device=device)
            flow = base_system.forward_from_encoded(
                t_zero,
                x0_noise,
                z_vil_past,
                lgt_past,
                z_vil_future_pred=z_vil_future_pred,
            )

            # Single Euler step: x_final = x0 + 1.0 * flow
            x_final = x0_noise + flow
            probs = (x_final > 0.5).float()

            truth = (lgt_future >= 0.5).float()
            batch_size_local = lgt_future.shape[0]
            brier_sum += float(torch.mean((probs - truth) ** 2).item()) * batch_size_local
            count += batch_size_local

    return brier_sum / max(count, 1)


parser = argparse.ArgumentParser(description="DDP training for lightning diffusion.")
parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/runner/flowcast/lightning_diffusion_config.yaml",
)
parser.add_argument("--train_vil_file", type=str, required=True)
parser.add_argument("--train_lightning_file", type=str, required=True)
parser.add_argument("--train_meta", type=str, required=True)
parser.add_argument("--val_vil_file", type=str, required=True)
parser.add_argument("--val_lightning_file", type=str, required=True)
parser.add_argument("--val_meta", type=str, required=True)


def main():
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = rank == 0

    debug_mode = config.run_params.debug_mode
    run_string = config.run_params.run_string
    model_family = str(config.model_params.get("model_family", "diffusion")).lower()
    artifact_family_dir = (
        "lightning_diffusion" if model_family == "diffusion" else f"lightning_{model_family}"
    )
    best_ckpt_filename = (
        "best_lightning_diffusion.pt"
        if model_family == "diffusion"
        else f"best_lightning_{model_family}.pt"
    )
    artifacts_dir = os.path.join(
        "artifacts",
        "sevir",
        artifact_family_dir,
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + run_string,
    )
    model_dir = os.path.join(artifacts_dir, "models")
    if is_main_process:
        os.makedirs(model_dir, exist_ok=True)

    batch_size = config.training_params.micro_batch_size
    num_workers = config.training_params.num_workers
    num_epochs = config.training_params.num_epochs
    fp16 = config.training_params.fp16
    grad_clip = config.training_params.gradient_clip_val
    init_checkpoint = config.training_params.get("init_checkpoint", None)
    resume_optimizer = bool(config.training_params.get("resume_optimizer", False))
    resume_scheduler = bool(config.training_params.get("resume_scheduler", False))
    resume_scaler = bool(config.training_params.get("resume_scaler", False))
    learning_rate = config.optimizer_params.learning_rate
    weight_decay = config.optimizer_params.weight_decay
    scheduler_min_lr_ratio = config.scheduler_params.lr_cosine_min_lr_ratio

    lag_time = config.data_params.lag_time
    lead_time = config.data_params.lead_time
    time_spacing = config.data_params.time_spacing
    raw_seq_len = config.data_params.raw_seq_len
    stride = config.data_params.stride

    sigma = config.flow_matching_params.sigma
    c_cap = config.loss_params.c_cap

    ae_config = config.autoencoder_params.autoencoder_config
    ae_checkpoint = config.autoencoder_params.autoencoder_checkpoint
    ae_chunk_size = config.autoencoder_params.ae_chunk_size
    finetune_vil_encoder = config.autoencoder_params.finetune_vil_encoder
    full_encoder_finetune = config.autoencoder_params.full_encoder_finetune
    encoder_lr_ratio = config.autoencoder_params.encoder_lr_ratio

    vil_dataset = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    target_hw = tuple(config.model_params.target_hw)
    hidden_channels = config.model_params.hidden_channels
    num_blocks = config.model_params.num_blocks
    dropout = config.model_params.dropout
    denoiser_backbone = str(config.model_params.get("denoiser_backbone", "resblock")).lower()
    cuboid_cfg = config.model_params.get("cuboid_params", {})
    simvp_cfg = config.model_params.get("simvp_params", {})
    simvp_hid_s = int(simvp_cfg.get("hid_S", 32))
    simvp_hid_t = int(simvp_cfg.get("hid_T", 256))
    simvp_n_s = int(simvp_cfg.get("N_S", 4))
    simvp_n_t = int(simvp_cfg.get("N_T", 4))
    simvp_spatio_kernel_enc = int(simvp_cfg.get("spatio_kernel_enc", 3))
    simvp_spatio_kernel_dec = int(simvp_cfg.get("spatio_kernel_dec", 3))
    simvp_pos_weight = simvp_cfg.get("pos_weight", None)
    simvp_focal_gamma = float(simvp_cfg.get("focal_gamma", 2.0))
    simvp_focal_alpha = simvp_cfg.get("focal_alpha", None)
    simvp_dice_weight = float(simvp_cfg.get("dice_weight", 1.0))
    simvp_use_hazy_loss = bool(simvp_cfg.get("use_hazy_loss", False))
    simvp_hazy_weight = float(simvp_cfg.get("hazy_weight", 1.0))
    simvp_use_hourly_target = bool(simvp_cfg.get("use_hourly_target", False))
    simvp_hourly_pos_weight = float(simvp_cfg.get("hourly_pos_weight", 20.0))
    simvp_hourly_hazy_weight = float(simvp_cfg.get("hourly_hazy_weight", 1.0))
    simvp_hourly_bce_weight = float(simvp_cfg.get("hourly_bce_weight", 1.0))
    if simvp_pos_weight is not None:
        simvp_pos_weight = float(simvp_pos_weight)
    if simvp_focal_alpha is not None:
        simvp_focal_alpha = float(simvp_focal_alpha)
    deeplight_cfg = config.model_params.get("deeplight_params", {})
    deeplight_stem_channels = int(deeplight_cfg.get("stem_channels", 32))
    deeplight_lstm_hidden = int(deeplight_cfg.get("lstm_hidden", 64))
    deeplight_pos_weight = float(deeplight_cfg.get("pos_weight", 20.0))
    deeplight_hazy_weight = float(deeplight_cfg.get("hazy_weight", 1.0))
    deeplight_bce_weight = float(deeplight_cfg.get("bce_weight", 1.0))
    future_vil_cfg = config.get("future_vil_condition_params", {})
    use_future_vil_condition = bool(future_vil_cfg.get("enable", False))
    future_vil_flowcast_config = future_vil_cfg.get("flowcast_config", None)
    future_vil_flowcast_checkpoint = future_vil_cfg.get("flowcast_checkpoint", None)
    future_vil_euler_steps = int(future_vil_cfg.get("euler_steps", 10))
    future_vil_probabilistic_samples = int(
        future_vil_cfg.get("probabilistic_samples", 1)
    )
    precomputed_cfg = config.get("precomputed_vil_condition_params", {})
    use_precomputed_vil = bool(precomputed_cfg.get("enable", False))
    train_precomputed_latent_file = precomputed_cfg.get("train_latent_file", None)
    val_precomputed_latent_file = precomputed_cfg.get("val_latent_file", None)
    latent_past_dataset = precomputed_cfg.get("latent_past_dataset", "z_vil_past")
    latent_future_pred_dataset = precomputed_cfg.get(
        "latent_future_pred_dataset", "z_vil_future_pred"
    )
    if use_precomputed_vil and finetune_vil_encoder:
        raise ValueError(
            "Precomputed VIL conditions are incompatible with VIL encoder fine-tuning. "
            "Disable finetune_vil_encoder or switch back to the online conditioning path."
        )
    if model_family not in {"diffusion", "simvp", "deeplight"}:
        raise ValueError(
            f"Unsupported model_family={model_family!r}. Expected 'diffusion', 'simvp', or 'deeplight'."
        )
    if model_family in {"simvp", "deeplight"}:
        if use_precomputed_vil:
            raise ValueError(
                f"{model_family} uses raw past VIL + past lightning inputs and does not "
                "support precomputed latent conditions."
            )
        if use_future_vil_condition:
            raise ValueError(
                f"{model_family} does not use the future-VIL conditioning branch. "
                f"Disable future_vil_condition_params.enable for model_family={model_family}."
            )

    local_past_frames = batch_size * lag_time
    global_batch_size = batch_size * world_size
    global_past_frames = local_past_frames * world_size
    if is_main_process:
        print(f"Lightning model family: {model_family}")
        print(
            "Lightning diffusion batch setup | "
            f"micro_batch_size_per_gpu={batch_size} | world_size={world_size} | "
            f"global_batch_size={global_batch_size} | "
            f"past_frames_per_gpu={local_past_frames} | "
            f"global_past_frames={global_past_frames} | "
            f"ae_chunk_size_config={ae_chunk_size!r}"
        )
        print(
            "VIL encoder chunk policy | "
            "runtime chunk = min(configured ae_chunk_size, local B*T_in); "
            "None/0/'auto' use the full local B*T_in on each GPU."
        )
        if use_future_vil_condition:
            print(
                "Future VIL condition | enabled=True | "
                f"flowcast_config={future_vil_flowcast_config} | "
                f"flowcast_checkpoint={future_vil_flowcast_checkpoint} | "
                f"flowcast_euler_steps={future_vil_euler_steps} | "
                f"flowcast_probabilistic_samples={future_vil_probabilistic_samples}"
            )
            if finetune_vil_encoder:
                print(
                    "Warning: future-VIL conditioning uses a frozen pretrained FlowCast. "
                    "Fine-tuning the VIL encoder changes the latent distribution fed to that "
                    "frozen forecaster."
                )
        if use_precomputed_vil:
            print(
                "Precomputed VIL conditions | enabled=True | "
                f"train_latent_file={train_precomputed_latent_file} | "
                f"val_latent_file={val_precomputed_latent_file} | "
                f"latent_past_dataset={latent_past_dataset} | "
                f"latent_future_pred_dataset={latent_future_pred_dataset}"
            )
        if model_family == "simvp":
            print(
                "SimVP params | "
                f"hid_S={simvp_hid_s} | hid_T={simvp_hid_t} | "
                f"N_S={simvp_n_s} | N_T={simvp_n_t} | "
                f"spatio_kernel_enc={simvp_spatio_kernel_enc} | "
                f"spatio_kernel_dec={simvp_spatio_kernel_dec} | "
                f"pos_weight={simvp_pos_weight} | "
                f"focal_gamma={simvp_focal_gamma} | "
                f"focal_alpha={simvp_focal_alpha} | "
                f"dice_weight={simvp_dice_weight} | "
                f"use_hazy_loss={simvp_use_hazy_loss} | "
                f"hazy_weight={simvp_hazy_weight} | "
                f"use_hourly_target={simvp_use_hourly_target}"
            )
        if model_family == "deeplight":
            print(
                "DeepLight params | "
                f"stem_channels={deeplight_stem_channels} | "
                f"lstm_hidden={deeplight_lstm_hidden} | "
                f"pos_weight={deeplight_pos_weight} | "
                f"bce_weight={deeplight_bce_weight} | "
                f"hazy_weight={deeplight_hazy_weight} | "
                "target=1-hour aggregated binary"
            )

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if use_precomputed_vil:
        if not train_precomputed_latent_file or not val_precomputed_latent_file:
            raise ValueError(
                "precomputed_vil_condition_params.enable=true requires both "
                "train_latent_file and val_latent_file."
            )
        train_dataset = PrecomputedPairedVilLightningDataset(
            meta_csv=args.train_meta,
            lightning_file=args.train_lightning_file,
            latent_file=train_precomputed_latent_file,
            lightning_dataset=lightning_dataset,
            lightning_row_key=lightning_row_key,
            latent_past_dataset=latent_past_dataset,
            latent_future_pred_dataset=latent_future_pred_dataset,
            raw_seq_len=raw_seq_len,
            lag_time=lag_time,
            lead_time=lead_time,
            time_spacing=time_spacing,
            stride=stride,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )
        val_dataset = PrecomputedPairedVilLightningDataset(
            meta_csv=args.val_meta,
            lightning_file=args.val_lightning_file,
            latent_file=val_precomputed_latent_file,
            lightning_dataset=lightning_dataset,
            lightning_row_key=lightning_row_key,
            latent_past_dataset=latent_past_dataset,
            latent_future_pred_dataset=latent_future_pred_dataset,
            raw_seq_len=raw_seq_len,
            lag_time=lag_time,
            lead_time=lead_time,
            time_spacing=time_spacing,
            stride=stride,
            c_cap=c_cap,
            debug_mode=debug_mode,
        )
        collate_fn = precomputed_paired_vil_lightning_collate
    else:
        # DeepLight pre-resizes VIL to target_hw in the DataLoader worker (CPU),
        # avoiding large 384×384 tensors on the GPU and the F.interpolate overhead.
        vil_output_hw = tuple(target_hw) if model_family == "deeplight" else None
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
            vil_output_hw=vil_output_hw,
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
            vil_output_hw=vil_output_hw,
        )
        collate_fn = dynamic_paired_vil_lightning_collate

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=seed
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=not debug_mode,
        persistent_workers=not debug_mode and num_workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=not debug_mode,
        persistent_workers=not debug_mode and num_workers > 0,
        drop_last=False,
    )

    if model_family == "diffusion":
        system = LightningDiffusionSystem(
            ae_config_path=ae_config,
            ae_checkpoint_path=ae_checkpoint,
            lag_time=lag_time,
            lead_time=lead_time,
            target_hw=target_hw,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            dropout=dropout,
            use_future_vil_condition=use_future_vil_condition,
            future_vil_online_forecaster=not use_precomputed_vil,
            future_vil_flowcast_config_path=future_vil_flowcast_config,
            future_vil_flowcast_checkpoint_path=future_vil_flowcast_checkpoint,
            future_vil_euler_steps=future_vil_euler_steps,
            future_vil_probabilistic_samples=future_vil_probabilistic_samples,
            denoiser_backbone=denoiser_backbone,
            # cuboid backbone params (ignored when denoiser_backbone="resblock")
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
        ).to(device)

        if finetune_vil_encoder:
            system.enable_vil_encoder_finetune(full_finetune=full_encoder_finetune)
        else:
            system.freeze_vil_encoder()

        model_params = [p for p in system.denoiser.parameters() if p.requires_grad]
        encoder_params = [p for p in system.vil_encoder.parameters() if p.requires_grad]
        param_groups = [{"params": model_params, "lr": learning_rate}]
        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": learning_rate * encoder_lr_ratio}
            )
    elif model_family == "simvp":
        system = LightningSimVPSystem(
            lag_time=lag_time,
            lead_time=1 if simvp_use_hourly_target else lead_time,
            target_hw=target_hw,
            hid_S=simvp_hid_s,
            hid_T=simvp_hid_t,
            N_S=simvp_n_s,
            N_T=simvp_n_t,
            spatio_kernel_enc=simvp_spatio_kernel_enc,
            spatio_kernel_dec=simvp_spatio_kernel_dec,
        ).to(device)
        simvp_hourly_loss_fn = DeepLightLoss(
            pos_weight=simvp_hourly_pos_weight,
            w1=simvp_hourly_bce_weight,
            w3=simvp_hourly_hazy_weight,
        ).to(device) if simvp_use_hourly_target else None
        param_groups = [{"params": [p for p in system.parameters() if p.requires_grad], "lr": learning_rate}]
    else:  # deeplight
        system = LightningDeepLightSystem(
            lag_time=lag_time,
            target_hw=target_hw,
            stem_channels=deeplight_stem_channels,
            lstm_hidden=deeplight_lstm_hidden,
        ).to(device)
        deeplight_loss_fn = DeepLightLoss(
            pos_weight=deeplight_pos_weight,
            w1=deeplight_bce_weight,
            w3=deeplight_hazy_weight,
        ).to(device)
        param_groups = [{"params": [p for p in system.parameters() if p.requires_grad], "lr": learning_rate}]

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=weight_decay)
    start_epoch = 0
    best_val_loss = float("inf")
    checkpoint = None
    if init_checkpoint:
        checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model_state_dict = strip_module_prefix_local(checkpoint["model_state_dict"])
        system.load_state_dict(model_state_dict, strict=True)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_val_loss = float(
            checkpoint.get("best_val_brier", checkpoint.get("best_val_loss", float("inf")))
        )
        if resume_optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if is_main_process:
            print(
                "Warm start enabled | "
                f"init_checkpoint={init_checkpoint} | "
                f"start_epoch={start_epoch} | "
                f"resume_optimizer={resume_optimizer} | "
                f"resume_scheduler={resume_scheduler} | "
                f"resume_scaler={resume_scaler}"
            )

    scheduler_last_epoch = -1
    if start_epoch > 0 and not resume_scheduler:
        for group in optimizer.param_groups:
            group.setdefault("initial_lr", group["lr"])
        scheduler_last_epoch = start_epoch - 1
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, num_epochs),
        eta_min=learning_rate * scheduler_min_lr_ratio,
        last_epoch=scheduler_last_epoch,
    )

    if world_size > 1:
        system = DDP(
            system,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    flow_matcher = ConditionalFlowMatcher(sigma=sigma)
    scaler = torch.amp.GradScaler(enabled=fp16 and device.type == "cuda")
    if checkpoint is not None:
        if resume_scheduler and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if resume_scaler and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

    best_ckpt = os.path.join(model_dir, best_ckpt_filename)

    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        system.train()
        train_loss_meter = 0.0
        train_count = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Train Epoch {epoch} (Rank {rank})",
            disable=not is_main_process,
            leave=False,
        )

        for batch in train_bar:
            (
                x_vil_past,
                z_vil_past,
                z_vil_future_pred,
                lgt_past,
                lgt_future,
                _,
            ) = unpack_condition_batch(batch, device, use_precomputed_vil)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=fp16):
                if model_family == "diffusion":
                    x0_noise = torch.randn_like(lgt_future, device=device)
                    t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                        x0_noise, lgt_future
                    )
                    flow_pred = system(
                        t,
                        x_t,
                        x_vil_past,
                        lgt_past,
                        ae_chunk_size=ae_chunk_size,
                        z_vil_past=z_vil_past,
                        z_vil_future_pred=z_vil_future_pred,
                    )
                    loss = compute_loss(flow_pred, u_t)
                elif model_family == "simvp":
                    logits = system(x_vil_past, lgt_past)
                    if simvp_use_hourly_target:
                        # logits: (B, 1, 1, H, W) → (B, 1, H, W)
                        target_1h = prepare_deeplight_target(lgt_future)
                        loss = simvp_hourly_loss_fn(logits.squeeze(1), target_1h)
                    else:
                        loss = compute_simvp_loss(
                            logits,
                            lgt_future,
                            simvp_pos_weight,
                            focal_gamma=simvp_focal_gamma,
                            focal_alpha=simvp_focal_alpha,
                            dice_weight=simvp_dice_weight,
                            use_hazy_loss=simvp_use_hazy_loss,
                            hazy_weight=simvp_hazy_weight,
                        )
                else:  # deeplight
                    target_1h = prepare_deeplight_target(lgt_future)
                    logits = system(x_vil_past, lgt_past)
                    loss = deeplight_loss_fn(logits, target_1h)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(system.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_size_local = lgt_future.size(0)
            train_loss_meter += float(loss.item()) * batch_size_local
            train_count += batch_size_local

            if is_main_process:
                train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        train_loss_sum = torch.tensor(train_loss_meter, device=device)
        train_count_tensor = torch.tensor(float(train_count), device=device)
        if world_size > 1:
            dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(train_count_tensor, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_sum / train_count_tensor.clamp_min(1.0)).item()

        system.eval()
        val_loss_meter = 0.0
        val_brier_meter = 0.0
        val_count = 0

        with torch.no_grad():
            val_bar = tqdm(
                val_loader,
                desc=f"Val Epoch {epoch} (Rank {rank})",
                disable=not is_main_process,
                leave=False,
            )
            for batch in val_bar:
                (
                    x_vil_past,
                    z_vil_past,
                    z_vil_future_pred,
                    lgt_past,
                    lgt_future,
                    _,
                ) = unpack_condition_batch(batch, device, use_precomputed_vil)

                with torch.amp.autocast(device_type=device.type, enabled=fp16):
                    if model_family == "diffusion":
                        x0_noise = torch.randn_like(lgt_future, device=device)
                        t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                            x0_noise, lgt_future
                        )
                        flow_pred = system(
                            t,
                            x_t,
                            x_vil_past,
                            lgt_past,
                            ae_chunk_size=ae_chunk_size,
                            z_vil_past=z_vil_past,
                            z_vil_future_pred=z_vil_future_pred,
                        )
                        loss = compute_loss(flow_pred, u_t)
                        probs_for_brier = None
                        brier_target = None
                    elif model_family == "simvp":
                        logits = system(x_vil_past, lgt_past)
                        if simvp_use_hourly_target:
                            target_1h = prepare_deeplight_target(lgt_future)
                            loss = simvp_hourly_loss_fn(logits.squeeze(1), target_1h)
                            probs_for_brier = torch.sigmoid(logits.squeeze(1))
                            brier_target = target_1h
                        else:
                            loss = compute_simvp_loss(
                                logits,
                                lgt_future,
                                simvp_pos_weight,
                                focal_gamma=simvp_focal_gamma,
                                focal_alpha=simvp_focal_alpha,
                                dice_weight=simvp_dice_weight,
                                use_hazy_loss=simvp_use_hazy_loss,
                                hazy_weight=simvp_hazy_weight,
                            )
                            probs_for_brier = torch.sigmoid(logits)
                            brier_target = lgt_future.permute(0, 1, 4, 2, 3).contiguous()
                    else:  # deeplight
                        target_1h = prepare_deeplight_target(lgt_future)
                        logits = system(x_vil_past, lgt_past)
                        loss = deeplight_loss_fn(logits, target_1h)
                        probs_for_brier = torch.sigmoid(logits)
                        brier_target = target_1h

                batch_size_local = lgt_future.size(0)
                val_loss_meter += float(loss.item()) * batch_size_local
                if model_family in {"simvp", "deeplight"}:
                    if probs_for_brier.shape != brier_target.shape:
                        raise ValueError(
                            f"{model_family} validation probability/target shape mismatch: "
                            f"probs={tuple(probs_for_brier.shape)} vs target={tuple(brier_target.shape)}"
                        )
                    batch_brier = torch.mean((probs_for_brier - brier_target) ** 2)
                    val_brier_meter += float(batch_brier.item()) * batch_size_local
                val_count += batch_size_local

                if is_main_process:
                    val_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        val_loss_sum = torch.tensor(val_loss_meter, device=device)
        val_brier_sum = torch.tensor(val_brier_meter, device=device)
        val_count_tensor = torch.tensor(float(val_count), device=device)
        if world_size > 1:
            dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_brier_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_count_tensor, op=dist.ReduceOp.SUM)
        val_loss = (val_loss_sum / val_count_tensor.clamp_min(1.0)).item()
        val_brier_simvp = (val_brier_sum / val_count_tensor.clamp_min(1.0)).item()

        # Quick Brier estimate on rank 0 only (single Euler step proxy).
        # Used for checkpoint selection — lower Brier = better calibrated probability.
        # Run the definitive multi-sample Brier / reliability evaluation with test_lightning_diffusion.py.
        val_brier = float("inf")
        if is_main_process:
            if model_family == "diffusion":
                brier_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                )
                val_brier = quick_brier_estimate(
                    system,
                    brier_loader,
                    device,
                    ae_chunk_size,
                    use_precomputed_vil=use_precomputed_vil,
                    max_batches=20,
                )
            elif model_family in {"simvp", "deeplight"}:
                val_brier = val_brier_simvp
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_brier(proxy)={val_brier:.4f}"
            )

            if val_brier < best_val_loss:
                best_val_loss = val_brier
                model_to_save = system.module if hasattr(system, "module") else system
                torch.save(
                    {
                        "epoch": epoch,
                        "best_val_brier": best_val_loss,
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "config": OmegaConf.to_container(config, resolve=True),
                    },
                    best_ckpt,
                )
                print(f"Saved best checkpoint -> {best_ckpt}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
