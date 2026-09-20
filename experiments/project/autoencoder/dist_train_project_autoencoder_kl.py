"""
Distributed trainer for a radar VAE on the Singapore project dataset.

This is the project-data analogue of `experiments/sevir/autoencoder/dist_train_autoencoder_kl.py`.
It keeps the same AutoencoderKL + LPIPSWithDiscriminator training stack, but
loads radar frames from `nowcast_dataset` NetCDF samples.
"""

from __future__ import annotations

import argparse
import datetime
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders import AutoencoderKL
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

try:
    import wandb
except ImportError:
    wandb = None

from common.autoencoder.losses.lpips import LPIPSWithDiscriminator
from common.autoencoder.utils.early_stopping import EarlyStopping
from common.utils.utils import warmup_lambda
from experiments.project.dataset.project_nowcast_dataset import (
    DEFAULT_PROJECT_DATASET_ROOT,
    ProjectRadarAutoencoderDataset,
    get_project_file_list,
    project_autoencoder_collate,
)


def setup_ddp():
    """Initialize distributed training if `torchrun` environment variables exist."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
        torch.cuda.set_device(local_rank)
        dist.barrier()
        return rank, world_size, local_rank, torch.device(f"cuda:{local_rank}")
    return (
        0,
        1,
        0,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def flatten_dict(d, parent_key="", sep="."):
    """Flatten an OmegaConf container for logging."""
    items = []
    for key, value in d.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def save_reconstruction_examples(
    inputs: torch.Tensor,
    recon: torch.Tensor,
    metadata,
    output_dir: str,
    epoch: int,
):
    """Save a few input/reconstruction pairs from the validation set."""
    os.makedirs(output_dir, exist_ok=True)
    inputs_np = inputs.detach().cpu().numpy()
    recon_np = recon.detach().cpu().numpy()

    n_examples = min(4, inputs_np.shape[0])
    for idx in range(n_examples):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(inputs_np[idx, 0], origin="lower", cmap="turbo", vmin=0.0, vmax=1.0)
        axes[0].set_title("Radar Input")
        axes[1].imshow(recon_np[idx, 0], origin="lower", cmap="turbo", vmin=0.0, vmax=1.0)
        axes[1].set_title("Reconstruction")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        sample_id = metadata[idx].get("sample_id", metadata[idx].get("filename", f"sample_{idx}"))
        fig.suptitle(f"Epoch {epoch} | {sample_id}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"epoch_{epoch:03d}_sample_{idx}.png"), dpi=150)
        plt.close(fig)


def maybe_wrap_ddp(model, world_size: int, local_rank: int, device: torch.device):
    if world_size == 1:
        return model
    if device.type == "cuda":
        return DDP(model, device_ids=[local_rank], output_device=local_rank)
    return DDP(model)


def resize_frames_if_needed(
    frames: torch.Tensor,
    target_hw: Optional[Tuple[int, int]],
) -> torch.Tensor:
    """Resize `(B, C, H, W)` radar frames only when a target size is requested."""
    if target_hw is None:
        return frames
    target_hw = tuple(int(v) for v in target_hw)
    if frames.shape[-2:] == target_hw:
        return frames
    return F.interpolate(
        frames,
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    )


def main():
    parser = argparse.ArgumentParser(description="Project radar AutoencoderKL trainer.")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/project/autoencoder/project_autoencoder_kl_config.yaml",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=str(DEFAULT_PROJECT_DATASET_ROOT),
    )
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    rank, world_size, local_rank, device = setup_ddp()
    is_main_process = rank == 0

    run_params = config.run_params
    dataset_params = config.dataset_params
    training_params = config.training_params
    optimizer_params = config.optimizer_params
    scheduler_params = config.scheduler_params
    model_params = config.model_params
    loss_params = config.loss_params

    debug_mode = bool(run_params.debug_mode)
    enable_wandb = bool(run_params.enable_wandb) and is_main_process and wandb is not None
    run_string = str(run_params.run_string)

    dataset_root = Path(args.dataset_root)
    radar_max = float(dataset_params.radar_max)
    include_future_frames = bool(dataset_params.include_future_frames)
    include_radar_only = bool(dataset_params.include_radar_only)
    ro_fraction = float(dataset_params.ro_fraction)
    model_input_hw_cfg = dataset_params.get("model_input_hw", None)
    model_input_hw = (
        tuple(int(v) for v in model_input_hw_cfg)
        if model_input_hw_cfg is not None
        else None
    )

    batch_size = int(training_params.micro_batch_size)
    num_epochs = int(training_params.num_epochs)
    num_workers = int(training_params.num_workers)
    warmup_generator_epochs = int(training_params.warmup_generator_epochs)
    early_stopping_patience = int(training_params.early_stopping_patience)
    gradient_clip_val = float(training_params.gradient_clip_val)
    preload_model = training_params.preload_model

    learning_rate = float(optimizer_params.learning_rate)
    optimizer_type = str(optimizer_params.optimizer_type)
    weight_decay = float(optimizer_params.weight_decay)

    scheduler_type = str(scheduler_params.scheduler_type)
    lr_plateau_factor = float(scheduler_params.lr_plateau_factor)
    lr_plateau_patience = int(scheduler_params.lr_plateau_patience)
    lr_cosine_warmup_iter_percentage = float(scheduler_params.lr_cosine_warmup_iter_percentage)
    lr_cosine_min_warmup_lr_ratio = float(scheduler_params.lr_cosine_min_warmup_lr_ratio)
    lr_cosine_min_lr_ratio = float(scheduler_params.lr_cosine_min_lr_ratio)

    latent_channels = int(model_params.latent_channels)
    norm_num_groups = int(model_params.norm_num_groups)
    layers_per_block = int(model_params.layers_per_block)
    act_fn = str(model_params.act_fn)
    block_out_channels = list(model_params.block_out_channels)
    down_block_types = list(model_params.down_block_types)
    up_block_types = list(model_params.up_block_types)

    kl_weight = float(loss_params.kl_weight)
    disc_weight = float(loss_params.disc_weight)

    random.seed(42)
    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + rank)

    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + run_string
    artifacts_dir = os.path.join("artifacts", "project", "autoencoder_kl", run_id)
    plots_dir = os.path.join(artifacts_dir, "plots")
    examples_dir = os.path.join(plots_dir, "examples")
    model_dir = os.path.join(artifacts_dir, "models")
    model_save_path = os.path.join(model_dir, "early_stopping_model.pt")

    if is_main_process:
        os.makedirs(examples_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        print(f"Run ID: {run_id}")
        print(f"Dataset root: {dataset_root}")
        print(f"Include radar-only: {include_radar_only} | ro_fraction={ro_fraction}")
        print(f"Model input resize: {model_input_hw}")
        print(f"Device: {device}")

    train_files = get_project_file_list(
        "train",
        dataset_root=dataset_root,
        include_radar_only=include_radar_only,
        ro_fraction=ro_fraction,
        seed=42,
    )
    val_files = get_project_file_list(
        "val",
        dataset_root=dataset_root,
        include_radar_only=include_radar_only,
        ro_fraction=ro_fraction,
        seed=42,
    )

    train_dataset = ProjectRadarAutoencoderDataset(
        train_files,
        radar_max=radar_max,
        include_future_frames=include_future_frames,
    )
    val_dataset = ProjectRadarAutoencoderDataset(
        val_files,
        radar_max=radar_max,
        include_future_frames=include_future_frames,
    )

    if debug_mode:
        train_dataset = Subset(train_dataset, range(min(64, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(32, len(val_dataset))))

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=project_autoencoder_collate,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=not debug_mode,
        persistent_workers=(not debug_mode and num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=project_autoencoder_collate,
        num_workers=num_workers if not debug_mode else 0,
        pin_memory=not debug_mode,
        persistent_workers=(not debug_mode and num_workers > 0),
    )

    if is_main_process:
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    total_num_steps = len(train_loader) * num_epochs
    if total_num_steps == 0:
        raise RuntimeError("Training loader is empty; cannot start VAE training.")

    input_shape = None
    for batch in train_loader:
        frames, _ = batch
        input_shape = frames.shape
        break
    if input_shape is None:
        raise RuntimeError("Could not determine radar autoencoder input shape.")

    if enable_wandb:
        primitive_config = OmegaConf.to_container(config, resolve=True)
        wandb.init(
            project="project-radar-nowcasting",
            config=flatten_dict(primitive_config),
            name=run_id,
        )
    elif is_main_process and bool(run_params.enable_wandb) and wandb is None:
        print("wandb is not installed; continuing with wandb disabled.")

    model = AutoencoderKL(
        in_channels=int(input_shape[1]),
        out_channels=int(input_shape[1]),
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        block_out_channels=block_out_channels,
        act_fn=act_fn,
        latent_channels=latent_channels,
        norm_num_groups=norm_num_groups,
        layers_per_block=layers_per_block,
    )

    if preload_model is not None:
        checkpoint = torch.load(preload_model, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if is_main_process:
            print(f"Loaded preload checkpoint: {preload_model}")

    model = model.to(device)
    model = maybe_wrap_ddp(model, world_size, local_rank, device)

    criterion = LPIPSWithDiscriminator(
        kl_weight=kl_weight,
        disc_weight=disc_weight,
        disc_in_channels=1,
    ).to(device)

    if optimizer_type == "adam":
        gen_optimizer = torch.optim.Adam(model.module.parameters() if hasattr(model, "module") else model.parameters(), lr=learning_rate)
        disc_optimizer = torch.optim.Adam(criterion.discriminator.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        model_params_iter = model.module.parameters() if hasattr(model, "module") else model.parameters()
        gen_optimizer = torch.optim.AdamW(
            model_params_iter,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
        disc_optimizer = torch.optim.AdamW(
            criterion.discriminator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )
    else:
        raise ValueError(f"Unsupported optimizer_type={optimizer_type!r}")

    warmup_iter = int(np.round(lr_cosine_warmup_iter_percentage * total_num_steps))
    if scheduler_type == "cosine":
        gen_warmup_scheduler = LambdaLR(
            gen_optimizer,
            lr_lambda=warmup_lambda(
                warmup_steps=warmup_iter,
                min_lr_ratio=lr_cosine_min_warmup_lr_ratio,
            ),
        )
        gen_cosine_scheduler = CosineAnnealingLR(
            gen_optimizer,
            T_max=max(total_num_steps - warmup_iter, 1),
            eta_min=lr_cosine_min_lr_ratio * learning_rate,
        )
        gen_scheduler = SequentialLR(
            gen_optimizer,
            schedulers=[gen_warmup_scheduler, gen_cosine_scheduler],
            milestones=[warmup_iter],
        )
        disc_warmup_scheduler = LambdaLR(
            disc_optimizer,
            lr_lambda=warmup_lambda(
                warmup_steps=warmup_iter,
                min_lr_ratio=lr_cosine_min_warmup_lr_ratio,
            ),
        )
        disc_cosine_scheduler = CosineAnnealingLR(
            disc_optimizer,
            T_max=max(total_num_steps - warmup_iter, 1),
            eta_min=lr_cosine_min_lr_ratio * learning_rate,
        )
        disc_scheduler = SequentialLR(
            disc_optimizer,
            schedulers=[disc_warmup_scheduler, disc_cosine_scheduler],
            milestones=[warmup_iter],
        )
    elif scheduler_type == "plateau":
        gen_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            gen_optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
        )
        disc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            disc_optimizer,
            mode="min",
            factor=lr_plateau_factor,
            patience=lr_plateau_patience,
        )
    else:
        raise ValueError(f"Unsupported scheduler_type={scheduler_type!r}")

    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        verbose=True,
        path=model_save_path,
        val_loss_min=float("inf"),
    )

    global_step = 0

    def model_module():
        return model.module if hasattr(model, "module") else model

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        model.train()
        criterion.train()
        train_iter = tqdm(train_loader, desc=f"Train Epoch {epoch}", disable=not is_main_process)
        train_gen_loss_sum = 0.0
        train_disc_loss_sum = 0.0
        train_count = 0

        for frames, _ in train_iter:
            frames = frames.to(device, non_blocking=True)
            model_inputs = resize_frames_if_needed(frames, model_input_hw)
            gen_optimizer.zero_grad()
            disc_optimizer.zero_grad()

            outputs = model(model_inputs)
            recon = outputs.sample
            posterior = model_module().encode(model_inputs)
            recon_for_loss = resize_frames_if_needed(recon, tuple(frames.shape[-2:]))

            loss_gen, _ = criterion(
                frames,
                recon_for_loss,
                posterior,
                optimizer_idx=0,
                mask=None,
                last_layer=model_module().decoder.conv_out.weight,
                split="train",
            )
            loss_gen.backward()
            torch.nn.utils.clip_grad_norm_(model_module().parameters(), gradient_clip_val)
            gen_optimizer.step()

            loss_disc, _ = criterion(
                frames,
                recon_for_loss,
                posterior,
                optimizer_idx=1,
                mask=None,
                last_layer=model_module().decoder.conv_out.weight,
                split="train",
            )
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(criterion.discriminator.parameters(), gradient_clip_val)
            disc_optimizer.step()

            if scheduler_type == "cosine":
                gen_scheduler.step()
                disc_scheduler.step()

            train_gen_loss_sum += loss_gen.item()
            train_disc_loss_sum += loss_disc.item()
            train_count += 1
            global_step += frames.shape[0] * world_size

            if is_main_process:
                train_iter.set_postfix(
                    train_gen=f"{loss_gen.item():.3f}",
                    train_disc=f"{loss_disc.item():.3f}",
                )
                if enable_wandb:
                    wandb.log(
                        {
                            "train_gen_loss": loss_gen.item(),
                            "train_disc_loss": loss_disc.item(),
                        },
                        step=global_step,
                    )

            if debug_mode and train_count >= 3:
                break

        avg_train_gen = train_gen_loss_sum / max(train_count, 1)
        avg_train_disc = train_disc_loss_sum / max(train_count, 1)
        avg_train_total = 0.5 * (avg_train_gen + avg_train_disc)

        model.eval()
        criterion.eval()
        val_iter = tqdm(val_loader, desc=f"Val Epoch {epoch}", disable=not is_main_process)
        val_gen_loss_sum = 0.0
        val_disc_loss_sum = 0.0
        val_count = 0
        first_val_batch = None

        with torch.no_grad():
            for frames, metadata in val_iter:
                frames = frames.to(device, non_blocking=True)
                model_inputs = resize_frames_if_needed(frames, model_input_hw)
                outputs = model(model_inputs)
                recon = outputs.sample
                posterior = model_module().encode(model_inputs)
                recon_for_loss = resize_frames_if_needed(recon, tuple(frames.shape[-2:]))

                loss_gen, _ = criterion(
                    frames,
                    recon_for_loss,
                    posterior,
                    optimizer_idx=0,
                    mask=None,
                    last_layer=model_module().decoder.conv_out.weight,
                    split="val",
                )
                loss_disc, _ = criterion(
                    frames,
                    recon_for_loss,
                    posterior,
                    optimizer_idx=1,
                    mask=None,
                    last_layer=model_module().decoder.conv_out.weight,
                    split="val",
                )

                val_gen_loss_sum += loss_gen.item()
                val_disc_loss_sum += loss_disc.item()
                val_count += 1

                if first_val_batch is None:
                    first_val_batch = (
                        frames[:4].detach().cpu(),
                        recon_for_loss[:4].detach().cpu(),
                        metadata[:4],
                    )

                if is_main_process:
                    val_iter.set_postfix(
                        val_gen=f"{loss_gen.item():.3f}",
                        val_disc=f"{loss_disc.item():.3f}",
                    )

                if debug_mode and val_count >= 3:
                    break

        val_gen_tensor = torch.tensor(val_gen_loss_sum / max(val_count, 1), device=device)
        val_disc_tensor = torch.tensor(val_disc_loss_sum / max(val_count, 1), device=device)
        if world_size > 1:
            dist.all_reduce(val_gen_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(val_disc_tensor, op=dist.ReduceOp.SUM)
            val_gen_tensor /= world_size
            val_disc_tensor /= world_size

        avg_val_gen = val_gen_tensor.item()
        avg_val_disc = val_disc_tensor.item()
        avg_val_total = 0.5 * (avg_val_gen + avg_val_disc)

        if scheduler_type == "plateau":
            gen_scheduler.step(avg_val_total)
            disc_scheduler.step(avg_val_total)

        if is_main_process:
            print(
                f"Epoch {epoch:03d} | train_total={avg_train_total:.4f} | "
                f"train_gen={avg_train_gen:.4f} | train_disc={avg_train_disc:.4f} | "
                f"val_total={avg_val_total:.4f} | val_gen={avg_val_gen:.4f} | val_disc={avg_val_disc:.4f}"
            )
            if enable_wandb:
                wandb.log(
                    {
                        "val_total_loss": avg_val_total,
                        "val_gen_loss": avg_val_gen,
                        "val_disc_loss": avg_val_disc,
                        "learning_rate": gen_optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            if first_val_batch is not None:
                save_reconstruction_examples(
                    first_val_batch[0],
                    first_val_batch[1],
                    first_val_batch[2],
                    examples_dir,
                    epoch,
                )

        if epoch >= warmup_generator_epochs:
            if not criterion.discriminator_active:
                if is_main_process:
                    print("Activating discriminator after warmup.")
                criterion.activate_discriminator()
            elif is_main_process:
                early_stopping(
                    avg_val_gen,
                    model,
                    gen_optimizer,
                    disc_optimizer,
                    epoch,
                    global_step,
                )

        stop_signal = torch.tensor(0, device=device)
        if is_main_process and early_stopping.early_stop:
            print("Early stopping triggered.")
            stop_signal = torch.tensor(1, device=device)
        if world_size > 1:
            dist.broadcast(stop_signal, src=0)
        if int(stop_signal.item()) == 1:
            break

    if is_main_process:
        print(f"Finished project radar VAE training: {run_id}")
        if enable_wandb:
            wandb.finish()

    cleanup_ddp()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
