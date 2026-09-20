"""
Single-GPU training script for conditional lightning diffusion.

This script follows the design in `LIGHTNING_FLOWCAST_BINARY_REVISION.md`:

    v_t = f_theta(t, x_t, z_vil_past, z_vil_future_pred, lgt_past)

where:
    - x_t is the noisy future lightning tensor on the native 48x48 grid
    - z_vil_past is the encoded past VIL condition
    - z_vil_future_pred is the frozen predicted future VIL latent condition
    - lgt_past is the transformed-count past lightning condition

Notes
-----
- Uses LightningDiffusionSystem from lightning_condition_diffusion.py so that
  checkpoints are compatible with dist_train_lightning_diffusion.py and
  test_lightning_diffusion.py.
- Saves under the key "model_state_dict" to match the test script.
"""

import argparse
import math
import os
import sys
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from common.cfm.cfm import ConditionalFlowMatcher
from common.models.flowcast.lightning_condition_diffusion import LightningDiffusionSystem
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
)


def train_or_eval_epoch(
    *,
    loader: DataLoader,
    system: LightningDiffusionSystem,
    flow_matcher: ConditionalFlowMatcher,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
    ae_chunk_size: Optional[Union[int, str]],
) -> float:
    if train:
        system.train()
        if not any(p.requires_grad for p in system.vil_encoder.parameters()):
            system.vil_encoder.eval()
    else:
        system.eval()

    loss_meter = 0.0
    count = 0

    for x_vil_past, lgt_past, lgt_future, _ in loader:
        x_vil_past = x_vil_past.to(device, non_blocking=True)
        lgt_past = lgt_past.to(device, non_blocking=True)
        lgt_future = lgt_future.to(device, non_blocking=True)

        x0_noise = torch.randn_like(lgt_future)

        if train:
            optimizer.zero_grad(set_to_none=True)

        encoder_needs_grad = train and any(
            p.requires_grad for p in system.vil_encoder.parameters()
        )
        encode_ctx = torch.enable_grad() if encoder_needs_grad else torch.no_grad()
        with encode_ctx:
            z_vil_past, z_vil_future_pred = system.prepare_vil_conditions(
                x_vil_past,
                ae_chunk_size=ae_chunk_size,
            )

        fwd_ctx = torch.enable_grad() if train else torch.no_grad()
        with fwd_ctx:
            t, x_t, u_t = flow_matcher.sample_location_and_conditional_flow(
                x0_noise, lgt_future
            )
            flow_pred = system.forward_from_encoded(
                t,
                x_t,
                z_vil_past,
                lgt_past,
                z_vil_future_pred=z_vil_future_pred,
            )
            loss = F.mse_loss(flow_pred, u_t)
            if train:
                loss.backward()
                optimizer.step()

        batch_size = x_vil_past.shape[0]
        loss_meter += float(loss.item()) * batch_size
        count += batch_size

    return loss_meter / max(count, 1)


def build_optimizer(
    system: LightningDiffusionSystem,
    learning_rate: float,
    encoder_lr_ratio: float,
    finetune_vil_encoder: bool,
) -> torch.optim.Optimizer:
    model_params = [p for p in system.denoiser.parameters() if p.requires_grad]
    if finetune_vil_encoder:
        encoder_params = [p for p in system.vil_encoder.parameters() if p.requires_grad]
        param_groups = [
            {"params": model_params, "lr": learning_rate},
            {"params": encoder_params, "lr": learning_rate * encoder_lr_ratio},
        ]
    else:
        param_groups = [{"params": model_params, "lr": learning_rate}]
    return torch.optim.AdamW(param_groups, lr=learning_rate, weight_decay=1.0e-4)


def parse_args():
    def parse_ae_chunk_size_arg(value: str):
        normalized = value.strip().lower()
        if normalized in {"", "auto", "max", "full"}:
            return "auto"
        return int(value)

    parser = argparse.ArgumentParser(
        description="Single-GPU conditional lightning diffusion training script."
    )
    parser.add_argument("--train-meta", type=str, required=True)
    parser.add_argument("--val-meta", type=str, required=True)
    parser.add_argument("--train-vil-h5", type=str, required=True)
    parser.add_argument("--val-vil-h5", type=str, required=True)
    parser.add_argument("--train-lght-h5", type=str, required=True)
    parser.add_argument("--val-lght-h5", type=str, required=True)
    parser.add_argument("--ae-config", type=str, required=True)
    parser.add_argument("--ae-checkpoint", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="artifacts/sevir/lightning_diffusion_draft")

    parser.add_argument("--vil-dataset", type=str, default="vil")
    parser.add_argument("--lght-dataset", type=str, default="lght")
    parser.add_argument("--vil-row-key", type=str, default=None)
    parser.add_argument("--lght-row-key", type=str, default=None)

    parser.add_argument("--lag-time", type=int, default=13)
    parser.add_argument("--lead-time", type=int, default=12)
    parser.add_argument("--time-spacing", type=int, default=1)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--raw-seq-len", type=int, default=49)

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--encoder-lr-ratio", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--c-cap", type=float, default=20.0)
    parser.add_argument("--hidden-channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--ae-chunk-size", type=parse_ae_chunk_size_arg, default="auto")
    parser.add_argument("--target-hw", type=int, nargs=2, default=[48, 48])
    parser.add_argument("--use-future-vil-condition", action="store_true")
    parser.add_argument("--future-vil-flowcast-config", type=str, default=None)
    parser.add_argument("--future-vil-flowcast-checkpoint", type=str, default=None)
    parser.add_argument("--future-vil-euler-steps", type=int, default=10)
    parser.add_argument("--future-vil-probabilistic-samples", type=int, default=1)

    parser.add_argument("--finetune-vil-encoder", action="store_true")
    parser.add_argument("--debug-mode", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    train_dataset = DynamicPairedVilLightningDataset(
        meta_csv=args.train_meta,
        vil_file=args.train_vil_h5,
        lightning_file=args.train_lght_h5,
        vil_dataset=args.vil_dataset,
        lightning_dataset=args.lght_dataset,
        vil_row_key=args.vil_row_key,
        lightning_row_key=args.lght_row_key,
        raw_seq_len=args.raw_seq_len,
        lag_time=args.lag_time,
        lead_time=args.lead_time,
        time_spacing=args.time_spacing,
        stride=args.stride,
        c_cap=args.c_cap,
        debug_mode=args.debug_mode,
    )
    val_dataset = DynamicPairedVilLightningDataset(
        meta_csv=args.val_meta,
        vil_file=args.val_vil_h5,
        lightning_file=args.val_lght_h5,
        vil_dataset=args.vil_dataset,
        lightning_dataset=args.lght_dataset,
        vil_row_key=args.vil_row_key,
        lightning_row_key=args.lght_row_key,
        raw_seq_len=args.raw_seq_len,
        lag_time=args.lag_time,
        lead_time=args.lead_time,
        time_spacing=args.time_spacing,
        stride=args.stride,
        c_cap=args.c_cap,
        debug_mode=args.debug_mode,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers if not args.debug_mode else 0,
        collate_fn=dynamic_paired_vil_lightning_collate,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers if not args.debug_mode else 0,
        collate_fn=dynamic_paired_vil_lightning_collate,
        pin_memory=torch.cuda.is_available(),
    )

    system = LightningDiffusionSystem(
        ae_config_path=args.ae_config,
        ae_checkpoint_path=args.ae_checkpoint,
        lag_time=args.lag_time,
        lead_time=args.lead_time,
        target_hw=tuple(args.target_hw),
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        use_future_vil_condition=args.use_future_vil_condition,
        future_vil_flowcast_config_path=args.future_vil_flowcast_config,
        future_vil_flowcast_checkpoint_path=args.future_vil_flowcast_checkpoint,
        future_vil_euler_steps=args.future_vil_euler_steps,
        future_vil_probabilistic_samples=args.future_vil_probabilistic_samples,
    ).to(device)

    if args.finetune_vil_encoder:
        system.enable_vil_encoder_finetune(full_finetune=False)
    else:
        system.freeze_vil_encoder()

    optimizer = build_optimizer(
        system=system,
        learning_rate=args.learning_rate,
        encoder_lr_ratio=args.encoder_lr_ratio,
        finetune_vil_encoder=args.finetune_vil_encoder,
    )

    flow_matcher = ConditionalFlowMatcher(sigma=args.sigma)

    best_val = math.inf
    best_path = os.path.join(args.out_dir, "best_lightning_diffusion.pt")

    for epoch in range(args.num_epochs):
        train_loss = train_or_eval_epoch(
            loader=train_loader,
            system=system,
            flow_matcher=flow_matcher,
            optimizer=optimizer,
            device=device,
            train=True,
            ae_chunk_size=args.ae_chunk_size,
        )
        val_loss = train_or_eval_epoch(
            loader=val_loader,
            system=system,
            flow_matcher=flow_matcher,
            optimizer=None,
            device=device,
            train=False,
            ae_chunk_size=args.ae_chunk_size,
        )

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_loss": best_val,
                    "model_state_dict": system.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved checkpoint -> {best_path}")


if __name__ == "__main__":
    main()
