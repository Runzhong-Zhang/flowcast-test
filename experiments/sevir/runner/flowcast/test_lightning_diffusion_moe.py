"""
Evaluation script for FC Cuboid Sparse MoE lightning diffusion.

Mirrors test_lightning_diffusion.py exactly, with the only difference being
that the model is loaded as LightningDiffusionMoESystem (reading moe_num_experts
and moe_top_k from the checkpoint saved by dist_train_lightning_diffusion_moe.py).

Output format (Markdown results file, console metrics) is identical to
test_lightning_diffusion.py so results can be directly compared.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from common.models.flowcast.lightning_condition_diffusion_moe import (
    LightningDiffusionMoESystem,
)
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
)


# ---------------------------------------------------------------------------
# Metric utilities  (identical to test_lightning_diffusion.py)
# ---------------------------------------------------------------------------

def max_pool_spatial(arr: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size <= 1:
        return arr
    h, w = arr.shape[-2], arr.shape[-1]
    pad_h = (-h) % pool_size
    pad_w = (-w) % pool_size
    if pad_h or pad_w:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[-2] = (0, pad_h)
        pad_width[-1] = (0, pad_w)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
        h, w = arr.shape[-2], arr.shape[-1]
    new_shape = arr.shape[:-2] + (h // pool_size, pool_size, w // pool_size, pool_size)
    return arr.reshape(new_shape).max(axis=(-3, -1))


def compute_binary_metrics(
    scores: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    pool_size: int = 1,
) -> Dict[str, float]:
    pred = max_pool_spatial(scores >= threshold, pool_size)
    true = max_pool_spatial(truth >= 0.5,       pool_size)
    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, ~true).sum()
    fn = np.logical_and(~pred, true).sum()
    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    far       = fp / (tp + fp + eps)
    csi       = tp / (tp + fp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    return dict(
        tp=float(tp), fp=float(fp), fn=float(fn),
        precision=float(precision), recall=float(recall),
        pod=float(recall), far=float(far), csi=float(csi), f1=float(f1),
    )


def compute_probabilistic_metrics(probs: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    true    = truth >= 0.5
    brier   = float(np.mean((probs - true.astype(np.float32)) ** 2))
    clim    = true.mean()
    brier_ref = float(clim * (1.0 - clim))
    bss     = 1.0 - brier / (brier_ref + 1e-8)
    return dict(brier=brier, bss=float(bss))


def compute_per_lead_metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    pool_size: int = 1,
) -> List[Dict[str, float]]:
    results = []
    for lead_idx in range(probs.shape[1]):
        m = compute_binary_metrics(
            probs[:, lead_idx], truth[:, lead_idx], threshold, pool_size=pool_size
        )
        m["lead_idx"] = lead_idx
        results.append(m)
    return results


def aggregate_hourly_binary_target(truth: np.ndarray) -> np.ndarray:
    return (truth >= 0.5).max(axis=1).astype(np.float32)


def aggregate_hourly_forecast_score(mean_raw: np.ndarray) -> np.ndarray:
    return mean_raw.max(axis=1)


def build_threshold_sweep(
    start: float = 0.1, stop: float = 0.9, step: float = 0.1
) -> np.ndarray:
    thresholds = np.arange(start, stop + 0.5 * step, step, dtype=np.float32)
    return thresholds[(thresholds > 0.0) & (thresholds < 1.0)]


def find_best_csi_threshold(
    probs: np.ndarray,
    truth: np.ndarray,
    thresholds: Sequence[float],
    pool_size: int = 1,
):
    best = None
    sweep = []
    for thr in thresholds:
        m = compute_binary_metrics(probs, truth, float(thr), pool_size=pool_size)
        m["threshold"] = float(thr)
        sweep.append(m)
        if best is None or m["csi"] > best["csi"]:
            best = m
    return best, sweep


def unwrap_base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


# ---------------------------------------------------------------------------
# Shard I/O  (identical interface to test_lightning_diffusion.py)
# ---------------------------------------------------------------------------

def shard_result_path(out_dir: str, shard_rank: int, num_shards: int) -> str:
    return os.path.join(
        out_dir,
        f"lightning_moe_test_results.shard{shard_rank:02d}_of_{num_shards:02d}.npz",
    )


def save_shard_outputs(
    out_dir: str,
    shard_rank: int,
    num_shards: int,
    selected_start: int,
    selected_end: int,
    total_selected: int,
    mean_raw_forecast: np.ndarray,
    truth_array: np.ndarray,
) -> str:
    path = shard_result_path(out_dir, shard_rank, num_shards)
    np.savez_compressed(
        path,
        shard_rank=np.int32(shard_rank),
        num_shards=np.int32(num_shards),
        selected_start=np.int32(selected_start),
        selected_end=np.int32(selected_end),
        total_selected=np.int32(total_selected),
        mean_raw_forecast=mean_raw_forecast,
        truth=truth_array,
    )
    return path


def merge_shard_outputs(out_dir: str, num_shards: int):
    shards = []
    for r in range(num_shards):
        path = shard_result_path(out_dir, r, num_shards)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing shard {r}/{num_shards}: {path}")
        with np.load(path) as d:
            shards.append(dict(
                selected_start=int(d["selected_start"]),
                mean_raw_forecast=d["mean_raw_forecast"],
                truth=d["truth"],
            ))
    shards.sort(key=lambda s: s["selected_start"])
    mean_raw = np.concatenate([s["mean_raw_forecast"] for s in shards], axis=0)
    truth    = np.concatenate([s["truth"]             for s in shards], axis=0)
    return mean_raw, truth


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------

def save_markdown_summary(
    out_dir: str,
    checkpoint_path: str,
    test_data_percentage: float,
    probabilistic_samples: int,
    euler_steps: int,
    pool_size: int,
    default_threshold: float,
    prob_metrics: Dict[str, float],
    overall: Dict[str, float],
    pooled_overall: Dict[str, float],
    hourly_overall: Dict[str, float],
    pooled_hourly_overall: Dict[str, float],
    per_lead: Sequence[Dict[str, float]],
    pooled_per_lead: Sequence[Dict[str, float]],
):
    summary_path = os.path.join(out_dir, "TEST_RESULTS_best_checkpoint.md")
    lines = [
        "# Lightning Nowcast Test Results (FC Cuboid MoE)",
        "",
        "## Setup",
        "",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Test fraction: `{test_data_percentage}`",
        f"- Ensemble members: `{probabilistic_samples}`",
        f"- ODE Euler steps: `{euler_steps}`",
        f"- Spatial pooling size for pooled metrics: `{pool_size}x{pool_size}`",
        "- Frame-wise case: evaluate all future leads 1..12 separately, then pool TP/FP/FN across all samples and all leads.",
        "- Hourly aggregated case: target is `any lightning over leads 1..12`; forecast score is `max(mean raw forecast over leads 1..12)`.",
        "",
        "## Probabilistic Summary",
        "",
        f"- Brier score: `{prob_metrics['brier']:.6f}`",
        f"- Brier skill score: `{prob_metrics['bss']:.6f}`",
        "",
        "## Case 1: Frame-Wise Over 12 Leads",
        "",
        "| Metric view | Threshold | CSI | POD | FAR | F1 | Precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Exact | {overall['threshold']:.1f} | {overall['csi']:.6f} | "
            f"{overall['pod']:.6f} | {overall['far']:.6f} | "
            f"{overall['f1']:.6f} | {overall['precision']:.6f} |"
        ),
        (
            f"| Pooled {pool_size}x{pool_size} | {pooled_overall['threshold']:.1f} | "
            f"{pooled_overall['csi']:.6f} | {pooled_overall['pod']:.6f} | "
            f"{pooled_overall['far']:.6f} | {pooled_overall['f1']:.6f} | "
            f"{pooled_overall['precision']:.6f} |"
        ),
        "",
        "### Per-Lead Frame-Wise Metrics",
        "",
        "| Lead | Exact CSI | POD | FAR | Pooled CSI | Pooled POD | Pooled FAR |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ex, po in zip(per_lead, pooled_per_lead):
        lines.append(
            f"| {ex['lead_idx'] + 1} | "
            f"{ex['csi']:.6f} | {ex['pod']:.6f} | {ex['far']:.6f} | "
            f"{po['csi']:.6f} | {po['pod']:.6f} | {po['far']:.6f} |"
        )
    lines.extend([
        "",
        "## Case 2: Aggregated 1-Hour Target",
        "",
        "| Metric view | Threshold | CSI | POD | FAR | F1 | Precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Exact hourly | {hourly_overall['threshold']:.1f} | {hourly_overall['csi']:.6f} | "
            f"{hourly_overall['pod']:.6f} | {hourly_overall['far']:.6f} | "
            f"{hourly_overall['f1']:.6f} | {hourly_overall['precision']:.6f} |"
        ),
        (
            f"| Pooled hourly {pool_size}x{pool_size} | {pooled_hourly_overall['threshold']:.1f} | "
            f"{pooled_hourly_overall['csi']:.6f} | {pooled_hourly_overall['pod']:.6f} | "
            f"{pooled_hourly_overall['far']:.6f} | {pooled_hourly_overall['f1']:.6f} | "
            f"{pooled_hourly_overall['precision']:.6f} |"
        ),
        "",
    ])
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved results -> {summary_path}")


def finalize_results(
    out_dir: str,
    checkpoint_path: str,
    test_data_percentage: float,
    probabilistic_samples: int,
    euler_steps: int,
    pooled_pool_size: int,
    default_threshold: float,
    threshold_candidates: np.ndarray,
    mean_raw_forecast: np.ndarray,
    truth_array: np.ndarray,
):
    if mean_raw_forecast.shape[-1] == 1:
        mean_raw_forecast = mean_raw_forecast[..., 0]
    if truth_array.shape[-1] == 1:
        truth_array = truth_array[..., 0]

    mean_probs   = np.clip(mean_raw_forecast, 0.0, 1.0)
    hourly_truth = aggregate_hourly_binary_target(truth_array)
    hourly_score = aggregate_hourly_forecast_score(mean_raw_forecast)

    prob_metrics = compute_probabilistic_metrics(mean_probs, truth_array)
    print(f"\nBrier: {prob_metrics['brier']:.6f}  BSS: {prob_metrics['bss']:.6f}")

    best_overall,        _ = find_best_csi_threshold(mean_raw_forecast, truth_array,  threshold_candidates)
    pooled_best_overall, _ = find_best_csi_threshold(mean_raw_forecast, truth_array,  threshold_candidates, pool_size=pooled_pool_size)
    hourly_best_overall, _ = find_best_csi_threshold(hourly_score,      hourly_truth, threshold_candidates)
    pooled_hourly_best,  _ = find_best_csi_threshold(hourly_score,      hourly_truth, threshold_candidates, pool_size=pooled_pool_size)

    per_lead = compute_per_lead_metrics(
        mean_raw_forecast, truth_array, threshold=best_overall["threshold"]
    )
    pooled_per_lead = compute_per_lead_metrics(
        mean_raw_forecast, truth_array,
        threshold=pooled_best_overall["threshold"],
        pool_size=pooled_pool_size,
    )

    print(f"Frame-wise  Exact   CSI={best_overall['csi']:.6f}  POD={best_overall['pod']:.6f}  FAR={best_overall['far']:.6f}")
    print(f"Frame-wise  Pooled  CSI={pooled_best_overall['csi']:.6f}  POD={pooled_best_overall['pod']:.6f}  FAR={pooled_best_overall['far']:.6f}")
    print(f"Hourly      Exact   CSI={hourly_best_overall['csi']:.6f}  POD={hourly_best_overall['pod']:.6f}  FAR={hourly_best_overall['far']:.6f}")
    print(f"Hourly      Pooled  CSI={pooled_hourly_best['csi']:.6f}  POD={pooled_hourly_best['pod']:.6f}  FAR={pooled_hourly_best['far']:.6f}")

    save_markdown_summary(
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        test_data_percentage=test_data_percentage,
        probabilistic_samples=probabilistic_samples,
        euler_steps=euler_steps,
        pool_size=pooled_pool_size,
        default_threshold=default_threshold,
        prob_metrics=prob_metrics,
        overall=best_overall,
        pooled_overall=pooled_best_overall,
        hourly_overall=hourly_best_overall,
        pooled_hourly_overall=pooled_hourly_best,
        per_lead=per_lead,
        pooled_per_lead=pooled_per_lead,
    )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Test FC Cuboid Sparse MoE lightning diffusion model."
)
parser.add_argument("--checkpoint",          type=str, required=True)
parser.add_argument(
    "--config", type=str,
    default="experiments/sevir/runner/flowcast/lightning_diffusion_moe_config.yaml",
)
parser.add_argument("--test_vil_file",       type=str, required=True)
parser.add_argument("--test_lightning_file", type=str, required=True)
parser.add_argument("--test_meta",           type=str, required=True)
parser.add_argument("--test_data_percentage", type=float, default=1.0)
parser.add_argument("--num_workers",         type=int, default=None)
parser.add_argument("--batch_size",          type=int, default=None)
parser.add_argument("--ae_chunk_size",       type=int, default=None)
parser.add_argument("--num_shards",          type=int, default=1)
parser.add_argument("--shard_rank",          type=int, default=0)
parser.add_argument("--merge_shards",        action="store_true")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parser.parse_args()
    config = OmegaConf.load(args.config)

    out_dir = os.path.join(os.path.dirname(args.checkpoint), "..", "test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    probabilistic_samples = config.test_params.probabilistic_samples
    euler_steps           = config.test_params.euler_steps
    default_threshold     = config.test_params.probability_threshold
    threshold_candidates  = build_threshold_sweep(0.1, 0.9, 0.1)
    pooled_pool_size      = 2

    # ── Shard merge mode ─────────────────────────────────────────────────────
    if args.merge_shards:
        mean_raw, truth = merge_shard_outputs(out_dir, args.num_shards)
        finalize_results(
            out_dir=out_dir,
            checkpoint_path=args.checkpoint,
            test_data_percentage=args.test_data_percentage,
            probabilistic_samples=probabilistic_samples,
            euler_steps=euler_steps,
            pooled_pool_size=pooled_pool_size,
            default_threshold=default_threshold,
            threshold_candidates=threshold_candidates,
            mean_raw_forecast=mean_raw,
            truth_array=truth,
        )
        return

    # ── Load checkpoint ───────────────────────────────────────────────────────
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size  = config.test_params.micro_batch_size
    if args.batch_size  is not None: batch_size  = args.batch_size
    num_workers = config.test_params.num_workers
    if args.num_workers is not None: num_workers = args.num_workers

    lag_time     = config.data_params.lag_time
    lead_time    = config.data_params.lead_time
    time_spacing = config.data_params.time_spacing
    raw_seq_len  = config.data_params.raw_seq_len
    stride       = config.data_params.stride
    c_cap        = config.loss_params.c_cap

    ae_config     = config.autoencoder_params.autoencoder_config
    ae_checkpoint = config.autoencoder_params.autoencoder_checkpoint
    ae_chunk_size = config.autoencoder_params.ae_chunk_size
    if args.ae_chunk_size is not None:
        ae_chunk_size = args.ae_chunk_size

    vil_dataset       = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key       = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    target_hw         = tuple(config.model_params.target_hw)

    future_vil_cfg  = config.get("future_vil_condition_params", {})
    use_future_vil  = bool(future_vil_cfg.get("enable", False))
    future_vil_fc_config = future_vil_cfg.get("flowcast_config", None)
    future_vil_fc_ckpt   = future_vil_cfg.get("flowcast_checkpoint", None)
    future_vil_euler     = int(future_vil_cfg.get("euler_steps", 10))
    future_vil_prob      = int(future_vil_cfg.get("probabilistic_samples", 1))

    cuboid_cfg = config.model_params.get("cuboid_params", {})

    # MoE params: prefer checkpoint-saved values (authoritative), fall back to config
    moe_num_experts = int(checkpoint.get("moe_num_experts",
                          config.get("moe_params", {}).get("num_experts", 8)))
    moe_top_k       = int(checkpoint.get("moe_top_k",
                          config.get("moe_params", {}).get("top_k", 2)))
    moe_strategy    = str(checkpoint.get("moe_strategy",
                          config.get("moe_params", {}).get("strategy", "up_l0_t")))
    moe_cross_modal = bool(checkpoint.get("cross_modal_routing",
                          config.get("moe_params", {}).get("cross_modal_routing", False)))
    moe_vil_cond_dim = int(config.get("moe_params", {}).get("vil_cond_dim", 128))
    moe_router_type = str(checkpoint.get("cross_modal_router_type",
                          config.get("moe_params", {}).get("cross_modal_router_type", "concat")))
    moe_shared_expert = bool(checkpoint.get("shared_expert",
                          config.get("moe_params", {}).get("shared_expert", False)))
    print(f"MoE params: num_experts={moe_num_experts}  top_k={moe_top_k}  "
          f"strategy={moe_strategy}  cross_modal={moe_cross_modal}  "
          f"router_type={moe_router_type}  shared_expert={moe_shared_expert}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = DynamicPairedVilLightningDataset(
        meta_csv=args.test_meta,
        vil_file=args.test_vil_file,
        lightning_file=args.test_lightning_file,
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
        debug_mode=config.run_params.debug_mode,
    )

    if args.test_data_percentage < 1.0:
        n_subset = max(1, int(len(dataset) * args.test_data_percentage))
        dataset  = Subset(dataset, list(range(n_subset)))

    total_selected = len(dataset)
    shard_start, shard_end = 0, total_selected
    if args.num_shards > 1:
        shard_size  = (total_selected + args.num_shards - 1) // args.num_shards
        shard_start = args.shard_rank * shard_size
        shard_end   = min(total_selected, shard_start + shard_size)
        if shard_start >= total_selected:
            raise ValueError(f"Shard {args.shard_rank} has no samples (total={total_selected}).")
        dataset = Subset(dataset, list(range(shard_start, shard_end)))

    print(
        f"Testing shard {args.shard_rank}/{args.num_shards} | "
        f"samples={len(dataset)} [{shard_start}:{shard_end}]"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dynamic_paired_vil_lightning_collate,
        num_workers=num_workers if not config.run_params.debug_mode else 0,
        pin_memory=torch.cuda.is_available(),
    )

    # ── Model (the only difference vs test_lightning_diffusion.py) ────────────
    system = LightningDiffusionMoESystem(
        ae_config_path=ae_config,
        ae_checkpoint_path=ae_checkpoint,
        lag_time=lag_time,
        lead_time=lead_time,
        target_hw=target_hw,
        use_future_vil_condition=use_future_vil,
        future_vil_online_forecaster=True,
        future_vil_flowcast_config_path=future_vil_fc_config,
        future_vil_flowcast_checkpoint_path=future_vil_fc_ckpt,
        future_vil_euler_steps=future_vil_euler,
        future_vil_probabilistic_samples=future_vil_prob,
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
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_strategy=moe_strategy,
        cross_modal_routing=moe_cross_modal,
        vil_cond_dim=moe_vil_cond_dim,
        cross_modal_router_type=moe_router_type,
        moe_shared_expert=moe_shared_expert,
    ).to(device)

    missing, unexpected = system.load_state_dict(
        checkpoint["model_state_dict"], strict=False
    )
    disallowed_missing = [k for k in missing if not k.startswith("future_vil_forecaster.")]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch.  Missing: {disallowed_missing}. "
            f"Unexpected: {unexpected}."
        )
    system.freeze_vil_encoder()
    system.eval()

    # ── Inference loop (aligned with test_lightning_diffusion.py) ────────────
    all_mean_raw: List[np.ndarray] = []
    all_truth:    List[np.ndarray] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Inference", leave=True)):
            x_vil_past, lgt_past, lgt_future, _ = batch
            x_vil_past = x_vil_past.to(device)
            lgt_past   = lgt_past.to(device)
            lgt_future = lgt_future.to(device)

            batch_size_local = lgt_future.shape[0]
            t_future = lgt_future.shape[1]
            hz, wz = lgt_future.shape[2], lgt_future.shape[3]
            channels = lgt_future.shape[4]

            z_vil_past, z_vil_future_pred = system.prepare_vil_conditions(
                x_vil_past, ae_chunk_size=ae_chunk_size
            )

            raw_members = []
            for sample_idx in range(probabilistic_samples):
                # Deterministic seed per sample — matches baseline exactly
                torch.manual_seed(batch_idx * probabilistic_samples + sample_idx)
                x0_noise = torch.randn_like(lgt_future, device=device)
                x0_flat = x0_noise.view(batch_size_local * t_future, hz, wz, channels)

                def flow_dynamics(t_scalar, x_flat):
                    x_flow = x_flat.view(batch_size_local, t_future, hz, wz, channels)
                    t_batched = t_scalar * torch.ones(batch_size_local, device=device)
                    flow = system.forward_from_encoded(
                        t_batched, x_flow, z_vil_past, lgt_past,
                        z_vil_future_pred=z_vil_future_pred,
                    )
                    return flow.contiguous().view(batch_size_local * t_future, hz, wz, channels)

                # ODE integration — matches baseline: t_span=[0,1], step_size, rtol/atol
                t_span = torch.tensor([0.0, 1.0], device=device)
                step_size = 1.0 / float(euler_steps)
                solution = odeint(
                    flow_dynamics, x0_flat, t_span,
                    method="euler",
                    options=dict(step_size=step_size),
                    rtol=1e-2, atol=1e-3,
                    adjoint_params=tuple(system.parameters()),
                )
                x_final = solution[-1].view(batch_size_local, t_future, hz, wz, channels)
                raw_members.append(x_final.unsqueeze(1))

            # Ensemble mean — NO sigmoid, matches baseline (clip in finalize_results)
            raw_ensemble = torch.cat(raw_members, dim=1)        # (B, S, T, H, W, 1)
            mean_raw_batch = raw_ensemble.mean(dim=1).cpu().numpy().astype(np.float32)

            all_mean_raw.append(mean_raw_batch)
            all_truth.append(lgt_future.cpu().numpy().astype(np.float32))

    mean_raw_forecast = np.concatenate(all_mean_raw, axis=0)   # (N, T, H, W, 1)
    truth_array       = np.concatenate(all_truth,    axis=0)   # (N, T, H, W, 1)

    # ── Save shard or finalize ────────────────────────────────────────────────
    if args.num_shards > 1:
        save_shard_outputs(
            out_dir=out_dir,
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
            selected_start=shard_start,
            selected_end=shard_end,
            total_selected=total_selected,
            mean_raw_forecast=mean_raw_forecast,
            truth_array=truth_array,
        )
        print(f"Shard {args.shard_rank} saved. Run with --merge_shards to finalize.")
    else:
        finalize_results(
            out_dir=out_dir,
            checkpoint_path=args.checkpoint,
            test_data_percentage=args.test_data_percentage,
            probabilistic_samples=probabilistic_samples,
            euler_steps=euler_steps,
            pooled_pool_size=pooled_pool_size,
            default_threshold=default_threshold,
            threshold_candidates=threshold_candidates,
            mean_raw_forecast=mean_raw_forecast,
            truth_array=truth_array,
        )


if __name__ == "__main__":
    main()
