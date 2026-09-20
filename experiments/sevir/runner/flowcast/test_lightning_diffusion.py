"""
Evaluation / inference script for conditional lightning diffusion.

This follows the same high-level idea as `test_flowcast.py`:

- load a trained checkpoint
- iterate over the test set
- generate ensemble forecasts with ODE integration from noise
- average the raw ensemble predictions
- sweep thresholds on that mean forecast to report the best CSI
"""

import argparse
import os
import sys
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from common.models.flowcast.lightning_condition_diffusion import (
    LightningDiffusionSystem,
)
from common.models.simvp_lightning import LightningSimVPSystem
from common.models.deeplight_lightning import prepare_deeplight_target
from experiments.sevir.display.display import get_cmap
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
)


VIL_CMAP, VIL_NORM, _, _ = get_cmap("vil")
LIGHTNING_CONT_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "lightning_white_darkfire",
    ["#ffffff", "#ffe082", "#ff8f00", "#d84315", "#8b0000", "#220000"],
)
LIGHTNING_CONT_NORM = mpl.colors.PowerNorm(gamma=0.7, vmin=0.0, vmax=1.0)


def _format_past_label(col: int, n_past: int) -> str:
    offset = col - (n_past - 1)
    return str(offset)


def _plot_vil(ax, frame: np.ndarray, title: str):
    ax.imshow(
        np.clip(frame, 0.0, 1.0) * 255.0,
        origin="lower",
        cmap=VIL_CMAP,
        norm=VIL_NORM,
        interpolation="nearest",
    )
    ax.set_title(title)


def _plot_lightning_continuous(ax, frame: np.ndarray, title: str):
    ax.imshow(
        np.clip(frame, 0.0, 1.0),
        origin="lower",
        cmap=LIGHTNING_CONT_CMAP,
        norm=LIGHTNING_CONT_NORM,
        interpolation="nearest",
    )
    ax.set_facecolor("white")
    ax.set_title(title)


def _plot_lightning_binary(ax, frame: np.ndarray, title: str):
    ax.imshow(
        frame,
        origin="lower",
        cmap="gray_r",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    ax.set_facecolor("white")
    ax.set_title(title)


def decode_vil_latents_for_preview(
    system: LightningDiffusionSystem,
    z_vil_future_pred: torch.Tensor,
    ae_chunk_size=None,
) -> np.ndarray:
    """Decode future VIL latents to VIL pixels for preview only."""
    decoded = system.decode_vil_latents(
        z_vil_future_pred,
        ae_chunk_size=ae_chunk_size,
    )
    return decoded[0].detach().cpu().numpy().astype(np.float32)


def max_pool_spatial(arr: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size <= 1:
        return arr
    if arr.ndim < 2:
        raise ValueError("Expected at least 2 spatial dimensions for max pooling.")

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
    pred = scores >= threshold
    true = truth >= 0.5
    pred = max_pool_spatial(pred, pool_size)
    true = max_pool_spatial(true, pool_size)

    tp = np.logical_and(pred, true).sum()
    fp = np.logical_and(pred, np.logical_not(true)).sum()
    fn = np.logical_and(np.logical_not(pred), true).sum()
    tn = np.logical_and(np.logical_not(pred), np.logical_not(true)).sum()

    eps = 1.0e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    far = fp / (tp + fp + eps)
    csi = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
        "precision": float(precision),
        "recall": float(recall),
        "pod": float(recall),
        "far": float(far),
        "csi": float(csi),
        "f1": float(f1),
    }


def compute_probabilistic_metrics(probs: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    true = truth >= 0.5
    eps = 1.0e-8
    brier = np.mean((probs - true.astype(np.float32)) ** 2)
    clim = true.mean()
    brier_ref = float(clim * (1.0 - clim))
    bss = 1.0 - brier / (brier_ref + eps)
    return {
        "brier": float(brier),
        "bss": float(bss),
    }


def compute_reliability_diagram(
    probs: np.ndarray,
    truth: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Compute reliability diagram data (calibration curve).

    Returns bin_centers, mean_predicted_prob, observed_freq, and bin_counts.
    A perfectly calibrated model has mean_predicted_prob == observed_freq.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    mean_pred = np.zeros(n_bins)
    obs_freq = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)

    p_flat = probs.ravel()
    y_flat = (truth >= 0.5).ravel().astype(np.float32)

    for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (p_flat >= lo) & (p_flat < hi)
        if mask.sum() > 0:
            mean_pred[i] = p_flat[mask].mean()
            obs_freq[i] = y_flat[mask].mean()
            counts[i] = mask.sum()
        else:
            mean_pred[i] = bin_centers[i]
            obs_freq[i] = np.nan

    return {
        "bin_centers": bin_centers,
        "mean_predicted_prob": mean_pred,
        "observed_freq": obs_freq,
        "counts": counts,
    }


def save_reliability_plot(out_dir: str, reliability: Dict[str, np.ndarray], title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    valid = ~np.isnan(reliability["observed_freq"])
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(
        reliability["mean_predicted_prob"][valid],
        reliability["observed_freq"][valid],
        "o-", color="steelblue", label="Model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"Reliability diagram {title}")
    ax.legend()

    ax2 = axes[1]
    ax2.bar(reliability["bin_centers"], reliability["counts"],
            width=1.0 / len(reliability["bin_centers"]) * 0.9,
            color="steelblue", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Forecast histogram")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"reliability{title.replace(' ', '_')}.png"), dpi=140)
    plt.close(fig)


def compute_per_lead_metrics(
    probs: np.ndarray,
    truth: np.ndarray,
    threshold: float,
    pool_size: int = 1,
):
    lead_metrics = []
    for lead_idx in range(probs.shape[1]):
        metrics = compute_binary_metrics(
            probs[:, lead_idx],
            truth[:, lead_idx],
            threshold,
            pool_size=pool_size,
        )
        metrics["lead_idx"] = lead_idx
        lead_metrics.append(metrics)
    return lead_metrics


def aggregate_hourly_binary_target(truth: np.ndarray) -> np.ndarray:
    return (truth >= 0.5).max(axis=1).astype(np.float32)


def aggregate_hourly_forecast_score(mean_raw_forecast: np.ndarray) -> np.ndarray:
    return mean_raw_forecast.max(axis=1)


def build_threshold_sweep(start: float = 0.1, stop: float = 0.9, step: float = 0.1) -> np.ndarray:
    if step <= 0:
        raise ValueError("Threshold sweep step must be positive.")
    thresholds = np.arange(start, stop + 0.5 * step, step, dtype=np.float32)
    thresholds = thresholds[(thresholds > 0.0) & (thresholds < 1.0)]
    if thresholds.size == 0:
        raise ValueError("Threshold sweep produced no valid thresholds in (0, 1).")
    return thresholds


def find_best_csi_threshold(
    probs: np.ndarray,
    truth: np.ndarray,
    thresholds: Sequence[float],
    pool_size: int = 1,
):
    sweep_results = []
    best_metrics = None
    for threshold in thresholds:
        metrics = compute_binary_metrics(probs, truth, float(threshold), pool_size=pool_size)
        metrics["threshold"] = float(threshold)
        sweep_results.append(metrics)
        if best_metrics is None or metrics["csi"] > best_metrics["csi"]:
            best_metrics = metrics
    return best_metrics, sweep_results


def unwrap_base_dataset(dataset):
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def pack_optional_array(arr: np.ndarray | None) -> np.ndarray:
    if arr is None:
        return np.empty((0,), dtype=np.float32)
    return arr


def unpack_optional_array(arr: np.ndarray) -> np.ndarray | None:
    if arr.size == 0:
        return None
    return arr


def shard_result_path(out_dir: str, shard_rank: int, num_shards: int) -> str:
    return os.path.join(
        out_dir,
        f"lightning_test_results.shard{shard_rank:02d}_of_{num_shards:02d}.npz",
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
    preview_vil_past: np.ndarray | None,
    preview_vil_future: np.ndarray | None,
    preview_vil_future_pred: np.ndarray | None,
    preview_lgt_past: np.ndarray | None,
    preview_raw_ensemble: np.ndarray | None,
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
        preview_vil_past=pack_optional_array(preview_vil_past),
        preview_vil_future=pack_optional_array(preview_vil_future),
        preview_vil_future_pred=pack_optional_array(preview_vil_future_pred),
        preview_lgt_past=pack_optional_array(preview_lgt_past),
        preview_raw_ensemble=pack_optional_array(preview_raw_ensemble),
    )
    return path


def merge_shard_outputs(out_dir: str, num_shards: int):
    shard_payloads = []
    for shard_rank in range(num_shards):
        path = shard_result_path(out_dir, shard_rank, num_shards)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing shard output for rank {shard_rank}/{num_shards}: {path}"
            )
        with np.load(path) as shard_npz:
            shard_payloads.append(
                {
                    "path": path,
                    "shard_rank": int(shard_npz["shard_rank"]),
                    "num_shards": int(shard_npz["num_shards"]),
                    "selected_start": int(shard_npz["selected_start"]),
                    "selected_end": int(shard_npz["selected_end"]),
                    "total_selected": int(shard_npz["total_selected"]),
                    "mean_raw_forecast": shard_npz["mean_raw_forecast"],
                    "truth": shard_npz["truth"],
                    "preview_vil_past": unpack_optional_array(shard_npz["preview_vil_past"]),
                    "preview_vil_future": unpack_optional_array(shard_npz["preview_vil_future"]),
                    "preview_vil_future_pred": unpack_optional_array(
                        shard_npz["preview_vil_future_pred"]
                    ),
                    "preview_lgt_past": unpack_optional_array(shard_npz["preview_lgt_past"]),
                    "preview_raw_ensemble": unpack_optional_array(
                        shard_npz["preview_raw_ensemble"]
                    ),
                }
            )

    shard_payloads.sort(key=lambda item: item["selected_start"])
    mean_raw_forecast = np.concatenate(
        [item["mean_raw_forecast"] for item in shard_payloads], axis=0
    )
    truth_array = np.concatenate([item["truth"] for item in shard_payloads], axis=0)
    preview_source = next(
        (
            item
            for item in shard_payloads
            if item["preview_vil_past"] is not None
        ),
        None,
    )
    if preview_source is None:
        preview_vil_past = None
        preview_vil_future = None
        preview_vil_future_pred = None
        preview_lgt_past = None
        preview_raw_ensemble = None
    else:
        preview_vil_past = preview_source["preview_vil_past"]
        preview_vil_future = preview_source["preview_vil_future"]
        preview_vil_future_pred = preview_source["preview_vil_future_pred"]
        preview_lgt_past = preview_source["preview_lgt_past"]
        preview_raw_ensemble = preview_source["preview_raw_ensemble"]

    return (
        mean_raw_forecast,
        truth_array,
        preview_vil_past,
        preview_vil_future,
        preview_vil_future_pred,
        preview_lgt_past,
        preview_raw_ensemble,
    )


def finalize_results(
    out_dir: str,
    checkpoint_path: str,
    test_data_percentage: float,
    probabilistic_samples: int,
    euler_steps: int,
    pooled_pool_size: int,
    default_threshold: float,
    threshold_candidates: Sequence[float],
    mean_raw_forecast: np.ndarray,
    truth_array: np.ndarray,
    preview_vil_past: np.ndarray | None = None,
    preview_vil_future: np.ndarray | None = None,
    preview_vil_future_pred: np.ndarray | None = None,
    preview_lgt_past: np.ndarray | None = None,
    preview_raw_ensemble: np.ndarray | None = None,
):
    if preview_raw_ensemble is not None and preview_raw_ensemble.shape[-1] == 1:
        preview_raw_ensemble = preview_raw_ensemble[..., 0]
    if mean_raw_forecast.shape[-1] == 1:
        mean_raw_forecast = mean_raw_forecast[..., 0]
    if truth_array.shape[-1] == 1:
        truth_array = truth_array[..., 0]

    mean_probs = np.clip(mean_raw_forecast, 0.0, 1.0)
    hourly_truth = aggregate_hourly_binary_target(truth_array)
    hourly_score = aggregate_hourly_forecast_score(mean_raw_forecast)

    best_overall, threshold_sweep = find_best_csi_threshold(
        mean_raw_forecast, truth_array, threshold_candidates
    )
    pooled_best_overall, pooled_threshold_sweep = find_best_csi_threshold(
        mean_raw_forecast,
        truth_array,
        threshold_candidates,
        pool_size=pooled_pool_size,
    )
    hourly_best_overall, hourly_threshold_sweep = find_best_csi_threshold(
        hourly_score, hourly_truth, threshold_candidates
    )
    pooled_hourly_best_overall, pooled_hourly_threshold_sweep = find_best_csi_threshold(
        hourly_score,
        hourly_truth,
        threshold_candidates,
        pool_size=pooled_pool_size,
    )
    best_threshold = best_overall["threshold"]
    pooled_best_threshold = pooled_best_overall["threshold"]
    hourly_best_threshold = hourly_best_overall["threshold"]
    pooled_hourly_best_threshold = pooled_hourly_best_overall["threshold"]
    overall = best_overall
    per_lead = compute_per_lead_metrics(
        mean_raw_forecast,
        truth_array,
        threshold=best_threshold,
    )
    pooled_per_lead = compute_per_lead_metrics(
        mean_raw_forecast,
        truth_array,
        threshold=pooled_best_threshold,
        pool_size=pooled_pool_size,
    )
    mean_binary_pred_array = (mean_raw_forecast >= best_threshold).astype(np.float32)
    prob_metrics = compute_probabilistic_metrics(mean_probs, truth_array)

    print("Clipped probabilistic metrics on mean raw forecast:")
    print(f"  Brier score : {prob_metrics['brier']:.6f}  (lower is better, 0=perfect)")
    print(f"  Brier skill : {prob_metrics['bss']:.6f}   (higher is better, 1=perfect, 0=climatology)")

    default_metrics = compute_binary_metrics(mean_raw_forecast, truth_array, threshold=default_threshold)
    pooled_default_metrics = compute_binary_metrics(
        mean_raw_forecast,
        truth_array,
        threshold=default_threshold,
        pool_size=pooled_pool_size,
    )
    hourly_default_metrics = compute_binary_metrics(
        hourly_score,
        hourly_truth,
        threshold=default_threshold,
    )
    pooled_hourly_default_metrics = compute_binary_metrics(
        hourly_score,
        hourly_truth,
        threshold=default_threshold,
        pool_size=pooled_pool_size,
    )
    print("\nThreshold sweep for mean raw forecast:")
    for item in threshold_sweep:
        print(f"  thr={item['threshold']:.1f} csi={item['csi']:.6f}")

    print(
        f"\nThreshold sweep for mean raw forecast after {pooled_pool_size}x{pooled_pool_size} max pooling:"
    )
    for item in pooled_threshold_sweep:
        print(f"  thr={item['threshold']:.1f} pooled_csi={item['csi']:.6f}")

    print("\nThreshold sweep for hourly aggregated forecast (max over 12 future leads):")
    for item in hourly_threshold_sweep:
        print(f"  thr={item['threshold']:.1f} hourly_csi={item['csi']:.6f}")

    print(
        f"\nThreshold sweep for hourly aggregated forecast after "
        f"{pooled_pool_size}x{pooled_pool_size} max pooling:"
    )
    for item in pooled_hourly_threshold_sweep:
        print(f"  thr={item['threshold']:.1f} pooled_hourly_csi={item['csi']:.6f}")

    print(
        f"\nBest threshold-based metrics "
        f"(best_mean_threshold={best_threshold:.1f}, default_threshold={default_threshold:.1f}):"
    )
    for key in ("csi", "pod", "recall", "far", "f1", "precision"):
        print(f"  {key}: {overall[key]:.6f}")
    print(
        f"\nReference metrics at default threshold={default_threshold:.1f}: "
        f"CSI={default_metrics['csi']:.6f}, POD={default_metrics['pod']:.6f}, FAR={default_metrics['far']:.6f}"
    )

    print(
        f"\nBest pooled threshold-based metrics "
        f"({pooled_pool_size}x{pooled_pool_size} max pooling, "
        f"best_mean_threshold={pooled_best_threshold:.1f}, default_threshold={default_threshold:.1f}):"
    )
    for key in ("csi", "pod", "recall", "far", "f1", "precision"):
        print(f"  pooled_{key}: {pooled_best_overall[key]:.6f}")
    print(
        f"\nReference pooled metrics at default threshold={default_threshold:.1f}: "
        f"pooled_CSI={pooled_default_metrics['csi']:.6f}, "
        f"pooled_POD={pooled_default_metrics['pod']:.6f}, "
        f"pooled_FAR={pooled_default_metrics['far']:.6f}"
    )

    print(
        f"\nBest hourly aggregated metrics "
        f"(max over 12 leads, best_threshold={hourly_best_threshold:.1f}, "
        f"default_threshold={default_threshold:.1f}):"
    )
    for key in ("csi", "pod", "far", "f1", "precision"):
        print(f"  hourly_{key}: {hourly_best_overall[key]:.6f}")
    print(
        f"\nReference hourly metrics at default threshold={default_threshold:.1f}: "
        f"hourly_CSI={hourly_default_metrics['csi']:.6f}, "
        f"hourly_POD={hourly_default_metrics['pod']:.6f}, "
        f"hourly_FAR={hourly_default_metrics['far']:.6f}"
    )

    print(
        f"\nBest pooled hourly aggregated metrics "
        f"({pooled_pool_size}x{pooled_pool_size} max pooling, "
        f"best_threshold={pooled_hourly_best_threshold:.1f}, "
        f"default_threshold={default_threshold:.1f}):"
    )
    for key in ("csi", "pod", "far", "f1", "precision"):
        print(f"  pooled_hourly_{key}: {pooled_hourly_best_overall[key]:.6f}")
    print(
        f"\nReference pooled hourly metrics at default threshold={default_threshold:.1f}: "
        f"pooled_hourly_CSI={pooled_hourly_default_metrics['csi']:.6f}, "
        f"pooled_hourly_POD={pooled_hourly_default_metrics['pod']:.6f}, "
        f"pooled_hourly_FAR={pooled_hourly_default_metrics['far']:.6f}"
    )

    print("\nPer-lead metrics:")
    for item in per_lead:
        print(
            f"  lead={item['lead_idx']:02d} "
            f"CSI={item['csi']:.4f} "
            f"POD={item['pod']:.4f} "
            f"Recall={item['recall']:.4f} "
            f"FAR={item['far']:.4f}"
        )

    print(f"\nPer-lead pooled metrics ({pooled_pool_size}x{pooled_pool_size} max pooling):")
    for item in pooled_per_lead:
        print(
            f"  lead={item['lead_idx']:02d} "
            f"pooled_CSI={item['csi']:.4f} "
            f"pooled_POD={item['pod']:.4f} "
            f"pooled_Recall={item['recall']:.4f} "
            f"pooled_FAR={item['far']:.4f}"
        )

    reliability = compute_reliability_diagram(mean_probs, truth_array, n_bins=10)
    save_reliability_plot(out_dir, reliability, title="overall")

    for lead_idx in range(mean_probs.shape[1]):
        rel = compute_reliability_diagram(
            mean_probs[:, lead_idx], truth_array[:, lead_idx], n_bins=10
        )
        save_reliability_plot(out_dir, rel, title=f"lead{lead_idx:02d}")

    np.savez_compressed(
        os.path.join(out_dir, "lightning_test_results.npz"),
        truth=truth_array,
        mean_raw_forecast=mean_raw_forecast,
        mean_clipped_prob=mean_probs,
        mean_binary_pred=mean_binary_pred_array,
        hourly_truth=hourly_truth,
        hourly_score=hourly_score,
        best_mean_threshold=best_threshold,
        pooled_best_mean_threshold=pooled_best_threshold,
        hourly_best_mean_threshold=hourly_best_threshold,
        pooled_hourly_best_mean_threshold=pooled_hourly_best_threshold,
        default_threshold=default_threshold,
        threshold_sweep=np.array([item["threshold"] for item in threshold_sweep], dtype=np.float32),
        threshold_sweep_csi=np.array([item["csi"] for item in threshold_sweep], dtype=np.float32),
        pooled_threshold_sweep=np.array(
            [item["threshold"] for item in pooled_threshold_sweep], dtype=np.float32
        ),
        pooled_threshold_sweep_csi=np.array(
            [item["csi"] for item in pooled_threshold_sweep], dtype=np.float32
        ),
        hourly_threshold_sweep=np.array(
            [item["threshold"] for item in hourly_threshold_sweep], dtype=np.float32
        ),
        hourly_threshold_sweep_csi=np.array(
            [item["csi"] for item in hourly_threshold_sweep], dtype=np.float32
        ),
        pooled_hourly_threshold_sweep=np.array(
            [item["threshold"] for item in pooled_hourly_threshold_sweep], dtype=np.float32
        ),
        pooled_hourly_threshold_sweep_csi=np.array(
            [item["csi"] for item in pooled_hourly_threshold_sweep], dtype=np.float32
        ),
        preview_vil_past=pack_optional_array(preview_vil_past),
        preview_vil_future=pack_optional_array(preview_vil_future),
        preview_vil_future_pred=pack_optional_array(preview_vil_future_pred),
        preview_lgt_past=pack_optional_array(preview_lgt_past),
        preview_raw_ensemble=pack_optional_array(preview_raw_ensemble),
        reliability_bin_centers=reliability["bin_centers"],
        reliability_mean_pred=reliability["mean_predicted_prob"],
        reliability_obs_freq=reliability["observed_freq"],
        reliability_counts=reliability["counts"],
    )
    save_markdown_summary(
        out_dir=out_dir,
        checkpoint_path=checkpoint_path,
        test_data_percentage=test_data_percentage,
        probabilistic_samples=probabilistic_samples,
        euler_steps=euler_steps,
        pool_size=pooled_pool_size,
        default_threshold=default_threshold,
        prob_metrics=prob_metrics,
        overall=overall,
        pooled_overall=pooled_best_overall,
        hourly_overall=hourly_best_overall,
        pooled_hourly_overall=pooled_hourly_best_overall,
        per_lead=per_lead,
        pooled_per_lead=pooled_per_lead,
    )
    if (
        preview_vil_past is not None
        and preview_vil_future is not None
        and preview_lgt_past is not None
        and preview_raw_ensemble is not None
    ):
        save_preview_plot(
            out_dir,
            preview_vil_past,
            preview_vil_future,
            preview_vil_future_pred,
            preview_lgt_past,
            preview_raw_ensemble,
            mean_raw_forecast[0],
            mean_binary_pred_array[0],
            truth_array[0],
            mean_threshold=best_threshold,
        )
    print(f"\nSaved outputs to {out_dir}")


def save_preview_plot(
    out_dir: str,
    vil_past: np.ndarray,
    vil_future: np.ndarray,
    vil_future_pred: np.ndarray,
    lgt_past: np.ndarray,
    preview_raw_ensemble: np.ndarray,
    mean_raw_forecast: np.ndarray,
    mean_binary_pred: np.ndarray,
    truth: np.ndarray,
    mean_threshold: float,
):
    os.makedirs(out_dir, exist_ok=True)
    n_past = vil_past.shape[0]
    n_vil_future = vil_future.shape[0]
    has_vil_future_pred = vil_future_pred is not None
    n_vil_future_pred = vil_future_pred.shape[0] if has_vil_future_pred else 0
    n_future = truth.shape[0]
    n_members = preview_raw_ensemble.shape[0]
    n_cols = max(n_past, n_future)
    base_rows = 4 if has_vil_future_pred else 3
    n_rows = base_rows + n_members + 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.0 * n_cols, 2.0 * n_rows))
    axes = np.atleast_2d(axes)

    def _clear_unused(ax):
        ax.axis("off")

    for col in range(n_cols):
        if col < n_past:
            past_label = _format_past_label(col, n_past)
            _plot_vil(axes[0, col], vil_past[col, :, :, 0], f"VIL {past_label}")
            _plot_lightning_continuous(
                axes[1, col],
                lgt_past[col, :, :, 0],
                f"LGT {past_label}",
            )
        else:
            _clear_unused(axes[0, col])
            _clear_unused(axes[1, col])

    for col in range(n_cols):
        if col < n_vil_future:
            _plot_vil(axes[2, col], vil_future[col, :, :, 0], f"VIL {col + 1}")
        else:
            _clear_unused(axes[2, col])

    ensemble_row_offset = 3
    if has_vil_future_pred:
        for col in range(n_cols):
            if col < n_vil_future_pred:
                _plot_vil(axes[3, col], vil_future_pred[col, :, :, 0], f"Pred VIL {col + 1}")
            else:
                _clear_unused(axes[3, col])
        ensemble_row_offset = 4

    for member_idx in range(n_members):
        row = ensemble_row_offset + member_idx
        for col in range(n_cols):
            if col < n_future:
                _plot_lightning_continuous(
                    axes[row, col],
                    preview_raw_ensemble[member_idx, col],
                    f"Ens {member_idx + 1} {col + 1}",
                )
            else:
                _clear_unused(axes[row, col])

    mean_row = ensemble_row_offset + n_members
    binary_row = mean_row + 1
    truth_row = binary_row + 1
    for col in range(n_cols):
        if col < n_future:
            lead_label = str(col + 1)
            _plot_lightning_continuous(
                axes[mean_row, col],
                mean_raw_forecast[col],
                f"Mean {lead_label}",
            )
            _plot_lightning_binary(
                axes[binary_row, col],
                mean_binary_pred[col],
                f"Mean >= {mean_threshold:.2f}",
            )
            _plot_lightning_binary(
                axes[truth_row, col],
                truth[col],
                f"Truth {lead_label}",
            )
        else:
            _clear_unused(axes[mean_row, col])
            _clear_unused(axes[binary_row, col])
            _clear_unused(axes[truth_row, col])

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("Past VIL\n384x384", fontsize=12)
    axes[1, 0].set_ylabel("Past Lightning\n48x48", fontsize=12)
    axes[2, 0].set_ylabel("Future VIL Ref Only\n384x384", fontsize=12)
    if has_vil_future_pred:
        axes[3, 0].set_ylabel("Future VIL Pred\n384x384", fontsize=12)
    for member_idx in range(n_members):
        axes[ensemble_row_offset + member_idx, 0].set_ylabel(
            f"Ens {member_idx + 1}\n48x48", fontsize=12
        )
    axes[mean_row, 0].set_ylabel("Mean Raw\n48x48", fontsize=12)
    axes[binary_row, 0].set_ylabel("Mean Binary\n48x48", fontsize=12)
    axes[truth_row, 0].set_ylabel("Truth\n48x48", fontsize=12)
    fig.suptitle(
        "Lightning panels are native 48x48 rendered with nearest-neighbor; subplot size matches layout only.",
        fontsize=12,
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(os.path.join(out_dir, "lightning_preview_multilead.png"), dpi=140)
    plt.close(fig)


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
        "# Lightning Nowcast Test Results",
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

    for exact_item, pooled_item in zip(per_lead, pooled_per_lead):
        lines.append(
            f"| {exact_item['lead_idx'] + 1} | "
            f"{exact_item['csi']:.6f} | {exact_item['pod']:.6f} | {exact_item['far']:.6f} | "
            f"{pooled_item['csi']:.6f} | {pooled_item['pod']:.6f} | {pooled_item['far']:.6f} |"
        )

    lines.extend(
        [
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
        ]
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


parser = argparse.ArgumentParser(description="Test SEVIR lightning nowcast models.")
parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to the trained checkpoint from dist_train_lightning_diffusion.py",
)
parser.add_argument(
    "--config",
    type=str,
    default="experiments/sevir/runner/flowcast/lightning_diffusion_config.yaml",
)
parser.add_argument("--test_vil_file", type=str, required=True)
parser.add_argument("--test_lightning_file", type=str, required=True)
parser.add_argument("--test_meta", type=str, required=True)
parser.add_argument("--test_data_percentage", type=float, default=1.0)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--ae_chunk_size", type=int, default=None)
parser.add_argument("--num_shards", type=int, default=1)
parser.add_argument("--shard_rank", type=int, default=0)
parser.add_argument("--merge_shards", action="store_true")


def main():
    args = parser.parse_args()
    if not (0.0 < args.test_data_percentage <= 1.0):
        raise ValueError("test_data_percentage must be in (0, 1].")
    if args.num_shards <= 0:
        raise ValueError("num_shards must be >= 1.")
    if not (0 <= args.shard_rank < args.num_shards):
        raise ValueError("shard_rank must satisfy 0 <= shard_rank < num_shards.")

    config = OmegaConf.load(args.config)
    out_dir = os.path.join(os.path.dirname(args.checkpoint), "..", "test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    probabilistic_samples = config.test_params.probabilistic_samples
    euler_steps = config.test_params.euler_steps
    default_threshold = config.test_params.probability_threshold
    threshold_candidates = build_threshold_sweep(0.1, 0.9, 0.1)
    pooled_pool_size = 2

    if args.merge_shards:
        (
            mean_raw_forecast,
            truth_array,
            preview_vil_past,
            preview_vil_future,
            preview_vil_future_pred,
            preview_lgt_past,
            preview_raw_ensemble,
        ) = merge_shard_outputs(out_dir, args.num_shards)
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
            preview_vil_past=preview_vil_past,
            preview_vil_future=preview_vil_future,
            preview_vil_future_pred=preview_vil_future_pred,
            preview_lgt_past=preview_lgt_past,
            preview_raw_ensemble=preview_raw_ensemble,
        )
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = config.test_params.micro_batch_size
    if args.batch_size is not None:
        batch_size = args.batch_size
    num_workers = config.test_params.num_workers
    if args.num_workers is not None:
        num_workers = args.num_workers

    lag_time = config.data_params.lag_time
    lead_time = config.data_params.lead_time
    time_spacing = config.data_params.time_spacing
    raw_seq_len = config.data_params.raw_seq_len
    stride = config.data_params.stride
    c_cap = config.loss_params.c_cap

    ae_config = config.autoencoder_params.autoencoder_config
    ae_checkpoint = config.autoencoder_params.autoencoder_checkpoint
    ae_chunk_size = config.autoencoder_params.ae_chunk_size
    if args.ae_chunk_size is not None:
        ae_chunk_size = args.ae_chunk_size

    vil_dataset = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    target_hw = tuple(config.model_params.target_hw)
    model_family = str(config.model_params.get("model_family", "diffusion")).lower()
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
    future_vil_cfg = config.get("future_vil_condition_params", {})
    use_future_vil_condition = bool(future_vil_cfg.get("enable", False))
    future_vil_flowcast_config = future_vil_cfg.get("flowcast_config", None)
    future_vil_flowcast_checkpoint = future_vil_cfg.get("flowcast_checkpoint", None)
    future_vil_euler_steps = int(future_vil_cfg.get("euler_steps", 10))
    future_vil_probabilistic_samples = int(
        future_vil_cfg.get("probabilistic_samples", 1)
    )
    simvp_use_hourly_target = bool(simvp_cfg.get("use_hourly_target", False))
    if model_family not in {"diffusion", "simvp", "deeplight"}:
        raise ValueError(
            f"Unsupported model_family={model_family!r}. Expected 'diffusion', 'simvp', or 'deeplight'."
        )
    if model_family in {"simvp", "deeplight"}:
        probabilistic_samples = 1
        euler_steps = 0
        use_future_vil_condition = False

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
        n_total = len(dataset)
        n_subset = max(1, int(n_total * args.test_data_percentage))
        selected_indices = list(range(n_subset))
    else:
        selected_indices = list(range(len(dataset)))

    total_selected = len(selected_indices)
    shard_start = 0
    shard_end = total_selected
    if args.num_shards > 1:
        shard_size = (total_selected + args.num_shards - 1) // args.num_shards
        shard_start = args.shard_rank * shard_size
        shard_end = min(total_selected, (args.shard_rank + 1) * shard_size)
        selected_indices = selected_indices[shard_start:shard_end]
        if len(selected_indices) == 0:
            raise ValueError(
                f"Shard {args.shard_rank}/{args.num_shards} received no samples "
                f"(total_selected={total_selected})."
            )
    dataset = Subset(dataset, selected_indices)
    print(
        "Testing shard selection | "
        f"num_shards={args.num_shards} | shard_rank={args.shard_rank} | "
        f"selected_start={shard_start} | selected_end={shard_end} | "
        f"num_samples={len(selected_indices)}"
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dynamic_paired_vil_lightning_collate,
        num_workers=num_workers if not config.run_params.debug_mode else 0,
        pin_memory=torch.cuda.is_available(),
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
            future_vil_flowcast_config_path=future_vil_flowcast_config,
            future_vil_flowcast_checkpoint_path=future_vil_flowcast_checkpoint,
            future_vil_euler_steps=future_vil_euler_steps,
            future_vil_probabilistic_samples=future_vil_probabilistic_samples,
            denoiser_backbone=denoiser_backbone,
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
        missing_keys, unexpected_keys = system.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        allowed_missing_prefixes = ("future_vil_forecaster.",)
        disallowed_missing = [
            key for key in missing_keys if not key.startswith(allowed_missing_prefixes)
        ]
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch while loading lightning diffusion weights. "
                f"Missing keys: {disallowed_missing}. Unexpected keys: {unexpected_keys}."
            )
        if missing_keys:
            print(
                "Loaded checkpoint with frozen online future-VIL forecaster initialized "
                f"from its separate pretrained checkpoint. Missing checkpoint keys: {len(missing_keys)}"
            )
    else:
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
        missing_keys, unexpected_keys = system.load_state_dict(
            checkpoint["model_state_dict"], strict=True
        )
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "Checkpoint/model mismatch while loading SimVP lightning weights. "
                f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
            )
    system.eval()

    base_dataset = unwrap_base_dataset(dataset)
    all_mean_raw_forecast = []
    all_truth = []
    preview_vil_past = None
    preview_vil_future = None
    preview_vil_future_pred = None
    preview_lgt_past = None
    preview_raw_ensemble = None

    with torch.no_grad():
        test_bar = tqdm(loader, desc="Testing", leave=False)
        for batch_idx, (x_vil_past, lgt_past, lgt_future, meta_batch) in enumerate(test_bar):
            if preview_vil_past is None:
                preview_vil_past = x_vil_past[0].cpu().numpy().astype(np.float32)
                preview_lgt_past = lgt_past[0].cpu().numpy().astype(np.float32)
                preview_meta = meta_batch[0]
                vil_row = int(preview_meta["_vil_row_index"])
                y_indices_abs = [int(idx) for idx in preview_meta["_y_indices_abs"]]
                # Reference-only visualization: this future VIL ground truth is never
                # passed into the lightning model or used for conditioning.
                base_dataset._open_hdf5()
                vil_event = base_dataset.vil_h5[vil_dataset][vil_row].astype(np.float32)
                if base_dataset.normalize_vil:
                    vil_event /= 255.0
                vil_future = vil_event[..., y_indices_abs]
                preview_vil_future = np.moveaxis(vil_future, -1, 0)[..., None].astype(np.float32)
            x_vil_past = x_vil_past.to(device)
            lgt_past = lgt_past.to(device)
            lgt_future = lgt_future.to(device)

            batch_size_local = x_vil_past.shape[0]
            t_future = lgt_future.shape[1]
            hz = lgt_future.shape[2]
            wz = lgt_future.shape[3]
            channels = lgt_future.shape[4]

            if model_family == "diffusion":
                z_vil_past, z_vil_future_pred = system.prepare_vil_conditions(
                    x_vil_past,
                    ae_chunk_size=ae_chunk_size,
                )
                if preview_vil_future_pred is None and z_vil_future_pred is not None:
                    preview_vil_future_pred = decode_vil_latents_for_preview(
                        system,
                        z_vil_future_pred[:1],
                        ae_chunk_size=ae_chunk_size,
                    )

                raw_members = []
                for sample_idx in range(probabilistic_samples):
                    torch.manual_seed(batch_idx * probabilistic_samples + sample_idx)
                    x0_noise = torch.randn_like(lgt_future, device=device)
                    x0_flat = x0_noise.view(batch_size_local * t_future, hz, wz, channels)

                    def flow_dynamics(t_scalar, x_flat):
                        x_flow_local = x_flat.view(batch_size_local, t_future, hz, wz, channels)
                        t_batched = t_scalar * torch.ones(batch_size_local, device=x_flow_local.device)
                        flow = system.forward_from_encoded(
                            t_batched,
                            x_flow_local,
                            z_vil_past,
                            lgt_past,
                            z_vil_future_pred=z_vil_future_pred,
                        )
                        return flow.contiguous().view(batch_size_local * t_future, hz, wz, channels)

                    t_span = torch.tensor([0.0, 1.0], device=device)
                    if euler_steps <= 0:
                        solution = odeint(
                            flow_dynamics,
                            x0_flat,
                            t_span,
                            method="adaptive_heun",
                            rtol=1e-2,
                            atol=1e-3,
                            adjoint_params=system.parameters(),
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
                            adjoint_params=system.parameters(),
                        )

                    x_final = solution[-1].view(batch_size_local, t_future, hz, wz, channels)
                    raw_members.append(x_final.unsqueeze(1))

                raw_ensemble = torch.cat(raw_members, dim=1)  # (B, S, T, H, W, 1)
            else:
                logits = system(x_vil_past, lgt_past)
                if simvp_use_hourly_target:
                    # logits: (B, 1, 1, H, W) → probs: (B, 1, H, W, 1)
                    probs = torch.sigmoid(logits.squeeze(1)).permute(0, 2, 3, 1).unsqueeze(1).contiguous()
                    # truth: aggregate 12 future frames → (B, 1, H, W, 1)
                    truth_1h = prepare_deeplight_target(lgt_future).permute(0, 2, 3, 1).unsqueeze(1).contiguous()
                    raw_ensemble = probs.unsqueeze(1)  # (B, 1, 1, H, W, 1)
                    all_truth.append(truth_1h.cpu().numpy().astype(np.float32))
                else:
                    expected_logits_shape = (
                        batch_size_local,
                        t_future,
                        1,
                        hz,
                        wz,
                    )
                    if tuple(logits.shape) != expected_logits_shape:
                        raise ValueError(
                            "Unexpected SimVP test logits shape: "
                            f"{tuple(logits.shape)} vs expected {expected_logits_shape}"
                        )
                    probs = torch.sigmoid(logits).permute(0, 1, 3, 4, 2).contiguous()
                    raw_ensemble = probs.unsqueeze(1)  # deterministic single-member forecast
                    all_truth.append(lgt_future.cpu().numpy().astype(np.float32))
            if preview_raw_ensemble is None:
                preview_raw_ensemble = raw_ensemble[0].cpu().numpy().astype(np.float32)
            mean_raw_batch = raw_ensemble.mean(dim=1).cpu().numpy().astype(np.float32)
            all_mean_raw_forecast.append(mean_raw_batch)
            if model_family == "diffusion":
                all_truth.append(lgt_future.cpu().numpy().astype(np.float32))

    mean_raw_forecast = np.concatenate(all_mean_raw_forecast, axis=0)
    truth_array = np.concatenate(all_truth, axis=0)

    if args.num_shards > 1:
        shard_path = save_shard_outputs(
            out_dir=out_dir,
            shard_rank=args.shard_rank,
            num_shards=args.num_shards,
            selected_start=shard_start,
            selected_end=shard_end,
            total_selected=total_selected,
            mean_raw_forecast=mean_raw_forecast,
            truth_array=truth_array,
            preview_vil_past=preview_vil_past,
            preview_vil_future=preview_vil_future,
            preview_vil_future_pred=preview_vil_future_pred,
            preview_lgt_past=preview_lgt_past,
            preview_raw_ensemble=preview_raw_ensemble,
        )
        print(f"\nSaved shard outputs to {shard_path}")
        return

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
        preview_vil_past=preview_vil_past,
        preview_vil_future=preview_vil_future,
        preview_vil_future_pred=preview_vil_future_pred,
        preview_lgt_past=preview_lgt_past,
        preview_raw_ensemble=preview_raw_ensemble,
    )


if __name__ == "__main__":
    main()
