"""
Evaluate a simple lightning persistence baseline on the SEVIR test split.

This script intentionally follows the same evaluation style as
`test_lightning_diffusion.py`:

- use the same paired SEVIR lightning test set
- report frame-wise metrics over all 12 leads
- report hourly aggregated metrics
- report exact and 2x2 pooled metrics
- save a markdown summary and npz bundle

Default persistence definition:

- take the last observed lightning frame from the past window
- convert it to binary occupancy
- repeat that same map across all 12 future leads
"""

import argparse
import os
import re
import sys
from typing import Dict, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from experiments.sevir.dataset.sevirlightningdataset import (
    get_row_index,
    lightning_counts_to_binary,
    lightning_counts_to_condition,
    resolve_event_and_seq_index,
)
from experiments.sevir.runner.flowcast.test_lightning_diffusion import (
    aggregate_hourly_binary_target,
    aggregate_hourly_forecast_score,
    build_threshold_sweep,
    compute_binary_metrics,
    compute_per_lead_metrics,
    compute_probabilistic_metrics,
    find_best_csi_threshold,
    pack_optional_array,
    save_preview_plot,
)


def _format_setup_value(value):
    if isinstance(value, float):
        return f"{value}"
    return str(value)


def parse_flowcast_summary(summary_path: str) -> Dict[str, Dict[str, float]]:
    """Parse the fixed-format FlowCast markdown summary for comparison."""
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Could not find FlowCast summary: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    results: Dict[str, Dict[str, float]] = {}
    metric_map = {
        "frame_exact": "| Exact |",
        "frame_pooled": "| Pooled 2x2 |",
        "hourly_exact": "| Exact hourly |",
        "hourly_pooled": "| Pooled hourly 2x2 |",
    }
    for key, prefix in metric_map.items():
        row = next((line for line in lines if line.startswith(prefix)), None)
        if row is None:
            raise ValueError(f"Missing row {prefix!r} in FlowCast summary {summary_path}")
        parts = [part.strip() for part in row.split("|")[1:-1]]
        # metric view, threshold, CSI, POD, FAR, F1, Precision
        results[key] = {
            "threshold": float(parts[1]),
            "csi": float(parts[2]),
            "pod": float(parts[3]),
            "far": float(parts[4]),
            "f1": float(parts[5]),
            "precision": float(parts[6]),
        }

    for line in lines:
        if line.startswith("- Brier score:"):
            results.setdefault("prob", {})["brier"] = float(
                re.findall(r"`([0-9.]+)`", line)[0]
            )
        if line.startswith("- Brier skill score:"):
            results.setdefault("prob", {})["bss"] = float(
                re.findall(r"`([0-9.]+)`", line)[0]
            )
    if "prob" not in results or "brier" not in results["prob"] or "bss" not in results["prob"]:
        raise ValueError(f"Could not parse probabilistic metrics from {summary_path}")
    return results


def build_persistence_forecast(
    lgt_past_count: np.ndarray,
    lead_time: int,
    c_cap: float,
    mode: str,
) -> np.ndarray:
    if mode == "last_binary":
        base = lightning_counts_to_binary(lgt_past_count[..., [-1]])
    elif mode == "last_condition":
        base = lightning_counts_to_condition(lgt_past_count[..., [-1]], c_cap)
    elif mode == "past_any_binary":
        base = lightning_counts_to_binary(lgt_past_count.max(axis=-1, keepdims=True))
    else:
        raise ValueError(f"Unsupported persistence mode: {mode}")
    return np.repeat(base.astype(np.float32), lead_time, axis=-1)


class LightningPersistenceDataset(Dataset):
    """Test dataset that loads only the fields needed for persistence evaluation."""

    def __init__(
        self,
        meta_csv: str,
        lightning_file: str,
        vil_file: Optional[str],
        lightning_dataset: str = "lght",
        vil_dataset: str = "vil",
        lightning_row_key: Optional[str] = None,
        vil_row_key: Optional[str] = None,
        raw_seq_len: int = 49,
        lag_time: int = 13,
        lead_time: int = 12,
        time_spacing: int = 1,
        stride: int = 12,
        c_cap: float = 20.0,
        persistence_mode: str = "last_binary",
        normalize_vil: bool = True,
        debug_mode: bool = False,
    ):
        self.meta_csv = meta_csv
        self.lightning_file = lightning_file
        self.vil_file = vil_file
        self.lightning_dataset = lightning_dataset
        self.vil_dataset = vil_dataset
        self.lightning_row_key = lightning_row_key
        self.vil_row_key = vil_row_key
        self.raw_seq_len = int(raw_seq_len)
        self.lag_time = int(lag_time)
        self.lead_time = int(lead_time)
        self.time_spacing = int(time_spacing)
        self.seq_len = (self.lag_time + self.lead_time) * self.time_spacing
        self.stride = int(stride)
        self.c_cap = float(c_cap)
        self.persistence_mode = persistence_mode
        self.normalize_vil = bool(normalize_vil)
        self.debug_mode = bool(debug_mode)

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in metadata file: {self.meta_csv}")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        self.lightning_h5 = h5py.File(self.lightning_file, "r")
        self.vil_h5 = h5py.File(self.vil_file, "r") if self.vil_file else None

        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        if self.debug_mode:
            self.n_seq_per_event = 1

        self.event_seq_counts = np.full(
            len(self.metadata), self.n_seq_per_event, dtype=np.int32
        )
        self.cum_counts = np.cumsum(self.event_seq_counts)

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, index):
        event_idx, seq_idx = resolve_event_and_seq_index(self.cum_counts, index)
        row = self.metadata.iloc[event_idx]
        lightning_row = get_row_index(row, self.lightning_row_key)
        vil_row = get_row_index(row, self.vil_row_key) if self.vil_h5 is not None else -1

        lightning_event = self.lightning_h5[self.lightning_dataset][lightning_row].astype(
            np.float32
        )
        start = seq_idx * self.stride
        end = start + self.seq_len
        lightning_segment = lightning_event[..., start:end]

        x_indices = [i * self.time_spacing for i in range(self.lag_time)]
        y_end = self.seq_len - 1
        y_indices = [y_end - i * self.time_spacing for i in range(self.lead_time)]
        y_indices.reverse()

        lgt_past_count = lightning_segment[..., x_indices]
        lgt_future_count = lightning_segment[..., y_indices]

        persistence_score = build_persistence_forecast(
            lgt_past_count=lgt_past_count,
            lead_time=self.lead_time,
            c_cap=self.c_cap,
            mode=self.persistence_mode,
        )
        lgt_past_condition = lightning_counts_to_condition(lgt_past_count, self.c_cap)
        lgt_future = lightning_counts_to_binary(lgt_future_count)

        persistence_score = torch.from_numpy(persistence_score).float().permute(2, 0, 1).unsqueeze(0)
        lgt_past_condition = torch.from_numpy(lgt_past_condition).float().permute(2, 0, 1).unsqueeze(0)
        lgt_future = torch.from_numpy(lgt_future).float().permute(2, 0, 1).unsqueeze(0)

        persistence_score = persistence_score.permute(1, 2, 3, 0)
        lgt_past_condition = lgt_past_condition.permute(1, 2, 3, 0)
        lgt_future = lgt_future.permute(1, 2, 3, 0)

        meta = row.to_dict()
        meta["_lightning_row_index"] = int(lightning_row)
        meta["_vil_row_index"] = int(vil_row)
        meta["_sequence_start_index"] = int(start)
        meta["_sequence_end_index"] = int(end)
        meta["_x_indices_abs"] = [int(start + idx) for idx in x_indices]
        meta["_y_indices_abs"] = [int(start + idx) for idx in y_indices]
        return persistence_score, lgt_past_condition, lgt_future, meta


def persistence_collate(batch):
    persistence_score, lgt_past, lgt_future, meta = zip(*batch)
    return (
        torch.stack(persistence_score, dim=0),
        torch.stack(lgt_past, dim=0),
        torch.stack(lgt_future, dim=0),
        list(meta),
    )


def save_persistence_summary(
    out_dir: str,
    baseline_name: str,
    persistence_mode: str,
    test_data_percentage: float,
    pool_size: int,
    default_threshold: float,
    prob_metrics: Dict[str, float],
    overall: Dict[str, float],
    pooled_overall: Dict[str, float],
    hourly_overall: Dict[str, float],
    pooled_hourly_overall: Dict[str, float],
    per_lead: Sequence[Dict[str, float]],
    pooled_per_lead: Sequence[Dict[str, float]],
    flowcast_summary: Optional[Dict[str, Dict[str, float]]] = None,
):
    summary_path = os.path.join(out_dir, "TEST_RESULTS_persistence_baseline.md")
    lines = [
        "# Lightning Persistence Baseline Test Results",
        "",
        "## Setup",
        "",
        f"- Baseline name: `{baseline_name}`",
        f"- Persistence mode: `{persistence_mode}`",
        f"- Test fraction: `{test_data_percentage}`",
        f"- Default threshold: `{default_threshold}`",
        f"- Spatial pooling size for pooled metrics: `{pool_size}x{pool_size}`",
        "- Frame-wise case: evaluate all future leads 1..12 separately, then pool TP/FP/FN across all samples and all leads.",
        "- Hourly aggregated case: target is `any lightning over leads 1..12`; persistence forecast score is repeated from the chosen past-lightning summary.",
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

    if flowcast_summary is not None:
        lines.extend(
            [
                "## Comparison With Current Lightning FlowCast",
                "",
                "| Model | Frame-wise CSI | Frame-wise pooled CSI | Hourly CSI | Hourly pooled CSI | Brier |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| Persistence ({persistence_mode}) | {overall['csi']:.6f} | "
                    f"{pooled_overall['csi']:.6f} | {hourly_overall['csi']:.6f} | "
                    f"{pooled_hourly_overall['csi']:.6f} | {prob_metrics['brier']:.6f} |"
                ),
                (
                    f"| Lightning FlowCast | {flowcast_summary['frame_exact']['csi']:.6f} | "
                    f"{flowcast_summary['frame_pooled']['csi']:.6f} | "
                    f"{flowcast_summary['hourly_exact']['csi']:.6f} | "
                    f"{flowcast_summary['hourly_pooled']['csi']:.6f} | "
                    f"{flowcast_summary['prob']['brier']:.6f} |"
                ),
                "",
                "### Delta: Persistence minus Lightning FlowCast",
                "",
                "| Metric | Delta |",
                "| --- | ---: |",
                f"| Frame-wise CSI | {overall['csi'] - flowcast_summary['frame_exact']['csi']:+.6f} |",
                f"| Frame-wise pooled CSI | {pooled_overall['csi'] - flowcast_summary['frame_pooled']['csi']:+.6f} |",
                f"| Hourly CSI | {hourly_overall['csi'] - flowcast_summary['hourly_exact']['csi']:+.6f} |",
                f"| Hourly pooled CSI | {pooled_hourly_overall['csi'] - flowcast_summary['hourly_pooled']['csi']:+.6f} |",
                f"| Brier score | {prob_metrics['brier'] - flowcast_summary['prob']['brier']:+.6f} |",
                "",
            ]
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a lightning persistence baseline.")
    parser.add_argument(
        "--test_vil_file",
        type=str,
        required=True,
        help="Optional VIL file is used only for preview/reference plotting.",
    )
    parser.add_argument("--test_lightning_file", type=str, required=True)
    parser.add_argument("--test_meta", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/sevir/runner/flowcast/lightning_diffusion_config.yaml",
    )
    parser.add_argument("--test_data_percentage", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--persistence_mode",
        type=str,
        default="last_binary",
        choices=("last_binary", "last_condition", "past_any_binary"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/sevir/persistence_baseline/test_outputs_last_binary",
    )
    parser.add_argument(
        "--flowcast_summary_md",
        type=str,
        default=(
            "artifacts/sevir/lightning_diffusion/"
            "2026-04-09_10-40-19_lightning_diffusion_first_model_g4567_mb24_ae96/"
            "test_outputs/TEST_RESULTS_best_checkpoint.md"
        ),
    )
    args = parser.parse_args()

    if not (0.0 < args.test_data_percentage <= 1.0):
        raise ValueError("test_data_percentage must be in (0, 1].")

    from omegaconf import OmegaConf

    config = OmegaConf.load(args.config)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    lag_time = config.data_params.lag_time
    lead_time = config.data_params.lead_time
    time_spacing = config.data_params.time_spacing
    raw_seq_len = config.data_params.raw_seq_len
    stride = config.data_params.stride
    c_cap = config.loss_params.c_cap
    vil_dataset = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    default_threshold = config.test_params.probability_threshold
    threshold_candidates = build_threshold_sweep(0.1, 0.9, 0.1)
    pooled_pool_size = 2

    dataset = LightningPersistenceDataset(
        meta_csv=args.test_meta,
        lightning_file=args.test_lightning_file,
        vil_file=args.test_vil_file,
        lightning_dataset=lightning_dataset,
        vil_dataset=vil_dataset,
        lightning_row_key=lightning_row_key,
        vil_row_key=vil_row_key,
        raw_seq_len=raw_seq_len,
        lag_time=lag_time,
        lead_time=lead_time,
        time_spacing=time_spacing,
        stride=stride,
        c_cap=c_cap,
        persistence_mode=args.persistence_mode,
        debug_mode=bool(config.run_params.debug_mode),
    )

    if args.test_data_percentage < 1.0:
        n_total = len(dataset)
        n_subset = max(1, int(n_total * args.test_data_percentage))
        dataset = Subset(dataset, list(range(n_subset)))

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=persistence_collate,
        num_workers=args.num_workers if not config.run_params.debug_mode else 0,
        pin_memory=False,
    )

    all_scores = []
    all_truth = []
    preview_vil_past = None
    preview_vil_future = None
    preview_lgt_past = None
    preview_raw_ensemble = None

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    for batch_idx, (persistence_score, lgt_past, lgt_future, meta_batch) in enumerate(loader):
        if preview_vil_past is None:
            preview_lgt_past = lgt_past[0].cpu().numpy().astype(np.float32)
            preview_raw_ensemble = persistence_score[0].cpu().numpy().astype(np.float32)[
                None, ...
            ]
            if base_dataset.vil_h5 is not None:
                preview_meta = meta_batch[0]
                vil_row = int(preview_meta["_vil_row_index"])
                x_indices_abs = [int(idx) for idx in preview_meta["_x_indices_abs"]]
                y_indices_abs = [int(idx) for idx in preview_meta["_y_indices_abs"]]
                vil_event = base_dataset.vil_h5[base_dataset.vil_dataset][vil_row].astype(
                    np.float32
                )
                if base_dataset.normalize_vil:
                    vil_event /= 255.0
                vil_past = vil_event[..., x_indices_abs]
                vil_future = vil_event[..., y_indices_abs]
                preview_vil_past = np.moveaxis(vil_past, -1, 0)[..., None].astype(np.float32)
                preview_vil_future = np.moveaxis(vil_future, -1, 0)[..., None].astype(
                    np.float32
                )

        all_scores.append(persistence_score.numpy().astype(np.float32))
        all_truth.append(lgt_future.numpy().astype(np.float32))

    persistence_forecast = np.concatenate(all_scores, axis=0)
    truth_array = np.concatenate(all_truth, axis=0)

    if persistence_forecast.shape[-1] == 1:
        persistence_forecast = persistence_forecast[..., 0]
    if truth_array.shape[-1] == 1:
        truth_array = truth_array[..., 0]
    if preview_raw_ensemble is not None and preview_raw_ensemble.shape[-1] == 1:
        preview_raw_ensemble = preview_raw_ensemble[..., 0]

    persistence_probs = np.clip(persistence_forecast, 0.0, 1.0)
    hourly_truth = aggregate_hourly_binary_target(truth_array)
    hourly_score = aggregate_hourly_forecast_score(persistence_forecast)

    best_overall, threshold_sweep = find_best_csi_threshold(
        persistence_forecast, truth_array, threshold_candidates
    )
    pooled_best_overall, pooled_threshold_sweep = find_best_csi_threshold(
        persistence_forecast, truth_array, threshold_candidates, pool_size=pooled_pool_size
    )
    hourly_best_overall, hourly_threshold_sweep = find_best_csi_threshold(
        hourly_score, hourly_truth, threshold_candidates
    )
    pooled_hourly_best_overall, pooled_hourly_threshold_sweep = find_best_csi_threshold(
        hourly_score, hourly_truth, threshold_candidates, pool_size=pooled_pool_size
    )

    per_lead = compute_per_lead_metrics(
        persistence_forecast,
        truth_array,
        threshold=best_overall["threshold"],
    )
    pooled_per_lead = compute_per_lead_metrics(
        persistence_forecast,
        truth_array,
        threshold=pooled_best_overall["threshold"],
        pool_size=pooled_pool_size,
    )
    prob_metrics = compute_probabilistic_metrics(persistence_probs, truth_array)
    mean_binary_pred = (persistence_forecast >= best_overall["threshold"]).astype(np.float32)

    np.savez_compressed(
        os.path.join(out_dir, "lightning_persistence_results.npz"),
        truth=truth_array,
        persistence_forecast=persistence_forecast,
        persistence_prob=persistence_probs,
        persistence_binary_pred=mean_binary_pred,
        hourly_truth=hourly_truth,
        hourly_score=hourly_score,
        best_mean_threshold=best_overall["threshold"],
        pooled_best_mean_threshold=pooled_best_overall["threshold"],
        hourly_best_mean_threshold=hourly_best_overall["threshold"],
        pooled_hourly_best_mean_threshold=pooled_hourly_best_overall["threshold"],
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
        preview_lgt_past=pack_optional_array(preview_lgt_past),
        preview_raw_ensemble=pack_optional_array(preview_raw_ensemble),
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
            None,
            preview_lgt_past,
            preview_raw_ensemble,
            persistence_forecast[0],
            mean_binary_pred[0],
            truth_array[0],
            mean_threshold=best_overall["threshold"],
        )

    flowcast_summary = None
    if args.flowcast_summary_md:
        flowcast_summary = parse_flowcast_summary(args.flowcast_summary_md)

    save_persistence_summary(
        out_dir=out_dir,
        baseline_name="persistence",
        persistence_mode=args.persistence_mode,
        test_data_percentage=args.test_data_percentage,
        pool_size=pooled_pool_size,
        default_threshold=default_threshold,
        prob_metrics=prob_metrics,
        overall=best_overall,
        pooled_overall=pooled_best_overall,
        hourly_overall=hourly_best_overall,
        pooled_hourly_overall=pooled_hourly_best_overall,
        per_lead=per_lead,
        pooled_per_lead=pooled_per_lead,
        flowcast_summary=flowcast_summary,
    )

    print(f"Saved persistence baseline outputs to {out_dir}")
    print(
        "Main persistence metrics | "
        f"frame_CSI={best_overall['csi']:.6f} | "
        f"frame_pooled_CSI={pooled_best_overall['csi']:.6f} | "
        f"hourly_CSI={hourly_best_overall['csi']:.6f} | "
        f"hourly_pooled_CSI={pooled_hourly_best_overall['csi']:.6f}"
    )


if __name__ == "__main__":
    main()
