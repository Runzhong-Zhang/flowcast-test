"""
Preprocess paired SEVIR VIL + lightning data for conditional lightning diffusion.

This script builds aligned train/validation/test splits where each row contains:

- one VIL event cube stored under dataset key ``vil``
- one native lightning count cube stored under dataset key ``lght``
- one shared metadata row with a common ``file_row`` index

Pairing rule
------------
The most reliable pairing uses the shared SEVIR event ``id``. Some VIL random-event
ids appear multiple times with slightly different bounding boxes. For those cases,
we keep the VIL row whose bounding box best matches the lightning catalog row for
the same ``id``.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, Tuple

sys.path.append(os.getcwd())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd

LIGHTNING_FRAME_TIMES = np.arange(-120.0, 125.0, 5.0) * 60.0
BBOX_COLUMNS = ["llcrnrlat", "llcrnrlon", "urcrnrlat", "urcrnrlon"]
PREPROCESSING_COMPLETE_FILENAME = "preprocessing_complete.json"
META_COLUMNS = [
    "id",
    "time_utc",
    "episode_id",
    "event_id",
    "event_type",
    "minute_offsets",
    "llcrnrlat",
    "llcrnrlon",
    "urcrnrlat",
    "urcrnrlon",
    "proj",
    "size_x",
    "size_y",
    "height_m",
    "width_m",
    "data_min",
    "data_max",
    "pct_missing",
    "file_name",
    "file_index",
    "lght_file_name",
    "lght_file_index",
    "lght_time_utc",
    "bbox_diff",
]


def apply_bicubic_interpolation(data: np.ndarray, downsample_factor: int) -> np.ndarray:
    """
    Downsample (H, W, T) data with the same bicubic helper used elsewhere in the repo.

    The torch dependency is loaded lazily so the default paired preprocessing path
    can run without torch when ``downsample_factor == 1``.
    """
    try:
        import torch

        import common.utils.bicubic_interpolation as bicubic_interpolation
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "downsample_factor > 1 requires torch and common.utils.bicubic_interpolation"
        ) from exc

    data_t = torch.from_numpy(data).permute(2, 0, 1)
    if torch.cuda.is_available():
        data_t = data_t.cuda()

    data_downsampled = bicubic_interpolation.imresize(
        data_t, scale=(1 / downsample_factor)
    )

    if torch.cuda.is_available():
        data_downsampled = data_downsampled.cpu()

    return data_downsampled.permute(1, 2, 0).numpy()


def flashes_to_grid(
    flash_data: np.ndarray,
    frame_times: np.ndarray = LIGHTNING_FRAME_TIMES,
    grid_size: int = 48,
) -> np.ndarray:
    """
    Convert SEVIR raw lightning rows with shape (N, 5) into a native count cube.

    This mirrors the official SEVIR binning approach:
    - col0 = time offset in seconds
    - col3 = x pixel in native 48x48 lightning space
    - col4 = y pixel in native 48x48 lightning space
    """
    n_frames = len(frame_times)
    grid = np.zeros((grid_size, grid_size, n_frames), dtype=np.int16)
    if flash_data.shape[0] == 0:
        return grid

    x = flash_data[:, 3].astype(np.int64)
    y = flash_data[:, 4].astype(np.int64)
    valid = (x >= 0) & (x < grid_size) & (y >= 0) & (y < grid_size)
    if not np.any(valid):
        return grid

    data = flash_data[valid]
    x = data[:, 3].astype(np.int64)
    y = data[:, 4].astype(np.int64)
    t = data[:, 0]

    z = np.digitize(t, frame_times) - 1
    z[z == -1] = 0
    z = np.clip(z, 0, n_frames - 1)

    k = np.ravel_multi_index(np.array([y, x, z]), (grid_size, grid_size, n_frames))
    counts = np.bincount(k, minlength=grid_size * grid_size * n_frames)
    return counts.reshape(grid_size, grid_size, n_frames).astype(np.int16)


def build_matched_catalog(
    catalog_csv_path: str,
    max_pct_missing: float = 0.05,
    bbox_tol: float = 1e-3,
) -> pd.DataFrame:
    """
    Build a paired VIL/lightning catalog.

    The returned dataframe contains one matched row per usable event. It keeps the
    original VIL metadata columns and appends the lightning source file info.
    """
    catalog = pd.read_csv(catalog_csv_path, low_memory=False, parse_dates=["time_utc"])

    vil = catalog[catalog["img_type"] == "vil"].copy().reset_index(drop=True)
    vil = vil[vil["pct_missing"] <= max_pct_missing].copy().reset_index(drop=True)
    vil["vil_catalog_row"] = np.arange(len(vil), dtype=np.int64)

    lght_cols = ["id", "file_name", "file_index", "time_utc", *BBOX_COLUMNS]
    lght = catalog[catalog["img_type"] == "lght"][lght_cols].copy().reset_index(drop=True)

    merged = vil.merge(lght, on="id", how="inner", suffixes=("_vil", "_lght"))
    if merged.empty:
        raise ValueError("No paired VIL/lightning events were found in the catalog.")

    for col in BBOX_COLUMNS:
        merged[f"{col}_absdiff"] = (
            merged[f"{col}_vil"].astype(np.float64)
            - merged[f"{col}_lght"].astype(np.float64)
        ).abs()
    merged["bbox_diff"] = merged[[f"{c}_absdiff" for c in BBOX_COLUMNS]].sum(axis=1)

    # One lightning event id maps to one lightning crop. If a VIL id is duplicated,
    # keep the VIL crop whose bbox best matches the lightning bbox.
    merged = (
        merged.sort_values(["id", "bbox_diff", "file_index_vil", "vil_catalog_row"])
        .groupby("id", as_index=False)
        .first()
    )
    matched = merged[merged["bbox_diff"] <= bbox_tol].copy()
    matched = matched.sort_values("vil_catalog_row").reset_index(drop=True)

    if matched.empty:
        raise ValueError(
            "No paired VIL/lightning events remained after bbox disambiguation. "
            f"Try increasing --bbox_tol (current {bbox_tol})."
        )

    matched = matched.rename(
        columns={
            "time_utc_vil": "time_utc",
            "llcrnrlat_vil": "llcrnrlat",
            "llcrnrlon_vil": "llcrnrlon",
            "urcrnrlat_vil": "urcrnrlat",
            "urcrnrlon_vil": "urcrnrlon",
            "file_name_vil": "file_name",
            "file_index_vil": "file_index",
            "file_name_lght": "lght_file_name",
            "file_index_lght": "lght_file_index",
            "time_utc_lght": "lght_time_utc",
        }
    )

    keep_cols = META_COLUMNS + ["vil_catalog_row"]
    return matched[keep_cols].copy()


def split_catalog_by_time(
    matched_catalog: pd.DataFrame,
    validation_cutoff_date,
    testing_cutoff_date,
) -> Dict[str, pd.DataFrame]:
    """Split a matched catalog into train/val/test by VIL event time."""
    if isinstance(validation_cutoff_date, str):
        validation_cutoff_date = pd.to_datetime(validation_cutoff_date)
    if isinstance(testing_cutoff_date, str):
        testing_cutoff_date = pd.to_datetime(testing_cutoff_date)

    return {
        "training": matched_catalog[
            matched_catalog["time_utc"] < validation_cutoff_date
        ].copy(),
        "validation": matched_catalog[
            (matched_catalog["time_utc"] >= validation_cutoff_date)
            & (matched_catalog["time_utc"] < testing_cutoff_date)
        ].copy(),
        "testing": matched_catalog[
            matched_catalog["time_utc"] >= testing_cutoff_date
        ].copy(),
    }


def append_event(ds: h5py.Dataset, event: np.ndarray, current_count: int) -> int:
    """Append one event along the first axis of a resizable HDF5 dataset."""
    new_count = current_count + 1
    ds.resize((new_count,) + ds.shape[1:])
    ds[current_count] = event
    return new_count


def get_h5_handle(cache: Dict[str, h5py.File], path: str) -> h5py.File:
    """Open and cache an HDF5 handle."""
    if path not in cache:
        cache[path] = h5py.File(path, "r")
    return cache[path]


def cleanup_file_if_exists(path: str):
    """Best-effort cleanup for temporary preprocessing artifacts."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def process_split(
    split_name: str,
    split_df: pd.DataFrame,
    data_dir: str,
    output_dir: str,
    keep_dtype: bool,
    downsample_factor: int,
    max_events_per_split: int = None,
    progress_every: int = 500,
) -> Tuple[str, str, str, int]:
    """Write one aligned VIL H5, one lightning H5, and one shared META CSV."""
    split_df = split_df.sort_values("vil_catalog_row").reset_index(drop=True)
    if max_events_per_split is not None:
        split_df = split_df.iloc[:max_events_per_split].copy().reset_index(drop=True)

    vil_h5_path = os.path.join(output_dir, f"nowcast_{split_name}_vil.h5")
    lght_h5_path = os.path.join(output_dir, f"nowcast_{split_name}_lght.h5")
    meta_csv_path = os.path.join(output_dir, f"nowcast_{split_name}_META.csv")
    tmp_vil_h5_path = vil_h5_path + ".tmp"
    tmp_lght_h5_path = lght_h5_path + ".tmp"
    tmp_meta_csv_path = meta_csv_path + ".tmp"

    vil_src_handles: Dict[str, h5py.File] = {}
    lght_src_handles: Dict[str, h5py.File] = {}
    meta_records = []
    vil_h5 = None
    lght_h5 = None
    vil_ds = None
    lght_ds = None
    kept_count = 0
    split_committed = False

    cleanup_file_if_exists(tmp_vil_h5_path)
    cleanup_file_if_exists(tmp_lght_h5_path)
    cleanup_file_if_exists(tmp_meta_csv_path)

    try:
        for idx, row in split_df.iterrows():
            vil_src_path = os.path.join(data_dir, row["file_name"])
            lght_src_path = os.path.join(data_dir, row["lght_file_name"])

            try:
                vil_src_h5 = get_h5_handle(vil_src_handles, vil_src_path)
                vil_event = vil_src_h5["vil"][int(row["file_index"])]
                if not keep_dtype:
                    vil_event = vil_event.astype(np.float32)
                else:
                    vil_event = np.array(vil_event)

                if downsample_factor > 1:
                    vil_event = apply_bicubic_interpolation(vil_event, downsample_factor)

                lght_src_h5 = get_h5_handle(lght_src_handles, lght_src_path)
                if row["id"] not in lght_src_h5:
                    raise KeyError(
                        f"Lightning key {row['id']} not found in {lght_src_path}"
                    )
                flash_data = lght_src_h5[row["id"]][:]
                lght_event = flashes_to_grid(flash_data)
            except Exception as exc:
                print(f"[{split_name}] skip row {idx} id={row['id']}: {exc}")
                continue

            if vil_ds is None:
                vil_h5 = h5py.File(tmp_vil_h5_path, "w")
                h, w, t = vil_event.shape
                vil_ds = vil_h5.create_dataset(
                    "vil",
                    shape=(0, h, w, t),
                    maxshape=(None, h, w, t),
                    dtype=vil_event.dtype,
                    chunks=(1, h, w, t),
                    compression="gzip",
                    compression_opts=4,
                )

            if lght_ds is None:
                lght_h5 = h5py.File(tmp_lght_h5_path, "w")
                h_l, w_l, t_l = lght_event.shape
                lght_ds = lght_h5.create_dataset(
                    "lght",
                    shape=(0, h_l, w_l, t_l),
                    maxshape=(None, h_l, w_l, t_l),
                    dtype=lght_event.dtype,
                    chunks=(1, h_l, w_l, t_l),
                    compression="gzip",
                    compression_opts=4,
                )

            kept_count = append_event(vil_ds, vil_event, kept_count)
            append_event(lght_ds, lght_event, kept_count - 1)

            meta = {col: row[col] for col in META_COLUMNS}
            meta["file_row"] = kept_count - 1
            meta_records.append(meta)

            if (idx + 1) % progress_every == 0:
                print(
                    f"[{split_name}] processed {idx + 1}/{len(split_df)} rows "
                    f"(kept {kept_count})"
                )

        if kept_count == 0:
            raise RuntimeError(f"No paired events were written for split '{split_name}'.")
    except Exception:
        cleanup_file_if_exists(tmp_vil_h5_path)
        cleanup_file_if_exists(tmp_lght_h5_path)
        cleanup_file_if_exists(tmp_meta_csv_path)
        raise
    finally:
        for handle in vil_src_handles.values():
            handle.close()
        for handle in lght_src_handles.values():
            handle.close()
        if vil_h5 is not None:
            vil_h5.close()
        if lght_h5 is not None:
            lght_h5.close()

    try:
        pd.DataFrame(meta_records, columns=META_COLUMNS + ["file_row"]).to_csv(
            tmp_meta_csv_path, index=False
        )
        os.replace(tmp_vil_h5_path, vil_h5_path)
        os.replace(tmp_lght_h5_path, lght_h5_path)
        os.replace(tmp_meta_csv_path, meta_csv_path)
        split_committed = True
    finally:
        if not split_committed:
            cleanup_file_if_exists(tmp_vil_h5_path)
            cleanup_file_if_exists(tmp_lght_h5_path)
            cleanup_file_if_exists(tmp_meta_csv_path)

    print(
        f"[{split_name}] saved {kept_count} paired events -> "
        f"{os.path.basename(vil_h5_path)}, {os.path.basename(lght_h5_path)}"
    )
    return vil_h5_path, lght_h5_path, meta_csv_path, kept_count


def convert_sevir_vil_lightning_nowcasting(
    catalog_csv_path: str = "/mnt/data/public_datasets/SEVIR/data/CATALOG.csv",
    data_dir: str = "/mnt/data/public_datasets/SEVIR/data",
    validation_cutoff_date: str = "2019-01-01 00:00:00",
    testing_cutoff_date: str = "2019-06-01 00:00:00",
    output_dir: str = "sevir_vil_lght",
    keep_dtype: bool = True,
    downsample_factor: int = 1,
    max_pct_missing: float = 0.05,
    bbox_tol: float = 1e-3,
    max_events_per_split: int = None,
    progress_every: int = 500,
):
    """Build aligned SEVIR VIL/lightning train/val/test splits."""
    os.makedirs(output_dir, exist_ok=True)
    completion_path = os.path.join(output_dir, PREPROCESSING_COMPLETE_FILENAME)
    completion_tmp_path = completion_path + ".tmp"
    cleanup_file_if_exists(completion_path)
    cleanup_file_if_exists(completion_tmp_path)

    print("Reading and matching catalog...")
    matched = build_matched_catalog(
        catalog_csv_path=catalog_csv_path,
        max_pct_missing=max_pct_missing,
        bbox_tol=bbox_tol,
    )

    splits = split_catalog_by_time(
        matched_catalog=matched,
        validation_cutoff_date=validation_cutoff_date,
        testing_cutoff_date=testing_cutoff_date,
    )

    print(
        "Matched paired events: "
        f"total={len(matched)}, "
        f"training={len(splits['training'])}, "
        f"validation={len(splits['validation'])}, "
        f"testing={len(splits['testing'])}"
    )

    split_counts = {}
    for split_name, split_df in splits.items():
        _, _, _, kept_count = process_split(
            split_name=split_name,
            split_df=split_df,
            data_dir=data_dir,
            output_dir=output_dir,
            keep_dtype=keep_dtype,
            downsample_factor=downsample_factor,
            max_events_per_split=max_events_per_split,
            progress_every=progress_every,
        )
        split_counts[split_name] = int(kept_count)

    completion_payload = {
        "created_at_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        + "Z",
        "catalog_csv_path": catalog_csv_path,
        "data_dir": data_dir,
        "validation_cutoff_date": str(validation_cutoff_date),
        "testing_cutoff_date": str(testing_cutoff_date),
        "output_dir": output_dir,
        "keep_dtype": bool(keep_dtype),
        "downsample_factor": int(downsample_factor),
        "max_pct_missing": float(max_pct_missing),
        "bbox_tol": float(bbox_tol),
        "matched_total": int(len(matched)),
        "split_counts": split_counts,
    }
    with open(completion_tmp_path, "w", encoding="utf-8") as fp:
        json.dump(completion_payload, fp, indent=2, sort_keys=True)
    os.replace(completion_tmp_path, completion_path)

    print("Paired VIL/lightning preprocessing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess paired SEVIR VIL + lightning nowcasting splits."
    )
    parser.add_argument(
        "--catalog_csv_path",
        type=str,
        default="/mnt/data/public_datasets/SEVIR/data/CATALOG.csv",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/mnt/data/public_datasets/SEVIR/data",
    )
    parser.add_argument("--val_cutoff", type=str, default="2019-01-01 00:00:00")
    parser.add_argument("--test_cutoff", type=str, default="2019-06-01 00:00:00")
    parser.add_argument("--output_dir", type=str, default="sevir_vil_lght")
    parser.add_argument(
        "--keep_dtype",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--downsample_factor", type=int, default=1)
    parser.add_argument("--max_pct_missing", type=float, default=0.05)
    parser.add_argument("--bbox_tol", type=float, default=1e-3)
    parser.add_argument("--max_events_per_split", type=int, default=None)
    parser.add_argument("--progress_every", type=int, default=500)

    args = parser.parse_args()

    convert_sevir_vil_lightning_nowcasting(
        catalog_csv_path=args.catalog_csv_path,
        data_dir=args.data_dir,
        validation_cutoff_date=args.val_cutoff,
        testing_cutoff_date=args.test_cutoff,
        output_dir=args.output_dir,
        keep_dtype=args.keep_dtype,
        downsample_factor=args.downsample_factor,
        max_pct_missing=args.max_pct_missing,
        bbox_tol=args.bbox_tol,
        max_events_per_split=args.max_events_per_split,
        progress_every=args.progress_every,
    )
