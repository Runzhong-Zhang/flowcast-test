"""
Dataset utilities for the Singapore radar-lightning nowcasting project.

The project dataset already stores one complete sample per NetCDF file:

- 13 radar input frames
- 12 radar target frames
- 13 IC / CG lightning input frames
- 12 IC / CG lightning target frames

This differs from the SEVIR loaders in this repo, which slice multiple windows
from a longer event cube. Here, each `.nc` file is already one model sample, so
the dataset classes map one file -> one item directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import h5py

try:
    import netCDF4 as nc
except ImportError:
    nc = None


RADAR_MAX_MMPH: float = 128.0
LIGHTNING_TARGET_MODES = ("ic_cg", "cg_only")
DEFAULT_PROJECT_DATASET_ROOT = Path("/mnt/data/users/wanglu/nowcast_dataset")


def lightning_counts_to_binary(
    ic_counts: np.ndarray,
    cg_counts: np.ndarray,
    target_mode: Literal["ic_cg", "cg_only"] = "ic_cg",
) -> np.ndarray:
    """
    Convert project lightning tensors into a binary occupancy mask.

    Modes:

    - `ic_cg`: positive if either IC or CG is present
    - `cg_only`: positive only if CG is present
    """
    if target_mode == "ic_cg":
        return ((ic_counts > 0) | (cg_counts > 0)).astype(np.float32)
    if target_mode == "cg_only":
        return (cg_counts > 0).astype(np.float32)
    raise ValueError(
        f"Unsupported lightning target mode {target_mode!r}. "
        f"Expected one of {LIGHTNING_TARGET_MODES}."
    )


def lightning_counts_to_condition(
    ic_counts: np.ndarray,
    cg_counts: np.ndarray,
    c_cap: float,
) -> np.ndarray:
    """
    Convert project IC / CG lightning tensors into a bounded conditioning field.

    We treat the per-pixel lightning count as IC count plus the CG hit indicator.
    This keeps the conditioning path similar to the SEVIR lightning loader:

        log1p(min(count, c_cap)) / log1p(c_cap)
    """
    total_counts = ic_counts.astype(np.float32) + cg_counts.astype(np.float32)
    clipped = np.minimum(total_counts, c_cap)
    return np.log1p(clipped) / np.log1p(c_cap)


def get_project_file_list(
    split: Literal["train", "val", "test"],
    dataset_root: str | Path = DEFAULT_PROJECT_DATASET_ROOT,
    include_radar_only: bool = False,
    ro_fraction: float = 1.0,
    seed: int = 42,
) -> list[Path]:
    """
    Return a sorted list of `.nc` files for one split.

    If `include_radar_only=True`, append files from `radar_only/<split>` after
    the main files. This mirrors the original project dataset design where
    radar-only negatives are injected as extra hard negatives.
    """
    root = Path(dataset_root)
    split_dir = root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    main_files = sorted(split_dir.glob("*.nc"))
    if not main_files:
        raise ValueError(f"No .nc files found under {split_dir}")

    radar_only_files: list[Path] = []
    if include_radar_only:
        radar_only_dir = root / "radar_only" / split
        if radar_only_dir.exists():
            radar_only_files = sorted(radar_only_dir.glob("*.nc"))
            if radar_only_files and ro_fraction < 1.0:
                rng = random.Random(seed)
                keep = max(1, int(len(radar_only_files) * ro_fraction))
                radar_only_files = sorted(rng.sample(radar_only_files, keep))

    return main_files + radar_only_files


def _to_channel_layout(array: np.ndarray, channel_last: bool) -> torch.Tensor:
    """
    Convert `(T, H, W)` arrays into either `(T, H, W, 1)` or `(1, T, H, W)`.
    """
    tensor = torch.from_numpy(array).float()
    tensor = tensor.unsqueeze(1)
    if channel_last:
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor


def _decode_attr(value):
    """Convert HDF5 / NetCDF attribute values into plain Python scalars/strings."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.bytes_):
        return value.tobytes().decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _decode_attr(value.reshape(-1)[0])
        return [_decode_attr(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_h5_variable(dataset) -> np.ndarray:
    """
    Read one HDF5-backed NetCDF variable and apply simple scale/offset decoding.
    """
    array = np.array(dataset[:])
    scale_factor = _decode_attr(dataset.attrs.get("scale_factor", 1.0))
    add_offset = _decode_attr(dataset.attrs.get("add_offset", 0.0))
    array = array.astype(np.float32) * float(scale_factor) + float(add_offset)
    return array.astype(np.float32)


def _read_sample(fpath: Path):
    """Read one NetCDF sample and return arrays plus selected metadata."""
    try:
        if nc is not None:
            with nc.Dataset(fpath) as ds:
                radar_input = np.array(ds.variables["radar_input"][:], dtype=np.float32)
                radar_target = np.array(ds.variables["radar_target"][:], dtype=np.float32)
                ic_input = np.array(ds.variables["ic_input"][:], dtype=np.float32)
                cg_input = np.array(ds.variables["cg_input"][:], dtype=np.float32)
                ic_target = np.array(ds.variables["ic_target"][:], dtype=np.float32)
                cg_target = np.array(ds.variables["cg_target"][:], dtype=np.float32)

                meta = {
                    "filename": fpath.name,
                    "anchor_time": str(getattr(ds, "anchor_time", "unknown")),
                    "split": str(getattr(ds, "split", "unknown")),
                    "sample_id": str(getattr(ds, "sample_id", fpath.stem)),
                    "sample_category": str(getattr(ds, "sample_category", "unknown")),
                    "session_id": int(getattr(ds, "session_id", -1)),
                    "sample_index": int(getattr(ds, "sample_index", -1)),
                    "input_steps": int(getattr(ds, "input_steps", radar_input.shape[0])),
                    "target_steps": int(getattr(ds, "target_steps", radar_target.shape[0])),
                    "step_min": int(getattr(ds, "step_min", 5)),
                }
        else:
            with h5py.File(fpath, "r") as ds:
                radar_input = _read_h5_variable(ds["radar_input"])
                radar_target = _read_h5_variable(ds["radar_target"])
                ic_input = np.array(ds["ic_input"][:], dtype=np.float32)
                cg_input = np.array(ds["cg_input"][:], dtype=np.float32)
                ic_target = np.array(ds["ic_target"][:], dtype=np.float32)
                cg_target = np.array(ds["cg_target"][:], dtype=np.float32)

                meta = {
                    "filename": fpath.name,
                    "anchor_time": str(_decode_attr(ds.attrs.get("anchor_time", "unknown"))),
                    "split": str(_decode_attr(ds.attrs.get("split", "unknown"))),
                    "sample_id": str(_decode_attr(ds.attrs.get("sample_id", fpath.stem))),
                    "sample_category": str(_decode_attr(ds.attrs.get("sample_category", "unknown"))),
                    "session_id": int(_decode_attr(ds.attrs.get("session_id", -1))),
                    "sample_index": int(_decode_attr(ds.attrs.get("sample_index", -1))),
                    "input_steps": int(_decode_attr(ds.attrs.get("input_steps", radar_input.shape[0]))),
                    "target_steps": int(_decode_attr(ds.attrs.get("target_steps", radar_target.shape[0]))),
                    "step_min": int(_decode_attr(ds.attrs.get("step_min", 5))),
                }
    except Exception as exc:
        raise RuntimeError(f"Failed to read {fpath}: {exc}") from exc

    return radar_input, radar_target, ic_input, cg_input, ic_target, cg_target, meta


class ProjectRadarFlowCastDataset(Dataset):
    """
    Dataset for radar-only FlowCast training on the project dataset.

    Returns one past radar sequence and one future radar sequence per `.nc` file.
    """

    def __init__(
        self,
        file_list: Iterable[str | Path],
        radar_max: float = RADAR_MAX_MMPH,
        channel_last: bool = True,
    ):
        self.files = [Path(fpath) for fpath in file_list]
        self.radar_max = radar_max
        self.channel_last = channel_last

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        fpath = self.files[index]
        radar_input, radar_target, _, _, _, _, meta = _read_sample(fpath)

        radar_input = np.clip(radar_input / self.radar_max, 0.0, 1.0)
        radar_target = np.clip(radar_target / self.radar_max, 0.0, 1.0)

        x = _to_channel_layout(radar_input, self.channel_last)
        y = _to_channel_layout(radar_target, self.channel_last)
        return x, y, meta


class ProjectRadarAutoencoderDataset(Dataset):
    """
    Dataset for radar VAE training on the project dataset.

    Each item returns one random radar frame sampled from the 25-frame
    input+target sequence stored in a single `.nc` file.
    """

    def __init__(
        self,
        file_list: Iterable[str | Path],
        radar_max: float = RADAR_MAX_MMPH,
        include_future_frames: bool = True,
    ):
        self.files = [Path(fpath) for fpath in file_list]
        self.radar_max = radar_max
        self.include_future_frames = include_future_frames

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        fpath = self.files[index]
        radar_input, radar_target, _, _, _, _, meta = _read_sample(fpath)
        frames = radar_input
        if self.include_future_frames:
            frames = np.concatenate([radar_input, radar_target], axis=0)

        frame_index = int(np.random.randint(0, frames.shape[0]))
        frame = np.clip(frames[frame_index] / self.radar_max, 0.0, 1.0)
        frame = torch.from_numpy(frame).float().unsqueeze(0)
        meta["frame_index"] = int(frame_index)
        return frame, meta


class ProjectPairedRadarLightningDataset(Dataset):
    """
    Dataset for Lightning FlowCast / FlowCast+ on the project dataset.

    Returns:

    - past radar sequence `(13, H, W, 1)` by default
    - past lightning condition `(13, H, W, 1)`
    - future binary lightning target `(12, H, W, 1)`
    - sample metadata
    """

    def __init__(
        self,
        file_list: Iterable[str | Path],
        radar_max: float = RADAR_MAX_MMPH,
        c_cap: float = 20.0,
        lightning_target_mode: Literal["ic_cg", "cg_only"] = "ic_cg",
        channel_last: bool = True,
    ):
        self.files = [Path(fpath) for fpath in file_list]
        self.radar_max = radar_max
        self.c_cap = c_cap
        if lightning_target_mode not in LIGHTNING_TARGET_MODES:
            raise ValueError(
                f"Unsupported lightning_target_mode={lightning_target_mode!r}. "
                f"Expected one of {LIGHTNING_TARGET_MODES}."
            )
        self.lightning_target_mode = lightning_target_mode
        self.channel_last = channel_last

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        fpath = self.files[index]
        radar_input, _, ic_input, cg_input, ic_target, cg_target, meta = _read_sample(fpath)

        radar_input = np.clip(radar_input / self.radar_max, 0.0, 1.0)
        lgt_past = lightning_counts_to_condition(ic_input, cg_input, self.c_cap)
        lgt_future = lightning_counts_to_binary(
            ic_target,
            cg_target,
            target_mode=self.lightning_target_mode,
        )

        x_radar_past = _to_channel_layout(radar_input, self.channel_last)
        lgt_past = _to_channel_layout(lgt_past, self.channel_last)
        lgt_future = _to_channel_layout(lgt_future, self.channel_last)
        meta["lightning_target_mode"] = self.lightning_target_mode
        return x_radar_past, lgt_past, lgt_future, meta


def project_sequential_collate(batch):
    """Collate function for `ProjectRadarFlowCastDataset`."""
    x_list, y_list, meta_list = zip(*batch)
    x = torch.stack(list(x_list), dim=0)
    y = torch.stack(list(y_list), dim=0)
    return x, y, list(meta_list)


def project_autoencoder_collate(batch):
    """Collate function for `ProjectRadarAutoencoderDataset`."""
    frame_list, meta_list = zip(*batch)
    frames = torch.stack(list(frame_list), dim=0)
    return frames, list(meta_list)


def project_paired_radar_lightning_collate(batch):
    """Collate function for `ProjectPairedRadarLightningDataset`."""
    radar_list, lgt_past_list, lgt_future_list, meta_list = zip(*batch)
    x_radar_past = torch.stack(list(radar_list), dim=0)
    lgt_past = torch.stack(list(lgt_past_list), dim=0)
    lgt_future = torch.stack(list(lgt_future_list), dim=0)
    return x_radar_past, lgt_past, lgt_future, list(meta_list)


def build_project_flowcast_dataloaders(
    dataset_root: str | Path = DEFAULT_PROJECT_DATASET_ROOT,
    batch_size: int = 16,
    num_workers: int = 4,
    radar_max: float = RADAR_MAX_MMPH,
    include_radar_only: bool = True,
    ro_fraction: float = 1.0,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Build train / val / test dataloaders for radar FlowCast training.
    """
    loaders = []
    for split in ("train", "val", "test"):
        dataset = ProjectRadarFlowCastDataset(
            get_project_file_list(
                split,
                dataset_root,
                include_radar_only=include_radar_only,
                ro_fraction=ro_fraction,
                seed=seed,
            ),
            radar_max=radar_max,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=project_sequential_collate,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
        )
    return tuple(loaders)


def build_project_autoencoder_dataloaders(
    dataset_root: str | Path = DEFAULT_PROJECT_DATASET_ROOT,
    batch_size: int = 16,
    num_workers: int = 4,
    radar_max: float = RADAR_MAX_MMPH,
    include_future_frames: bool = True,
    include_radar_only: bool = True,
    ro_fraction: float = 1.0,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Build train / val / test dataloaders for project radar autoencoder training.
    """
    loaders = []
    for split in ("train", "val", "test"):
        dataset = ProjectRadarAutoencoderDataset(
            get_project_file_list(
                split,
                dataset_root,
                include_radar_only=include_radar_only,
                ro_fraction=ro_fraction,
                seed=seed,
            ),
            radar_max=radar_max,
            include_future_frames=include_future_frames,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=project_autoencoder_collate,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
        )
    return tuple(loaders)


def build_project_lightning_dataloaders(
    dataset_root: str | Path = DEFAULT_PROJECT_DATASET_ROOT,
    batch_size: int = 16,
    num_workers: int = 4,
    radar_max: float = RADAR_MAX_MMPH,
    c_cap: float = 20.0,
    lightning_target_mode: Literal["ic_cg", "cg_only"] = "ic_cg",
    include_radar_only: bool = True,
    ro_fraction: float = 1.0,
    seed: int = 42,
    pin_memory: bool = True,
):
    """
    Build train / val / test dataloaders for Lightning FlowCast training.
    """
    loaders = []
    for split in ("train", "val", "test"):
        dataset = ProjectPairedRadarLightningDataset(
            get_project_file_list(
                dataset_root,
                split,
                include_radar_only=include_radar_only,
                ro_fraction=ro_fraction,
                seed=seed,
            ),
            radar_max=radar_max,
            c_cap=c_cap,
            lightning_target_mode=lightning_target_mode,
        )
        loaders.append(
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=project_paired_radar_lightning_collate,
                pin_memory=pin_memory,
                persistent_workers=(num_workers > 0),
            )
        )
    return tuple(loaders)
