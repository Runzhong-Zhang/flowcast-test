"""
Dataset utilities for conditional lightning diffusion on SEVIR.

This module follows the general style of `sevirfulldataset.py`, but pairs:

- past VIL sequences
- past lightning transformed-count sequences
- future binary lightning sequences

Assumptions
-----------
- VIL is stored as `(N, H, W, T)` under a dataset key such as `"vil"`.
- Lightning is stored as native count cubes `(N, 48, 48, T)` under a dataset key
  such as `"lght"`.
- The metadata CSV contains row indices for the VIL and lightning HDF5 datasets.
"""

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def get_row_index(row: pd.Series, preferred_key=None) -> int:
    """Resolve an HDF5 row index from a metadata row."""
    if preferred_key is not None and preferred_key in row:
        return int(row[preferred_key])
    for key in ("file_row", "file_index"):
        if key in row:
            return int(row[key])
    raise KeyError(
        "Could not find a row index column. Expected one of "
        f"{preferred_key!r}, 'file_row', or 'file_index'."
    )


def lightning_counts_to_binary(counts: np.ndarray) -> np.ndarray:
    """Convert native lightning counts into binary occupancy."""
    return (counts > 0).astype(np.float32)


def lightning_counts_to_condition(counts: np.ndarray, c_cap: float) -> np.ndarray:
    """Transform native lightning counts into a bounded log-conditioning field."""
    clipped = np.minimum(counts.astype(np.float32), c_cap)
    return np.log1p(clipped) / np.log1p(c_cap)


def resolve_event_and_seq_index(cum_counts: np.ndarray, index: int):
    """Map a flattened sample index to `(event_idx, seq_idx)`."""
    event_idx = int(np.searchsorted(cum_counts, index, side="right"))
    if event_idx == 0:
        seq_idx = index
    else:
        seq_idx = index - int(cum_counts[event_idx - 1])
    return event_idx, seq_idx


class DynamicPairedVilLightningDataset(Dataset):
    """
    Sliding-window SEVIR dataset for:

    - past VIL condition
    - past lightning condition
    - future binary lightning target
    """

    def __init__(
        self,
        meta_csv,
        vil_file,
        lightning_file,
        vil_dataset="vil",
        lightning_dataset="lght",
        vil_row_key=None,
        lightning_row_key=None,
        raw_seq_len=49,
        lag_time=13,
        lead_time=12,
        time_spacing=1,
        stride=12,
        normalize_vil=True,
        c_cap=20.0,
        channel_last=True,
        debug_mode=False,
        vil_output_hw=None,
    ):
        self.meta_csv = meta_csv
        self.vil_file = vil_file
        self.lightning_file = lightning_file
        self.vil_dataset = vil_dataset
        self.lightning_dataset = lightning_dataset
        self.vil_row_key = vil_row_key
        self.lightning_row_key = lightning_row_key
        self.raw_seq_len = raw_seq_len
        self.lag_time = lag_time
        self.lead_time = lead_time
        self.time_spacing = time_spacing
        self.seq_len = (lag_time + lead_time) * time_spacing
        self.stride = stride
        self.normalize_vil = normalize_vil
        self.c_cap = c_cap
        self.channel_last = channel_last
        self.debug_mode = debug_mode
        self.vil_output_hw = tuple(vil_output_hw) if vil_output_hw is not None else None

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in metadata file: {self.meta_csv}")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        if self.raw_seq_len < self.seq_len:
            raise ValueError(
                "raw_seq_len must be >= (lag_time + lead_time) * time_spacing"
            )

        # Opened lazily in __getitem__ so each DataLoader worker gets its own
        # file handle — h5py file handles must not be shared across forked processes.
        self.vil_h5 = None
        self.lightning_h5 = None

        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        if self.debug_mode:
            self.n_seq_per_event = 1

        self.event_seq_counts = np.full(
            len(self.metadata), self.n_seq_per_event, dtype=np.int32
        )
        self.cum_counts = np.cumsum(self.event_seq_counts)

    def __len__(self):
        return int(self.cum_counts[-1])

    def _open_hdf5(self):
        if self.vil_h5 is None:
            self.vil_h5 = h5py.File(self.vil_file, "r")
        if self.lightning_h5 is None:
            self.lightning_h5 = h5py.File(self.lightning_file, "r")

    def __getitem__(self, index):
        self._open_hdf5()
        event_idx, seq_idx = resolve_event_and_seq_index(self.cum_counts, index)

        row = self.metadata.iloc[event_idx]
        vil_row = get_row_index(row, self.vil_row_key)
        lightning_row = get_row_index(row, self.lightning_row_key)

        vil_event = self.vil_h5[self.vil_dataset][vil_row].astype(np.float32)
        lightning_event = self.lightning_h5[self.lightning_dataset][lightning_row].astype(
            np.float32
        )

        if vil_event.ndim != 3:
            raise ValueError(
                f"Expected VIL event with shape (H, W, T), got {vil_event.shape}"
            )
        if lightning_event.ndim != 3:
            raise ValueError(
                "Expected lightning event with shape (48, 48, T) containing native "
                f"count cubes, got {lightning_event.shape}"
            )

        if self.normalize_vil:
            vil_event /= 255.0

        start = seq_idx * self.stride
        end = start + self.seq_len

        vil_segment = vil_event[..., start:end]
        lightning_segment = lightning_event[..., start:end]

        x_indices = [i * self.time_spacing for i in range(self.lag_time)]
        y_end = self.seq_len - 1
        y_indices = [y_end - i * self.time_spacing for i in range(self.lead_time)]
        y_indices.reverse()

        x_vil_past = vil_segment[..., x_indices]
        lgt_past_count = lightning_segment[..., x_indices]
        lgt_future_count = lightning_segment[..., y_indices]

        lgt_past = lightning_counts_to_condition(lgt_past_count, self.c_cap)
        lgt_future = lightning_counts_to_binary(lgt_future_count)

        x_vil_past = torch.from_numpy(x_vil_past).float()
        lgt_past = torch.from_numpy(lgt_past).float()
        lgt_future = torch.from_numpy(lgt_future).float()

        # Convert from (H, W, T) to either (T, H, W, C) or (C, T, H, W).
        x_vil_past = x_vil_past.permute(2, 0, 1).unsqueeze(0)

        # Optional CPU-side resize: avoids sending large 384×384 tensors to GPU.
        # Each DataLoader worker resizes independently — much cheaper than GPU interpolation.
        if self.vil_output_hw is not None:
            h, w = x_vil_past.shape[2], x_vil_past.shape[3]
            if (h, w) != self.vil_output_hw:
                x_vil_past = torch.nn.functional.interpolate(
                    x_vil_past,
                    size=self.vil_output_hw,
                    mode="bilinear",
                    align_corners=False,
                )
        lgt_past = lgt_past.permute(2, 0, 1).unsqueeze(0)
        lgt_future = lgt_future.permute(2, 0, 1).unsqueeze(0)

        if self.channel_last:
            x_vil_past = x_vil_past.permute(1, 2, 3, 0)
            lgt_past = lgt_past.permute(1, 2, 3, 0)
            lgt_future = lgt_future.permute(1, 2, 3, 0)

        meta = row.to_dict()
        meta["_vil_row_index"] = int(vil_row)
        meta["_lightning_row_index"] = int(lightning_row)
        meta["_sequence_start_index"] = int(start)
        meta["_sequence_end_index"] = int(end)
        meta["_x_indices_abs"] = [int(start + idx) for idx in x_indices]
        meta["_y_indices_abs"] = [int(start + idx) for idx in y_indices]

        # Intentionally do not return future VIL here. The lightning model is trained
        # only with past VIL, past lightning, and future lightning targets so there is
        # no train-time leakage from future VIL ground truth.
        return x_vil_past, lgt_past, lgt_future, meta


class PrecomputedPairedVilLightningDataset(Dataset):
    """
    Sliding-window SEVIR dataset that reuses cached VIL latent conditions.

    Each sample returns:

    - cached past VIL latent condition
    - cached future VIL latent forecast condition
    - past lightning transformed-count condition
    - future binary lightning target
    """

    def __init__(
        self,
        meta_csv,
        lightning_file,
        latent_file,
        lightning_dataset="lght",
        lightning_row_key=None,
        latent_past_dataset="z_vil_past",
        latent_future_pred_dataset="z_vil_future_pred",
        raw_seq_len=49,
        lag_time=13,
        lead_time=12,
        time_spacing=1,
        stride=12,
        c_cap=20.0,
        debug_mode=False,
    ):
        self.meta_csv = meta_csv
        self.lightning_file = lightning_file
        self.latent_file = latent_file
        self.lightning_dataset = lightning_dataset
        self.lightning_row_key = lightning_row_key
        self.latent_past_dataset = latent_past_dataset
        self.latent_future_pred_dataset = latent_future_pred_dataset
        self.raw_seq_len = raw_seq_len
        self.lag_time = lag_time
        self.lead_time = lead_time
        self.time_spacing = time_spacing
        self.seq_len = (lag_time + lead_time) * time_spacing
        self.stride = stride
        self.c_cap = c_cap
        self.debug_mode = debug_mode

        self.metadata = pd.read_csv(self.meta_csv, parse_dates=["time_utc"])
        if self.metadata.empty:
            raise ValueError(f"No events found in metadata file: {self.meta_csv}")
        if self.debug_mode:
            self.metadata = self.metadata.iloc[:10].reset_index(drop=True)

        if self.raw_seq_len < self.seq_len:
            raise ValueError(
                "raw_seq_len must be >= (lag_time + lead_time) * time_spacing"
            )

        self.lightning_h5 = h5py.File(self.lightning_file, "r")
        self.latent_h5 = h5py.File(self.latent_file, "r")

        if self.latent_past_dataset not in self.latent_h5:
            raise KeyError(
                f"Missing latent dataset {self.latent_past_dataset!r} in {self.latent_file}"
            )
        if self.latent_future_pred_dataset not in self.latent_h5:
            raise KeyError(
                f"Missing latent dataset {self.latent_future_pred_dataset!r} in {self.latent_file}"
            )

        self.n_seq_per_event = 1 + (self.raw_seq_len - self.seq_len) // self.stride
        if self.debug_mode:
            self.n_seq_per_event = 1

        self.event_seq_counts = np.full(
            len(self.metadata), self.n_seq_per_event, dtype=np.int32
        )
        self.cum_counts = np.cumsum(self.event_seq_counts)
        expected_len = int(self.cum_counts[-1])
        if self.latent_h5[self.latent_past_dataset].shape[0] != expected_len:
            raise ValueError(
                "Cached past latent count does not match expected sample count: "
                f"{self.latent_h5[self.latent_past_dataset].shape[0]} vs {expected_len}"
            )
        if self.latent_h5[self.latent_future_pred_dataset].shape[0] != expected_len:
            raise ValueError(
                "Cached future latent count does not match expected sample count: "
                f"{self.latent_h5[self.latent_future_pred_dataset].shape[0]} vs {expected_len}"
            )

    def __len__(self):
        return int(self.cum_counts[-1])

    def __getitem__(self, index):
        event_idx, seq_idx = resolve_event_and_seq_index(self.cum_counts, index)

        row = self.metadata.iloc[event_idx]
        lightning_row = get_row_index(row, self.lightning_row_key)
        lightning_event = self.lightning_h5[self.lightning_dataset][lightning_row].astype(
            np.float32
        )
        if lightning_event.ndim != 3:
            raise ValueError(
                "Expected lightning event with shape (48, 48, T) containing native "
                f"count cubes, got {lightning_event.shape}"
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

        lgt_past = lightning_counts_to_condition(lgt_past_count, self.c_cap)
        lgt_future = lightning_counts_to_binary(lgt_future_count)
        z_vil_past = self.latent_h5[self.latent_past_dataset][index].astype(np.float32)
        z_vil_future_pred = self.latent_h5[self.latent_future_pred_dataset][index].astype(
            np.float32
        )

        z_vil_past = torch.from_numpy(z_vil_past).float()
        z_vil_future_pred = torch.from_numpy(z_vil_future_pred).float()
        lgt_past = torch.from_numpy(lgt_past).float()
        lgt_future = torch.from_numpy(lgt_future).float()

        lgt_past = lgt_past.permute(2, 0, 1).unsqueeze(0).permute(1, 2, 3, 0)
        lgt_future = lgt_future.permute(2, 0, 1).unsqueeze(0).permute(1, 2, 3, 0)

        meta = row.to_dict()
        meta["_lightning_row_index"] = int(lightning_row)
        meta["_sequence_start_index"] = int(start)
        meta["_sequence_end_index"] = int(end)
        meta["_x_indices_abs"] = [int(start + idx) for idx in x_indices]
        meta["_y_indices_abs"] = [int(start + idx) for idx in y_indices]
        meta["_precomputed_latent_index"] = int(index)
        return z_vil_past, z_vil_future_pred, lgt_past, lgt_future, meta


def dynamic_paired_vil_lightning_collate(batch):
    """Collate function matching the style of other SEVIR dataset modules."""
    x_vil_past, lgt_past, lgt_future, meta = zip(*batch)
    x_vil_past = torch.stack(list(x_vil_past), dim=0)
    lgt_past = torch.stack(list(lgt_past), dim=0)
    lgt_future = torch.stack(list(lgt_future), dim=0)
    return x_vil_past, lgt_past, lgt_future, list(meta)


def precomputed_paired_vil_lightning_collate(batch):
    """Collate function for cached VIL latent conditions."""
    z_vil_past, z_vil_future_pred, lgt_past, lgt_future, meta = zip(*batch)
    z_vil_past = torch.stack(list(z_vil_past), dim=0)
    z_vil_future_pred = torch.stack(list(z_vil_future_pred), dim=0)
    lgt_past = torch.stack(list(lgt_past), dim=0)
    lgt_future = torch.stack(list(lgt_future), dim=0)
    return z_vil_past, z_vil_future_pred, lgt_past, lgt_future, list(meta)
