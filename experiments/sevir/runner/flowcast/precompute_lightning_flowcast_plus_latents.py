"""
Offline latent-cache builder for FlowCast+ lightning nowcasting.

This script precomputes, for every train/validation sample:

- past VIL VAE latents
- future VIL VAE latents from ground-truth future VIL
- future VIL latents forecast by the frozen pretrained VIL FlowCast

The resulting HDF5 file can be consumed by the lightning trainer so train/val no
longer need to run VAE encoding + VIL FlowCast inference online every batch.
The online test/inference path should stay unchanged to measure real deployment
costs.
"""

import argparse
import os
import sys
from typing import Sequence

import h5py
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.getcwd())
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from common.models.flowcast.lightning_condition_diffusion import (  # noqa: E402
    FrozenVILFlowCastForecaster,
    PretrainedVILEncoder,
)
from experiments.sevir.dataset.sevirlightningdataset import (  # noqa: E402
    DynamicPairedVilLightningDataset,
    dynamic_paired_vil_lightning_collate,
)


def resolve_output_dtype(name: str):
    normalized = name.strip().lower()
    if normalized == "float16":
        return np.float16, torch.float16
    if normalized == "float32":
        return np.float32, torch.float32
    raise ValueError(f"Unsupported output dtype {name!r}. Use float16 or float32.")


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, world_size, local_rank, device
    return 0, 1, 0, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


class IndexedSubsetDataset(Dataset):
    """Wrap a dataset and expose original sample indices for sharded precompute."""

    def __init__(self, dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, local_idx):
        global_idx = int(self.indices[local_idx])
        sample = self.dataset[global_idx]
        return global_idx, *sample


def indexed_dynamic_collate(batch):
    indices, x_vil_past, lgt_past, lgt_future, meta = zip(*batch)
    x_vil_past = torch.stack(list(x_vil_past), dim=0)
    lgt_past = torch.stack(list(lgt_past), dim=0)
    lgt_future = torch.stack(list(lgt_future), dim=0)
    index_tensor = torch.tensor(indices, dtype=torch.int64)
    return index_tensor, x_vil_past, lgt_past, lgt_future, list(meta)


def load_future_vil_batch(base_dataset, meta_batch):
    frames = []
    for meta in meta_batch:
        vil_row = int(meta["_vil_row_index"])
        y_indices_abs = [int(idx) for idx in meta["_y_indices_abs"]]
        vil_event = base_dataset.vil_h5[base_dataset.vil_dataset][vil_row].astype(np.float32)
        if base_dataset.normalize_vil:
            vil_event /= 255.0
        vil_future = vil_event[..., y_indices_abs]
        frames.append(np.moveaxis(vil_future, -1, 0)[..., None])
    return torch.from_numpy(np.stack(frames, axis=0)).float()


def main():
    parser = argparse.ArgumentParser(
        description="Precompute cached VIL latent conditions for FlowCast+ lightning."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/sevir/runner/flowcast/lightning_flowcast_plus_config.yaml",
    )
    parser.add_argument("--meta_csv", type=str, required=True)
    parser.add_argument("--vil_file", type=str, required=True)
    parser.add_argument("--lightning_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug_mode", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_file) and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_file}. Pass --overwrite to replace it."
        )

    rank, world_size, local_rank, ddp_device = setup_ddp()
    is_main_process = rank == 0
    config = OmegaConf.load(args.config)
    debug_mode = bool(config.run_params.debug_mode) or args.debug_mode

    lag_time = int(config.data_params.lag_time)
    lead_time = int(config.data_params.lead_time)
    time_spacing = int(config.data_params.time_spacing)
    raw_seq_len = int(config.data_params.raw_seq_len)
    stride = int(config.data_params.stride)
    c_cap = float(config.loss_params.c_cap)

    vil_dataset = config.dataset_params.vil_dataset
    lightning_dataset = config.dataset_params.lightning_dataset
    vil_row_key = config.dataset_params.vil_row_key
    lightning_row_key = config.dataset_params.lightning_row_key
    target_hw = tuple(config.model_params.target_hw)

    ae_config = config.autoencoder_params.autoencoder_config
    ae_checkpoint = config.autoencoder_params.autoencoder_checkpoint
    ae_chunk_size = config.autoencoder_params.ae_chunk_size

    future_vil_cfg = config.get("future_vil_condition_params", {})
    if not bool(future_vil_cfg.get("enable", False)):
        raise ValueError(
            "This precompute script is for FlowCast+ and expects future_vil_condition_params.enable=true."
        )
    future_vil_flowcast_config = future_vil_cfg.get("flowcast_config", None)
    future_vil_flowcast_checkpoint = future_vil_cfg.get("flowcast_checkpoint", None)
    future_vil_euler_steps = int(future_vil_cfg.get("euler_steps", 10))
    future_vil_probabilistic_samples = int(
        future_vil_cfg.get("probabilistic_samples", 1)
    )

    if args.device == "cuda" and torch.cuda.is_available():
        device = ddp_device
    else:
        device = torch.device(args.device)
    out_np_dtype, out_torch_dtype = resolve_output_dtype(args.dtype)

    base_dataset = DynamicPairedVilLightningDataset(
        meta_csv=args.meta_csv,
        vil_file=args.vil_file,
        lightning_file=args.lightning_file,
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
    all_indices = np.arange(len(base_dataset), dtype=np.int64)
    local_indices = all_indices[rank::world_size]
    dataset = IndexedSubsetDataset(base_dataset, local_indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=indexed_dynamic_collate,
        num_workers=args.num_workers if not debug_mode else 0,
        pin_memory=not debug_mode,
        persistent_workers=not debug_mode and args.num_workers > 0,
        drop_last=False,
    )

    encoder = PretrainedVILEncoder(ae_config, ae_checkpoint).to(device)
    encoder.freeze_all()
    encoder.eval()
    latent_channels = encoder.latent_channels

    forecaster = FrozenVILFlowCastForecaster(
        config_path=future_vil_flowcast_config,
        checkpoint_path=future_vil_flowcast_checkpoint,
        input_shape=(lag_time, target_hw[0], target_hw[1], latent_channels),
        target_shape=(lead_time, target_hw[0], target_hw[1], latent_channels),
    ).to(device)
    forecaster.eval()

    total_samples = len(base_dataset)
    local_samples = len(dataset)
    local_past_shape = (
        local_samples,
        lag_time,
        target_hw[0],
        target_hw[1],
        latent_channels,
    )
    local_future_shape = (
        local_samples,
        lead_time,
        target_hw[0],
        target_hw[1],
        latent_channels,
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    shard_output = args.output_file + f".rank{rank}.tmp.h5"
    if os.path.exists(shard_output):
        os.remove(shard_output)

    with h5py.File(shard_output, "w") as out_h5:
        sample_index_ds = out_h5.create_dataset(
            "sample_index",
            shape=(local_samples,),
            dtype=np.int64,
        )
        z_vil_past_ds = out_h5.create_dataset(
            "z_vil_past",
            shape=local_past_shape,
            dtype=out_np_dtype,
            chunks=(1,) + local_past_shape[1:],
        )
        z_vil_future_pred_ds = out_h5.create_dataset(
            "z_vil_future_pred",
            shape=local_future_shape,
            dtype=out_np_dtype,
            chunks=(1,) + local_future_shape[1:],
        )
        z_vil_future_gt_ds = out_h5.create_dataset(
            "z_vil_future_gt",
            shape=local_future_shape,
            dtype=out_np_dtype,
            chunks=(1,) + local_future_shape[1:],
        )

        out_h5.attrs["meta_csv"] = args.meta_csv
        out_h5.attrs["vil_file"] = args.vil_file
        out_h5.attrs["lightning_file"] = args.lightning_file
        out_h5.attrs["config"] = args.config
        out_h5.attrs["ae_chunk_size"] = str(ae_chunk_size)
        out_h5.attrs["future_vil_euler_steps"] = future_vil_euler_steps
        out_h5.attrs["future_vil_probabilistic_samples"] = future_vil_probabilistic_samples
        out_h5.attrs["saved_dtype"] = args.dtype

        write_index = 0
        with torch.no_grad():
            progress = tqdm(
                loader,
                desc=f"Precomputing FlowCast+ latents rank {rank}",
                leave=False,
                disable=not is_main_process and world_size > 1,
            )
            for sample_indices, x_vil_past, _, _, meta_batch in progress:
                x_vil_past = x_vil_past.to(device, non_blocking=True)
                x_vil_future_gt = load_future_vil_batch(base_dataset, meta_batch).to(
                    device, non_blocking=True
                )

                z_vil_past = encoder.encode_sequence(
                    x_vil_past,
                    target_hw=target_hw,
                    chunk_size=ae_chunk_size,
                )
                z_vil_future_pred = forecaster.forecast(
                    z_vil_past,
                    probabilistic_samples=future_vil_probabilistic_samples,
                    euler_steps=future_vil_euler_steps,
                )
                z_vil_future_gt = encoder.encode_sequence(
                    x_vil_future_gt,
                    target_hw=target_hw,
                    chunk_size=ae_chunk_size,
                )

                batch_size_local = z_vil_past.shape[0]
                write_slice = slice(write_index, write_index + batch_size_local)
                sample_index_ds[write_slice] = sample_indices.numpy()
                z_vil_past_ds[write_slice] = (
                    z_vil_past.detach().to(dtype=out_torch_dtype).cpu().numpy()
                )
                z_vil_future_pred_ds[write_slice] = (
                    z_vil_future_pred.detach().to(dtype=out_torch_dtype).cpu().numpy()
                )
                z_vil_future_gt_ds[write_slice] = (
                    z_vil_future_gt.detach().to(dtype=out_torch_dtype).cpu().numpy()
                )
                write_index += batch_size_local

        if write_index != total_samples:
            if write_index != local_samples:
                raise RuntimeError(
                    f"Expected to write {local_samples} local samples on rank {rank}, "
                    f"but wrote {write_index}."
                )

    if world_size > 1:
        dist.barrier()

    # Non-main ranks are done once shard writing has completed and synchronized.
    # Keeping them inside the process group while rank 0 performs a long serial
    # HDF5 merge can trigger NCCL watchdog timeouts.
    if not is_main_process:
        cleanup_ddp()
        return

    if is_main_process:
        final_tmp = args.output_file + ".tmp"
        if os.path.exists(final_tmp):
            os.remove(final_tmp)
        past_shape = (
            total_samples,
            lag_time,
            target_hw[0],
            target_hw[1],
            latent_channels,
        )
        future_shape = (
            total_samples,
            lead_time,
            target_hw[0],
            target_hw[1],
            latent_channels,
        )
        with h5py.File(final_tmp, "w") as final_h5:
            final_h5.attrs["meta_csv"] = args.meta_csv
            final_h5.attrs["vil_file"] = args.vil_file
            final_h5.attrs["lightning_file"] = args.lightning_file
            final_h5.attrs["config"] = args.config
            final_h5.attrs["ae_chunk_size"] = str(ae_chunk_size)
            final_h5.attrs["future_vil_euler_steps"] = future_vil_euler_steps
            final_h5.attrs["future_vil_probabilistic_samples"] = future_vil_probabilistic_samples
            final_h5.attrs["saved_dtype"] = args.dtype
            final_h5.attrs["world_size"] = world_size

            final_past = final_h5.create_dataset(
                "z_vil_past",
                shape=past_shape,
                dtype=out_np_dtype,
                chunks=(1,) + past_shape[1:],
            )
            final_future_pred = final_h5.create_dataset(
                "z_vil_future_pred",
                shape=future_shape,
                dtype=out_np_dtype,
                chunks=(1,) + future_shape[1:],
            )
            final_future_gt = final_h5.create_dataset(
                "z_vil_future_gt",
                shape=future_shape,
                dtype=out_np_dtype,
                chunks=(1,) + future_shape[1:],
            )

            for shard_rank in range(world_size):
                shard_path = args.output_file + f".rank{shard_rank}.tmp.h5"
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Missing shard file: {shard_path}")
                with h5py.File(shard_path, "r") as shard_h5:
                    shard_indices = shard_h5["sample_index"][:].astype(np.int64)
                    block = 128
                    for start in range(0, shard_indices.shape[0], block):
                        end = min(start + block, shard_indices.shape[0])
                        idx = shard_indices[start:end]
                        final_past[idx] = shard_h5["z_vil_past"][start:end]
                        final_future_pred[idx] = shard_h5["z_vil_future_pred"][start:end]
                        final_future_gt[idx] = shard_h5["z_vil_future_gt"][start:end]

            os.replace(final_tmp, args.output_file)
            for shard_rank in range(world_size):
                shard_path = args.output_file + f".rank{shard_rank}.tmp.h5"
                if os.path.exists(shard_path):
                    os.remove(shard_path)
        print(f"Saved precomputed FlowCast+ latents -> {args.output_file}")

    cleanup_ddp()
 

if __name__ == "__main__":
    main()
