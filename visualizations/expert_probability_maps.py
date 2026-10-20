"""Streaming full-softmax routing analysis; model setup mirrors the existing MoE test runner."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchdiffeq import odeint_adjoint as odeint
from tqdm import tqdm
from common.models.flowcast.lightning_condition_diffusion_moe import LightningDiffusionMoESystem
from common.models.flowcast.moe_ffn import CrossModalSparseMoEFFN
from experiments.sevir.dataset.sevirlightningdataset import (
    DynamicPairedVilLightningDataset, dynamic_paired_vil_lightning_collate,
)


class RoutingAccumulator:
    """Reduce logits on device immediately; never retain per-token routing tensors."""
    def __init__(self, lag, lead):
        self.lag, self.lead = lag, lead
        self.sums, self.counts, self.calls, self.shapes = {}, {}, {}, {}
        self.handles = []

    def attach(self, system):
        layers = [(n, m) for n, m in system.denoiser.named_modules()
                  if isinstance(m, CrossModalSparseMoEFFN)]
        if len(layers) != 4 or any(m.num_experts != 8 for _, m in layers):
            raise ValueError("Expected four cross-modal MoE layers, each with eight experts")
        for name, layer in layers:
            self.sums[name] = torch.zeros(self.lead, 8, dtype=torch.float64,
                                          device=next(layer.parameters()).device)
            self.counts[name] = 0
            self.calls[name] = 0
            def before(module, inputs, name=name):
                shape = tuple(inputs[0].shape)
                if len(shape) != 5 or shape[1] != self.lag + self.lead:
                    raise ValueError(f"Unexpected routing shape for {name}: {shape}")
                self.shapes[name] = shape
            def logits_hook(module, inputs, output, name=name):
                b, t, h, w, _ = self.shapes[name]
                # Match the actual router's softmax dtype before summing in float64.
                p = torch.softmax(output.detach(), dim=-1).reshape(b, t, h, w, 8)
                self.sums[name] += p[:, self.lag:].sum(dim=(0, 2, 3), dtype=torch.float64)
                self.counts[name] += b * h * w
                self.calls[name] += 1
            self.handles.append(layer.register_forward_pre_hook(before))
            self.handles.append(layer.router.register_forward_hook(logits_hook))

    def save(self, path, metadata, expected_calls):
        names = list(self.sums)
        if any(self.calls[n] != expected_calls for n in names):
            raise RuntimeError(f"Unexpected router evaluation counts: {self.calls}")
        sums = np.stack([self.sums[n].cpu().numpy() for n in names])
        counts = np.array([self.counts[n] for n in names], dtype=np.int64)
        means = sums / counts[:, None, None]
        if not np.isfinite(means).all() or not np.allclose(means.sum(-1), 1, atol=1e-5):
            raise RuntimeError("Invalid mean routing probabilities")
        metadata.update(layer_names=names, router_calls=self.calls,
                        averaging="token-weighted over samples, pixels, members, and ODE evaluations")
        np.savez_compressed(path, sums=sums, counts=counts,
                            metadata=json.dumps(metadata, sort_keys=True))
        for handle in self.handles:
            handle.remove()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer(args):
    os.chdir(ROOT)
    config = OmegaConf.load(args.config)
    seed_all(args.seed)
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

    if (moe_num_experts, moe_top_k, moe_strategy, moe_cross_modal,
            moe_router_type, moe_shared_expert) != (8, 2, "up_l0_t", True, "gated", False):
        raise ValueError("Checkpoint is not the expected eight-expert gated up_l0_t model")

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

    collector = RoutingAccumulator(lag_time, lead_time)
    collector.attach(system)
    if lead_time != 12:
        raise ValueError("Expected 12 forecast leads")
    evaluations = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=args.label)):
            # Same batch/shard seeds for both models, including future VIL sampling.
            batch_seed = args.seed + args.shard_rank * 1000000 + batch_idx * 100
            seed_all(batch_seed)
            x_vil_past, lgt_past, lgt_future, _ = batch
            x_vil_past, lgt_past = x_vil_past.to(device), lgt_past.to(device)
            lgt_future = lgt_future.to(device)
            z_past, z_future = system.prepare_vil_conditions(x_vil_past, ae_chunk_size=ae_chunk_size)
            b, t, h, w, c = lgt_future.shape
            for member in range(args.members):
                seed_all(batch_seed + member + 1)
                x0 = torch.randn_like(lgt_future).reshape(b * t, h, w, c)
                def dynamics(time, state):
                    nonlocal evaluations
                    evaluations += 1
                    velocity = system.forward_from_encoded(
                        time * torch.ones(b, device=device), state.reshape(b, t, h, w, c),
                        z_past, lgt_past, z_vil_future_pred=z_future)
                    return velocity.contiguous().reshape(b * t, h, w, c)
                odeint(dynamics, x0, torch.tensor([0., 1.], device=device),
                       method="euler", options={"step_size": 1. / args.euler_steps},
                       rtol=1e-2, atol=1e-3, adjoint_params=tuple(system.parameters()))
    metadata = dict(label=args.label, checkpoint=str(Path(args.checkpoint).resolve()),
                    config=OmegaConf.to_container(config, resolve=True), seed=args.seed,
                    euler_steps=args.euler_steps, members=args.members,
                    future_vil_euler_steps=future_vil_euler,
                    selected_total=total_selected, shard_start=shard_start, shard_end=shard_end,
                    num_shards=args.num_shards, shard_rank=args.shard_rank,
                    batch_size=batch_size, torch_version=str(torch.__version__))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    collector.save(out / f"{args.label}_shard_{args.shard_rank}.npz", metadata, evaluations)


def aggregate_probabilities(shards):
    """Combine shard sums using their true token counts (including partial batches)."""
    sums = np.stack([v[0] for v in shards]).sum(0)
    counts = np.stack([v[1] for v in shards]).sum(0)
    if np.any(counts <= 0):
        raise ValueError("Empty routing accumulator")
    means = sums / counts[:, None, None]
    if means.shape != (4, 12, 8) or not np.isfinite(means).all():
        raise ValueError("Invalid probability array")
    if np.any(means < 0) or not np.allclose(means.sum(-1), 1, atol=1e-5):
        raise ValueError("Probabilities must be nonnegative and sum to one")
    return means


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.output_dir)
    maps, records = {}, {}
    for label in ("cross_modal_gated", "cmi_gated"):
        shards = []
        for rank in range(args.num_shards):
            with np.load(out / f"{label}_shard_{rank}.npz", allow_pickle=False) as data:
                shards.append((data["sums"].copy(), data["counts"].copy(),
                               json.loads(str(data["metadata"]))))
        meta = [v[2] for v in shards]
        reference = meta[0]
        for rank, m in enumerate(meta):
            for key in ("checkpoint", "config", "layer_names", "seed", "euler_steps", "members",
                        "num_shards", "selected_total", "batch_size"):
                if m[key] != reference[key]:
                    raise ValueError(f"Inconsistent shard metadata: {label}, {key}")
            if m["shard_rank"] != rank:
                raise ValueError("Incorrect shard rank")
        if meta[0]["shard_start"] != 0 or meta[-1]["shard_end"] != reference["selected_total"]:
            raise ValueError("Incomplete sample coverage")
        if any(a["shard_end"] != b["shard_start"] for a, b in zip(meta, meta[1:])):
            raise ValueError("Overlapping or missing shard samples")
        maps[label] = aggregate_probabilities(shards)
        records[label] = meta
        if maps[label].shape != (4, 12, 8) or not np.allclose(maps[label].sum(-1), 1, atol=1e-5):
            raise ValueError("Invalid aggregated map")
        np.save(out / f"{label}_mean_probabilities.npy", maps[label])
        for layer, matrix in enumerate(maps[label], 1):
            np.savetxt(out / f"{label}_layer_{layer}.csv", matrix, delimiter=",",
                       header=",".join(f"expert_{e}" for e in range(1, 9)), comments="")
    for key in ("seed", "euler_steps", "members", "selected_total", "batch_size", "layer_names",
                "future_vil_euler_steps"):
        if records["cross_modal_gated"][0][key] != records["cmi_gated"][0][key]:
            raise ValueError(f"Model comparison settings differ: {key}")
    vmax = max(float(v.max()) for v in maps.values())
    for layer in range(4):
        fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)
        for ax, (label, matrix) in zip(axes, maps.items()):
            im = ax.imshow(matrix[layer], vmin=0, vmax=vmax, cmap="viridis", aspect="auto")
            ax.set(title=label, xlabel="Expert (IDs are model-specific)", ylabel="Forecast lead (frame)",
                   xticks=range(8), xticklabels=range(1, 9), yticks=range(12), yticklabels=range(1, 13))
        fig.suptitle(f"MoE layer {layer + 1}: mean router probability\n"
                     f"{records['cmi_gated'][0]['euler_steps']} Euler steps; "
                     f"{records['cmi_gated'][0]['members']} ensemble members")
        fig.colorbar(im, ax=axes, label="Mean pre-top-2 probability")
        for extension in ("png", "pdf"):
            fig.savefig(out / f"layer_{layer + 1}_comparison.{extension}", dpi=200)
        plt.close(fig)
    (out / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved four comparison pairs and probability matrices to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("infer", "plot"))
    p.add_argument("--label", choices=("cross_modal_gated", "cmi_gated"))
    p.add_argument("--checkpoint")
    p.add_argument("--config")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--test_vil_file", default="data/nowcast_testing_vil.h5")
    p.add_argument("--test_lightning_file", default="data/nowcast_testing_lght.h5")
    p.add_argument("--test_meta", default="data/nowcast_testing_META.csv")
    p.add_argument("--num_shards", type=int, default=4)
    p.add_argument("--shard_rank", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ae_chunk_size", type=int)
    p.add_argument("--test_data_percentage", type=float, default=1.)
    p.add_argument("--euler_steps", type=int, default=4)
    p.add_argument("--members", type=int, default=8)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_rank < args.num_shards:
        p.error("Invalid shard settings")
    if min(args.euler_steps, args.members, args.batch_size) < 1 or args.num_workers < 0:
        p.error("Invalid inference settings")
    if not 0 < args.test_data_percentage <= 1:
        p.error("test_data_percentage must be in (0, 1]")
    if args.mode == "infer":
        if not all((args.checkpoint, args.config, args.label)):
            p.error("infer requires --checkpoint, --config, and --label")
        infer(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
