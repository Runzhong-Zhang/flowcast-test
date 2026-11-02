"""Lead-resolved Grad-CAM for the final lightning Euler evaluation."""
import argparse
import csv
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np


def cam_weights(p, weight):
    """Spatially pooled gradient of mean expert probability, for each expert.

    p is (batch, lead, space, expert). For one token the softmax Jacobian
    is diag(p) - p p^T. Divide twice by space: target mean and Grad-CAM pool.
    """
    import torch
    sites = p.shape[-2]
    jac = torch.diag_embed(p.sum(-2)) - p.transpose(-2, -1) @ p
    return (jac @ weight) / (sites * sites)


class CamCollector:
    def __init__(self, lag, lead, *, steps, rank):
        self.lag, self.lead, self.steps, self.rank = lag, lead, steps, rank
        self.calls, self.counts, self.sums, self.examples = {}, {}, {}, {}
        self.handles = []

    def attach(self, system):
        from common.models.flowcast.moe_ffn import CrossModalSparseMoEFFN
        for name, layer in system.denoiser.named_modules():
            if isinstance(layer, CrossModalSparseMoEFFN):
                self.calls[name] = self.counts[name] = 0
                self.handles.append(layer.register_forward_pre_hook(partial(self.observe, name)))
        if len(self.calls) != 4:
            raise ValueError("Expected four MoE layers")

    def observe(self, name, layer, inputs):
        import torch
        self.calls[name] += 1
        if self.calls[name] % self.steps:
            return
        x = inputs[0]
        b, t, h, w, c = x.shape
        if t != self.lag + self.lead or layer._vil_cond is None:
            raise ValueError("Unexpected lead layout or missing VIL condition")
        # Recompute only current-layer future tokens under the runner's no_grad.
        hidden = x[:, self.lag:].reshape(b, self.lead, h*w, c)
        vil = layer._vil_cond.reshape(b, t, h, w, -1)[:, self.lag:]
        vil = layer.vil_proj(vil.reshape(b, self.lead, h*w, -1))
        gate = layer.gate_proj(torch.cat((hidden, vil), -1)).sigmoid()
        injection = gate * layer.vil_value_proj(vil)
        p = layer.router(hidden + injection).float().softmax(-1)
        weights = cam_weights(p, layer.router.weight.float())
        stats = [p.mean(-2).sum(0)]
        example = []
        for branch in (hidden, injection):
            maps = (branch.float() @ weights.transpose(-2, -1)).relu()
            stats.append(maps.mean(-2).sum(0))
            if self.rank == 0 and not self.counts[name]:
                example.append(maps[0].reshape(self.lead, h, w, -1).cpu().numpy())
        if example:
            self.examples[name] = np.stack(example)
        values = torch.stack(stats).double()
        if name not in self.sums:
            self.sums[name] = torch.zeros_like(values)
        self.sums[name] += values
        self.counts[name] += b

    def save(self, path, metadata, expected_calls):
        for handle in self.handles:
            handle.remove()
        count = (metadata["shard_end"] - metadata["shard_start"]) * metadata["members"]
        names = list(self.calls)
        if any(self.calls[n] != expected_calls or self.counts[n] != count for n in names):
            raise ValueError("Unexpected final-step capture count")
        sums = np.stack([self.sums[n].cpu().numpy() for n in names])
        if sums.shape != (4, 3, 12, 8) or not np.isfinite(sums).all():
            raise ValueError("Invalid CAM summaries")
        metadata.update(layer_names=names, example_index=0, example_member=0)
        np.savez_compressed(path, sums=sums, count=count, metadata=json.dumps(metadata),
                            **{f"example_{i}": self.examples[n] for i,n in enumerate(names)
                               if n in self.examples})


def plot(out, num_shards):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    records, examples, count = [], [], 0
    sums = np.zeros((4, 3, 12, 8))
    end = 0
    for rank in range(num_shards):
        with np.load(out / f"cmi_gated_shard_{rank}.npz") as data:
            meta = json.loads(str(data["metadata"]))
            if (meta["shard_rank"] != rank or meta["num_shards"] != num_shards
                    or meta["shard_start"] != end or (records and any(
                        meta[k] != records[0][k] for k in ("config", "checkpoint", "seed",
                        "members", "euler_steps", "layer_names", "selected_total", "batch_size")))):
                raise ValueError("Inconsistent shard metadata")
            end = meta["shard_end"]
            records.append(meta)
            sums += data["sums"]
            count += int(data["count"])
            if rank == 0:
                examples = [data[f"example_{i}"] for i in range(4)]
    if end != meta["selected_total"] or count != end * meta["members"]:
        raise ValueError("Incomplete coverage")
    means = sums / count
    leads = np.arange(1, 13)
    with (out / "cam_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "lead", "expert", "probability", "hidden_cam", "vil_cam"])
        for i, (prob, hidden, vil) in enumerate(means):
            expert = int(prob.mean(0).argmax())
            maps = examples[i][..., expert]
            fig, axes = plt.subplots(2, 12, figsize=(18, 4), constrained_layout=True)
            for branch, row in enumerate(axes):
                for j, ax in enumerate(row):
                    im = ax.imshow(maps[branch,j], vmin=0, vmax=max(float(maps.max()), 1e-30),
                                   cmap="magma", interpolation="nearest")
                    ax.set(xticks=[], yticks=[])
                    if branch == 0:
                        ax.set_title(f"Lead {j+1}")
                row[0].set_ylabel(("Hidden", "Gated VIL")[branch])
            fig.colorbar(im, ax=axes.ravel().tolist(), label="Positive Grad-CAM")
            fig.suptitle(f"Layer {i+1}, expert {expert+1} | test case 0, member 1 | final Euler call")
            fig.savefig(out / f"layer_{i+1}_spatial.png", dpi=160)
            plt.close(fig)
            fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True,
                                     constrained_layout=True)
            for e, ax in enumerate(axes.flat):
                ax.plot(leads, hidden[:,e], label="Hidden")
                ax.plot(leads, vil[:,e], label="Gated VIL")
                ax.set_title(f"Expert {e+1}")
                writer.writerows((i+1,j+1,e+1,prob[j,e],hidden[j,e],vil[j,e]) for j in range(12))
            axes[0,0].legend()
            fig.supxlabel("Forecast lead")
            fig.supylabel("Mean positive Grad-CAM")
            fig.suptitle(f"Layer {i+1} | final Euler call")
            fig.savefig(out / f"layer_{i+1}_summary.png", dpi=160)
            plt.close(fig)
    (out / "metadata.json").write_text(json.dumps(records, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["infer", "plot"])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--shard_rank", type=int, default=0)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--test_data_percentage", type=float, default=1.)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    if args.mode == "plot":
        plot(Path(args.output_dir), args.num_shards)
        return
    if (not args.checkpoint or not 0 <= args.shard_rank < args.num_shards
            or min(args.members, args.batch_size) < 1 or not 0 < args.test_data_percentage <= 1):
        parser.error("Invalid inference settings or missing checkpoint")
    import expert_probability_maps as runner
    args.label = "cmi_gated"
    args.config = "lightning_diffusion_moe_config_server_cmi_gated.yaml"
    args.euler_steps, args.seed, args.num_workers, args.ae_chunk_size = 4, 2026, 4, None
    args.test_meta = "data/nowcast_testing_META.csv"
    args.test_vil_file = "data/nowcast_testing_vil.h5"
    args.test_lightning_file = "data/nowcast_testing_lght.h5"
    # Use the production visualization runner's model loading, seeds and sampling.
    with patch.object(runner, "RoutingAccumulator",
                      partial(CamCollector, steps=args.euler_steps, rank=args.shard_rank)):
        runner.infer(args)


if __name__ == "__main__":
    main()
