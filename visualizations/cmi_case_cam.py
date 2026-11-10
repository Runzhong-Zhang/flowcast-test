"""Capture one paired lightning forecast and its layer-4 CMI routing maps."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np


def capture(args, label, fields):
    import torch
    import expert_probability_maps as runner
    from cmi_router_cam import cam_weights
    from common.models.flowcast.moe_ffn import CrossModalSparseMoEFFN

    def array(value):
        return value.detach().float().cpu().numpy()

    def load_case(dataset, **kwargs):
        if not 0 <= args.sample < len(dataset):
            raise ValueError(f"Sample {args.sample} outside dataset of size {len(dataset)}")
        batch = kwargs["collate_fn"]([dataset[args.sample]])
        vil, past, future, _ = batch
        fields.update(observed_vil=array(vil[0, -1, ..., 0]),
                      observed_lightning=array(past[0, -1, ..., 0]),
                      truth=array(future[0, -1, ..., 0]))
        return [batch]

    solve = runner.odeint

    def forecast(*pos, **kw):
        trajectory = solve(*pos, **kw)
        fields[label] = array(trajectory[-1, -1, ..., 0])
        return trajectory

    class Collector:
        def __init__(self, lag, lead):
            self.calls = 0

        def attach(self, system):
            if label != "cmi_gated":
                return
            layers = [(n, m) for n, m in system.denoiser.named_modules()
                      if isinstance(m, CrossModalSparseMoEFFN)]
            if len(layers) != 4:
                raise ValueError("Expected four MoE layers")
            self.name, layer = layers[-1]
            layer.record_routing = True
            layer.register_forward_hook(self.observe)
            prepare = system.prepare_vil_conditions

            def conditions(*pos, **kw):
                past, future = prepare(*pos, **kw)
                decoded = system.decode_vil_latents(future[:, -1:])
                fields["predicted_vil"] = array(decoded.squeeze())
                return past, future
            system.prepare_vil_conditions = conditions

        def observe(self, layer, inputs, output):
            self.calls += 1
            if self.calls != 4:
                return
            x = inputs[0]
            b, t, h, w, c = x.shape
            hidden = x[:, -1].reshape(b, h*w, c)
            vil = layer.vil_proj(layer._vil_cond.reshape(b,t,h*w,-1)[:, -1])
            gate = layer.gate_proj(torch.cat((hidden, vil), -1)).sigmoid()
            injection = gate * layer.vil_value_proj(vil)
            p = layer.router(hidden + injection).float().softmax(-1)
            weights = cam_weights(p, layer.router.weight.float())
            fields["cam"] = np.stack([array((branch.float() @ weights.transpose(-2,-1))
                                         .relu().reshape(h,w,8)) for branch in (hidden,injection)])
            fields["probability"] = array(p.reshape(h,w,8))
            fields["top2_ids"] = layer._last_topk_idx.reshape(b,t,h,w,2)[0,-1].numpy() + 1
            fields["top2_weights"] = array(layer._last_topk_w.reshape(b,t,h,w,2)[0,-1])

        def save(self, path, metadata, evaluations):
            if evaluations != 4:
                raise ValueError("Expected four Euler evaluations")
            if label == "cmi_gated":
                metadata["layer_name"] = self.name
            args.metadata[label] = {k: metadata[k] for k in ("checkpoint","config","seed","euler_steps")}
            args.metadata[label].update(layer_name=metadata.get("layer_name"))

    options = argparse.Namespace(label=label, config=f"lightning_diffusion_moe_config_server_{label}.yaml",
        checkpoint=args.cmi if label == "cmi_gated" else args.baseline, seed=args.seed,
        members=1, euler_steps=4, batch_size=1, num_workers=0, ae_chunk_size=None,
        num_shards=1, shard_rank=0, test_data_percentage=1, output_dir=args.output,
        test_meta="data/nowcast_testing_META.csv", test_vil_file="data/nowcast_testing_vil.h5",
        test_lightning_file="data/nowcast_testing_lght.h5")
    with patch.object(runner,"DataLoader",load_case), patch.object(runner,"odeint",forecast), \
            patch.object(runner,"RoutingAccumulator",Collector):
        runner.infer(options)


def plot(fields, stem, sample):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    fig = plt.figure(figsize=(15,9), layout="constrained")
    sections = fig.subfigures(3,1,height_ratios=[1,1.5,1])
    def panel(ax, values, title, **kw):
        im = ax.imshow(values, interpolation="nearest", **kw)
        ax.set(title=title, xticks=[], yticks=[])
        return im
    for ax,key,title in zip(sections[0].subplots(1,4), 
            ("observed_vil","observed_lightning","predicted_vil","truth"),
            ("Last observed VIL","Last observed lightning","Forecast VIL condition","Lead-12 lightning truth")):
        panel(ax,fields[key],title,cmap="gray")
    axes = sections[1].subplots(2,8)
    cams = fields["cam"]
    for e in range(8):
        coverage = (fields["top2_ids"] == e+1).any(-1).mean()
        title = f"E{e+1}   p={fields['probability'][...,e].mean():.2f}\nselected {coverage:.0%}"
        for row in range(2):
            im = panel(axes[row,e],cams[row,...,e],title if row==0 else "",
                       cmap="magma",vmin=0,vmax=max(cams.max(),1e-30))
    for ax,name in zip(axes[:,0],("Hidden","Gated VIL")):
        ax.set_ylabel(name)
    sections[1].colorbar(im, ax=axes.ravel().tolist(),label="Positive router CAM")
    bottom = sections[2].subplots(1,4)
    colors = ListedColormap(plt.get_cmap("tab10").colors[:8])
    for rank in range(2):
        im = panel(bottom[rank],fields["top2_ids"][...,rank],f"Expert rank {rank+1}",
                   cmap=colors,norm=BoundaryNorm(np.arange(.5,9),8))
    sections[2].colorbar(im,ax=bottom[:2].tolist(),ticks=range(1,9),label="Expert ID")
    for ax,key,title in zip(bottom[2:],("cross_modal_gated","cmi_gated"),("Baseline forecast","CMI forecast")):
        im = panel(ax,fields[key],title,cmap="gray",vmin=0,vmax=1)
    sections[2].colorbar(im,ax=bottom[2:].tolist(),label="Lightning forecast")
    fig.suptitle(f"Test sample {sample} | Lead 12 | MoE layer 4 | Final Euler evaluation")
    for ext in (".pdf",".png"):
        fig.savefig(stem.with_suffix(ext),dpi=250)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("baseline","cmi","output"):
        parser.add_argument("--"+flag,required=True)
    parser.add_argument("--sample",type=int,default=0)
    parser.add_argument("--seed",type=int,default=2026)
    args = parser.parse_args()
    args.metadata = dict(sample=args.sample, member=1, layer=4, lead=12, evaluation=4)
    fields = {}
    for label in ("cross_modal_gated","cmi_gated"):
        capture(args,label,fields)
    if not all(np.isfinite(v).all() for v in fields.values()):
        raise ValueError("Nonfinite visualization data")
    stem = Path(args.output) / f"case_{args.sample}_lead12_layer4"
    np.savez_compressed(stem.with_suffix(".npz"),**fields)
    stem.with_suffix(".json").write_text(json.dumps(args.metadata,indent=2))
    plot(fields,stem,args.sample)
