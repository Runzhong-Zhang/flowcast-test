# Expert probability maps

From the server project root, submit:

```bash
qsub submit_moe_expert_probability_maps.sh
```

By default, the launcher chooses the newest matching checkpoint for each experiment.
To pin checkpoints:

```bash
qsub -v CROSS_CKPT=/absolute/path/cross_modal_gated/best_lightning_diffusion_moe.pt,CMI_CKPT=/absolute/path/cmi_gated/best_lightning_diffusion_moe.pt submit_moe_expert_probability_maps.sh
```

Four GPUs process disjoint test shards. Each GPU runs the two models sequentially.
Both use four Euler steps for lightning generation, eight ensemble members, the
same batch boundaries, and matching seeds. The upstream VIL forecaster retains
its configured ten steps and one member for both models. Existing source and
configuration files are not edited. The environment must include matplotlib
in addition to the existing inference dependencies.

Results are written under `visualizations/outputs/<timestamp>_<job-id>/`:

- Four `layer_N_comparison.png` / `.pdf` figures, each containing two 12-by-8 maps.
- Per-model `.npy` arrays of shape `(4, 12, 8)` and per-layer CSV matrices.
- Streaming shard sums/counts in `.npz` files and resolved run metadata in JSON.
- Submission and per-shard logs.

Rows are future leads 1–12; columns are experts 1–8. Color is the mean full
softmax probability **before** top-2 selection and renormalization. Historical
slots are excluded. Probabilities are averaged over test samples, spatial
locations, ensemble members, and actual ODE evaluations. Shards are combined
using token counts, not an unweighted average of shard means. All four figure
pairs share one color scale. No raw per-token tensors or lightning forecasts
are saved; this keeps memory and storage bounded by batch size.

Expert identities are independent between layers and trained models. Compare
patterns of lead dependence, not the semantic identity of equal-numbered
experts. These maps describe routing preference, not top-2 selection frequency
or causal influence of VIL.

For a small server smoke run, invoke the Python `infer` command twice with
`--num_shards 1 --test_data_percentage 0.001 --members 1` and each label/config/
checkpoint, using the same output directory; then run `plot --num_shards 1`.
Use a fresh output directory for every analysis. All CLI options are available
with `python visualizations/expert_probability_maps.py --help`.
