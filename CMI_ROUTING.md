# CMI-inspired gated routing

This experiment adds MCR-style latent permutation JSD to the existing gated
cross-modal router. It is a routing-sensitivity surrogate, not a calibrated
estimate of conditional mutual information or target relevance.

For each MoE layer, keep the hidden state fixed and compare expert probabilities
for the original VIL sequence and another example's VIL sequence. Minimize:

```text
CFM loss + load_balance_weight * load_balance_loss
         - cmi_weight * mean_router_JSD
```

JSD is computed before top-k in float32 and averaged over tokens and layers.
The auxiliary branch detaches hidden states and raw VIL features, so it updates
only the VIL projection, gate, value projection, and router. Normal task
gradients are unchanged. One random nonzero batch shift is shared across layers
and retained for checkpoint recomputation. Local batches of size one skip JSD.
No extra expert evaluations, parameters, or inference computation are introduced.

The new config matches the existing cross-modal gated experiment except for its
run name and two settings:

- `cmi_weight: 0.01`: conservative starting coefficient; JSD is bounded by ln(2),
  so the auxiliary loss magnitude is at most approximately 0.00693 at full weight.
- `cmi_warmup_epochs: 5`: weights 0.002, 0.004, 0.006, 0.008, then 0.01.
  Resume uses the absolute epoch. Zero warm-up applies the full weight immediately.

Existing configs default to weight zero. Their scripts are unchanged. Checkpoint
parameter names are unchanged, so the existing inference runner loads CMI models
without an auxiliary inference branch. Training logs include `routing_jsd` and
the effective `cmi_weight`; select models by forecast metrics, not JSD alone.

## Server execution

From the same server repository used by the existing PBS scripts:

```bash
mkdir -p logs/cmi_gated/train logs/cmi_gated/test
qsub submit_moe_train_cmi_gated.sh
# After training completes:
qsub submit_moe_test_cmi_gated.sh
```

These scripts retain the existing server paths, conda environment, four-GPU
training, four-shard testing, and metric merging. The test script selects the
newest `best_lightning_diffusion_moe.pt` under a `cmi_gated` run. This name does
not match the original script's `cross_modal_gated` checkpoint filter.

Run focused checks in an environment with the repository dependencies:

```bash
python -m unittest discover -s tests -p test_cmi_routing.py -v
```

Reference: [MCR standalone implementation](https://github.com/kkontras/MCR/blob/main/models/MCR_Model_StandAlone.py),
particularly `_build_shuffled_batches` and `_get_lmipds`. The adaptation uses one
VIL perturbation and omits the symmetric encoder game, contrastive loss, and
reconstruction loss.
