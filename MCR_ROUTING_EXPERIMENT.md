# Gate-side greedy MCR experiment

Launch: `qsub submit_moe_train_mcr_greedy_gated.sh`.
Evaluate: `qsub submit_moe_test_mcr_greedy_gated.sh`.
Only the dedicated MCR runner, helper, and config implement these changes.
The baseline source, configuration, architecture, checkpoint keys and inference
path are unchanged. Other settings match the current cross-modal gated config.

## MIPD adaptation

At each MoE layer, compare full expert softmax distributions for original,
VIL-permuted, and H-permuted router inputs. Compute symmetric categorical JSD
in float32 log space before top-k selection, average over tokens, then average
equally over layers. JSD uses natural logs and lies between zero and log(2).
A whole-event cyclic batch permutation is shared across layers and positions.
All comparisons use the same original layer context; auxiliary counterfactuals
do not alter dispatch or propagate into later layers. Hidden inputs and raw VIL
conditioning are detached before local routing projections.

This adapts the JSD construction in https://arxiv.org/pdf/2411.07335 to expert
routing distributions. The paper compares task predictions; this experiment
measures routing sensitivity and is not a task-output CMI estimator. Training
interpolation contains target information, so sensitivity is conditional on
that state and time. No velocity surrogate or velocity scale is used.

Greedy k=-1 applies separate objectives to existing parameter groups:

- VIL (`vil_proj`, `vil_value_proj`): `weight*(-D_V + D_H)`.
- H (`gate_proj`, `router`): `weight*(-D_H + D_V)`.

Differentiate each objective against its own group. Summing the scalar objectives
would cancel them. H already contains mixed context and the gate/router see both
inputs, so these are routing branches, not independent modality encoders.

## CON

Pool the H-only block of the existing gate_proj and
vil_value_proj(vil_proj(z)), normalize, and use symmetric paired-instance InfoNCE.
No projection head or parameter is added. H is detached before projection.
CON updates existing gate-side projections in both groups, never transformer
parameters. Matching events are positives; other local batch events are negatives.
This is instance contrastive, not the paper's supervised class contrastive loss;
similar storms can be false negatives.

## Defaults and execution

Both weights are fixed at 0.001 from the first training batch. Auxiliary updates
run every batch, with no task-only warmup or ramp, matching the authors'
standalone scheduling design. Microbatch 8
matches training and supplies seven CON negatives per event. One auxiliary
model forward collects all local routing comparisons; reduce microbatch if needed
(minimum 2). Temperature is 0.1.

Task CFM and load-balance backward is unchanged. Auxiliary passes temporarily
disable dropout/checkpoint recomputation and freeze non-routing parameters.
Auxiliary gradients are explicitly averaged across DDP ranks and added in scaled
units before unscaling and original global gradient clipping. Modes, parameter
flags, routing methods, and condition caches are restored. Task transformer
gradients remain intact; global clipping may still rescale the combined update.

Logs include routing_jsd_vil, routing_jsd_hidden, CON, ramp factor, auxiliary/task routing
gradient norms and cosine similarity. train_total remains CFM + load balance:
the greedy game has no single scalar total.

Run focused CPU checks:
`python -m unittest discover -s tests -p test_mcr_routing_experiment.py -v`.

Compare baseline and experiment using identical evaluation sampling and seeds.
Small weights do not guarantee comparable Brier/CSI. No cluster result is claimed.
