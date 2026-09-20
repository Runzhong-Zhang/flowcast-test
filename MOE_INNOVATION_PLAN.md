# FlowCast MoE Innovation Plan

See the full document in the conversation history or regenerate from the detailed discussion.

## Current Experiment: Standard Token-level Sparse MoE (Step 1)

- Strategy: up_all (24 MoE layers in Up-path decoder)
- Experts: 8, top_k: 2
- Baseline: FC Cuboid (BSS=0.255, Frame CSI=0.300)
- Training: 50 epochs, precomputed latents, all params aligned with baseline
