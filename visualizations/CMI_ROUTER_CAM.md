# CMI router Grad-CAM

From the server repository root, submit a four-GPU smoke job with
`qsub -v TEST_FRACTION=0.01,MEMBERS=1,BATCH_SIZE=1 submit_cmi_router_cam.sh`.
Use `qsub submit_cmi_router_cam.sh` for the full test set with eight members.
The launcher pins the CMI checkpoint from the October 22 probability-map run.
Override it with `qsub -v CMI_CKPT=/absolute/path/to/checkpoint submit_cmi_router_cam.sh`.

Each run writes eight PNGs, cam_summary.csv, metadata.json, four shard archives
and logs under visualizations/outputs/cmi_cam_<run-id>. Each summary panel
shows one expert's mean positive Grad-CAM for the hidden and gated VIL branches
over 12 forecast leads, averaging spatially within each case and then across
cases and ensemble members. Each spatial figure uses test index zero, first
member, and the layer's expert with highest dataset mean probability across
leads. Both branches and all leads share a color scale within that figure.

Capture occurs on call four of each four-step lightning Euler solve.
The target is the spatial mean pre-top-2 probability for one case, lead and
expert. The additive branches have identical gradients at their sum. The
collector computes these gradients analytically from the linear router and
softmax, pools over space, then projects each branch activation with those
weights and applies ReLU separately. Values measure positive support at the
router. The hidden representation already contains VIL information.
