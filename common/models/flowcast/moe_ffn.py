"""
Sparse Mixture-of-Experts FFN for CuboidTransformer-based models.

Provides
--------
SparseMoEFFN
    Drop-in replacement for PositionwiseFFN with identical forward signature
    ``forward(data: Tensor[B, S, C]) -> Tensor[B, S, C]``.
patch_ffn_with_moe_selective
    Selectively replaces PositionwiseFFN layers in a CuboidTransformerUNet
    according to a physics-motivated strategy.  Default strategy ``"up_l0_t"``
    replaces only the T-axis FFN in the high-resolution decoder — the layer
    where convective vs. clear-sky specialisation matters most.
patch_ffn_with_moe
    Legacy recursive replacement (all main FFNs).  Kept for backward
    compatibility; prefer ``patch_ffn_with_moe_selective`` for new work.
collect_moe_aux_loss
    Sums the differentiable load-balance auxiliary losses across all
    SparseMoEFFN layers in a model.  Add the result to the main loss:
        total_loss = cfm_loss + load_balance_weight * collect_moe_aux_loss(model)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cuboid_transformer import PositionwiseFFN


class SparseMoEFFN(nn.Module):
    """
    Top-k sparse Mixture-of-Experts feed-forward layer.

    Drop-in replacement for PositionwiseFFN with identical interface::

        forward(data: Tensor[*leading, C]) -> Tensor[*leading, C]

    Supports arbitrary leading dimensions (e.g. ``(B, S, C)`` or ``(B, T, H, W, C)``).

    Each token is independently routed to ``top_k`` out of ``num_experts``
    experts via a learned linear router.  Expert outputs are combined with a
    re-normalised weighted sum.

    Because each expert (PositionwiseFFN) already applies its own residual
    connection, the weighted sum is mathematically equivalent to one shared
    residual applied to the mixture of FFN transformations:

        Σ_k w_k · (x + FFN_k(norm_k(x)))
        = x · Σ w_k  +  Σ_k w_k · FFN_k(norm_k(x))
        = x           +  Σ_k w_k · FFN_k(norm_k(x))   [Σ w_k = 1 by construction]

    Auxiliary load-balance loss
    ---------------------------
    During training the Switch-Transformer load-balance auxiliary loss is
    computed and stored in ``self._aux_loss_for_bp`` (differentiable) and
    ``self.last_aux_loss`` (float, for monitoring).  Collect the loss across
    all layers with ``collect_moe_aux_loss`` and add to the training objective.

    Parameters
    ----------
    units, hidden_size, activation_dropout, dropout, gated_proj,
    activation, normalization, layer_norm_eps, pre_norm,
    linear_init_mode, ffn2_linear_init_mode, norm_init_mode
        Forwarded verbatim to each PositionwiseFFN expert.  Keep these
        identical to the PositionwiseFFN being replaced.
    num_experts : int
        Total number of expert FFNs.
    top_k : int
        Number of experts each token is routed to (1 ≤ top_k ≤ num_experts).
    """

    def __init__(
        self,
        units: int = 512,
        hidden_size: int = 2048,
        activation_dropout: float = 0.0,
        dropout: float = 0.1,
        gated_proj: bool = False,
        activation: str = "relu",
        normalization: str = "layer_norm",
        layer_norm_eps: float = 1e-5,
        pre_norm: bool = False,
        linear_init_mode: str = "0",
        ffn2_linear_init_mode: str = "2",
        norm_init_mode: str = "0",
        num_experts: int = 8,
        top_k: int = 2,
        shared_expert: bool = False,
    ):
        super().__init__()
        if not (1 <= top_k <= num_experts):
            raise ValueError(
                f"top_k={top_k} must satisfy 1 ≤ top_k ≤ num_experts={num_experts}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.units = units
        self.shared_expert_enabled = shared_expert

        # Common FFN kwargs for creating expert instances
        _ffn_kwargs = dict(
            units=units, hidden_size=hidden_size,
            activation_dropout=activation_dropout, dropout=dropout,
            gated_proj=gated_proj, activation=activation,
            normalization=normalization, layer_norm_eps=layer_norm_eps,
            pre_norm=pre_norm, linear_init_mode=linear_init_mode,
            ffn2_linear_init_mode=ffn2_linear_init_mode,
            norm_init_mode=norm_init_mode,
        )

        if shared_expert:
            # 1 shared (always active) + (E-1) routed
            self.shared_expert_ffn = PositionwiseFFN(**_ffn_kwargs)
            self.num_routed = num_experts - 1
            self.experts = nn.ModuleList([
                PositionwiseFFN(**_ffn_kwargs)
                for _ in range(self.num_routed)
            ])
            self.router = nn.Linear(units, self.num_routed, bias=False)
        else:
            # All E experts are routed (current behavior)
            self.shared_expert_ffn = None
            self.num_routed = num_experts
            self.experts = nn.ModuleList([
                PositionwiseFFN(**_ffn_kwargs)
                for _ in range(num_experts)
            ])
            self.router = nn.Linear(units, num_experts, bias=False)

        # Monitoring (non-differentiable scalar, updated every training forward)
        self.last_aux_loss: float = 0.0
        # Differentiable handle for back-propagation (None during eval)
        self._aux_loss_for_bp: Optional[torch.Tensor] = None

        # Routing recording (off by default; enable for visualization)
        self.record_routing: bool = False
        self._last_topk_idx: Optional[torch.Tensor] = None
        self._last_topk_w: Optional[torch.Tensor] = None
        self._last_input_shape: Optional[tuple] = None

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        # Small noise encourages expert exploration early in training
        nn.init.normal_(self.router.weight, std=0.02)
        for expert in self.experts:
            expert.reset_parameters()
        if self.shared_expert_ffn is not None:
            self.shared_expert_ffn.reset_parameters()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        data : torch.Tensor
            Arbitrary shape ``(*leading, C)`` — identical to PositionwiseFFN
            which supports any leading dimensions via nn.Linear on the last dim.
            Typical shapes: ``(B, S, C)`` or ``(B, T, H, W, C)``.

        Returns
        -------
        torch.Tensor
            Same shape as input.
        """
        orig_shape = data.shape
        C = orig_shape[-1]
        N = data.numel() // C

        # Flatten all leading dims into one token axis
        flat = data.reshape(N, C)                               # (N, C)

        # ── Shared expert (processes ALL tokens, no routing) ────────────
        if self.shared_expert_ffn is not None:
            # shared_expert_ffn is a PositionwiseFFN which includes residual:
            #   shared_out = x + FFN_shared(norm(x))
            shared_out = self.shared_expert_ffn(
                flat.unsqueeze(0)
            ).squeeze(0).float()                                # (N, C) fp32
        else:
            shared_out = None

        # ── Router (over routed experts only) ───────────────────────────
        logits = self.router(flat)                              # (N, num_routed)
        probs  = F.softmax(logits, dim=-1)                      # (N, num_routed)

        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # (N, k)
        # Re-normalise so weights within a token sum to 1
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # ── Record routing decisions (for visualization) ────────────────
        if self.record_routing:
            self._last_topk_idx = topk_idx.detach().cpu()    # (N, k)
            self._last_topk_w = topk_w.detach().cpu()        # (N, k)
            self._last_input_shape = orig_shape               # (*leading, C)

        # ── Auxiliary load-balance loss (training only) ──────────────────
        if self.training:
            # f_i  : fraction of tokens dispatched to routed expert i
            topk_onehot = F.one_hot(topk_idx, self.num_routed).float()  # (N, k, num_routed)
            f = topk_onehot.mean(dim=[0, 1])   # (num_routed,)

            # P_i  : mean router probability for routed expert i
            P = probs.mean(dim=0)              # (num_routed,)

            # Switch-Transformer formulation (over routed experts only)
            aux = self.num_routed * (f.detach() * P).sum()
            self._aux_loss_for_bp = aux
            self.last_aux_loss    = aux.item()
        else:
            self._aux_loss_for_bp = None

        # ── Vectorized routed expert dispatch & weighted aggregation ─────
        dispatch_weights = torch.zeros(
            N, self.num_routed, dtype=topk_w.dtype, device=flat.device
        )
        dispatch_weights.scatter_add_(1, topk_idx, topk_w)     # (N, num_routed)

        Nk = N * self.top_k
        token_ids  = torch.arange(N, device=flat.device).unsqueeze(1).expand(
            -1, self.top_k
        ).reshape(Nk)                                           # (N*k,)
        expert_ids = topk_idx.reshape(Nk)                       # (N*k,)

        sort_idx     = expert_ids.argsort(stable=True)
        sorted_eids  = expert_ids[sort_idx]
        sorted_tids  = token_ids[sort_idx]

        counts  = torch.bincount(sorted_eids, minlength=self.num_routed)
        offsets = torch.zeros(
            self.num_routed + 1, dtype=torch.long, device=flat.device
        )
        torch.cumsum(counts, dim=0, out=offsets[1:])

        # Use float32 for accumulation to avoid precision loss under AMP
        routed_output = torch.zeros(N, C, dtype=torch.float32, device=flat.device)

        for e_idx in range(self.num_routed):
            start = offsets[e_idx].item()
            end   = offsets[e_idx + 1].item()
            if start == end:
                continue

            tids_e   = sorted_tids[start:end]                   # (n_e,)
            tokens_e = flat[tids_e]                              # (n_e, C) contiguous
            w_e      = dispatch_weights[tids_e, e_idx]           # (n_e,)

            # PositionwiseFFN expects (*leading, C)
            out_e = self.experts[e_idx](tokens_e.unsqueeze(0)).squeeze(0)  # (n_e, C)

            # Weighted scatter-add back to output
            routed_output.index_add_(0, tids_e, w_e.unsqueeze(-1).float() * out_e.float())

        # ── Combine shared + routed ─────────────────────────────────────
        if shared_out is not None:
            # Both shared_out and routed_output contain residual x:
            #   shared_out  = x + FFN_shared(norm(x))
            #   routed_output = Σ w_k * (x + FFN_k(norm(x))) = x + Σ w_k * FFN_k(norm(x))
            # Adding them gives 2x + ..., so subtract one copy of x.
            flat_f32 = flat.float()
            output = shared_out + (routed_output - flat_f32)
        else:
            output = routed_output

        return output.to(data.dtype).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Cross-Modal MoE FFN (VIL-conditioned routing)
# ---------------------------------------------------------------------------

class CrossModalSparseMoEFFN(SparseMoEFFN):
    """
    SparseMoEFFN with cross-modal routing conditioned on VIL features.

    The router input is ``cat([x_token, vil_proj], dim=-1)`` instead of
    just ``x_token``.  The VIL condition is injected externally via the
    ``_vil_cond`` attribute before each forward pass — no changes needed
    to CuboidTransformerUNet or StackCuboidSelfAttentionBlock interfaces.

    Injection protocol (handled by LightningDiffusionMoESystem)::

        # Before each denoiser forward:
        for layer in cross_modal_moe_layers:
            layer._vil_cond = vil_cond_at_this_resolution  # (*leading, C_vil)

        # During forward, CrossModalSparseMoEFFN reads self._vil_cond
        # and concatenates it with the token for routing.
        # After forward, _vil_cond is cleared to avoid stale references.

    Falls back to standard token-only routing when ``_vil_cond is None``
    (e.g. during standalone testing or when cross-modal is disabled).

    Parameters
    ----------
    vil_cond_dim : int
        Dimension of the projected VIL condition vector.
    router_type : str
        ``"concat"`` — cat([token, vil_proj]) -> Linear (default).
        ``"gated"``  — token + sigmoid(gate(token,vil)) * value(vil) -> Linear.
        Gated fusion allows the router to dynamically decide how much
        VIL information to use, conditioned on the token content.
    All other parameters
        Same as SparseMoEFFN.
    """

    def __init__(
        self,
        *args,
        vil_cond_dim: int = 128,
        router_type: str = "concat",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vil_cond_dim = vil_cond_dim
        self.router_type = router_type

        if router_type == "concat":
            # cat([token, vil_proj]) -> Linear -> scores
            self.router = nn.Linear(
                self.units + vil_cond_dim, self.num_routed, bias=False
            )
        elif router_type == "gated":
            # Gated fusion: token + gate * value(vil) -> Linear -> scores
            self.gate_proj = nn.Linear(self.units + vil_cond_dim, self.units)
            self.vil_value_proj = nn.Linear(vil_cond_dim, self.units)
            self.router = nn.Linear(self.units, self.num_routed, bias=False)
        else:
            raise ValueError(
                f"Unknown router_type={router_type!r}. Use 'concat' or 'gated'."
            )

        # Projection: reduce VIL raw channels to vil_cond_dim
        # Actual input dim set later by set_vil_proj() after we know C_z
        self.vil_proj: Optional[nn.Module] = None

        # Attribute for VIL condition injection (set externally before forward)
        self._vil_cond: Optional[torch.Tensor] = None

        self._reset_cross_modal_params()

    def _reset_cross_modal_params(self) -> None:
        nn.init.normal_(self.router.weight, std=0.02)
        if self.router_type == "gated":
            nn.init.normal_(self.gate_proj.weight, std=0.02)
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.normal_(self.vil_value_proj.weight, std=0.02)
            nn.init.zeros_(self.vil_value_proj.bias)

    def set_vil_proj(self, vil_raw_dim: int) -> None:
        """
        Initialize the VIL projection layer once we know the raw VIL
        channel dimension (C_z, typically 4 for the VAE latent).

        Called by the system during model construction.
        """
        self.vil_proj = nn.Sequential(
            nn.Linear(vil_raw_dim, self.vil_cond_dim),
            nn.SiLU(),
        )
        # Re-init
        for m in self.vil_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        orig_shape = data.shape
        C = orig_shape[-1]
        N = data.numel() // C

        flat = data.reshape(N, C)                               # (N, C)

        # ── Cross-modal router input ────────────────────────────────────
        # Note: do NOT clear _vil_cond — gradient checkpointing re-runs
        # forward during backward and needs the same _vil_cond.
        vil_cond = self._vil_cond
        has_vil = vil_cond is not None and self.vil_proj is not None

        if has_vil:
            vil_flat = vil_cond.reshape(N, -1)                  # (N, C_vil_raw)
            vil_proj = self.vil_proj(vil_flat)                  # (N, vil_cond_dim)
        else:
            vil_proj = torch.zeros(
                N, self.vil_cond_dim, device=flat.device, dtype=flat.dtype
            )

        if self.router_type == "concat":
            router_input = torch.cat([flat, vil_proj], dim=-1)  # (N, C + vil_cond_dim)
            logits = self.router(router_input)                  # (N, E)
        elif self.router_type == "gated":
            gate = torch.sigmoid(
                self.gate_proj(torch.cat([flat, vil_proj], dim=-1))
            )                                                   # (N, C)
            value = self.vil_value_proj(vil_proj)               # (N, C)
            router_input = flat + gate * value                  # (N, C)
            logits = self.router(router_input)                  # (N, E)
        probs  = F.softmax(logits, dim=-1)                      # (N, E)

        topk_w, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)

        # ── Record routing decisions (for visualization) ────────────────
        if self.record_routing:
            self._last_topk_idx = topk_idx.detach().cpu()
            self._last_topk_w = topk_w.detach().cpu()
            self._last_input_shape = orig_shape

        # ── Shared expert (processes ALL tokens, no routing) ────────────
        if self.shared_expert_ffn is not None:
            shared_out = self.shared_expert_ffn(
                flat.unsqueeze(0)
            ).squeeze(0).float()
        else:
            shared_out = None

        # ── Auxiliary load-balance loss (training only) ──────────────────
        if self.training:
            topk_onehot = F.one_hot(topk_idx, self.num_routed).float()
            f = topk_onehot.mean(dim=[0, 1])
            P = probs.mean(dim=0)
            aux = self.num_routed * (f.detach() * P).sum()
            self._aux_loss_for_bp = aux
            self.last_aux_loss    = aux.item()
        else:
            self._aux_loss_for_bp = None

        # ── Routed expert dispatch ──────────────────────────────────────
        dispatch_weights = torch.zeros(
            N, self.num_routed, dtype=topk_w.dtype, device=flat.device
        )
        dispatch_weights.scatter_add_(1, topk_idx, topk_w)

        Nk = N * self.top_k
        token_ids = torch.arange(N, device=flat.device).unsqueeze(1).expand(
            -1, self.top_k
        ).reshape(Nk)
        expert_ids = topk_idx.reshape(Nk)

        sort_idx     = expert_ids.argsort(stable=True)
        sorted_eids  = expert_ids[sort_idx]
        sorted_tids  = token_ids[sort_idx]

        counts  = torch.bincount(sorted_eids, minlength=self.num_routed)
        offsets = torch.zeros(
            self.num_routed + 1, dtype=torch.long, device=flat.device
        )
        torch.cumsum(counts, dim=0, out=offsets[1:])

        routed_output = torch.zeros(N, C, dtype=torch.float32, device=flat.device)

        for e_idx in range(self.num_routed):
            start = offsets[e_idx].item()
            end   = offsets[e_idx + 1].item()
            if start == end:
                continue
            tids_e   = sorted_tids[start:end]
            tokens_e = flat[tids_e]
            w_e      = dispatch_weights[tids_e, e_idx]
            out_e = self.experts[e_idx](tokens_e.unsqueeze(0)).squeeze(0)
            routed_output.index_add_(0, tids_e, w_e.unsqueeze(-1).float() * out_e.float())

        # ── Combine shared + routed ─────────────────────────────────────
        if shared_out is not None:
            flat_f32 = flat.float()
            output = shared_out + (routed_output - flat_f32)
        else:
            output = routed_output

        return output.to(data.dtype).reshape(orig_shape)


# ---------------------------------------------------------------------------
# Helper: replace a single PositionwiseFFN with SparseMoEFFN
# ---------------------------------------------------------------------------

def _make_moe_from_ffn(
    ffn: PositionwiseFFN,
    num_experts: int,
    top_k: int,
    cross_modal: bool = False,
    vil_cond_dim: int = 128,
    router_type: str = "concat",
    shared_expert: bool = False,
) -> SparseMoEFFN:
    """Create a SparseMoEFFN (or CrossModalSparseMoEFFN) that mirrors
    *ffn*'s hyper-parameters."""
    cls = CrossModalSparseMoEFFN if cross_modal else SparseMoEFFN
    kwargs = dict(
        **ffn._kwargs,
        linear_init_mode=ffn.linear_init_mode,
        ffn2_linear_init_mode=ffn.ffn2_linear_init_mode,
        norm_init_mode=ffn.norm_init_mode,
        num_experts=num_experts,
        top_k=top_k,
        shared_expert=shared_expert,
    )
    if cross_modal:
        kwargs["vil_cond_dim"] = vil_cond_dim
        kwargs["router_type"] = router_type
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Selective patching (recommended)
# ---------------------------------------------------------------------------

# Valid strategy names — used for input validation and documentation.
VALID_STRATEGIES = (
    "up_l0_t", "both_l0_t", "up_t", "up_all", "all", "dense_wider",
)


def patch_ffn_with_moe_selective(
    unet: nn.Module,
    strategy: str = "up_l0_t",
    num_experts: int = 8,
    top_k: int = 2,
    cross_modal: bool = False,
    vil_cond_dim: int = 128,
    router_type: str = "concat",
    shared_expert: bool = False,
) -> int:
    """
    Selectively replace PositionwiseFFN → SparseMoEFFN in a CuboidTransformerUNet.

    Replacement is **in-place**.  Only **main FFN layers** (``ffn_l``) are
    considered; global-vector FFN layers (``global_ffn_l``) are always skipped
    because they process only ``num_global_vectors`` (typically 8) tokens,
    which is too few for meaningful MoE routing.

    Strategies (physics-motivated, from most targeted to broadest)
    ---------------------------------------------------------------
    ``"up_l0_t"`` — **Strategy A (default, recommended)**
        Replace the T-axis FFN (``ffn_l[0]``) in Up-path Level-0 blocks only.
        These 4 layers sit in the high-resolution decoder where the model
        makes final per-pixel predictions.  After temporal attention, each
        spatial location carries a rich temporal-evolution representation;
        convective locations (with lightning) and clear-sky locations require
        fundamentally different transformations.
        → 4 MoE layers, ~6.5 M additional params (E=8).

    ``"up_t"`` — **Strategy B**
        Replace the T-axis FFN in Up-path blocks at *all* levels.
        Adds Level-1 (24×24, coarse) T-axis FFN for large-scale convective
        system routing on top of Strategy A.
        → 8 MoE layers.

    ``"up_all"`` — **Strategy C**
        Replace *all* main FFN (T+H+W axes) in the entire Up-path.
        Full decoder MoE.
        → 24 MoE layers.

    ``"all"`` — **Strategy D (ablation baseline)**
        Replace all main FFN in both Down-path and Up-path.
        Not recommended for production; useful as an ablation row to show
        that targeted placement outperforms blanket replacement.
        → 48 MoE layers.

    ``"dense_wider"`` — **Dense-Wider control**
        No MoE.  Instead, replace the same 4 FFN layers as ``"up_l0_t"``
        with wider ``PositionwiseFFN`` (hidden_size=4096, up from 512).
        Parameter increase (~3.68 M) matches ``"up_l0_t"`` with E=8 K=2
        (~3.70 M) to within 99.6 %.  Used to verify that MoE gains come
        from expert specialisation, not simply from extra parameters.
        → 0 MoE layers, 4 wider FFN layers.

    Parameters
    ----------
    unet : nn.Module
        A ``CuboidTransformerUNet`` instance (``denoiser.model``).
        Must have ``up_self_blocks`` and ``down_self_blocks`` attributes.
    strategy : str
        One of ``"up_l0_t"``, ``"up_t"``, ``"up_all"``, ``"all"``,
        ``"dense_wider"``.
    num_experts : int
        Number of experts per MoE layer (ignored for ``"dense_wider"``).
    top_k : int
        Number of experts each token is routed to (ignored for
        ``"dense_wider"``).

    Returns
    -------
    int
        Number of PositionwiseFFN layers that were replaced.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown MoE strategy {strategy!r}. "
            f"Valid options: {VALID_STRATEGIES}"
        )

    if not hasattr(unet, "up_self_blocks"):
        raise TypeError(
            "Expected a CuboidTransformerUNet with 'up_self_blocks', "
            f"got {type(unet).__name__}"
        )

    replaced = 0

    def _patch_block_ffn(block: nn.Module, axis_indices: list[int]) -> int:
        """Replace ffn_l[i] for each i in *axis_indices* within one
        StackCuboidSelfAttentionBlock.  Returns count of replacements."""
        count = 0
        for idx in axis_indices:
            old_ffn = block.ffn_l[idx]
            if isinstance(old_ffn, PositionwiseFFN):
                block.ffn_l[idx] = _make_moe_from_ffn(
                    old_ffn, num_experts, top_k,
                    cross_modal=cross_modal, vil_cond_dim=vil_cond_dim,
                    router_type=router_type, shared_expert=shared_expert,
                )
                count += 1
        return count

    # ------------------------------------------------------------------
    # Determine which blocks and which axis-FFN indices to patch
    # ------------------------------------------------------------------

    # Axial pattern produces 3 attention+FFN pairs per block:
    #   ffn_l[0] → after Attn-T (temporal)
    #   ffn_l[1] → after Attn-H (horizontal spatial)
    #   ffn_l[2] → after Attn-W (vertical spatial)
    T_AXIS = [0]
    ALL_AXES = [0, 1, 2]

    num_levels = len(unet.up_self_blocks)

    if strategy == "dense_wider":
        # Dense-Wider control: replace the same 4 T-axis FFN as up_l0_t
        # with a wider PositionwiseFFN (hidden_size=4096) instead of MoE.
        # Parameter increase matches up_l0_t E=8 K=2 to within 99.6%.
        _DENSE_WIDER_HIDDEN = 4096
        for block in unet.up_self_blocks[0]:
            old_ffn = block.ffn_l[0]
            if isinstance(old_ffn, PositionwiseFFN):
                block.ffn_l[0] = PositionwiseFFN(
                    units=old_ffn._kwargs["units"],
                    hidden_size=_DENSE_WIDER_HIDDEN,
                    activation_dropout=old_ffn._kwargs["activation_dropout"],
                    dropout=old_ffn._kwargs["dropout"],
                    gated_proj=old_ffn._kwargs["gated_proj"],
                    activation=old_ffn._kwargs["activation"],
                    normalization=old_ffn._kwargs["normalization"],
                    layer_norm_eps=old_ffn._kwargs["layer_norm_eps"],
                    pre_norm=old_ffn._kwargs["pre_norm"],
                    linear_init_mode=old_ffn.linear_init_mode,
                    ffn2_linear_init_mode=old_ffn.ffn2_linear_init_mode,
                    norm_init_mode=old_ffn.norm_init_mode,
                )
                replaced += 1
        return replaced

    if strategy == "up_l0_t":
        # Strategy A: Up-path, Level 0, T-axis only (4 layers)
        for block in unet.up_self_blocks[0]:
            replaced += _patch_block_ffn(block, T_AXIS)

    elif strategy == "both_l0_t":
        # Strategy A2: Down+Up path, Level 0, T-axis only (8 layers)
        for block in unet.down_self_blocks[0]:
            replaced += _patch_block_ffn(block, T_AXIS)
        for block in unet.up_self_blocks[0]:
            replaced += _patch_block_ffn(block, T_AXIS)

    elif strategy == "up_t":
        # Strategy B: Up-path, all levels, T-axis only
        for level_blocks in unet.up_self_blocks:
            for block in level_blocks:
                replaced += _patch_block_ffn(block, T_AXIS)

    elif strategy == "up_all":
        # Strategy C: Up-path, all levels, all axes
        for level_blocks in unet.up_self_blocks:
            for block in level_blocks:
                replaced += _patch_block_ffn(block, ALL_AXES)

    elif strategy == "all":
        # Strategy D: Both paths, all levels, all axes (main FFN only)
        for level_blocks in unet.down_self_blocks:
            for block in level_blocks:
                replaced += _patch_block_ffn(block, ALL_AXES)
        for level_blocks in unet.up_self_blocks:
            for block in level_blocks:
                replaced += _patch_block_ffn(block, ALL_AXES)

    return replaced


def init_cross_modal_vil_proj(model: nn.Module, vil_raw_dim: int) -> int:
    """
    Initialize the VIL projection layer in all CrossModalSparseMoEFFN layers.

    Must be called after ``patch_ffn_with_moe_selective(cross_modal=True)``
    and before the first forward pass.

    Parameters
    ----------
    model : nn.Module
        The model containing CrossModalSparseMoEFFN layers.
    vil_raw_dim : int
        Raw VIL latent channel dimension (typically 4 for VAE latent).

    Returns
    -------
    int
        Number of layers initialized.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, CrossModalSparseMoEFFN):
            module.set_vil_proj(vil_raw_dim)
            count += 1
    return count


def inject_vil_cond(model: nn.Module, vil_cond: torch.Tensor) -> int:
    """
    Inject VIL condition into all CrossModalSparseMoEFFN layers.

    Call this **before each forward pass** of the denoiser.  The VIL
    condition must have the same spatial shape as the data that will
    flow through the MoE layers.

    Parameters
    ----------
    model : nn.Module
        The model containing CrossModalSparseMoEFFN layers.
    vil_cond : torch.Tensor
        VIL condition tensor, shape ``(B, T, H, W, C_vil)``.
        Must match the spatial resolution of the MoE layer's input.

    Returns
    -------
    int
        Number of layers injected.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, CrossModalSparseMoEFFN):
            module._vil_cond = vil_cond
            count += 1
    return count


# ---------------------------------------------------------------------------
# Legacy recursive patching (backward compatibility)
# ---------------------------------------------------------------------------

def patch_ffn_with_moe(
    model: nn.Module,
    num_experts: int = 8,
    top_k: int = 2,
) -> nn.Module:
    """
    Recursively replace every ``PositionwiseFFN`` in *model* with a
    ``SparseMoEFFN``.

    .. deprecated::
        Use :func:`patch_ffn_with_moe_selective` instead.  This function
        replaces **all** PositionwiseFFN layers including ``global_ffn_l``
        (which operate on only ~8 global-vector tokens and are ill-suited
        for MoE routing).

    The replacement is performed **in-place**; the same model object is
    returned for convenience.
    """
    for attr_name, child in list(model.named_children()):
        if isinstance(child, PositionwiseFFN):
            moe = _make_moe_from_ffn(child, num_experts, top_k)
            setattr(model, attr_name, moe)
        else:
            patch_ffn_with_moe(child, num_experts=num_experts, top_k=top_k)
    return model


# ---------------------------------------------------------------------------
# Loss collection
# ---------------------------------------------------------------------------

def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    """
    Sum the differentiable auxiliary load-balance losses from every
    ``SparseMoEFFN`` layer in *model*.

    Add the result to the training objective::

        total_loss = cfm_loss + load_balance_weight * collect_moe_aux_loss(system)

    Returns a zero tensor **on the same device as the model** if no
    SparseMoEFFN layers are found or during evaluation.

    Parameters
    ----------
    model : nn.Module
        The model (or sub-module) to search.

    Returns
    -------
    torch.Tensor
        Scalar tensor, part of the computation graph during training.
    """
    total: Optional[torch.Tensor] = None
    for module in model.modules():
        if isinstance(module, SparseMoEFFN) and module._aux_loss_for_bp is not None:
            if total is None:
                total = module._aux_loss_for_bp
            else:
                total = total + module._aux_loss_for_bp
    if total is not None:
        return total
    # Return a zero tensor on the correct device
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    return torch.tensor(0.0, device=device)


def get_moe_layer_stats(model: nn.Module) -> dict[str, float]:
    """
    Return a ``{layer_name: last_aux_loss}`` dict for all SparseMoEFFN layers.
    Useful for monitoring expert utilisation during training.
    """
    return {
        name: module.last_aux_loss
        for name, module in model.named_modules()
        if isinstance(module, SparseMoEFFN)
    }
