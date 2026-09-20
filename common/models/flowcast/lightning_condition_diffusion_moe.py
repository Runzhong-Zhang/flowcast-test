"""
FC Cuboid MoE variant of the lightning diffusion system.

This module adds a Sparse Mixture-of-Experts (MoE) FFN layer to the
CuboidTransformerUNet backbone used by FC Cuboid, without touching any
existing source files.

New classes
-----------
LightningConditionedDenoiserCuboidMoE
    Subclass of LightningConditionedDenoiserCuboid.  After the parent
    __init__ builds the full CuboidTransformerUNet, selected PositionwiseFFN
    layers are replaced in-place by SparseMoEFFN via
    patch_ffn_with_moe_selective.

LightningDiffusionMoESystem
    Subclass of LightningDiffusionSystem.  Forces denoiser_backbone="cuboid"
    then patches the resulting denoiser with SparseMoEFFN.  Exposes
    collect_moe_aux_loss() so the training loop can add the load-balance
    penalty to the CFM loss.

Cross-modal routing
-------------------
When ``cross_modal_routing=True``, MoE layers use CrossModalSparseMoEFFN
instead of SparseMoEFFN.  The router input becomes
``cat([x_token, vil_proj], dim=-1)``, where ``vil_proj`` is a learned
projection of the VIL latent at the same spatial position.

VIL conditions are injected via module attributes before each forward
pass — no changes needed to CuboidTransformerUNet or
StackCuboidSelfAttentionBlock interfaces.

Design principles
-----------------
* Zero modifications to existing files.
* Data pipeline, VIL encoder, and FrozenVILFlowCastForecaster are reused
  unchanged from LightningDiffusionSystem (inherited).
* Cross-modal routing is controlled by a single boolean switch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .lightning_condition_diffusion import (
    LightningConditionedDenoiserCuboid,
    LightningDiffusionSystem,
)
from .moe_ffn import (
    patch_ffn_with_moe_selective,
    collect_moe_aux_loss,
    init_cross_modal_vil_proj,
    inject_vil_cond,
)


# ---------------------------------------------------------------------------
# MoE denoiser
# ---------------------------------------------------------------------------

class LightningConditionedDenoiserCuboidMoE(LightningConditionedDenoiserCuboid):
    """
    CuboidTransformerUNet denoiser with Sparse MoE FFN layers.

    Identical to ``LightningConditionedDenoiserCuboid`` except that after
    the parent ``__init__`` constructs the full UNet, selected
    ``PositionwiseFFN`` layers inside ``self.model`` are replaced in-place
    by ``SparseMoEFFN`` via :func:`patch_ffn_with_moe_selective`.

    Parameters
    ----------
    num_experts : int
        Number of experts per MoE FFN layer.
    top_k : int
        Number of experts each token is routed to (top-k sparse routing).
    moe_strategy : str
        Which FFN layers to replace.  Default ``"up_l0_t"`` (Strategy A):
        only T-axis FFN in Up-path Level-0 (4 layers).
    All other parameters
        Forwarded verbatim to ``LightningConditionedDenoiserCuboid``.
    """

    def __init__(
        self,
        *args,
        num_experts: int = 8,
        top_k: int = 2,
        moe_strategy: str = "up_l0_t",
        cross_modal: bool = False,
        vil_cond_dim: int = 128,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.moe_n_replaced = patch_ffn_with_moe_selective(
            self.model,
            strategy=moe_strategy,
            num_experts=num_experts,
            top_k=top_k,
            cross_modal=cross_modal,
            vil_cond_dim=vil_cond_dim,
        )


# ---------------------------------------------------------------------------
# MoE system
# ---------------------------------------------------------------------------

class LightningDiffusionMoESystem(LightningDiffusionSystem):
    """
    FC Cuboid Sparse MoE variant of ``LightningDiffusionSystem``.

    Parameters
    ----------
    moe_num_experts : int
        Number of experts per MoE FFN layer.  Default: 8.
    moe_top_k : int
        Number of experts each token is routed to.  Default: 2.
    moe_strategy : str
        Which FFN layers to replace.  Default: ``"up_l0_t"``.
    cross_modal_routing : bool
        If True, use CrossModalSparseMoEFFN with VIL-conditioned routing.
        VIL latent features are injected into the router input alongside
        the token features.  Default: False.
    vil_cond_dim : int
        Dimension of VIL condition projection for cross-modal routing.
        Default: 128.
    All other parameters
        Forwarded verbatim to ``LightningDiffusionSystem``.
        ``denoiser_backbone`` is always overridden to ``"cuboid"``.
    """

    def __init__(
        self,
        *args,
        moe_num_experts: int = 8,
        moe_top_k: int = 2,
        moe_strategy: str = "up_l0_t",
        cross_modal_routing: bool = False,
        vil_cond_dim: int = 128,
        cross_modal_router_type: str = "concat",
        moe_shared_expert: bool = False,
        **kwargs,
    ):
        # Always use the cuboid backbone; MoE patches are applied afterwards
        kwargs["denoiser_backbone"] = "cuboid"
        super().__init__(*args, **kwargs)

        # Selectively patch PositionwiseFFN -> SparseMoEFFN (or CrossModal)
        n_replaced = patch_ffn_with_moe_selective(
            self.denoiser.model,
            strategy=moe_strategy,
            num_experts=moe_num_experts,
            top_k=moe_top_k,
            cross_modal=cross_modal_routing,
            vil_cond_dim=vil_cond_dim,
            router_type=cross_modal_router_type,
            shared_expert=moe_shared_expert,
        )

        # If cross-modal, initialize VIL projection layers
        self.cross_modal_routing = cross_modal_routing
        if cross_modal_routing:
            vil_raw_dim = self.vil_encoder.latent_channels  # typically 4
            n_init = init_cross_modal_vil_proj(
                self.denoiser, vil_raw_dim
            )
            assert n_init == n_replaced, (
                f"Expected {n_replaced} CrossModalSparseMoEFFN layers, "
                f"but init_cross_modal_vil_proj found {n_init}"
            )

        # Store for checkpoint serialisation and logging
        self.moe_num_experts = moe_num_experts
        self.moe_top_k       = moe_top_k
        self.moe_strategy    = moe_strategy
        self.moe_n_replaced  = n_replaced
        self.vil_cond_dim    = vil_cond_dim
        self.cross_modal_router_type = cross_modal_router_type

    # ------------------------------------------------------------------
    # VIL condition injection for cross-modal routing
    # ------------------------------------------------------------------

    def _inject_vil_cond_for_cross_modal(
        self,
        z_vil_past: torch.Tensor,
        z_vil_future_pred: Optional[torch.Tensor],
    ) -> None:
        """
        Build the VIL condition tensor and inject it into all
        CrossModalSparseMoEFFN layers before the denoiser forward pass.

        The VIL condition is the raw VIL latent (C_z=4 channels) at the
        same spatiotemporal resolution as the denoiser's internal
        representation at Level 0.

        Inside CuboidTransformerUNet, cond and x are temporally
        concatenated: full_vil = cat([z_vil_past, z_vil_future_pred], dim=1)
        giving shape (B, T_in+T_out, H, W, C_z).  This matches the
        Level 0 spatial resolution (48x48) directly.
        """
        if not self.cross_modal_routing:
            return

        # Build full VIL temporal sequence matching UNet internal layout:
        #   cond occupies frames 0..T_in-1 (VIL past)
        #   x occupies frames T_in..T_in+T_out-1 (VIL future pred)
        if z_vil_future_pred is not None:
            vil_full = torch.cat([z_vil_past, z_vil_future_pred], dim=1)
        else:
            # When future VIL is unavailable, zero-pad to match channel count
            B, T_out = z_vil_past.shape[0], 12  # lead_time
            H, W, C_z = z_vil_past.shape[2], z_vil_past.shape[3], z_vil_past.shape[4]
            vil_zeros = torch.zeros(B, T_out, H, W, C_z,
                                    device=z_vil_past.device, dtype=z_vil_past.dtype)
            vil_full = torch.cat([z_vil_past, vil_zeros], dim=1)

        # Inject into all CrossModalSparseMoEFFN layers
        # All MoE layers at Level 0 share the same 48x48 resolution
        inject_vil_cond(self.denoiser, vil_full)

    # ------------------------------------------------------------------
    # Override forward_from_encoded to inject VIL before denoiser call
    # ------------------------------------------------------------------

    def forward_from_encoded(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        z_vil_past: torch.Tensor,
        lgt_past: torch.Tensor,
        z_vil_future_pred: Optional[torch.Tensor] = None,
    ):
        # Inject VIL condition into CrossModal MoE layers (no-op if disabled)
        self._inject_vil_cond_for_cross_modal(z_vil_past, z_vil_future_pred)

        # Standard denoiser forward (unchanged interface)
        return self.denoiser(
            t, x_t, z_vil_past, lgt_past,
            z_vil_future_pred=z_vil_future_pred,
        )

    # ------------------------------------------------------------------
    # Auxiliary loss collection
    # ------------------------------------------------------------------

    def collect_moe_aux_loss(self) -> torch.Tensor:
        """
        Return the sum of differentiable load-balance auxiliary losses from
        every SparseMoEFFN layer in the denoiser.
        """
        return collect_moe_aux_loss(self.denoiser)
