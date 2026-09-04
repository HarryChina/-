"""Frozen H4 helper for the R8D/R8E train-only diagonal-consistency study.

H4 has exactly the H3 architecture.  This module adds no parameters, buffers,
or inference-time correction; it only defines the future training objective.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import Tensor, nn

from p6_phaseA8MRN2R2_preregister import complex_mse
from p6_phaseA8MRN2R8B_boundary_aware_source_model import build_h3


LAMBDA_DIAG: Final[float] = 0.1
FIELD_PAIRS_PER_UPDATE: Final[int] = 32768
DIAGONAL_PAIRS_PER_UPDATE: Final[int] = 1024
DIAGONAL_LOCAL_INDICES: Final[tuple[int, ...]] = tuple(range(0, 8192, 32))


def build_h4(model_seed: int) -> nn.Module:
    """Return the unchanged H3 field architecture with fresh initialization."""

    return build_h3(model_seed, boundary_features_enabled=True)


def field_dual_loss(
    model: nn.Module,
    output_u: Tensor,
    source_r: Tensor,
    target: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Frozen dual-orientation field loss with coefficient exactly 1.0."""

    branch_a, _, branch_b = model.branches(output_u, source_r)
    loss_a = complex_mse(branch_a.float(), target.float())
    loss_b = complex_mse(branch_b.float(), target.float())
    return 0.5 * loss_a + 0.5 * loss_b, branch_a, branch_b


def diagonal_dual_loss(
    model: nn.Module,
    diagonal_q: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Evaluate (q,q,d=0) against the TRAIN-derived approximate target 1+0j."""

    branch_a, _, branch_b = model.branches(diagonal_q, diagonal_q)
    target = torch.zeros_like(branch_a, dtype=torch.float32)
    target[:, 0] = 1.0
    loss_a = complex_mse(branch_a.float(), target)
    loss_b = complex_mse(branch_b.float(), target)
    return 0.5 * loss_a + 0.5 * loss_b, branch_a, branch_b, target


def h4_total_loss(field_loss: Tensor, diagonal_loss: Tensor) -> Tensor:
    """Keep the field coefficient at 1.0 and add the fixed 0.1 prior."""

    return field_loss + LAMBDA_DIAG * diagonal_loss

