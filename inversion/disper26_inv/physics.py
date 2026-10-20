"""Stiffness-tensor positive-definiteness checks for VTI models.

Conditions per node (Love notation): rho > 0, L > 0, N > 0, C > 0, A > 0,
AC - F^2 > 0. Violations show up downstream as vanishing dispersion roots, so
every generated truth model, prior sample, LM trial step and surrogate
candidate is gated through here first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import ForwardGridModel


@dataclass
class PosdefReport:
    ok: bool
    min_margin: float          # smallest normalized margin over all nodes/conditions
    worst_node: int
    worst_condition: str
    violations: list = field(default_factory=list)  # (node, condition, margin)

    def __bool__(self) -> bool:
        return self.ok


def check_stiffness_posdef(model: ForwardGridModel) -> PosdefReport:
    stiff = model.stiffness()
    A, C, F, L, N = (stiff[k] for k in ("A", "C", "F", "L", "N"))
    rho = stiff["rho"]

    # Normalized margins: positive iff the condition holds, ~O(1) when healthy.
    margins = {
        "rho>0": rho / np.max(np.abs(rho)),
        "A>0": A / np.maximum(np.abs(A), 1e-300),
        "C>0": C / np.maximum(np.abs(C), 1e-300),
        "L>0": L / np.maximum(np.abs(L), 1e-300),
        "N>0": N / np.maximum(np.abs(N), 1e-300),
        "AC-F^2>0": (A * C - F**2) / np.maximum(np.abs(A * C), 1e-300),
    }

    violations = []
    min_margin = np.inf
    worst_node = -1
    worst_condition = ""
    for condition, margin in margins.items():
        idx = int(np.argmin(margin))
        val = float(margin[idx])
        if val < min_margin:
            min_margin = val
            worst_node = idx
            worst_condition = condition
        for node in np.nonzero(margin <= 0.0)[0]:
            violations.append((int(node), condition, float(margin[node])))

    return PosdefReport(
        ok=not violations,
        min_margin=float(min_margin),
        worst_node=worst_node,
        worst_condition=worst_condition,
        violations=violations,
    )
