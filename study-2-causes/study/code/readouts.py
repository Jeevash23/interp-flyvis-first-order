"""Direction-selectivity readouts of saved or simulated T4/T5 traces, with derivatives along trace paths.

`peak_dsi` is AI-06's index: the temporal peak of the rectified voltage of each trajectory, combined across the 12
directions of the readout's preferred polarity into |vector sum| / max(ON sum, OFF sum). `mean_dsi` replaces the
temporal peak by the temporal mean of the rectified voltage over the same window and changes nothing else. Both accept
leading batch dimensions: trace (..., 24, T, 8) -> (..., 8). The traces are AI-06 causal prefixes, which start at the
intervention onset and end at the last measured frame, so the whole trace is the measurement window.
"""
from __future__ import annotations

import numpy as np
import torch

KINDS = ("peak_dsi", "mean_dsi")


class Readout:
    def __init__(self, stim: dict, frames: int, dtype=torch.float64, device="cpu"):
        times = np.arange(frames, dtype=np.float64) * 0.005
        end = ((10 * 2.25 / 5.8 + 2.25 / 5.8) - (-10 * 2.25 / 5.8)) / float(stim["speed"])
        if not np.all((times >= 0) & (times <= end)):
            raise ValueError("Trace extends beyond the measurement window")
        self.angle = torch.as_tensor(np.deg2rad(stim["angle"].astype(np.float64)), dtype=dtype, device=device)
        self.polarity = np.asarray(stim["intensity"])
        self.dtype, self.device = dtype, device

    def responses(self, trace: torch.Tensor, kind: str) -> torch.Tensor:
        rectified = trace.clamp_min(0)
        if kind == "peak_dsi":
            return rectified.amax(-2)
        if kind == "mean_dsi":
            return rectified.mean(-2)
        raise ValueError(kind)

    def __call__(self, trace: torch.Tensor, kind: str) -> dict[str, torch.Tensor]:
        return self.ratio(self.responses(trace, kind))

    def ratio(self, response: torch.Tensor) -> dict[str, torch.Tensor]:
        """The ratio stage: responses (..., 24, 8) -> |vector sum| / larger polarity sum, per readout."""
        sums, vectors = [], []
        for polarity in (0, 1):
            select = torch.as_tensor(np.flatnonzero(self.polarity == polarity), device=response.device)
            values = response.index_select(-2, select)
            angle = self.angle[select].to(values.dtype)
            sums.append(values.sum(-2))
            vectors.append(torch.stack([(values * torch.cos(angle)[:, None]).sum(-2),
                                        (values * torch.sin(angle)[:, None]).sum(-2)], 0).norm(dim=0))
        denominator = torch.stack(sums).amax(0)
        both = torch.stack(vectors) / (denominator + 1e-15)  # (2, ..., 8)
        chosen = torch.stack([both[1 if r < 4 else 0, ..., r] for r in range(8)], -1)
        return {"dsi": chosen, "all_dsi": both, "denominator": denominator, "responses": response}

    def dsi(self, trace: torch.Tensor, kind: str) -> torch.Tensor:
        return self(trace, kind)["dsi"]

    def path_derivatives(self, base: torch.Tensor, first: torch.Tensor, second: torch.Tensor, kind: str):
        """First and second derivatives at s = 0 of s -> DSI(base + s*first + s^2/2*second).

        base (24, T, 8); first and second (n, 24, T, 8). With first = dv/da and second = d2v/da2 along one cell
        type's intervention coordinate, these are dDSI/da and d2DSI/da2 of the composed map (the readout's own
        curvature enters through the second derivative). Rectification and the temporal peak contribute zero second
        derivative away from ties, as in automatic differentiation.
        """
        from torch.func import jvp

        def path(s):
            return self.dsi(base + s * first + 0.5 * s * s * second, kind)

        zero = torch.zeros((), dtype=base.dtype, device=base.device)
        one = torch.ones((), dtype=base.dtype, device=base.device)

        def slope(s):
            return jvp(path, (s,), (one,))[1]

        g, h = jvp(slope, (zero,), (one,))
        return g, h  # each (n, 8)
