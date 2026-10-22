"""Mean replacement and first- and second-order trajectory tangents for AI-06's batched FlyVis simulator.

AI-06 silences a cell type by scaling the rectified output of each of its cells by (1 - a):
    release = relu(v) * (1 - a)                                   ("zero": towards zero output)
AI-09 adds mean replacement, which moves each cell's output towards its own average output m instead:
    release = relu(v) * (1 - a) + a * m                            ("mean")
m is a per-cell constant: the cell's rectified output averaged over the 24 trajectories of the stimulus and over the
intervention period of the intact network (the states from which each update's release is computed).

Tangents. Differentiating the explicit Euler update v' = v + dt/tau * (-v + b + W release(v, a) + x) along one cell
type's coordinate (indicator c over cells) at a = 0 gives, with J = dv/da, K = d2v/da2 and relu' = 1[v > 0]:
    d release   = relu'(v) J - c relu(v)            (zero)      relu'(v) J + c (m - relu(v))     (mean)
    d2 release  = relu'(v) K - 2 c relu'(v) J       (both; relu'' = 0, as in automatic differentiation)
    J' = J + dt/tau * (-J + W d release),   K' = K + dt/tau * (-K + W d2 release),   J = K = 0 at onset.
These are linear recursions driven by the intact trajectory, so one pass per cell type gives the tangents of all
eight readout traces at once. Readout tangents are recorded after each update, as AI-06 records voltages.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ai09_common import Core, health, sync


class ReplacementCore(Core):
    """AI-06's Core with an optional per-cell replacement value; with mean None it is AI-06's step exactly."""

    mean: torch.Tensor | None = None

    def step(self, v, x, atten, dt):
        if self.mean is None:
            return super().step(v, x, atten, dt)
        a = atten[:, self.types]
        release = torch.relu(v) * (1 - a) + a * self.mean
        edge = release[:, self.source] * self.weight
        current = torch.zeros_like(v).scatter_add(1, self.target[None].expand(v.shape[0], -1), edge)
        external = torch.zeros_like(v)
        external[:, self.input] = x[:, None, :].to(v.dtype)
        velocity = 1 / torch.maximum(self.tau, torch.tensor(dt, device=self.device, dtype=torch.float32)) * (
            -v + self.bias + current + external)
        return v + velocity * dt


def onset_state(core: Core, frames: np.ndarray, gray: torch.Tensor) -> torch.Tensor:
    """State at the intervention onset after the 200-frame stimulus preperiod (as in AI-06's run_model)."""
    _, state, _ = core.run(frames[:, :200], state=gray.expand(frames.shape[0], -1).clone())
    return state.detach()


def propagate(core: Core, release: torch.Tensor) -> torch.Tensor:
    edge = release[:, core.source] * core.weight
    return torch.zeros_like(release).scatter_add(1, core.target[None].expand(release.shape[0], -1), edge)


@torch.no_grad()
def node_means(core: Core, causal: torch.Tensor, state: torch.Tensor, dt: float = 0.005) -> torch.Tensor:
    """Per-cell rectified output averaged over trajectories and the intervention period of the intact network."""
    v = state.clone()
    zeros = torch.zeros((v.shape[0], core.c), device=core.device)
    total = torch.zeros(core.n, dtype=torch.float64, device=core.device)
    for k in range(causal.shape[1]):
        total += torch.relu(v).sum(0, dtype=torch.float64)
        v = core.step(v, causal[:, k], zeros, dt)
    return (total / (causal.shape[1] * v.shape[0])).to(torch.float32)


@torch.no_grad()
def tangent_bank(core: Core, causal: torch.Tensor, state: torch.Tensor, types: list[int], mode: str,
                 mean: torch.Tensor | None = None, second: bool = True, batch_types: int = 4,
                 dt: float = 0.005, dtype=torch.float32):
    """Readout-trace tangents J (and K) for each listed cell type: arrays (len(types), 24, T, 8)."""
    if mode not in ("zero", "mean") or (mode == "mean") != (mean is not None):
        raise ValueError("mode 'mean' needs the per-cell means; mode 'zero' takes none")
    trajectories, frames = causal.shape[0], causal.shape[1]
    out_j = np.empty((len(types), trajectories, frames, 8), np.float32)
    out_k = np.empty_like(out_j) if second else None
    inverse = (1 / torch.maximum(core.tau, torch.tensor(dt, device=core.device, dtype=torch.float32))).to(dtype)
    zeros = torch.zeros((trajectories, core.c), device=core.device)
    seconds = 0.0
    for begin in range(0, len(types), batch_types):
        health()
        chunk = types[begin:begin + batch_types]
        started = time.perf_counter()
        type_rows = torch.tensor([t for t in chunk for _ in range(trajectories)], device=core.device)
        rows = torch.tensor([j for _ in chunk for j in range(trajectories)], device=core.device)
        indicator = (core.types[None] == type_rows[:, None]).to(dtype)
        v = state.clone()
        J = torch.zeros((len(rows), core.n), dtype=dtype, device=core.device)
        K = torch.zeros_like(J) if second else None
        record_j = torch.empty((len(rows), frames, 8), dtype=dtype, device=core.device)
        record_k = torch.empty_like(record_j) if second else None
        for k in range(frames):
            voltage = v[rows].to(dtype)
            active = (voltage > 0).to(dtype)
            rectified = torch.relu(voltage)
            if mode == "zero":
                drive = active * J - indicator * rectified
            else:
                drive = active * J + indicator * (mean.to(dtype)[None] - rectified)
            if second:
                drive2 = active * K - 2 * indicator * active * J
                both = propagate(core, torch.cat([drive, drive2]).to(core.weight.dtype)).to(dtype)
                J = J + dt * inverse * (-J + both[:len(rows)])
                K = K + dt * inverse * (-K + both[len(rows):])
                record_k[:, k] = K[:, core.readout]
            else:
                J = J + dt * inverse * (-J + propagate(core, drive.to(core.weight.dtype)).to(dtype))
            record_j[:, k] = J[:, core.readout]
            v = core.step(v, causal[:, k], zeros, dt)
        sync(core.device)
        seconds += time.perf_counter() - started
        for position, cell_type in enumerate(chunk):
            span = slice(position * trajectories, (position + 1) * trajectories)
            out_j[begin + position] = record_j[span].cpu().numpy()
            if second:
                out_k[begin + position] = record_k[span].cpu().numpy()
        del J, K, record_j, record_k, indicator
    return out_j, out_k, seconds


@torch.no_grad()
def exact_bank(core: ReplacementCore, causal: torch.Tensor, state: torch.Tensor, conditions: list[tuple[int, float]],
               batch_rows: int = 48, dt: float = 0.005):
    """Exact readout traces for (cell type, strength) conditions under the core's current replacement setting."""
    trajectories, frames = causal.shape[0], causal.shape[1]
    queue = [(c, j) for c in range(len(conditions)) for j in range(trajectories)]
    out = np.empty((len(conditions), trajectories, frames, 8), np.float32)
    seconds = 0.0
    for begin in range(0, len(queue), batch_rows):
        health()
        jobs = queue[begin:begin + batch_rows]
        index = [j for _, j in jobs]
        attenuation = torch.zeros((len(jobs), core.c), device=core.device)
        for row, (c, _) in enumerate(jobs):
            cell_type, strength = conditions[c]
            attenuation[row, cell_type] = strength
        started = time.perf_counter()
        trace, _, _ = core.run(causal[index], attenuation, state=state[index].clone(), start=0)
        sync(core.device)
        seconds += time.perf_counter() - started
        bank = trace.cpu().numpy()
        for row, (c, j) in enumerate(jobs):
            out[c, j] = bank[row]
    return out, seconds
