"""Unit tests of AI-09's tangents, readouts and scoring (CPU, seconds). Run: python -B study/code/test_arms.py"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from ai09_common import ai06
import analyse_arms as arms
from readouts import Readout
from tangents import ReplacementCore, tangent_bank

F64 = torch.float64
RNG = np.random.default_rng(20261016)


def fake_stimulus():
    angles = np.tile(np.arange(0, 360, 30), 2).astype(np.float32)
    return {"angle": angles, "intensity": np.repeat([0, 1], 12), "speed": np.array(19.0)}


def test_peak_readout_matches_ai06():
    stim = fake_stimulus()
    readout = Readout(stim, 40)
    for _ in range(5):
        trace = RNG.normal(0, 1, (24, 40, 8))
        trace[RNG.random((24, 40, 8)) < 0.3] *= -1
        trace[3] = -np.abs(trace[3])  # a trajectory with no positive voltage
        mine = readout(torch.as_tensor(trace), "peak_dsi")
        theirs = ai06.independent_metrics(trace, stim)
        assert np.allclose(mine["dsi"].numpy(), theirs["dsi"], atol=1e-12, rtol=0)
        assert np.allclose(mine["denominator"].numpy(), theirs["denominator"], atol=1e-12, rtol=0)


def test_path_derivatives_match_finite_differences():
    stim = fake_stimulus()
    readout = Readout(stim, 30)
    base = torch.as_tensor(RNG.uniform(0.5, 2.0, (24, 30, 8)), dtype=F64)  # positive: rectifier inactive
    base[:, 10] += 3.0  # one clear temporal peak per trajectory
    first = torch.as_tensor(RNG.normal(0, 0.3, (3, 24, 30, 8)), dtype=F64)
    second = torch.as_tensor(RNG.normal(0, 0.3, (3, 24, 30, 8)), dtype=F64)
    for kind in ("peak_dsi", "mean_dsi"):
        g, h = readout.path_derivatives(base, first, second, kind)
        step = 1e-4
        value = lambda s: readout.dsi(base + s * first + 0.5 * s * s * second, kind)  # noqa: E731
        fd_g = (value(step) - value(-step)) / (2 * step)
        fd_h = (value(step) - 2 * value(0.0) + value(-step)) / step**2
        assert torch.allclose(g, fd_g, atol=1e-8), kind
        assert torch.allclose(h, fd_h, atol=1e-4), kind
        assert torch.allclose(arms.slope(readout, base, first, kind), g, atol=1e-12)


def tiny_core(nodes=12, edges=40, types=3):
    core = object.__new__(ReplacementCore)
    core.device = torch.device("cpu")
    core.n, core.c = nodes, types
    core.types = torch.as_tensor(RNG.integers(0, types, nodes))
    core.types[:types] = torch.arange(types)
    core.source = torch.as_tensor(RNG.integers(0, nodes, edges))
    core.target = torch.as_tensor(RNG.integers(0, nodes, edges))
    core.weight = torch.as_tensor(RNG.normal(0, 1.5, edges), dtype=F64)
    core.bias = torch.as_tensor(RNG.normal(0.2, 0.5, nodes), dtype=F64)
    core.tau = torch.as_tensor(RNG.uniform(0.01, 0.05, nodes), dtype=F64)
    core.readout = torch.arange(8)
    core.input = torch.as_tensor([[8, 9, 10, 11]])
    core.mean = None
    return core


def test_tangents_match_nested_autodiff():
    from torch.func import jvp
    import tangents
    tangents.health = lambda: None  # the disk and memory guards are not under test here
    core = tiny_core()
    frames = torch.as_tensor(RNG.uniform(0, 1, (3, 25, 4)), dtype=F64)
    state = torch.as_tensor(RNG.normal(0.1, 0.5, (3, core.n)), dtype=F64)
    mean = torch.as_tensor(RNG.uniform(0, 1, core.n), dtype=F64)
    zero, one = torch.zeros((), dtype=F64), torch.ones((), dtype=F64)
    for mode in ("zero", "mean"):
        core.mean = None if mode == "zero" else mean
        J, K, _ = tangent_bank(core, frames, state, [0, 1, 2], mode, mean=mean if mode == "mean" else None,
                               batch_types=2, dtype=F64)
        for cell in range(3):
            def trace(a, cell=cell):
                attenuation = a * torch.nn.functional.one_hot(torch.tensor(cell), core.c).to(F64)
                return core.run(frames, attenuation[None].expand(3, -1), state=state.clone(), start=0)[0]
            first = jvp(trace, (zero,), (one,))[1].detach().numpy()
            second = jvp(lambda a: jvp(trace, (a,), (one,))[1], (zero,), (one,))[1].detach().numpy()
            assert np.allclose(J[cell], first, atol=1e-6, rtol=1e-5), (mode, cell)  # J stored as float32
            assert np.allclose(K[cell], second, atol=1e-6, rtol=1e-5), (mode, cell)
    core.mean = torch.zeros_like(mean)  # replacement by zero equals AI-06's silencing exactly
    attenuation = torch.zeros((3, core.c), dtype=F64)
    attenuation[:, 1] = 1.0
    with_zero_mean = core.run(frames, attenuation, state=state.clone(), start=0)[0]
    core.mean = None
    silenced = core.run(frames, attenuation, state=state.clone(), start=0)[0]
    assert torch.equal(with_zero_mean, silenced)


def test_estimators_and_pooling():
    panel = SimpleNamespace(gradient=np.full((8, 65), 0.2),
                            first={("mean", "peak_dsi"): np.full((8, 65), 0.1)},
                            second={("zero", "peak_dsi"): np.full((8, 65), -0.4), ("mean", "peak_dsi"): np.ones((8, 65))})
    assert np.allclose(arms.estimate_for(panel, "zero", "peak_dsi", "first", 0.5, {}), 0.1)
    assert np.allclose(arms.estimate_for(panel, "zero", "peak_dsi", "second", 0.5, {}), 0.1 - 0.05)
    assert np.allclose(arms.estimate_for(panel, "mean", "peak_dsi", "second", 1.0, {}), 0.6)
    assert np.allclose(arms.estimate_for(panel, "mean", "peak_dsi", "mismatch", 1.0, {}), 0.2)
    assert np.allclose(arms.estimate_for(panel, "zero", "peak_dsi", "mismatch", 1.0, {}), 0.1)
    units = [dict(n=2, n_large=1, abs_error_sum=0.4, abs_exact_sum=0.2, clipped_abs_error_sum=0.1, signed_error_sum=0.2,
                  wrong_sign=1, zero_estimate=0, top10=0.5, anatomy_top10=0.25, rho=0.5, rho_large=None,
                  mean_denominator_dropped=0),
             dict(n=6, n_large=1, abs_error_sum=0.4, abs_exact_sum=1.4, clipped_abs_error_sum=0.3, signed_error_sum=-0.6,
                  wrong_sign=0, zero_estimate=1, top10=1.0, anatomy_top10=0.5, rho=None, rho_large=None,
                  mean_denominator_dropped=2)]
    pooled = arms.pool(units)
    assert np.isclose(pooled["mae"], 0.1) and np.isclose(pooled["zero_mae"], 0.2) and np.isclose(pooled["ratio"], 0.5)
    assert np.isclose(pooled["bias"], -0.05) and pooled["wrong_sign_rate"] == 0.5 and pooled["zero_estimate_rate"] == 0.5
    assert np.isclose(pooled["top10"], 0.75) and np.isclose(pooled["top10_minus_anatomy"], 0.375)
    assert pooled["rho"] == 0.5 and pooled["rho_large"] is None and pooled["mean_denominator_dropped"] == 2


def test_score_units_eligibility():
    stim = fake_stimulus()
    base_dsi = np.array([0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    panel = SimpleNamespace(key="dev_fast", base_denominator=np.array([1.0, 1.0, 0.0, 1, 1, 1, 1, 1]),
                            base_peak_dsi=base_dsi, base_dsi={"peak_dsi": base_dsi, "mean_dsi": base_dsi * 0.5},
                            anatomy=np.tile(np.arange(65, dtype=float), (8, 1)))
    change = np.zeros((8, 65))
    change[:, :5] = 0.3
    exact = {"dsi": {"peak_dsi": base_dsi[:, None] + change, "mean_dsi": base_dsi[:, None] * 0.5 + change},
             "peak_denominator": np.ones((8, 65)), "mean_denominator": np.ones((8, 65)),
             "amplification": np.ones(65)}
    exact["peak_denominator"][0, 10] = 0.0  # abolished
    exact["amplification"][11] = 1000.0  # amplified in every readout
    exact["mean_denominator"][0, 12] = 0.0  # dropped from the mean-response readout only
    estimate = np.zeros((8, 65))
    estimate[:, :5] = -0.3
    rows = arms.score_units(panel, exact, "peak_dsi", estimate, 1.0)
    assert [r["readout"] for r in rows] == ["T4a", "T4d", "T5a", "T5b", "T5c", "T5d"]  # T4b selectivity, T4c silent
    assert rows[0]["n"] == 63 and rows[1]["n"] == 64
    assert rows[0]["n_large"] == 5 and rows[0]["wrong_sign"] == 5 and rows[0]["top10"] == 0.5
    mean_rows = arms.score_units(panel, exact, "mean_dsi", estimate, 1.0)
    assert mean_rows[0]["n"] == 62 and mean_rows[0]["mean_denominator_dropped"] == 1


def test_flag_rule():
    cases = [("x", 0, 0.5, 0.4, 0.4, 0.4), ("x", 1, 0.1, 0.0, 0.0, 0.0), ("y", 0, 0.2, 0.3, 0.3, 0.3),
             ("y", 1, -0.4, 0.05, 0.05, 0.05)]
    wall = {("x", 0): 1.0, ("x", 1): 1.0, ("y", 0): 2.0, ("y", 1): 2.0}
    result = arms.flag_rule(cases, derivative_seconds=3.0, wall=wall, k=1, fallback="zero")
    # flags x/0 and y/0; remaining errors 0.1 and 0.4
    assert np.isclose(result["mae"], 0.125) and np.isclose(result["zero_mae"], 0.3)
    assert np.isclose(result["seconds"], 6.0) and np.isclose(result["exhaustive_seconds"], 6.0)
    assert np.isclose(result["recovered"], 1 - 0.125 / 0.3)


def test_interval_and_decisions():
    indices = arms.bootstrap_indices(46)
    values = np.linspace(-1, 0, 46)
    summary = arms.interval(values, indices)
    assert summary["interval"][1] < 0 and arms.sign_decision(summary, "below", "above", "neither") == "below"
    with_missing = arms.interval(np.r_[values[:45], np.nan], indices)
    assert with_missing["models"] == 45 and np.isfinite(with_missing["interval"]).all()
    reductions = {"a": np.full(46, 0.5) + np.linspace(0, 0.1, 46), "b": np.full(46, 0.1), "c": np.zeros(46)}
    assert arms.dominance(reductions, {"a": True, "b": True, "c": False}, indices)["decision"] == "(a) dominates"
    assert arms.dominance(reductions, {"a": False, "b": False, "c": False}, indices)["decision"] == "no arm improves"
    close = {"a": np.linspace(0, 1, 46), "b": np.linspace(0, 1, 46)[::-1], "c": np.zeros(46)}
    assert arms.dominance(close, {"a": True, "b": True, "c": False}, indices)["decision"] == "no single arm dominates"


def test_readout_split():
    stim = fake_stimulus()
    readout = Readout(stim, 30)
    v0 = RNG.uniform(1.0, 2.0, (24, 30, 8))
    J = torch.as_tensor(RNG.normal(0, 0.1, (4, 24, 30, 8)), dtype=F64)
    fake = SimpleNamespace(tangents={"zero": (np.array([0, 1, 2, 3]), J, None)}, v0=v0, readout=readout)
    base = torch.as_tensor(v0, dtype=F64)
    assert torch.allclose(readout.ratio(readout.responses(base, "peak_dsi"))["dsi"], readout.dsi(base, "peak_dsi"))
    strength = 0.5
    cf = arms.Panel.counterfactuals(fake, "zero", "mean_dsi", strength, v0[None].repeat(65, 0))
    first = strength * arms.slope(readout, base, J, "mean_dsi").numpy().T
    # traces stay positive, so the mean-response stage is linear: exact responses = first order, and
    # exact ratio = both stages exact.
    assert np.allclose(cf["linear_trajectory_exact_responses"][:, :4], first, atol=1e-12)
    assert np.allclose(cf["linear_trajectory_exact_ratio"], cf["linear_trajectory_exact_readout"], atol=1e-12)
    assert np.all(cf["linear_trajectory_exact_ratio"][:, 4:] == 0)


def test_replay_tolerance():
    import run_arms
    reference = np.zeros((2, 3, 8))
    reference[0, 0, 0] = 20.0
    assert np.isclose(run_arms.normalised_difference(reference + 1e-5, reference), 1e-5 / 20)
    assert np.isclose(run_arms.normalised_difference(np.full_like(reference, 1e-5), np.zeros_like(reference)), 1e-5)
    assert run_arms.normalised_difference(reference + 9e-6, reference) <= run_arms.REPLAY_TOLERANCE


if __name__ == "__main__":
    torch.set_num_threads(2)
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("pass", test.__name__, flush=True)
    print(f"{len(tests)} tests passed")
