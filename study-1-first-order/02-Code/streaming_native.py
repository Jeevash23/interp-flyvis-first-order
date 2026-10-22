"""Detached single-trial CPU execution retaining only central-cell outputs.

Uses FlyVis 1.2.0's parameter API, initial state and _next_state without replacing
its recurrence, edge gather, activation, or scatter reduction. No simulation runs
on import. This is infrastructure awaiting explicit native-equality validation.
Do not use NetworkView's clean response cache for instrumented trials.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import numpy as np
import torch


@dataclass
class StreamResult:
    responses: np.ndarray  # (frame, selected central cell)
    final_state: object
    metadata: dict


@dataclass
class EdgeGate:
    multiplier: torch.Tensor  # spatial edge array, NOT fitted parameter groups
    selected: torch.Tensor
    metadata: dict


def _check_network(network):
    parameters = list(network.parameters())
    if not parameters or any(p.device.type != "cpu" for p in parameters):
        raise ValueError("A recovered, entirely CPU network is required.")
    dtype = parameters[0].dtype
    if any(p.dtype != dtype for p in parameters):
        raise ValueError("Mixed parameter dtypes are not supported.")
    if torch.get_default_dtype() != dtype:
        raise ValueError("Native factories require default dtype to match parameters.")
    if torch.empty(0).device.type != "cpu":
        raise ValueError("Native factory default device must be CPU.")
    if getattr(network, "_state_hooks", ()):
        raise ValueError("This output-gate adapter forbids state-writing hooks.")
    for index in (network._source_indices, network._target_indices):
        if index.device.type != "cpu":
            raise ValueError("Native edge indices must be CPU tensors.")
    return dtype


def _readout(network, indices=None):
    if indices is None:
        indices = np.asarray(network.connectome.central_cells_index[:], dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0:
        raise ValueError("Readout indices must be a nonempty 1D array.")
    if indices.min() < 0 or indices.max() >= network.n_nodes:
        raise ValueError("Readout index is outside the native network.")
    if len(set(indices.tolist())) != len(indices):
        raise ValueError("Readout indices must not repeat.")
    types = np.asarray(network.connectome.nodes.type[:]).astype(str)
    return torch.as_tensor(indices, dtype=torch.long, device="cpu"), types[indices].tolist()


def _input_indices(network):
    indices = np.asarray(network.stimulus.input_index, dtype=np.int64)
    if indices.ndim != 2 or indices.shape[1] != network.stimulus.n_input_elements:
        raise ValueError("Unsupported native stimulus input mapping.")
    if len(set(indices.reshape(-1).tolist())) != indices.size:
        raise ValueError("Repeated input indices would need separate accumulation rules.")
    return torch.as_tensor(indices, dtype=torch.long, device="cpu")


def _hex_frames(stimulus, n_input_elements):
    frames = torch.as_tensor(stimulus, device="cpu").detach()
    # Native MovingEdge items: (time, hex_pixel). Saved trial input can retain
    # its singleton channel, but multi-trial batching is intentionally rejected.
    if frames.ndim == 3 and frames.shape[1] == 1:
        frames = frames[:, 0]
    if frames.ndim != 2 or frames.shape[1] != n_input_elements:
        raise ValueError("Expected one trial shaped (frame, hex_pixel), optionally channel1.")
    if frames.shape[0] == 0 or not bool(torch.isfinite(frames).all()):
        raise ValueError("Stimulus must be nonempty and finite.")
    return frames


def build_edge_gate(network, source_types=(), type_pairs=(), multiplier=0.0):
    """Select union of source-type and (source,target)-type spatial edges.

    Empty selections return an identity gate. A multiplier of one remains a
    useful identity control even when a nonempty edge set is selected.
    """
    dtype = _check_network(network)
    multiplier = float(multiplier)
    if not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError("Output multipliers must be finite and lie in [0,1].")
    source_types = tuple(sorted(set(source_types)))
    type_pairs = tuple(sorted(set(tuple(pair) for pair in type_pairs)))
    if any(len(pair) != 2 for pair in type_pairs):
        raise ValueError("Every connection selector must contain source and target type.")
    node_types = np.asarray(network.connectome.nodes.type[:]).astype(str)
    known = set(node_types.tolist())
    requested = set(source_types) | {cell for pair in type_pairs for cell in pair}
    if requested - known:
        raise ValueError(f"Unknown cell types: {sorted(requested - known)}")
    source = node_types[network._source_indices.detach().cpu().numpy()]
    target = node_types[network._target_indices.detach().cpu().numpy()]
    selected = np.isin(source, source_types)
    absent_pairs = []
    for source_type, target_type in type_pairs:
        pair_mask = (source == source_type) & (target == target_type)
        if not bool(pair_mask.any()):
            absent_pairs.append([source_type, target_type])
        selected |= pair_mask
    selected_tensor = torch.as_tensor(selected, dtype=torch.bool, device="cpu")
    values = torch.ones(network.n_edges, dtype=dtype, device="cpu")
    values[selected_tensor] = multiplier
    return EdgeGate(values, selected_tensor, {
        "semantics": "multiply derived spatial-edge current weights; no voltage writes",
        "source_types": list(source_types), "type_pairs": [list(p) for p in type_pairs],
        "selected_multiplier": multiplier, "selected_edges": int(selected.sum()),
        "total_edges": int(network.n_edges), "absent_type_pairs": absent_pairs,
        "mask_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "combination": "union", "dtype": str(dtype),
    })


@contextmanager
def synaptic_gate(network, gate):
    """Scope a derived-weight wrapper to this network instance; restore it."""
    dtype = _check_network(network)
    if not isinstance(gate, EdgeGate) or gate.multiplier.shape != (network.n_edges,):
        raise ValueError("Gate must correspond to this spatial-edge array.")
    if gate.multiplier.dtype != dtype or gate.multiplier.device.type != "cpu":
        raise ValueError("Gate dtype/device must match network parameters.")
    if not bool(torch.isfinite(gate.multiplier).all()):
        raise ValueError("Gate contains nonfinite weights.")
    dynamics = network.dynamics
    if getattr(dynamics, "_ai06_edge_gate_active", False):
        raise RuntimeError("Nested edge-gate contexts are forbidden.")
    had_override = "write_derived_params" in dynamics.__dict__
    original_override = dynamics.__dict__.get("write_derived_params")
    original = dynamics.write_derived_params
    fixed_multiplier = gate.multiplier.detach().clone()

    def write_gated_params(params, **kwargs):
        original(params, **kwargs)
        # Native derived weights have already expanded shared fitted parameter
        # groups to individual spatial edges. Do not mask grouped raw values.
        params.edges.weight = params.edges.weight * fixed_multiplier

    dynamics._ai06_edge_gate_active = True
    dynamics.write_derived_params = write_gated_params
    try:
        yield gate.metadata
    finally:
        if had_override:
            dynamics.write_derived_params = original_override
        else:
            del dynamics.write_derived_params
        del dynamics._ai06_edge_gate_active


@torch.no_grad()
def gray_initial_state(network, t_pre=1.0, dt=1/200, value=0.5):
    """Native rest initialization plus grey adaptation; no trajectory retained.

    This is the extra grey period in native stimulus_response, separate from
    MovingEdge's own preperiod. Gate context, if active, also affects adaptation.
    """
    dtype = _check_network(network)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(t_pre) or t_pre < 0:
        raise ValueError("dt must be positive and adaptation duration nonnegative.")
    if not math.isfinite(value):
        raise ValueError("Grey level must be finite.")
    network.clamp()
    params = network._param_api()
    state = network._initial_state(params, 1)
    input_indices = _input_indices(network)
    current = torch.zeros((1, network.n_nodes), dtype=dtype, device="cpu")
    current[:, input_indices] += value
    for frame in range(int(t_pre / dt)):
        state = network._next_state(params, state, current, dt)
        if not bool(torch.isfinite(state.nodes.activity).all()):
            raise FloatingPointError(f"Nonfinite grey-adaptation state at frame {frame}.")
    return state


@torch.no_grad()
def streaming_forward(network, stimulus, dt, state=None, indices=None):
    """Single-trial native update, storing central cells after each update.

    Pass the entire rendered trial, including its dataset preperiod. For native
    helper equivalence, first obtain gray_initial_state outside the gate context
    (common adaptation), or inside it (adaptation under intervention), explicitly.
    No implicit adaptation is added here. Returned times are post-update native
    state times, exactly as in Network.forward.
    """
    dtype = _check_network(network)
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite.")
    frames = _hex_frames(stimulus, network.stimulus.n_input_elements)
    input_indices = _input_indices(network)
    readout_indices, cell_types = _readout(network, indices)
    network.clamp()
    params = network._param_api()
    if state is None:
        state = network._initial_state(params, 1)
    if state.nodes.activity.shape != (1, network.n_nodes):
        raise ValueError("Initial state must be a native single-trial state.")
    if state.nodes.activity.dtype != dtype or state.nodes.activity.device.type != "cpu":
        raise ValueError("Initial state dtype/device mismatch.")
    current = torch.zeros((1, network.n_nodes), dtype=dtype, device="cpu")
    output = torch.empty((frames.shape[0], len(cell_types)), dtype=dtype, device="cpu")
    for frame_index, frame in enumerate(frames):
        current.zero_()
        # Identical input-index broadcasting and += casting as Stimulus.add_input.
        current[:, input_indices] += frame[None, None, :]
        state = network._next_state(params, state, current, dt)
        voltage = state.nodes.activity
        if not bool(torch.isfinite(voltage).all()):
            raise FloatingPointError(f"Nonfinite network state at frame {frame_index}.")
        output[frame_index] = voltage[0, readout_indices]
    return StreamResult(output.numpy(), state, {
        "execution": "native_next_state_streaming", "device": "cpu",
        "dtype": str(dtype), "dt": float(dt), "n_frames": int(frames.shape[0]),
        "readout_indices": readout_indices.tolist(), "cell_types": cell_types,
        "n_readout": len(cell_types), "n_nodes": int(network.n_nodes),
        "n_edges": int(network.n_edges), "implicit_adaptation_seconds": 0,
        "full_state_checked_finite_each_frame": True,
        "stimulus_sha256": hashlib.sha256(frames.numpy().tobytes()).hexdigest(),
    })


@torch.no_grad()
def validate_streaming(network, stimulus, dt=1/200, t_pre=0.02,
                       source_type="Mi1", atol=1e-7, rtol=1e-6, max_frames=8):
    """Explicit bounded control run; never called automatically.

    Independent native Stimulus maps short input and Network.forward retains the
    short full trajectory for comparison. Reports rest/grey-state equality,
    clean equality, identity gate, spatial-edge mask locality, raw-parameter
    immutability, and restoration. Only direct edge writes must be local; later
    state changes are allowed to propagate through the recurrent network.
    """
    _check_network(network)
    all_frames = _hex_frames(stimulus, network.stimulus.n_input_elements)
    if not 1 <= max_frames <= 32:
        raise ValueError("Validation frame cap must be 1..32.")
    if not 0 < t_pre <= 0.1:
        raise ValueError("Validation adaptation must be positive and at most 0.1s.")
    frames = all_frames[:max_frames]
    readout_indices, _ = _readout(network)
    network.clamp()
    parameters_before = {name: p.detach().clone() for name, p in network.named_parameters()}
    native_state = network.steady_state(t_pre, dt, batch_size=1, value=0.5)
    streamed_state = gray_initial_state(network, t_pre=t_pre, dt=dt, value=0.5)
    grey_error = float((native_state.nodes.activity - streamed_state.nodes.activity).abs().max())
    network.stimulus.zero(1, len(frames))
    network.stimulus.add_input(frames[None, :, None, :])
    native = network(network.stimulus(), dt, state=native_state)[0, :, readout_indices]
    streamed = streaming_forward(network, frames, dt, state=streamed_state)
    clean = torch.from_numpy(streamed.responses)
    baseline_weight = network._param_api().edges.weight.detach().clone()
    identity_gate = build_edge_gate(network, source_types=(source_type,), multiplier=1)
    with synaptic_gate(network, identity_gate):
        identity_weight = network._param_api().edges.weight.detach().clone()
        identity = streaming_forward(network, frames, dt, state=streamed_state)
    zero_gate = build_edge_gate(network, source_types=(source_type,), multiplier=0)
    with synaptic_gate(network, zero_gate):
        zero_weight = network._param_api().edges.weight.detach().clone()
        perturbed = streaming_forward(network, frames, dt, state=streamed_state)
    restored_weight = network._param_api().edges.weight.detach().clone()
    selected = zero_gate.selected
    checks = {
        "gray_equal": bool(torch.allclose(native_state.nodes.activity,
                                            streamed_state.nodes.activity, atol=atol, rtol=rtol)),
        "native_stream_equal": bool(torch.allclose(native, clean, atol=atol, rtol=rtol)),
        "identity_weight_equal": bool(torch.equal(baseline_weight, identity_weight)),
        "identity_trace_equal": bool(np.array_equal(streamed.responses, identity.responses)),
        "mask_nonempty": bool(selected.any()),
        "unselected_weights_unchanged": bool(torch.equal(zero_weight[~selected], baseline_weight[~selected])),
        "selected_weights_zero": bool(torch.equal(zero_weight[selected], torch.zeros_like(zero_weight[selected]))),
        "restored_weights_equal": bool(torch.equal(baseline_weight, restored_weight)),
        "no_state_hooks_written": not bool(getattr(network, "_state_hooks", ())),
        "parameters_unchanged": all(torch.equal(parameters_before[name], p)
                                    for name, p in network.named_parameters()),
        "finite": all(bool(np.isfinite(record.responses).all())
                      for record in (streamed, identity, perturbed)),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        "atol": atol, "rtol": rtol, "frames": len(frames), "adaptation_seconds": t_pre,
        "gray_max_absolute_error": grey_error,
        "native_stream_max_absolute_error": float((native - clean).abs().max()),
        "native_stream_bitwise_equal": bool(torch.equal(native, clean)),
        "source_gate": zero_gate.metadata, "stream_metadata": streamed.metadata,
        "perturbation_max_absolute_change": float(np.max(np.abs(perturbed.responses - streamed.responses))),
        "scope": "short local infrastructure controls; no biological or method ranking",
    }
