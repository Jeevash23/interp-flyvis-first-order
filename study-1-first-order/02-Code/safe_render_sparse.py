"""Faster optional receptor-pixel gather over unchanged native FlyVis geometry.

Separate from safe_render.py so a running dense renderer remains unaffected.
Same pixels, geometry dtype, rasterization and denominators. Omitting zero mask
entries changes floating-point summation order; tight numerical validation is
required and bitwise identity is not promised. No numerical work runs on import.
"""
from contextlib import contextmanager
import hashlib

import numpy as np
import torch

import safe_render as dense_adapter
from flyvis.datasets.rendering import eye as eye_module


_ACTIVE = False


def _sparse_init(self, *args, receptor_chunk=8, **kwargs):
    dense_adapter._bounded_init(self, *args, receptor_chunk=receptor_chunk, **kwargs)
    # One contiguous scan avoids 721 separate strided passes over the full mask.
    pixel_receptor = self.is_inside.nonzero(as_tuple=False)
    order = torch.argsort(pixel_receptor[:, 1], stable=True)
    sorted_pixels = pixel_receptor[order, 0].contiguous()
    counts = torch.bincount(pixel_receptor[:, 1], minlength=self.n_ommatidia)
    if not torch.equal(counts, self.kernel_sum):
        raise AssertionError("Sparse membership counts differ from native mask counts.")
    self._ai06_pixel_lists = sorted_pixels.split(counts.tolist())
    self._ai06_total_memberships = len(sorted_pixels)


def _sparse_call(self, stim, mode="mean", n_chunks=1):
    dense_adapter._cpu_device(stim.device)
    if mode not in ("mean", "sum"):
        raise ValueError("Only native mean/sum pooling is supported.")
    if stim.ndim == 3:
        h, w = stim.shape[1:]
        if h < self.monitor_height_px or w < self.monitor_width_px:
            stim = eye_module.ttf.resize(
                stim, [self.monitor_height_px, self.monitor_width_px]
            )
        stim = stim.reshape(stim.shape[0], -1)
    if stim.ndim != 2 or stim.shape[1] != self.is_inside.shape[0]:
        raise ValueError("Stimulus must contain the complete native monitor pixels.")
    if not bool(torch.isfinite(stim).all()):
        # Omitting mask-zero pixels would otherwise change NaN/Inf semantics.
        raise ValueError("Sparse rendering requires finite Cartesian pixel values.")
    values = []
    for pixels in self._ai06_pixel_lists:
        # Stable grouping above retains ascending native pixel order per receptor.
        # Incoming dtype is preserved, including float64 native moving bars.
        pooled = stim[:, pixels].sum(dim=1)
        if mode == "mean":
            pooled = pooled / len(pixels)
        values.append(pooled)
    return torch.stack(values, dim=1)


@contextmanager
def safe_sparse_renderer(receptor_chunk=8):
    """Runtime-only global class patch; single worker, no nested patching."""
    global _ACTIVE
    if _ACTIVE or dense_adapter._ACTIVE:
        raise RuntimeError("Concurrent/nested dense or sparse patches are forbidden.")
    if not isinstance(receptor_chunk, int) or not 1 <= receptor_chunk <= 16:
        raise ValueError("Geometry receptor chunk must be from 1 to 16.")
    original_init = eye_module.HexEye.__init__
    original_call = eye_module.HexEye.__call__

    def patched_init(self, *args, **kwargs):
        return _sparse_init(self, *args, receptor_chunk=receptor_chunk, **kwargs)

    eye_module.HexEye.__init__ = patched_init
    eye_module.HexEye.__call__ = _sparse_call
    _ACTIVE = True
    try:
        yield
    finally:
        eye_module.HexEye.__init__ = original_init
        eye_module.HexEye.__call__ = original_call
        _ACTIVE = False


def _images(eye, angles=(0, 30), include_arbitrary=True):
    stimuli = []
    if include_arbitrary:
        n = eye.monitor_height_px * eye.monitor_width_px
        stimuli.append((torch.arange(n, dtype=torch.float64) % 17 / 17)[None])
    for angle in angles:
        for intensity in (0.0, 1.0):
            cartesian = eye.render_bar(
                bar_width_rad=np.radians(180), bar_height_rad=np.radians(180),
                bar_loc_theta=np.radians(0), bar_loc_phi=np.radians(90),
                n_bars=1, bar_intensity=intensity, bg_intensity=0.5,
                moving_angle=angle, cartesian=True,
            )
            stimuli.append(torch.tensor(cartesian.reshape(1, -1), dtype=torch.float64))
    return torch.cat(stimuli, dim=0)


def validate_small_sparse_renderer(atol=1e-12, rtol=1e-12, receptor_chunk=8):
    """Original native small-eye geometry and pooling versus sparse path."""
    if _ACTIVE or dense_adapter._ACTIVE:
        raise RuntimeError("Run validation outside patch contexts.")
    original_init = eye_module.HexEye.__init__
    original_call = eye_module.HexEye.__call__
    native = eye_module.HexEye(19, 5, device=torch.device("cpu"))
    stimuli = _images(native, angles=(0, 30, 90))
    expected_mean = native(stimuli, mode="mean")
    expected_sum = native(stimuli, mode="sum")
    with safe_sparse_renderer(receptor_chunk):
        sparse = eye_module.HexEye(19, 5, device=torch.device("cpu"))
        actual_mean = sparse(stimuli, mode="mean")
        actual_sum = sparse(stimuli, mode="sum")
    checks = {
        "geometry_mask_equal": bool(torch.equal(native.is_inside, sparse.is_inside)),
        "kernel_count_equal": bool(torch.equal(native.kernel_sum, sparse.kernel_sum)),
        "mean_close": bool(torch.allclose(expected_mean, actual_mean, atol=atol, rtol=rtol)),
        "sum_close": bool(torch.allclose(expected_sum, actual_sum, atol=atol, rtol=rtol)),
        "finite": bool(torch.isfinite(actual_mean).all() and torch.isfinite(actual_sum).all()),
        "original_methods_restored": (eye_module.HexEye.__init__ is original_init
                                      and eye_module.HexEye.__call__ is original_call),
    }
    return {
        "status": "pass" if all(checks.values()) else "fail", "checks": checks,
        "n_ommatidia": 19, "ppo": 5, "atol": atol, "rtol": rtol,
        "mean_bitwise_equal": bool(torch.equal(expected_mean, actual_mean)),
        "sum_bitwise_equal": bool(torch.equal(expected_sum, actual_sum)),
        "max_mean_absolute_error": float((actual_mean - expected_mean).abs().max()),
        "max_sum_absolute_error": float((actual_sum - expected_sum).abs().max()),
        "memberships": sparse._ai06_total_memberships,
        "stimulus_sha256": hashlib.sha256(stimuli.numpy().tobytes()).hexdigest(),
    }


def validate_full_sparse_renderer(atol=1e-12, rtol=1e-12, receptor_chunk=8):
    """Full 721-receptor grid, four native ON/OFF frames, dense versus sparse.

    Reference uses previously small-eye-validated receptor-chunked dense pooling
    to avoid re-triggering the unsafe native giant allocation. This is explicitly
    a layered check, not a full-size original-native allocation.
    """
    if _ACTIVE or dense_adapter._ACTIVE:
        raise RuntimeError("Run validation outside patch contexts.")
    with safe_sparse_renderer(receptor_chunk):
        sparse = eye_module.HexEye(721, 25, device=torch.device("cpu"))
        stimuli = _images(sparse, angles=(0, 30), include_arbitrary=False)
        actual = sparse(stimuli)
        expected = dense_adapter._bounded_call(
            sparse, stimuli, mode="mean", receptor_chunk=receptor_chunk
        )
    finite = bool(torch.isfinite(actual).all())
    close = bool(torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return {
        "status": "pass" if finite and close else "fail", "finite": finite,
        "n_ommatidia": 721, "ppo": 25, "frames": len(stimuli),
        "reference": "receptor-chunked complete-pixel dense reduction",
        "atol": atol, "rtol": rtol, "close": close,
        "bitwise_equal": bool(torch.equal(actual, expected)),
        "max_absolute_error": float((actual - expected).abs().max()),
        "memberships": sparse._ai06_total_memberships,
        "stimulus_sha256": hashlib.sha256(stimuli.numpy().tobytes()).hexdigest(),
    }
