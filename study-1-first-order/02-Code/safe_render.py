"""Bounded-memory, CPU-only rendering with native FlyVis geometry and reductions.

Runtime patches only; never edits FlyVis or datamate's rendering cache. Receptor
chunking preserves each receptor's complete pixel reduction and float64 bar
values. Floating-point equality must be measured, not assumed. This module does
not run validation or build stimuli on import.

Derived from FlyVis 1.2.0 HexEye (MIT; Lappalainen et al., 2023). Its license is
retained in the project's installed flyvis distribution and source records.
"""
from contextlib import contextmanager
import hashlib

import numpy as np
import torch

from flyvis.datasets.rendering import eye as eye_module


_ACTIVE = False


def _cpu_device(device):
    device = torch.device("cpu") if device is None else torch.device(device)
    if device.type != "cpu":
        raise ValueError("The bounded qualification renderer is CPU-only.")
    return device


def _bounded_init(
    self, n_ommatidia=721, ppo=25, monitor_height_px=None,
    monitor_width_px=None, device=None, dtype=torch.float16, *, receptor_chunk=8,
):
    """Native constructor algebra; evaluate geometry and sums by receptor."""
    device = _cpu_device(device)
    n_hex_circfer = 2 * (-1 / 2 + np.sqrt(1 / 4 - ((1 - n_ommatidia) / 3))) + 1
    if n_hex_circfer % 1 != 0:
        raise ValueError(f"{n_ommatidia} does not fill a regular hex grid.")
    self.monitor_width_px = monitor_width_px or ppo * int(n_hex_circfer)
    self.monitor_height_px = monitor_height_px or ppo * int(n_hex_circfer)
    x_hc, y_hc, (dist_w, dist_h) = eye_module.hex_center_coordinates(
        n_ommatidia, self.monitor_width_px, self.monitor_height_px
    )
    # Same ordering as native itertools.product(x_pixels, y_pixels).
    x_img = np.repeat(np.arange(self.monitor_width_px), self.monitor_height_px)
    y_img = np.tile(np.arange(self.monitor_height_px), self.monitor_width_px)
    y_tensor = torch.tensor(y_img, dtype=dtype, device=device)
    x_tensor = torch.tensor(x_img, dtype=dtype, device=device)
    x_centers = torch.tensor(x_hc, dtype=dtype, device=device)
    y_centers = torch.tensor(y_hc, dtype=dtype, device=device)
    edge_distance = torch.tensor((dist_w + dist_h) / 4, dtype=dtype, device=device)
    tilt = torch.tensor(np.radians(0), dtype=dtype, device=device)
    self.is_inside = torch.empty(
        (len(x_img), n_ommatidia), dtype=torch.bool, device=device
    )
    self.kernel_sum = torch.empty(n_ommatidia, dtype=torch.int64, device=device)
    for first in range(0, n_ommatidia, receptor_chunk):
        last = min(first + receptor_chunk, n_ommatidia)
        # Preserve native x/y argument ordering, geometry dtype and algorithm.
        _, membership = eye_module.is_inside_hex(
            y_tensor, x_tensor, x_centers[first:last], y_centers[first:last],
            edge_distance, tilt, device=device, dtype=dtype,
        )
        self.is_inside[:, first:last] = membership
        self.kernel_sum[first:last] = membership.sum(dim=0)
    if bool((self.kernel_sum == 0).any()):
        raise ValueError("Geometry contains an empty receptor; cannot average safely.")
    self.n_ommatidia = n_ommatidia
    self.omm_width_rad = np.radians(5.8)
    self.omm_height_rad = np.radians(5.8)
    self.ppo = ppo or int(
        (self.monitor_width_px + self.monitor_width_px) / (2 * n_hex_circfer)
    )
    self.n_hex_circfer = n_hex_circfer
    self.device = device
    self.dtype = dtype


def _bounded_call(self, stim, mode="mean", n_chunks=1, *, receptor_chunk=8):
    """Reduce full pixel vectors; only receptor and frame axes are chunked."""
    _cpu_device(stim.device)
    if mode not in ("mean", "sum"):
        raise ValueError("Only native mean/sum rendering is qualified by this adapter.")
    if stim.ndim == 3:
        h, w = stim.shape[1:]
        if h < self.monitor_height_px or w < self.monitor_width_px:
            stim = eye_module.ttf.resize(
                stim, [self.monitor_height_px, self.monitor_width_px]
            )
        stim = stim.reshape(stim.shape[0], -1)
    if stim.ndim != 2 or stim.shape[1] != self.is_inside.shape[0]:
        raise ValueError("Stimulus must have the native monitor's complete pixels.")
    # Native render_bar produces float64. Deliberately preserve the incoming
    # dtype rather than trade precision for memory. No input is modified.
    frames = []
    for frame in stim.split(1, dim=0):
        receptor_values = []
        for first in range(0, self.n_ommatidia, receptor_chunk):
            last = min(first + receptor_chunk, self.n_ommatidia)
            membership = self.is_inside[:, first:last]
            value = (frame[:, :, None] * membership).sum(dim=1)
            if mode == "mean":
                value = value / self.kernel_sum[first:last]
            receptor_values.append(value)
        frames.append(torch.cat(receptor_values, dim=1))
    if not frames:
        raise ValueError("At least one frame is required.")
    return torch.cat(frames, dim=0)


@contextmanager
def safe_renderer(receptor_chunk=8):
    """Temporarily patch HexEye; restore exact originals even on failure.

    Use for MovingEdge construction only. Not thread-safe, not nestable, and
    intentionally rejects CUDA and unvalidated median pooling.
    """
    global _ACTIVE
    if not isinstance(receptor_chunk, int) or not 1 <= receptor_chunk <= 16:
        raise ValueError("receptor_chunk must be an integer from 1 to 16.")
    if _ACTIVE:
        raise RuntimeError("Nested/global concurrent renderer patches are forbidden.")
    original_init = eye_module.HexEye.__init__
    original_call = eye_module.HexEye.__call__

    def patched_init(self, *args, **kwargs):
        return _bounded_init(self, *args, **kwargs, receptor_chunk=receptor_chunk)

    def patched_call(self, stim, mode="mean", n_chunks=1):
        return _bounded_call(
            self, stim, mode=mode, n_chunks=n_chunks, receptor_chunk=receptor_chunk
        )

    _ACTIVE = True
    eye_module.HexEye.__init__ = patched_init
    eye_module.HexEye.__call__ = patched_call
    try:
        yield
    finally:
        eye_module.HexEye.__init__ = original_init
        eye_module.HexEye.__call__ = original_call
        _ACTIVE = False


def validate_small_renderer(receptor_chunk=8, atol=1e-12, rtol=1e-12):
    """Compare independent native and patched small-eye paths when called.

    Includes noninteger inputs and rotated bars. This executes only after an
    explicit caller invocation. A small-eye pass does not certify full stimuli.
    """
    if _ACTIVE:
        raise RuntimeError("Run native-versus-safe validation outside safe_renderer.")
    native = eye_module.HexEye(19, 5, device=torch.device("cpu"))
    pixels = native.monitor_height_px * native.monitor_width_px
    arbitrary = (torch.arange(pixels, dtype=torch.float64) % 17 / 17)[None]
    native_mask = native.is_inside.clone()
    native_kernel = native.kernel_sum.clone()
    stimuli = [arbitrary]
    for angle in (0, 30, 90):
        for intensity in (0.0, 1.0):
            image = native.render_bar(
                bar_width_rad=np.radians(20), bar_height_rad=np.radians(30),
                bar_loc_theta=np.radians(90), bar_loc_phi=np.radians(90),
                n_bars=1, bar_intensity=intensity, bg_intensity=0.5,
                moving_angle=angle, cartesian=True,
            )
            stimuli.append(torch.tensor(image.reshape(1, -1), dtype=torch.float64))
    stimuli = torch.cat(stimuli, dim=0)
    expected_mean = native(stimuli, mode="mean")
    expected_sum = native(stimuli, mode="sum")
    native_class_init = eye_module.HexEye.__init__
    native_class_call = eye_module.HexEye.__call__
    with safe_renderer(receptor_chunk):
        adapted = eye_module.HexEye(19, 5, device=torch.device("cpu"))
        actual_mean = adapted(stimuli, mode="mean")
        actual_sum = adapted(stimuli, mode="sum")
    mask_equal = bool(torch.equal(native_mask, adapted.is_inside))
    kernel_equal = bool(torch.equal(native_kernel, adapted.kernel_sum))
    finite = bool(torch.isfinite(actual_mean).all() and torch.isfinite(actual_sum).all())
    mean_close = bool(torch.allclose(actual_mean, expected_mean, atol=atol, rtol=rtol))
    sum_close = bool(torch.allclose(actual_sum, expected_sum, atol=atol, rtol=rtol))
    restored = (eye_module.HexEye.__init__ is native_class_init
                and eye_module.HexEye.__call__ is native_class_call)
    return {
        "status": "pass" if all((mask_equal, kernel_equal, finite, mean_close,
                                  sum_close, restored)) else "fail",
        "n_ommatidia": 19, "ppo": 5, "receptor_chunk": receptor_chunk,
        "dtype": str(stimuli.dtype), "geometry_mask_equal": mask_equal,
        "kernel_sum_equal": kernel_equal, "finite": finite,
        "mean_close": mean_close, "sum_close": sum_close,
        "mean_bitwise_equal": bool(torch.equal(actual_mean, expected_mean)),
        "sum_bitwise_equal": bool(torch.equal(actual_sum, expected_sum)),
        "max_mean_absolute_error": float((actual_mean - expected_mean).abs().max()),
        "max_sum_absolute_error": float((actual_sum - expected_sum).abs().max()),
        "atol": atol, "rtol": rtol, "original_methods_restored": restored,
        "stimulus_sha256": hashlib.sha256(stimuli.numpy().tobytes()).hexdigest(),
    }


def renderer_metadata(receptor_chunk=8):
    """Static provenance, without constructing an eye or numerical input."""
    return {
        "adapter": "native_flyvis_receptor_chunking",
        "native_geometry_dtype": "torch.float16",
        "native_bar_dtype": "torch.float64",
        "receptor_chunk": receptor_chunk, "frame_chunk": 1,
        "pixel_reduction": "complete native pixel vector per receptor",
        "changes_resolution": False, "changes_rasterization": False,
        "changes_installed_package_files": False,
        "full_size_boolean_mask_bytes": 775 * 775 * 721,
        "maximum_float64_product_bytes": 775 * 775 * receptor_chunk * 8,
    }
