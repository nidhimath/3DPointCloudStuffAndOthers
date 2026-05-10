"""Image-space metrics for rendered vs ground-truth frames.

Three families:

* ``psnr`` - pixel-wise PSNR. Pure numpy; handles ``uint8`` and float images
  transparently. Returns ``inf`` for identical frames (after casting MSE to
  float).

* ``ssim`` - Structural Similarity Index. Uses
  ``skimage.metrics.structural_similarity`` when available; otherwise falls
  back to a numpy implementation (Gaussian-window SSIM, identical formula).
  This keeps the package usable on systems where ``scikit-image`` is
  unavailable (e.g. minimal CI containers).

* ``lpips_distance`` - perceptual similarity using the official ``lpips``
  package + a pretrained network (default ``alex``). The package is
  *optional*: if it cannot be imported the function emits a single warning
  and returns ``None`` so the rest of the evaluation continues uninterrupted.

All metrics also expose a ``compute_*_sequence`` helper that returns
``{"per_frame": [...], "mean": ..., "median": ..., "std": ...}`` for a
list/array of frame pairs, which is the format the main CLI emits.

Inputs
------
Frames may be:

* ``numpy.ndarray`` of shape ``(H, W)`` or ``(H, W, 3)``
* ``uint8``  (range ``[0, 255]``) or float (range ``[0, 1]``)

The internal canonical form is float in ``[0, 1]``; everything else is
normalised in ``_to_float01``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------


try:
    from skimage.metrics import structural_similarity as _sk_ssim  # type: ignore
    _SKIMAGE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the host install
    _SKIMAGE_AVAILABLE = False


_LPIPS_WARNED = False
_LPIPS_MODEL_CACHE: dict = {}


def _try_load_lpips(net: str = "alex"):
    """Return (model, torch) on success, or None if LPIPS is unavailable."""
    global _LPIPS_WARNED
    try:
        import torch  # noqa: F401  (used by callers via the returned reference)
        import lpips  # type: ignore
    except Exception as e:
        if not _LPIPS_WARNED:
            warnings.warn(
                f"LPIPS unavailable ({type(e).__name__}: {e}). "
                "Install with `pip install lpips` to enable perceptual metrics. "
                "Skipping LPIPS computation.",
                RuntimeWarning,
                stacklevel=2,
            )
            _LPIPS_WARNED = True
        return None

    import torch  # re-import for the local binding

    if net not in _LPIPS_MODEL_CACHE:
        # ``lpips.LPIPS`` downloads weights on first use.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = lpips.LPIPS(net=net, verbose=False)
            model.eval()
        _LPIPS_MODEL_CACHE[net] = model
    return _LPIPS_MODEL_CACHE[net], torch


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _to_float01(img: np.ndarray) -> np.ndarray:
    """Normalise an image to float32 in ``[0, 1]``.

    Integer images (any int dtype) are interpreted as ``[0, 255]``. Float
    images are clipped to ``[0, 1]`` only if they appear to be already in
    that range; otherwise they're rescaled by ``max(|img|)`` with a tiny
    epsilon - this keeps HDR / linear renders from collapsing to all-zeros.
    """
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if np.issubdtype(img.dtype, np.integer):
        return img.astype(np.float32) / 255.0
    arr = img.astype(np.float32)
    if arr.size == 0:
        return arr
    if arr.max() <= 1.0 + 1e-3 and arr.min() >= -1e-3:
        return np.clip(arr, 0.0, 1.0)
    m = float(np.abs(arr).max())
    return np.clip(arr / max(m, 1e-12), 0.0, 1.0)


def _validate_pair(rendered: np.ndarray, gt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if rendered.shape != gt.shape:
        raise ValueError(
            f"Image shape mismatch: rendered {rendered.shape} vs gt {gt.shape}"
        )
    if rendered.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D image, got shape {rendered.shape}")
    return _to_float01(rendered), _to_float01(gt)


def load_image(path: PathLike) -> np.ndarray:
    """Load an image file as RGB uint8.

    Uses OpenCV (always installed in this project) and converts BGR->RGB.
    """
    import cv2  # local import keeps the module importable without OpenCV

    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def load_images_from_dir(
    directory: PathLike,
    pattern: str = "*",
) -> List[np.ndarray]:
    """Load every image in ``directory`` (sorted by filename)."""
    directory = Path(directory)
    paths = sorted(p for p in directory.glob(pattern)
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
    if not paths:
        raise ValueError(f"No images found in {directory}")
    return [load_image(p) for p in paths]


def load_video_frames(
    path: PathLike,
    max_frames: Optional[int] = None,
) -> List[np.ndarray]:
    """Decode a video into a list of RGB ``(H, W, 3)`` uint8 frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise ValueError(f"Video contained no decodable frames: {path}")
    return frames


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------


def psnr(rendered: np.ndarray, gt: np.ndarray, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB.

    PSNR = 10 * log10(MAX^2 / MSE).

    ``data_range`` is the dynamic range of the (already-normalised) input
    (defaults to 1.0 because we always work in ``[0, 1]`` internally).
    """
    r, g = _validate_pair(rendered, gt)
    mse = float(np.mean((r - g) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10((data_range ** 2) / mse))


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------


def _gaussian_kernel_1d(window_size: int, sigma: float) -> np.ndarray:
    coords = np.arange(window_size, dtype=np.float64) - (window_size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


def _filter2d(img: np.ndarray, kernel_1d: np.ndarray) -> np.ndarray:
    """Separable convolution with reflection padding (numpy-only)."""
    pad = len(kernel_1d) // 2
    if img.ndim == 2:
        padded = np.pad(img, pad, mode="reflect")
        # horizontal
        h = np.zeros_like(img, dtype=np.float64)
        for i, w in enumerate(kernel_1d):
            h += w * padded[pad:pad + img.shape[0], i:i + img.shape[1]]
        # vertical
        padded_v = np.pad(h, pad, mode="reflect")
        out = np.zeros_like(img, dtype=np.float64)
        for i, w in enumerate(kernel_1d):
            out += w * padded_v[i:i + img.shape[0], pad:pad + img.shape[1]]
        return out
    return np.stack(
        [_filter2d(img[..., c], kernel_1d) for c in range(img.shape[-1])],
        axis=-1,
    )


def _ssim_numpy(
    a: np.ndarray, b: np.ndarray,
    data_range: float = 1.0, win_size: int = 11, sigma: float = 1.5,
) -> float:
    """Reference SSIM implementation (Wang et al. 2004), numpy-only."""
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    kernel = _gaussian_kernel_1d(win_size, sigma)
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    mu_a = _filter2d(a, kernel)
    mu_b = _filter2d(b, kernel)
    mu_a2 = mu_a ** 2
    mu_b2 = mu_b ** 2
    mu_ab = mu_a * mu_b

    sigma_a2 = _filter2d(a * a, kernel) - mu_a2
    sigma_b2 = _filter2d(b * b, kernel) - mu_b2
    sigma_ab = _filter2d(a * b, kernel) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2)
    return float(np.mean(num / den))


def ssim(
    rendered: np.ndarray, gt: np.ndarray,
    data_range: float = 1.0, win_size: int = 11,
) -> float:
    """Structural similarity index in ``[-1, 1]`` (1 = identical).

    Uses ``skimage.metrics.structural_similarity`` when scikit-image is
    installed (it is faster and well-tested). Otherwise falls back to the
    numpy implementation in this file - same formula, so the values are
    numerically equivalent up to padding conventions.
    """
    r, g = _validate_pair(rendered, gt)

    if _SKIMAGE_AVAILABLE:
        if r.ndim == 2:
            return float(_sk_ssim(r, g, data_range=data_range, win_size=win_size))
        return float(
            _sk_ssim(r, g, data_range=data_range, win_size=win_size, channel_axis=-1)
        )
    return _ssim_numpy(r, g, data_range=data_range, win_size=win_size)


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------


def lpips_distance(
    rendered: np.ndarray, gt: np.ndarray,
    net: str = "alex",
) -> Optional[float]:
    """LPIPS perceptual distance. Returns ``None`` if the package is missing."""
    loaded = _try_load_lpips(net)
    if loaded is None:
        return None
    model, torch = loaded

    r, g = _validate_pair(rendered, gt)
    if r.ndim == 2:
        r = np.stack([r] * 3, axis=-1)
        g = np.stack([g] * 3, axis=-1)

    # LPIPS expects (N, 3, H, W) tensors normalised to [-1, 1].
    def _to_tensor(x):
        t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float()
        return t * 2.0 - 1.0

    with torch.no_grad():
        d = model(_to_tensor(r), _to_tensor(g))
    return float(d.item())


def lpips_batch(
    rendered: Sequence[np.ndarray],
    gt: Sequence[np.ndarray],
    net: str = "alex",
    batch_size: int = 8,
) -> Optional[List[float]]:
    """Batched LPIPS - much faster than per-frame calls.

    Returns a list of per-frame distances, or ``None`` if LPIPS is missing.
    """
    if len(rendered) != len(gt):
        raise ValueError(f"Length mismatch: {len(rendered)} vs {len(gt)}")
    loaded = _try_load_lpips(net)
    if loaded is None:
        return None
    model, torch = loaded

    out: List[float] = []
    with torch.no_grad():
        for s in range(0, len(rendered), batch_size):
            r_chunk = []
            g_chunk = []
            for r_i, g_i in zip(rendered[s:s + batch_size], gt[s:s + batch_size]):
                r, g = _validate_pair(r_i, g_i)
                if r.ndim == 2:
                    r = np.stack([r] * 3, axis=-1)
                    g = np.stack([g] * 3, axis=-1)
                r_chunk.append(r.transpose(2, 0, 1))
                g_chunk.append(g.transpose(2, 0, 1))

            r_t = torch.from_numpy(np.stack(r_chunk)).float() * 2.0 - 1.0
            g_t = torch.from_numpy(np.stack(g_chunk)).float() * 2.0 - 1.0
            d = model(r_t, g_t)
            out.extend([float(x) for x in d.flatten().cpu().numpy()])
    return out


# ---------------------------------------------------------------------------
# Sequence-level helpers
# ---------------------------------------------------------------------------


@dataclass
class SequenceStats:
    per_frame: List[float]
    mean: float
    median: float
    std: float
    min: float
    max: float
    n: int

    def to_dict(self) -> dict:
        return {
            "per_frame": self.per_frame,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "n": self.n,
        }


def _stats(values: Iterable[float]) -> SequenceStats:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return SequenceStats([], float("nan"), float("nan"), float("nan"),
                             float("nan"), float("nan"), 0)
    finite = np.isfinite(arr)
    finite_arr = arr[finite] if finite.any() else arr
    return SequenceStats(
        per_frame=arr.tolist(),
        mean=float(finite_arr.mean()),
        median=float(np.median(finite_arr)),
        std=float(finite_arr.std()),
        min=float(finite_arr.min()),
        max=float(finite_arr.max()),
        n=int(arr.size),
    )


def compute_image_metrics_sequence(
    rendered: Sequence[np.ndarray],
    gt: Sequence[np.ndarray],
    f_score_threshold: float = 0.0,  # unused, kept for symmetry
    lpips_net: str = "alex",
    include_lpips: bool = True,
    ssim_win_size: int = 11,
) -> dict:
    """Run PSNR + SSIM (+ optional LPIPS) over a sequence of frame pairs."""
    if len(rendered) != len(gt):
        raise ValueError(f"Length mismatch: {len(rendered)} vs {len(gt)}")
    if not rendered:
        raise ValueError("Empty image sequence")

    psnr_vals = [psnr(r, g) for r, g in zip(rendered, gt)]
    ssim_vals = [ssim(r, g, win_size=ssim_win_size) for r, g in zip(rendered, gt)]

    out = {
        "psnr": _stats(psnr_vals).to_dict(),
        "ssim": _stats(ssim_vals).to_dict(),
        "skimage_available": _SKIMAGE_AVAILABLE,
    }

    if include_lpips:
        lpips_vals = lpips_batch(rendered, gt, net=lpips_net)
        if lpips_vals is None:
            out["lpips"] = None
            out["lpips_available"] = False
        else:
            out["lpips"] = _stats(lpips_vals).to_dict()
            out["lpips_available"] = True
            out["lpips_net"] = lpips_net

    return out
