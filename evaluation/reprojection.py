"""Reprojection Consistency Error (RCE).

We project the reconstructed 3D point cloud back into each frame's image
plane and compare against what the camera actually saw. This is the most
direct test of *2D-3D consistency*: a geometrically faithful reconstruction
should reproject to colours that match the observed pixels in every frame.

Why this matters
----------------
* Pure geometric metrics (Chamfer / F-score) only care about positions, not
  appearance. A cloud with the right shape but the wrong colours, or with
  geometry that disagrees with where the cameras claim to have stood, can
  still get a great Chamfer score.
* Pose-only metrics (ATE / RPE) say nothing about geometry.
* RCE couples both: it requires the cloud, the intrinsics *and* the
  extrinsics to agree with the observed images. Any error in any of them
  raises the residual.

Two flavours are implemented
----------------------------
1. ``photometric_reprojection_error`` - for each frame, project every 3D
   point that falls in front of the camera and inside the image, sample
   the frame at that pixel, and compare with the point's stored colour.
   Returns mean L1 / RMSE residuals plus a coverage ratio.

2. ``rasterize_point_cloud`` - render a sparse splat of the cloud with a
   simple z-buffer. Useful as a quick "rendered frame" that can then be
   fed into PSNR/SSIM/LPIPS in ``image_metrics``.

Conventions
-----------
We work in OpenCV camera convention (z-forward, y-down). Trajectories
loaded from ``pose_metrics.load_trajectory`` are already in this form.
``c2w`` is camera-to-world, ``w2c`` is world-to-camera; converting between
them is just ``np.linalg.inv``. Intrinsics ``K`` is the standard

    [ fx  0  cx ]
    [  0 fy  cy ]
    [  0  0   1 ].
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import open3d as o3d


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


def intrinsics_from_transforms_json(path: PathLike) -> Tuple[np.ndarray, int, int]:
    """Load ``(K, W, H)`` from a Nerfstudio-style ``transforms.json``."""
    with open(path, "r") as f:
        data = json.load(f)

    fx = float(data["fl_x"])
    fy = float(data["fl_y"])
    cx = float(data["cx"])
    cy = float(data["cy"])
    W = int(data["w"])
    H = int(data["h"])

    K = np.array([[fx, 0.0, cx],
                  [0.0, fy, cy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return K, W, H


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def project_points(
    points_world: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    image_size: Tuple[int, int],   # (H, W)
    z_min: float = 1e-3,
    z_max: float = 1e6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points into a single image.

    Returns
    -------
    uv  : (N, 2) float pixel coordinates (column, row), -1 where invalid.
    z   : (N,)   camera-space depth (positive = in front of camera).
    visible : (N,) bool mask = (z in [z_min, z_max]) AND uv inside image.
    """
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"Expected (N, 3), got {points_world.shape}")
    if w2c.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) w2c, got {w2c.shape}")

    H, W = image_size
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # World -> camera.
    Pc = points_world @ R.T + t        # (N, 3)
    z = Pc[:, 2]

    in_front = (z > z_min) & (z < z_max)
    z_safe = np.where(in_front, z, 1.0)  # avoid divide-by-zero in masked entries

    uv = (Pc[:, :2] / z_safe[:, None])   # normalised image plane
    uv = uv @ K[:2, :2].T + K[:2, 2]     # apply fx,fy and cx,cy

    inside = (uv[:, 0] >= 0) & (uv[:, 0] < W) & \
             (uv[:, 1] >= 0) & (uv[:, 1] < H)
    visible = in_front & inside

    uv_out = np.where(visible[:, None], uv, -1.0)
    return uv_out, z, visible


def _sample_image(
    image: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
) -> np.ndarray:
    """Bilinearly sample ``image`` at ``uv`` for visible points.

    Returns (N, C) sampled colours; rows for invisible points are zero.
    """
    if image.ndim == 2:
        image = image[..., None]
    H, W, C = image.shape
    out = np.zeros((uv.shape[0], C), dtype=np.float64)
    if not visible.any():
        return out

    img = image.astype(np.float64)
    u = uv[visible, 0]
    v = uv[visible, 1]
    u0 = np.clip(np.floor(u).astype(np.int64), 0, W - 1)
    v0 = np.clip(np.floor(v).astype(np.int64), 0, H - 1)
    u1 = np.clip(u0 + 1, 0, W - 1)
    v1 = np.clip(v0 + 1, 0, H - 1)
    du = (u - u0)[:, None]
    dv = (v - v0)[:, None]

    Ia = img[v0, u0]
    Ib = img[v0, u1]
    Ic = img[v1, u0]
    Id = img[v1, u1]

    sampled = (Ia * (1 - du) * (1 - dv) +
               Ib * du       * (1 - dv) +
               Ic * (1 - du) * dv       +
               Id * du       * dv)
    out[visible] = sampled
    return out


# ---------------------------------------------------------------------------
# Rasterizer (sparse splat with z-buffer)
# ---------------------------------------------------------------------------


def rasterize_point_cloud(
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
    image_size: Tuple[int, int],   # (H, W)
    point_radius_px: int = 1,
    background: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Tiny z-buffer rasterizer for an ``(N, 3)`` coloured cloud.

    Each visible point is splatted into a ``(2r+1) x (2r+1)`` square; the
    nearest sample wins per pixel. This is *not* a production rasterizer
    (no ellipsoid splats, no anti-aliasing, no occlusion culling beyond the
    z-buffer), but it is a faithful "what does this point cloud look like
    from this camera?" preview suitable for PSNR/SSIM/LPIPS sanity checks.
    """
    H, W = image_size
    bg = np.asarray(background, dtype=np.float32)
    image = np.tile(bg[None, None, :], (H, W, 1))
    z_buffer = np.full((H, W), np.inf, dtype=np.float64)

    uv, z, visible = project_points(points_world, K, w2c, image_size)
    if not visible.any():
        return image

    uv_v = uv[visible]
    z_v = z[visible]
    c_v = colors[visible].astype(np.float32)

    # Sort by depth so closer points overwrite farther ones cleanly.
    order = np.argsort(-z_v)  # render far-to-near
    uv_v = uv_v[order]
    z_v = z_v[order]
    c_v = c_v[order]

    r = max(int(point_radius_px), 0)
    for (u, v), zi, ci in zip(uv_v, z_v, c_v):
        u0 = max(int(round(u)) - r, 0)
        u1 = min(int(round(u)) + r + 1, W)
        v0 = max(int(round(v)) - r, 0)
        v1 = min(int(round(v)) + r + 1, H)
        if u0 >= u1 or v0 >= v1:
            continue
        sub_z = z_buffer[v0:v1, u0:u1]
        mask = zi < sub_z
        sub_z[mask] = zi
        z_buffer[v0:v1, u0:u1] = sub_z
        sub_img = image[v0:v1, u0:u1]
        sub_img[mask] = ci
        image[v0:v1, u0:u1] = sub_img

    return np.clip(image, 0.0, 1.0)


def rasterize_sequence(
    points_world: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,           # (M, 4, 4) camera-to-world (OpenCV)
    image_size: Tuple[int, int],
    **kwargs,
) -> List[np.ndarray]:
    """Rasterize a coloured cloud once per camera in ``c2w``."""
    out = []
    for c in c2w:
        w2c = np.linalg.inv(c)
        out.append(rasterize_point_cloud(points_world, colors, K, w2c, image_size, **kwargs))
    return out


# ---------------------------------------------------------------------------
# Reprojection Consistency Error
# ---------------------------------------------------------------------------


@dataclass
class FrameRCE:
    frame_index: int
    n_visible: int
    coverage: float       # n_visible / n_points
    mean_l1: float        # mean L1 colour residual in [0, 1]
    rmse: float           # RMSE colour residual
    mean_pixel_l2: Optional[float]  # mean 2D distance to nearest non-bg pixel (if requested)


@dataclass
class RCEResult:
    per_frame: List[FrameRCE]
    mean_l1: float
    rmse: float
    coverage_mean: float
    n_frames: int
    n_points: int
    image_size: Tuple[int, int]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_frame"] = [asdict(f) for f in self.per_frame]
        return d


def reprojection_consistency_error(
    points_world: np.ndarray,
    point_colors: np.ndarray,        # (N, 3) in [0, 1]
    K: np.ndarray,
    c2w: np.ndarray,                 # (M, 4, 4) camera-to-world (OpenCV)
    images: Sequence[np.ndarray],    # list of (H, W, 3) frames
    image_size: Optional[Tuple[int, int]] = None,
    z_min: float = 1e-3,
    z_max: float = 1e6,
) -> RCEResult:
    """Photometric reprojection consistency.

    For each frame:
      1. Project all 3D points using ``K`` and the inverse of ``c2w[i]``.
      2. Keep points that are in front of the camera AND inside the image.
      3. Bilinearly sample the frame at each visible pixel (observed
         colour) and compare against the *stored* colour of the 3D point.
      4. Aggregate: mean L1 + RMSE residual, plus coverage ratio.

    Notes on what "small" means
    ---------------------------
    A perfect reconstruction would still have non-zero RCE because:
    * the stored colour of a 3D point is a single value but reality has
      view-dependent shading (specularities, etc.);
    * the trajectory comes from a different optimisation than the cloud,
      so even sub-pixel calibration error shows up as a colour mismatch.
    Treat RCE as a *relative* metric: compare ground-truth-trajectory
    reconstructions against alternative-trajectory reconstructions of the
    same scene; the smaller-RCE one is more 2D-3D-consistent.
    """
    n_points = points_world.shape[0]
    if point_colors.shape != points_world.shape:
        raise ValueError(
            f"colors {point_colors.shape} mismatched with points {points_world.shape}"
        )
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"c2w must be (M, 4, 4), got {c2w.shape}")

    if len(images) != len(c2w):
        n = min(len(images), len(c2w))
        c2w = c2w[np.linspace(0, len(c2w) - 1, n).round().astype(int)]
        images = list(images)
        if len(images) != n:
            idx = np.linspace(0, len(images) - 1, n).round().astype(int)
            images = [images[i] for i in idx]

    if image_size is None:
        h, w = images[0].shape[:2]
        image_size = (h, w)

    pt_colors = np.clip(point_colors.astype(np.float64), 0.0, 1.0)

    per_frame: List[FrameRCE] = []
    all_l1 = []
    all_sq = []

    from evaluation.image_metrics import _to_float01  # local: keep import optional

    for i, (c2w_i, img) in enumerate(zip(c2w, images)):
        w2c = np.linalg.inv(c2w_i)
        uv, _, visible = project_points(
            points_world, K, w2c, image_size, z_min=z_min, z_max=z_max
        )
        n_v = int(visible.sum())
        if n_v == 0:
            per_frame.append(FrameRCE(
                frame_index=i, n_visible=0, coverage=0.0,
                mean_l1=float("nan"), rmse=float("nan"),
                mean_pixel_l2=None,
            ))
            continue

        img_f = _to_float01(img)
        sampled = _sample_image(img_f, uv, visible)
        diff = pt_colors[visible] - sampled[visible]

        l1 = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        per_frame.append(FrameRCE(
            frame_index=i,
            n_visible=n_v,
            coverage=n_v / max(n_points, 1),
            mean_l1=l1,
            rmse=rmse,
            mean_pixel_l2=None,
        ))
        all_l1.append(l1)
        all_sq.append(rmse ** 2)

    if all_l1:
        mean_l1 = float(np.mean(all_l1))
        rmse = float(np.sqrt(np.mean(all_sq)))
    else:
        mean_l1 = float("nan")
        rmse = float("nan")
    cov_mean = float(np.mean([f.coverage for f in per_frame]))

    return RCEResult(
        per_frame=per_frame,
        mean_l1=mean_l1,
        rmse=rmse,
        coverage_mean=cov_mean,
        n_frames=len(per_frame),
        n_points=int(n_points),
        image_size=tuple(image_size),
    )


# ---------------------------------------------------------------------------
# Convenience: pull (xyz, rgb) out of an Open3D point cloud
# ---------------------------------------------------------------------------


def pcd_to_xyz_rgb(pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(N, 3)`` positions and ``(N, 3)`` colours in ``[0, 1]``.

    If the cloud has no colours, returns mid-grey for every point so
    photometric metrics still produce a finite (albeit uninformative) value.
    """
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if pcd.has_colors():
        rgb = np.asarray(pcd.colors, dtype=np.float64)
    else:
        rgb = np.full_like(xyz, 0.5)
    return xyz, rgb
