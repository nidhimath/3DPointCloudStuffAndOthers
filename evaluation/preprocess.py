"""Point cloud preprocessing utilities.

These helpers exist because raw reconstructions (especially those coming from
single-image-conditioned pipelines) tend to differ in:

* Density - the predicted cloud may be far denser or sparser than the GT.
* Origin - clouds are not generally centred at the same world point.
* Scale  - monocular depth pipelines produce up-to-scale geometry.
* Noise  - back-projected depth maps contain a lot of stray points around
  edges and at depth discontinuities.

Every metric in this package expects a *normalised, cleaned* cloud, so the
``prepare_for_metrics`` function bundles all of these steps into one call and
documents the order of operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import open3d as o3d


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# I/O and conversions
# ---------------------------------------------------------------------------


def load_point_cloud(path: PathLike) -> o3d.geometry.PointCloud:
    """Load a ``.ply`` (or any Open3D-supported format) into a PointCloud.

    Notes
    -----
    Open3D's reader silently returns an empty cloud when the file is malformed
    or the path is wrong, so we explicitly validate the result. This avoids
    the classic "all metrics are zero" failure mode where downstream code
    happily processes an empty cloud.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud not found: {path}")

    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        raise ValueError(f"Loaded point cloud is empty: {path}")
    return pcd


def numpy_to_pcd(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> o3d.geometry.PointCloud:
    """Convert ``(N, 3)`` numpy array (and optional colours) into a PointCloud."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) points, got {points.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if colors is not None:
        if colors.shape != points.shape:
            raise ValueError(
                f"colors shape {colors.shape} does not match points {points.shape}"
            )
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return pcd


def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Return the ``(N, 3)`` numpy view of an Open3D point cloud."""
    return np.asarray(pcd.points, dtype=np.float64)


# ---------------------------------------------------------------------------
# Cleanup primitives
# ---------------------------------------------------------------------------


def voxel_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Voxel downsample to ``voxel_size`` units.

    Voxel downsampling makes the per-point density approximately uniform,
    which is critical for a fair Chamfer / F-score comparison: dense regions
    would otherwise dominate the mean nearest-neighbour distance.
    """
    if voxel_size <= 0:
        return pcd
    return pcd.voxel_down_sample(voxel_size)


def remove_outliers(
    pcd: o3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
    """Statistical outlier removal.

    Points whose mean distance to their ``nb_neighbors`` nearest neighbours is
    further than ``std_ratio`` standard deviations from the global mean are
    discarded. This removes the classic "flying" points that come from depth
    discontinuities at object silhouettes.
    """
    if len(pcd.points) == 0:
        return pcd
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return cleaned


def center_to_origin(
    pcd: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """Translate the cloud so its centroid is at the origin.

    Returns the centred cloud plus the centroid that was subtracted, in case
    the caller needs to undo the transform later (e.g. for visualisation in
    the original frame).
    """
    pts = pcd_to_numpy(pcd)
    centroid = pts.mean(axis=0)
    translated = pts - centroid

    out = o3d.geometry.PointCloud(pcd)
    out.points = o3d.utility.Vector3dVector(translated)
    return out, centroid


def normalize_scale(
    pcd: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, float]:
    """Scale the cloud so that the RMS distance from the origin is 1.

    RMS is preferred over max-abs (which is dominated by outliers) and over
    the bounding-box diagonal (which is dominated by the most distant single
    point, again often an outlier).
    """
    pts = pcd_to_numpy(pcd)
    rms = float(np.sqrt(np.mean(np.sum(pts ** 2, axis=1))))
    if rms < 1e-12:
        return pcd, 1.0

    out = o3d.geometry.PointCloud(pcd)
    out.points = o3d.utility.Vector3dVector(pts / rms)
    return out, rms


def estimate_normals(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 30,
) -> o3d.geometry.PointCloud:
    """Estimate per-point normals.

    Normals are required for point-to-plane ICP, which is significantly more
    robust than plain point-to-point ICP on man-made (planar) scenes.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    return pcd


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------


@dataclass
class PreprocessConfig:
    """Configuration for ``prepare_for_metrics``."""

    voxel_size: float = 0.01            # 1 cm in metric scenes; tune per-dataset
    remove_outliers: bool = True
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    center: bool = True
    normalize_scale: bool = False       # off by default - geometry metrics
                                        # are usually reported in scene units
    estimate_normals: bool = True
    normal_radius_factor: float = 2.0   # normal-search radius = factor * voxel
    max_normal_nn: int = 30


def prepare_for_metrics(
    pcd: o3d.geometry.PointCloud,
    config: Optional[PreprocessConfig] = None,
) -> Tuple[o3d.geometry.PointCloud, dict]:
    """Run the standard preprocessing pipeline.

    Order matters:
        downsample -> outlier removal -> center -> (optional) scale norm -> normals

    We downsample *before* outlier removal because statistical outlier removal
    on an extremely dense cloud is wasteful and the outlier statistics become
    unstable when neighbourhoods are tiny. Centering must happen before
    optional scale normalisation so that the RMS is computed about the
    centroid rather than about an arbitrary origin.
    """
    config = config or PreprocessConfig()

    info: dict = {"input_points": len(pcd.points)}

    pcd = voxel_downsample(pcd, config.voxel_size)
    info["after_voxel"] = len(pcd.points)

    if config.remove_outliers:
        pcd = remove_outliers(
            pcd,
            nb_neighbors=config.outlier_neighbors,
            std_ratio=config.outlier_std_ratio,
        )
        info["after_outliers"] = len(pcd.points)

    if config.center:
        pcd, centroid = center_to_origin(pcd)
        info["centroid"] = centroid.tolist()

    if config.normalize_scale:
        pcd, rms = normalize_scale(pcd)
        info["scale_rms"] = rms

    if config.estimate_normals:
        radius = max(config.normal_radius_factor * config.voxel_size, 1e-3)
        pcd = estimate_normals(pcd, radius=radius, max_nn=config.max_normal_nn)

    return pcd, info
