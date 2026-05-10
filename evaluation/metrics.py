"""Geometry-quality metrics.

All functions in this module operate on plain ``(N, 3)`` numpy arrays so
they are easy to use outside the Open3D ecosystem (e.g. in unit tests, in
ablation tables, or when comparing meshes against sampled point clouds).

Every metric assumes that the two clouds are already in the same coordinate
frame. In practice this means: run the preprocessing pipeline, run ICP, and
then call these. Unaligned clouds will report nonsense.

Metrics
-------
* ``chamfer_distance`` - symmetric mean nearest-neighbour distance.
* ``f_score``           - precision/recall/F1 at a distance threshold.
* ``hausdorff_distance``- maximum (or robust quantile) deviation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree

import open3d as o3d


ArrayLike = Union[np.ndarray, "o3d.geometry.PointCloud"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_array(x: ArrayLike) -> np.ndarray:
    """Return an ``(N, 3)`` float64 array from numpy or Open3D inputs."""
    if isinstance(x, np.ndarray):
        arr = x
    elif isinstance(x, o3d.geometry.PointCloud):
        arr = np.asarray(x.points)
    else:
        raise TypeError(f"Unsupported point cloud type: {type(x)!r}")

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) array, got {arr.shape}")
    if arr.shape[0] == 0:
        raise ValueError("Empty point cloud passed to a metric")
    return arr.astype(np.float64, copy=False)


def _maybe_voxel_downsample(
    arr: np.ndarray,
    voxel_size: Optional[float],
) -> np.ndarray:
    """Optional voxel downsample for very large clouds.

    KDTree queries scale ~O(N log N), so for clouds with millions of points
    it is significantly faster (and statistically equivalent) to evaluate on
    a uniformly-downsampled subset.
    """
    if voxel_size is None or voxel_size <= 0:
        return arr

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    pcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(pcd.points, dtype=np.float64)


def nearest_neighbor_distances(
    query: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """For each point in ``query`` return the distance to its NN in ``reference``."""
    tree = cKDTree(reference)
    dists, _ = tree.query(query, k=1, workers=-1)
    return dists.astype(np.float64)


# ---------------------------------------------------------------------------
# Chamfer distance
# ---------------------------------------------------------------------------


@dataclass
class ChamferResult:
    chamfer_distance: float
    pred_to_gt_mean: float
    gt_to_pred_mean: float
    n_pred: int
    n_gt: int

    def to_dict(self) -> dict:
        return asdict(self)


def chamfer_distance(
    pred: ArrayLike,
    gt: ArrayLike,
    voxel_size: Optional[float] = None,
    squared: bool = False,
) -> ChamferResult:
    """Symmetric Chamfer distance between two point clouds.

    Definition
    ----------
        CD(P, G) = mean_{p in P} min_{g in G} ||p - g||
                 + mean_{g in G} min_{p in P} ||g - p||

    The symmetric form penalises both *missing* (low recall) and
    *spurious* (low precision) geometry, which is essential when one of the
    clouds may be denser than the other.

    Parameters
    ----------
    pred, gt
        Predicted and ground-truth clouds.
    voxel_size
        If given, both clouds are voxel-downsampled to this resolution
        before the computation. Recommended for clouds > ~1M points.
    squared
        If True return the *squared* form (sum of squared NN distances).
        The non-squared form is more interpretable and is the default.
    """
    p = _maybe_voxel_downsample(_as_array(pred), voxel_size)
    g = _maybe_voxel_downsample(_as_array(gt), voxel_size)

    d_pg = nearest_neighbor_distances(p, g)
    d_gp = nearest_neighbor_distances(g, p)

    if squared:
        d_pg = d_pg ** 2
        d_gp = d_gp ** 2

    return ChamferResult(
        chamfer_distance=float(d_pg.mean() + d_gp.mean()),
        pred_to_gt_mean=float(d_pg.mean()),
        gt_to_pred_mean=float(d_gp.mean()),
        n_pred=int(p.shape[0]),
        n_gt=int(g.shape[0]),
    )


# ---------------------------------------------------------------------------
# F-score (precision / recall at threshold)
# ---------------------------------------------------------------------------


@dataclass
class FScoreResult:
    threshold: float
    precision: float
    recall: float
    f_score: float
    n_pred: int
    n_gt: int

    def to_dict(self) -> dict:
        return asdict(self)


def f_score(
    pred: ArrayLike,
    gt: ArrayLike,
    threshold: float = 0.01,
    voxel_size: Optional[float] = None,
) -> FScoreResult:
    """Precision / recall / F1 of ``pred`` against ``gt`` at distance ``threshold``.

    A *predicted* point counts as correct (true positive for precision) if
    it has at least one GT point within ``threshold``. A *GT* point counts
    as recovered (true positive for recall) if it has at least one
    predicted point within ``threshold``. F1 is the harmonic mean.

    The default threshold of 0.01 assumes scene-scale units of metres
    (i.e. 1 cm), which matches the voxel size used in the existing
    ``gaussian_splatting_from_image_trajectory`` pipeline. Pick a threshold
    that is several voxels wide for stable numbers.
    """
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    p = _maybe_voxel_downsample(_as_array(pred), voxel_size)
    g = _maybe_voxel_downsample(_as_array(gt), voxel_size)

    d_pg = nearest_neighbor_distances(p, g)
    d_gp = nearest_neighbor_distances(g, p)

    precision = float((d_pg < threshold).mean())
    recall = float((d_gp < threshold).mean())

    if precision + recall < 1e-12:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return FScoreResult(
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f_score=float(f1),
        n_pred=int(p.shape[0]),
        n_gt=int(g.shape[0]),
    )


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------


@dataclass
class HausdorffResult:
    hausdorff: float
    directed_pred_to_gt: float
    directed_gt_to_pred: float
    quantile: float
    pred_to_gt_quantile: float
    gt_to_pred_quantile: float

    def to_dict(self) -> dict:
        return asdict(self)


def hausdorff_distance(
    pred: ArrayLike,
    gt: ArrayLike,
    voxel_size: Optional[float] = None,
    quantile: float = 0.95,
) -> HausdorffResult:
    """Symmetric (and robust) Hausdorff distance.

    The classical Hausdorff distance is

        H(P, G) = max( max_p min_g ||p - g||, max_g min_p ||g - p|| )

    which is extremely sensitive to a single outlier. We additionally
    return a robust *quantile* variant (default 95th percentile of the NN
    distances). Use the quantile version for any quantitative comparison;
    the raw maximum is mostly useful for diagnostics.
    """
    p = _maybe_voxel_downsample(_as_array(pred), voxel_size)
    g = _maybe_voxel_downsample(_as_array(gt), voxel_size)

    d_pg = nearest_neighbor_distances(p, g)
    d_gp = nearest_neighbor_distances(g, p)

    return HausdorffResult(
        hausdorff=float(max(d_pg.max(), d_gp.max())),
        directed_pred_to_gt=float(d_pg.max()),
        directed_gt_to_pred=float(d_gp.max()),
        quantile=float(quantile),
        pred_to_gt_quantile=float(np.quantile(d_pg, quantile)),
        gt_to_pred_quantile=float(np.quantile(d_gp, quantile)),
    )


# ---------------------------------------------------------------------------
# One-shot bundle
# ---------------------------------------------------------------------------


def compute_geometry_metrics(
    pred: ArrayLike,
    gt: ArrayLike,
    f_score_threshold: float = 0.01,
    voxel_size: Optional[float] = None,
    include_hausdorff: bool = True,
) -> dict:
    """Compute Chamfer + F-score (+ optional Hausdorff) in one call."""
    cd = chamfer_distance(pred, gt, voxel_size=voxel_size)
    fs = f_score(pred, gt, threshold=f_score_threshold, voxel_size=voxel_size)

    out: dict = {
        "chamfer": cd.to_dict(),
        "f_score": fs.to_dict(),
    }
    if include_hausdorff:
        out["hausdorff"] = hausdorff_distance(pred, gt, voxel_size=voxel_size).to_dict()
    return out
