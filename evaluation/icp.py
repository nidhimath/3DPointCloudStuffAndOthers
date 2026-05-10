"""ICP-based registration metrics.

Predicted and ground-truth point clouds live in different coordinate frames
(arbitrary rotation, translation, and possibly scale). Iterative Closest
Point (ICP) finds a rigid transform that snaps the predicted cloud into the
GT frame so that subsequent metrics (Chamfer, F-score, ...) measure
*shape* error rather than *frame* error.

The metrics returned here also act as a registration-quality summary in
their own right:

* ``fitness``     - fraction of source points with a correspondence inside
  the search radius. Effectively the predicted-to-GT overlap ratio.
* ``inlier_rmse`` - RMSE of those correspondences. Lower is better; this is
  the residual geometric error after alignment.
* ``inliers``     - raw count of correspondences (useful when comparing
  clouds of very different sizes).
* ``inlier_ratio`` - ``inliers / |source|``. Equivalent to fitness for a
  fully-converged ICP, but kept separately for clarity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass
class ICPResult:
    """Container for ICP outputs."""

    transformation: np.ndarray
    fitness: float
    inlier_rmse: float
    num_correspondences: int
    inlier_ratio: float
    threshold: float
    method: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["transformation"] = self.transformation.tolist()
        return d


def _ensure_normals(pcd: o3d.geometry.PointCloud, radius: float) -> None:
    """Ensure ``pcd`` has normals (required by point-to-plane ICP)."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
        )


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float,
    init: Optional[np.ndarray] = None,
    method: str = "point_to_plane",
    max_iterations: int = 200,
    relative_fitness: float = 1e-7,
    relative_rmse: float = 1e-7,
) -> ICPResult:
    """Align ``source`` to ``target`` with Open3D ICP.

    Parameters
    ----------
    source, target
        Source = predicted cloud, target = GT cloud (this is the convention
        used by every metric downstream).
    threshold
        Maximum correspondence distance. Should scale with the voxel size of
        the preprocessing pipeline (a common default is ``5 * voxel_size``).
    init
        4x4 initial transform. Default is the identity, which works well
        when both clouds have already been centred (see ``preprocess.py``).
    method
        ``"point_to_point"`` or ``"point_to_plane"``. Point-to-plane is
        faster-converging and more accurate for piecewise-planar scenes
        (rooms, buildings, ...) but requires normals on the target.
    """
    if init is None:
        init = np.eye(4, dtype=np.float64)

    if method == "point_to_plane":
        _ensure_normals(target, radius=threshold * 2.0)
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    elif method == "point_to_point":
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(
            with_scaling=False
        )
    else:
        raise ValueError(f"Unknown ICP method: {method!r}")

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=relative_fitness,
        relative_rmse=relative_rmse,
        max_iteration=max_iterations,
    )

    reg = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init,
        estimator,
        criteria,
    )

    n_src = max(len(source.points), 1)
    n_corr = len(reg.correspondence_set)
    return ICPResult(
        transformation=np.asarray(reg.transformation, dtype=np.float64),
        fitness=float(reg.fitness),
        inlier_rmse=float(reg.inlier_rmse),
        num_correspondences=int(n_corr),
        inlier_ratio=float(n_corr) / float(n_src),
        threshold=float(threshold),
        method=method,
    )


def apply_transform(
    pcd: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> o3d.geometry.PointCloud:
    """Return a transformed *copy* of ``pcd``."""
    out = o3d.geometry.PointCloud(pcd)
    out.transform(transformation)
    return out
