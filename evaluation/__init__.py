"""Research-grade evaluation framework for trajectory-conditioned 3D
reconstruction.

The package implements three families of metrics:

1. Geometry quality - point cloud similarity (Chamfer distance, F-score,
   optional Hausdorff distance).
2. Registration quality - ICP-based alignment metrics (fitness, RMSE,
   correspondence inlier ratio, transformation matrix).
3. Trajectory quality - SE(3)-aligned camera-pose metrics inspired by the
   "True Pose Similarity" (TPS) idea from trajectory-conditioned novel-view
   synthesis (ATE, RPE, and a combined Pose Consistency Score).

All comparisons are performed *after* explicit alignment in both the geometry
domain (ICP) and the trajectory domain (Umeyama / SE(3) Procrustes), which is
the central scientific requirement: predicted reconstructions live in their
own arbitrary coordinate frame, scale and orientation, so any direct numeric
comparison without alignment is meaningless.
"""

from evaluation import (  # noqa: F401
    icp,
    image_metrics,
    metrics,
    pose_metrics,
    preprocess,
    reprojection,
)

__all__ = [
    "icp",
    "image_metrics",
    "metrics",
    "pose_metrics",
    "preprocess",
    "reprojection",
]
