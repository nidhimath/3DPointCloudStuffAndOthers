"""Open3D-based visualisations.

Two debugging views are exposed:

* ``visualize_alignment`` - GT cloud (green) + predicted-before-ICP (red) +
  predicted-after-ICP (blue). This is the single most useful sanity check:
  if the blue cloud has clearly snapped onto the green one and the red
  cloud is offset, ICP did its job.

* ``visualize_trajectories`` - GT and predicted trajectory polylines as 3D
  line strips, with little coordinate frames at every Nth pose. Optionally
  overlays a point cloud.

Both functions are no-ops on machines without a display; they exit cleanly
rather than crashing the evaluation pipeline.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import open3d as o3d


_GT_COLOR = (0.10, 0.70, 0.20)        # green
_PRED_BEFORE_COLOR = (0.85, 0.10, 0.10)  # red
_PRED_AFTER_COLOR = (0.10, 0.30, 0.95)   # blue


def _paint(pcd: o3d.geometry.PointCloud, color) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud(pcd)
    out.paint_uniform_color(color)
    return out


def _safe_draw(geometries, window_name: str) -> None:
    try:
        o3d.visualization.draw_geometries(
            geometries, window_name=window_name, width=1280, height=800
        )
    except Exception as e:  # pragma: no cover - depends on the host environment
        print(f"[visualize] could not open Open3D window: {e}")


# ---------------------------------------------------------------------------
# Point clouds
# ---------------------------------------------------------------------------


def visualize_alignment(
    gt: o3d.geometry.PointCloud,
    pred_before: o3d.geometry.PointCloud,
    pred_after: o3d.geometry.PointCloud,
    window_name: str = "ICP alignment",
) -> None:
    """Show GT (green) + predicted before (red) + predicted after (blue)."""
    geometries = [
        _paint(gt, _GT_COLOR),
        _paint(pred_before, _PRED_BEFORE_COLOR),
        _paint(pred_after, _PRED_AFTER_COLOR),
    ]
    _safe_draw(geometries, window_name)


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------


def _trajectory_lineset(
    centers: np.ndarray, color: Sequence[float]
) -> o3d.geometry.LineSet:
    """Build a coloured polyline from a (N, 3) array of camera centres."""
    if centers.shape[0] < 2:
        return o3d.geometry.LineSet()

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(centers.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(
        np.stack([np.arange(len(centers) - 1), np.arange(1, len(centers))], axis=1)
    )
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(centers) - 1, 1))
    )
    return ls


def visualize_trajectories(
    gt_c2w: np.ndarray,
    pred_c2w: np.ndarray,
    point_cloud: Optional[o3d.geometry.PointCloud] = None,
    frame_every: int = 8,
    frame_size: float = 0.05,
    window_name: str = "Trajectories",
) -> None:
    """Overlay two camera trajectories in 3D, optionally on a point cloud.

    The little RGB triads drawn at every ``frame_every``-th pose are full
    camera frames (extracted from the c2w matrices), so you can also see
    rotation drift, not just position drift.
    """
    geometries = []
    if point_cloud is not None:
        geometries.append(point_cloud)

    gt_centers = gt_c2w[:, :3, 3]
    pred_centers = pred_c2w[:, :3, 3]

    geometries.append(_trajectory_lineset(gt_centers, _GT_COLOR))
    geometries.append(_trajectory_lineset(pred_centers, _PRED_AFTER_COLOR))

    for c2w in gt_c2w[::max(frame_every, 1)]:
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        f.transform(c2w)
        geometries.append(f)
    for c2w in pred_c2w[::max(frame_every, 1)]:
        f = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 0.7)
        f.transform(c2w)
        geometries.append(f)

    _safe_draw(geometries, window_name)
