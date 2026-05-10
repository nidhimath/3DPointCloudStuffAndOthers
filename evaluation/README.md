# Evaluation framework for trajectory-conditioned 3D reconstruction

This module compares predicted point clouds, camera trajectories, and rendered
images against ground-truth references. It implements five metric families:

| Family         | Metrics                                          | File                |
|----------------|--------------------------------------------------|---------------------|
| Geometry       | Chamfer distance, F-score, Hausdorff             | `metrics.py`        |
| Registration   | ICP fitness / RMSE / inlier ratio                | `icp.py`            |
| Trajectory     | ATE, RPE (translation + rotation), PCS (TPS-ish) | `pose_metrics.py`   |
| Image quality  | PSNR, SSIM, LPIPS                                | `image_metrics.py`  |
| 2D-3D consistency | Reprojection Consistency Error (RCE)          | `reprojection.py`   |

All comparisons run **after explicit alignment** in both the geometry domain
(ICP) and the trajectory domain (Umeyama / SE(3) Procrustes), which is the
central scientific requirement: predicted reconstructions live in their own
arbitrary coordinate frame, scale and orientation.

## Layout

```
evaluation/
├── __init__.py
├── preprocess.py     # voxel down-sample, outlier removal, normals, ...
├── icp.py            # Open3D ICP wrapper + metrics
├── metrics.py        # Chamfer / F-score / Hausdorff (numpy + scipy KDTree)
├── pose_metrics.py   # Trajectory loaders, Umeyama, ATE, RPE, PCS
├── image_metrics.py  # PSNR / SSIM / LPIPS (skimage + lpips, with fallbacks)
├── reprojection.py   # Project 3D->2D, sparse rasterizer, RCE
├── visualize.py      # Optional Open3D visualisations
├── evaluate.py       # CLI entry point (single + batch modes)
└── README.md
```

## Quick start

### Single pair

```bash
python -m evaluation.evaluate \
    --gt   path/to/gt.ply \
    --pred path/to/pred.ply \
    --gt_traj   path/to/gt_traj.json \
    --pred_traj path/to/pred_traj.json \
    --output    results/run01.json
```

Trajectories may be either Nerfstudio-style `transforms.json` (camera-to-world,
OpenGL convention) or RealEstate10K `.txt` (world-to-camera, OpenCV).

### Batch

```bash
python -m evaluation.evaluate \
    --gt_dir   data/gt/ \
    --pred_dir data/pred/ \
    --gt_traj_dir   data/gt_trajs/ \
    --pred_traj_dir data/pred_trajs/ \
    --output_dir    results/
```

Files are paired by basename (`scene01.ply` ↔ `scene01.ply` ↔ `scene01.json`).

### Visualise an alignment

```bash
python -m evaluation.evaluate --gt ... --pred ... --visualize
```

GT cloud is shown in **green**, predicted-before-ICP in **red**, and
predicted-after-ICP in **blue**. The trajectory window draws GT and predicted
camera centres as line strips with little RGB camera frames at every Nth pose.

## Image-quality + RCE flags

```bash
python -m evaluation.evaluate \
    --gt   gt.ply --pred pred.ply \
    --gt_traj gt.json --pred_traj pred.json \
    --rendered  results/render_seq/  \
    --gt_images data/gt_seq/         \
    --rce_intrinsics data/transforms.json \
    --rce_video      data/observed.mp4    \
    --output results/run.json
```

* `--rendered` / `--gt_images` accept a directory of images, a single image,
  or a video file. Sequences must have the same length; mismatched
  resolutions are auto-resized to match the GT.
* `--rce_intrinsics` is any Nerfstudio-style `transforms.json` (we read
  `fl_x`, `fl_y`, `cx`, `cy`, `w`, `h`). RCE then projects the GT cloud
  through every pose in `--gt_traj` and compares against the matching frame
  in `--rce_video` / `--rce_image_dir`.
* LPIPS is on by default. `--no_lpips` skips it; if the `lpips` package is
  not installed the framework also skips it gracefully with a warning.

## Output schema

The CLI emits JSON with the canonical keys (top-level `summary`):

```json
{
  "chamfer_distance": 0.0123,
  "f_score": 0.7421,
  "icp_fitness": 0.9132,
  "icp_rmse":    0.0078,
  "inlier_ratio":0.9100,
  "ate":         0.0421,
  "rpe_translation": 0.0067,
  "rpe_rotation":    0.42,
  "psnr": 28.94,
  "ssim": 0.873,
  "lpips": 0.142,
  "rce_l1": 0.103,
  "rce_rmse": 0.156,
  "rce_coverage": 0.81
}
```

A second `detailed` block contains the full structured output of every metric
module (preprocessing counts, the 4×4 ICP transform, alignment Umeyama
parameters, per-direction Chamfer means, precision/recall, etc.) so all
downstream analysis (ablation tables, plots) can be done from a single file.

## Programmatic use

```python
from pathlib import Path
from evaluation.evaluate import evaluate_pair, EvalConfig

result = evaluate_pair(
    gt_ply=Path("gt.ply"),
    pred_ply=Path("pred.ply"),
    gt_traj=Path("gt.json"),
    pred_traj=Path("pred.json"),
    config=EvalConfig(voxel_size=0.01, f_score_threshold=0.02),
)
print(result["summary"])
```

Or call the modules directly:

```python
from evaluation import preprocess, icp, metrics, pose_metrics

gt   = preprocess.load_point_cloud("gt.ply")
pred = preprocess.load_point_cloud("pred.ply")
gt_p, _   = preprocess.prepare_for_metrics(gt)
pred_p, _ = preprocess.prepare_for_metrics(pred)

reg = icp.run_icp(pred_p, gt_p, threshold=0.05)
pred_aligned = icp.apply_transform(pred_p, reg.transformation)

geom = metrics.compute_geometry_metrics(pred_aligned, gt_p, f_score_threshold=0.02)
pose = pose_metrics.compute_pose_metrics(
    pred_c2w=pose_metrics.load_trajectory("pred.json"),
    gt_c2w=pose_metrics.load_trajectory("gt.json"),
)
```

## Notes on TPS

In the trajectory-conditioned novel-view-synthesis literature the term *True
Pose Similarity* (TPS) sometimes refers to the cross-scene transferability of
a pose representation. We do **not** implement that full notion. The
`pose_consistency_score` exposed here is a deliberately simple,
task-specific surrogate combining the well-understood ATE and RPE metrics
into a single bounded scalar in (0, 1].
