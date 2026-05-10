"""Main evaluation script.

Two modes:

1. **Single**::

       python -m evaluation.evaluate \
           --gt   path/to/gt.ply \
           --pred path/to/pred.ply \
           --gt_traj   path/to/gt.json \
           --pred_traj path/to/pred.json \
           --output    results/run01.json

2. **Batch**::

       python -m evaluation.evaluate \
           --gt_dir   data/gt/ \
           --pred_dir data/pred/ \
           --output_dir results/

   In batch mode the script pairs files by *basename* (everything before the
   first dot) so ``scene01.ply`` is matched with ``scene01.ply`` /
   ``scene01.json`` etc.

The script always emits a JSON file matching the canonical schema described
in the task brief:

    {
      "chamfer_distance": ...,
      "f_score": ...,
      "icp_fitness": ...,
      "icp_rmse": ...,
      "inlier_ratio": ...,
      "ate": ...,
      "rpe_translation": ...,
      "rpe_rotation": ...
    }

Plus a ``"detailed"`` block with the full structured output of each metric
module, for downstream analysis.

Pipeline (per pair):
    load -> preprocess -> ICP align -> chamfer/f-score/hausdorff
                 (optional) load trajectories -> ATE / RPE / PCS
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import open3d as o3d

# Allow ``python evaluation/evaluate.py ...`` (when ``evaluation/`` is not on
# the path) as well as ``python -m evaluation.evaluate ...``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation import icp as icp_mod
from evaluation import image_metrics as image_mod
from evaluation import metrics as metrics_mod
from evaluation import pose_metrics as pose_mod
from evaluation import preprocess as prep_mod
from evaluation import reprojection as reproj_mod


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalConfig:
    voxel_size: float = 0.01
    f_score_threshold: float = 0.02
    icp_threshold_factor: float = 5.0
    icp_method: str = "point_to_plane"
    icp_max_iterations: int = 200
    include_hausdorff: bool = True
    pose_with_scaling: bool = True
    rpe_delta: int = 1
    # Image metrics
    include_lpips: bool = True
    lpips_net: str = "alex"
    ssim_win_size: int = 11
    # Reprojection consistency
    rce_z_min: float = 1e-3
    rce_z_max: float = 1e6


# ---------------------------------------------------------------------------
# Core single-pair evaluation
# ---------------------------------------------------------------------------


def evaluate_pair(
    gt_ply: Path,
    pred_ply: Path,
    gt_traj: Optional[Path] = None,
    pred_traj: Optional[Path] = None,
    config: Optional[EvalConfig] = None,
    verbose: bool = True,
    rendered_images: Optional[Path] = None,
    gt_images: Optional[Path] = None,
    rce_intrinsics_json: Optional[Path] = None,
    rce_video: Optional[Path] = None,
    rce_image_dir: Optional[Path] = None,
) -> dict:
    """Run the full geometry + trajectory evaluation on one pair."""
    config = config or EvalConfig()

    if verbose:
        print(f"[load] gt   = {gt_ply}")
        print(f"[load] pred = {pred_ply}")

    gt = prep_mod.load_point_cloud(gt_ply)
    pred = prep_mod.load_point_cloud(pred_ply)

    prep_cfg = prep_mod.PreprocessConfig(voxel_size=config.voxel_size)

    t0 = time.time()
    gt_p, gt_info = prep_mod.prepare_for_metrics(gt, prep_cfg)
    pred_p, pred_info = prep_mod.prepare_for_metrics(pred, prep_cfg)
    if verbose:
        print(
            f"[prep] gt: {gt_info['input_points']:>8} -> {len(gt_p.points):>6}   "
            f"pred: {pred_info['input_points']:>8} -> {len(pred_p.points):>6}   "
            f"({time.time() - t0:.2f}s)"
        )

    icp_threshold = config.icp_threshold_factor * config.voxel_size
    t0 = time.time()
    icp_res = icp_mod.run_icp(
        source=pred_p,
        target=gt_p,
        threshold=icp_threshold,
        method=config.icp_method,
        max_iterations=config.icp_max_iterations,
    )
    if verbose:
        print(
            f"[icp]  fitness={icp_res.fitness:.4f}  rmse={icp_res.inlier_rmse:.4f}  "
            f"inliers={icp_res.num_correspondences}  "
            f"({time.time() - t0:.2f}s)"
        )

    pred_aligned = icp_mod.apply_transform(pred_p, icp_res.transformation)

    t0 = time.time()
    geom = metrics_mod.compute_geometry_metrics(
        pred=pred_aligned,
        gt=gt_p,
        f_score_threshold=config.f_score_threshold,
        include_hausdorff=config.include_hausdorff,
    )
    if verbose:
        cd = geom["chamfer"]["chamfer_distance"]
        fs = geom["f_score"]["f_score"]
        print(f"[geom] chamfer={cd:.5f}  f1@{config.f_score_threshold}={fs:.4f}  "
              f"({time.time() - t0:.2f}s)")

    pose: Optional[dict] = None
    if gt_traj is not None and pred_traj is not None:
        if verbose:
            print(f"[load] gt_traj   = {gt_traj}")
            print(f"[load] pred_traj = {pred_traj}")
        gt_c2w = pose_mod.load_trajectory(gt_traj)
        pred_c2w = pose_mod.load_trajectory(pred_traj)

        t0 = time.time()
        pose = pose_mod.compute_pose_metrics(
            pred_c2w=pred_c2w,
            gt_c2w=gt_c2w,
            with_scaling=config.pose_with_scaling,
            rpe_delta=config.rpe_delta,
        )
        if verbose:
            ate = pose["ate"]["ate_rmse"]
            rt = pose["rpe"]["rpe_translation_rmse"]
            rr = pose["rpe"]["rpe_rotation_rmse_deg"]
            pcs = pose["pose_consistency"]["pose_consistency_score"]
            print(
                f"[pose] ATE={ate:.4f}  RPE_t={rt:.4f}  "
                f"RPE_r={rr:.3f}deg  PCS={pcs:.4f}  "
                f"({time.time() - t0:.2f}s)"
            )

    image: Optional[dict] = None
    if rendered_images is not None and gt_images is not None:
        if verbose:
            print(f"[load] rendered = {rendered_images}")
            print(f"[load] gt_imgs  = {gt_images}")
        rendered = _load_image_sequence(rendered_images)
        gt_seq = _load_image_sequence(gt_images)
        n = min(len(rendered), len(gt_seq))
        rendered, gt_seq = rendered[:n], gt_seq[:n]
        rendered, gt_seq = _resize_to_match(rendered, gt_seq)

        t0 = time.time()
        image = image_mod.compute_image_metrics_sequence(
            rendered, gt_seq,
            include_lpips=config.include_lpips,
            lpips_net=config.lpips_net,
            ssim_win_size=config.ssim_win_size,
        )
        if verbose:
            psnr_v = image["psnr"]["mean"]
            ssim_v = image["ssim"]["mean"]
            lpips_v = image.get("lpips", {}).get("mean") if image.get("lpips") else None
            lpips_str = f"  LPIPS={lpips_v:.4f}" if lpips_v is not None else "  LPIPS=skipped"
            print(f"[img]  PSNR={psnr_v:.3f}  SSIM={ssim_v:.4f}{lpips_str}  "
                  f"({time.time() - t0:.2f}s)")

    rce: Optional[dict] = None
    if (rce_intrinsics_json is not None
            and (rce_video is not None or rce_image_dir is not None)
            and gt_traj is not None):
        if verbose:
            src = rce_video if rce_video is not None else rce_image_dir
            print(f"[load] rce_intrinsics = {rce_intrinsics_json}")
            print(f"[load] rce_frames     = {src}")
        K, W, H = reproj_mod.intrinsics_from_transforms_json(rce_intrinsics_json)

        if rce_video is not None:
            frames = image_mod.load_video_frames(rce_video)
        else:
            frames = image_mod.load_images_from_dir(rce_image_dir)

        gt_c2w = pose_mod.load_trajectory(gt_traj)

        gt_xyz, gt_rgb = reproj_mod.pcd_to_xyz_rgb(gt)
        t0 = time.time()
        rce_obj = reproj_mod.reprojection_consistency_error(
            points_world=gt_xyz,
            point_colors=gt_rgb,
            K=K,
            c2w=gt_c2w,
            images=frames,
            image_size=(H, W),
            z_min=config.rce_z_min,
            z_max=config.rce_z_max,
        )
        rce = rce_obj.to_dict()
        if verbose:
            print(f"[rce]  L1={rce_obj.mean_l1:.4f}  RMSE={rce_obj.rmse:.4f}  "
                  f"coverage={rce_obj.coverage_mean:.3f}  "
                  f"({time.time() - t0:.2f}s)")

    summary = {
        "chamfer_distance": geom["chamfer"]["chamfer_distance"],
        "f_score": geom["f_score"]["f_score"],
        "icp_fitness": icp_res.fitness,
        "icp_rmse": icp_res.inlier_rmse,
        "inlier_ratio": icp_res.inlier_ratio,
    }
    if pose is not None:
        summary["ate"] = pose["ate"]["ate_rmse"]
        summary["rpe_translation"] = pose["rpe"]["rpe_translation_rmse"]
        summary["rpe_rotation"] = pose["rpe"]["rpe_rotation_rmse_deg"]
    else:
        summary["ate"] = None
        summary["rpe_translation"] = None
        summary["rpe_rotation"] = None

    if image is not None:
        summary["psnr"] = image["psnr"]["mean"]
        summary["ssim"] = image["ssim"]["mean"]
        summary["lpips"] = image.get("lpips", {}).get("mean") if image.get("lpips") else None
    else:
        summary["psnr"] = None
        summary["ssim"] = None
        summary["lpips"] = None

    if rce is not None:
        summary["rce_l1"] = rce["mean_l1"]
        summary["rce_rmse"] = rce["rmse"]
        summary["rce_coverage"] = rce["coverage_mean"]
    else:
        summary["rce_l1"] = None
        summary["rce_rmse"] = None
        summary["rce_coverage"] = None

    detailed = {
        "icp": icp_res.to_dict(),
        "geometry": geom,
        "preprocess": {"gt": gt_info, "pred": pred_info},
        "config": {
            "voxel_size": config.voxel_size,
            "f_score_threshold": config.f_score_threshold,
            "icp_threshold": icp_threshold,
            "icp_method": config.icp_method,
            "pose_with_scaling": config.pose_with_scaling,
            "rpe_delta": config.rpe_delta,
            "include_lpips": config.include_lpips,
            "lpips_net": config.lpips_net,
        },
        "inputs": {
            "gt": str(gt_ply),
            "pred": str(pred_ply),
            "gt_traj": str(gt_traj) if gt_traj else None,
            "pred_traj": str(pred_traj) if pred_traj else None,
            "rendered_images": str(rendered_images) if rendered_images else None,
            "gt_images": str(gt_images) if gt_images else None,
            "rce_intrinsics_json": str(rce_intrinsics_json) if rce_intrinsics_json else None,
            "rce_video": str(rce_video) if rce_video else None,
            "rce_image_dir": str(rce_image_dir) if rce_image_dir else None,
        },
    }
    if pose is not None:
        detailed["pose"] = pose
    if image is not None:
        detailed["image"] = image
    if rce is not None:
        detailed["rce"] = rce

    return {"summary": summary, "detailed": detailed}


def _load_image_sequence(path: Path) -> list:
    """Load images from a directory, a single file, or a video."""
    p = Path(path)
    if p.is_dir():
        return image_mod.load_images_from_dir(p)
    if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        return image_mod.load_video_frames(p)
    return [image_mod.load_image(p)]


def _resize_to_match(a: list, b: list) -> tuple:
    """Resize each image in ``a`` to match the shape of the corresponding ``b``.

    PSNR/SSIM/LPIPS require identical shapes. When rendered frames come at a
    different resolution than the GT (very common - e.g. 256x256 splat
    renders against 1080p photos), we resize the *rendered* side to match
    the GT.
    """
    import cv2

    out_a = []
    for img_a, img_b in zip(a, b):
        if img_a.shape[:2] != img_b.shape[:2]:
            img_a = cv2.resize(
                img_a, (img_b.shape[1], img_b.shape[0]),
                interpolation=cv2.INTER_AREA,
            )
        out_a.append(img_a)
    return out_a, b


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------


def _index_directory(d: Path, exts: Iterable[str]) -> dict:
    """Map basename (stem) -> path for every file in ``d`` with a given extension."""
    index = {}
    for ext in exts:
        for p in sorted(d.glob(f"*{ext}")):
            index.setdefault(p.stem, p)
    return index


def evaluate_batch(
    gt_dir: Path,
    pred_dir: Path,
    gt_traj_dir: Optional[Path] = None,
    pred_traj_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    config: Optional[EvalConfig] = None,
) -> dict:
    """Pair files by stem and evaluate each pair."""
    gt_idx = _index_directory(gt_dir, [".ply"])
    pred_idx = _index_directory(pred_dir, [".ply"])
    gt_traj_idx = _index_directory(gt_traj_dir, [".json", ".txt"]) if gt_traj_dir else {}
    pred_traj_idx = _index_directory(pred_traj_dir, [".json", ".txt"]) if pred_traj_dir else {}

    common = sorted(set(gt_idx) & set(pred_idx))
    if not common:
        raise ValueError(
            f"No overlapping basenames between {gt_dir} and {pred_dir}"
        )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for stem in common:
        print("=" * 72)
        print(f"== {stem}")
        print("=" * 72)
        result = evaluate_pair(
            gt_ply=gt_idx[stem],
            pred_ply=pred_idx[stem],
            gt_traj=gt_traj_idx.get(stem),
            pred_traj=pred_traj_idx.get(stem),
            config=config,
        )
        all_results[stem] = result

        if output_dir is not None:
            (output_dir / f"{stem}.json").write_text(json.dumps(result, indent=2))

    aggregate = _aggregate_summaries(all_results)
    if output_dir is not None:
        (output_dir / "_aggregate.json").write_text(json.dumps(aggregate, indent=2))
    return {"per_scene": all_results, "aggregate": aggregate}


def _aggregate_summaries(all_results: dict) -> dict:
    """Mean / median across all scene summaries (skipping ``None`` entries)."""
    keys = [
        "chamfer_distance",
        "f_score",
        "icp_fitness",
        "icp_rmse",
        "inlier_ratio",
        "ate",
        "rpe_translation",
        "rpe_rotation",
        "psnr",
        "ssim",
        "lpips",
        "rce_l1",
        "rce_rmse",
        "rce_coverage",
    ]
    out = {}
    for k in keys:
        values = [r["summary"][k] for r in all_results.values()
                  if r["summary"].get(k) is not None]
        if not values:
            out[k] = None
            continue
        arr = np.asarray(values, dtype=np.float64)
        out[k] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }
    return out


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def print_summary(summary: dict) -> None:
    """Compact human-readable view of a single ``summary`` dict."""
    print("\n=== Evaluation summary ===")

    def fmt(v):
        if v is None:
            return "  n/a"
        return f"{v: .6f}" if isinstance(v, float) else f"{v}"

    rows = [
        ("Chamfer distance", summary["chamfer_distance"]),
        ("F-score",          summary["f_score"]),
        ("ICP fitness",      summary["icp_fitness"]),
        ("ICP RMSE",         summary["icp_rmse"]),
        ("Inlier ratio",     summary["inlier_ratio"]),
        ("ATE",              summary.get("ate")),
        ("RPE translation",  summary.get("rpe_translation")),
        ("RPE rotation [deg]", summary.get("rpe_rotation")),
        ("PSNR [dB]",        summary.get("psnr")),
        ("SSIM",             summary.get("ssim")),
        ("LPIPS",            summary.get("lpips")),
        ("RCE (L1)",         summary.get("rce_l1")),
        ("RCE (RMSE)",       summary.get("rce_rmse")),
        ("RCE coverage",     summary.get("rce_coverage")),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name.ljust(width)}  {fmt(value)}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate trajectory-conditioned 3D reconstructions: "
                    "ICP, Chamfer, F-score, ATE, RPE, PCS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    single = p.add_argument_group("Single-pair mode")
    single.add_argument("--gt",        type=Path, help="GT point cloud (.ply)")
    single.add_argument("--pred",      type=Path, help="Predicted point cloud (.ply)")
    single.add_argument("--gt_traj",   type=Path, help="GT trajectory (.json or RE10K .txt)")
    single.add_argument("--pred_traj", type=Path, help="Predicted trajectory (.json or RE10K .txt)")
    single.add_argument("--output",    type=Path, help="Output JSON file")

    batch = p.add_argument_group("Batch mode")
    batch.add_argument("--gt_dir",        type=Path, help="Directory of GT .ply files")
    batch.add_argument("--pred_dir",      type=Path, help="Directory of predicted .ply files")
    batch.add_argument("--gt_traj_dir",   type=Path, help="Directory of GT trajectories")
    batch.add_argument("--pred_traj_dir", type=Path, help="Directory of predicted trajectories")
    batch.add_argument("--output_dir",    type=Path, help="Directory to save per-scene + aggregate JSON")

    cfg = p.add_argument_group("Evaluation parameters")
    cfg.add_argument("--voxel_size",        type=float, default=0.01)
    cfg.add_argument("--f_score_threshold", type=float, default=0.02)
    cfg.add_argument("--icp_threshold_factor", type=float, default=5.0,
                     help="ICP correspondence threshold = factor * voxel_size")
    cfg.add_argument("--icp_method", choices=["point_to_plane", "point_to_point"],
                     default="point_to_plane")
    cfg.add_argument("--icp_max_iterations", type=int, default=200)
    cfg.add_argument("--no_hausdorff", action="store_true")
    cfg.add_argument("--no_scale_alignment", action="store_true",
                     help="Disable Sim(3) scale in trajectory alignment (use strict SE(3)).")
    cfg.add_argument("--rpe_delta", type=int, default=1)

    img = p.add_argument_group("Image-quality metrics (PSNR / SSIM / LPIPS)")
    img.add_argument("--rendered", type=Path,
                     help="Rendered frames: directory of images, a single image, or a video (.mp4).")
    img.add_argument("--gt_images", type=Path,
                     help="Ground-truth frames: directory, single image, or video.")
    img.add_argument("--no_lpips", action="store_true",
                     help="Skip LPIPS even if the package is installed.")
    img.add_argument("--lpips_net", choices=["alex", "vgg", "squeeze"], default="alex")
    img.add_argument("--ssim_win_size", type=int, default=11)

    rce = p.add_argument_group("Reprojection Consistency Error (RCE)")
    rce.add_argument("--rce_intrinsics", type=Path,
                     help="transforms.json with fl_x/fl_y/cx/cy/w/h to define the camera intrinsics.")
    rce.add_argument("--rce_video", type=Path,
                     help="Video file containing the observed frames (one per pose in --gt_traj).")
    rce.add_argument("--rce_image_dir", type=Path,
                     help="Directory of observed frames (alternative to --rce_video).")
    rce.add_argument("--rce_z_min", type=float, default=1e-3)
    rce.add_argument("--rce_z_max", type=float, default=1e6)

    cfg.add_argument("--visualize", action="store_true",
                     help="Open Open3D windows for ICP and trajectories (single mode only).")
    return p


def _config_from_args(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        voxel_size=args.voxel_size,
        f_score_threshold=args.f_score_threshold,
        icp_threshold_factor=args.icp_threshold_factor,
        icp_method=args.icp_method,
        icp_max_iterations=args.icp_max_iterations,
        include_hausdorff=not args.no_hausdorff,
        pose_with_scaling=not args.no_scale_alignment,
        rpe_delta=args.rpe_delta,
        include_lpips=not args.no_lpips,
        lpips_net=args.lpips_net,
        ssim_win_size=args.ssim_win_size,
        rce_z_min=args.rce_z_min,
        rce_z_max=args.rce_z_max,
    )


def _maybe_visualize(
    gt_ply: Path,
    pred_ply: Path,
    config: EvalConfig,
    gt_traj: Optional[Path],
    pred_traj: Optional[Path],
) -> None:
    """Re-run the alignment just to populate the visualisation windows."""
    from evaluation import visualize as viz  # local import (display optional)

    gt = prep_mod.load_point_cloud(gt_ply)
    pred = prep_mod.load_point_cloud(pred_ply)
    prep_cfg = prep_mod.PreprocessConfig(voxel_size=config.voxel_size)
    gt_p, _ = prep_mod.prepare_for_metrics(gt, prep_cfg)
    pred_p, _ = prep_mod.prepare_for_metrics(pred, prep_cfg)

    icp_res = icp_mod.run_icp(
        source=pred_p,
        target=gt_p,
        threshold=config.icp_threshold_factor * config.voxel_size,
        method=config.icp_method,
        max_iterations=config.icp_max_iterations,
    )
    pred_aligned = icp_mod.apply_transform(pred_p, icp_res.transformation)
    viz.visualize_alignment(gt_p, pred_p, pred_aligned)

    if gt_traj is not None and pred_traj is not None:
        gt_c2w = pose_mod.load_trajectory(gt_traj)
        pred_c2w = pose_mod.load_trajectory(pred_traj)
        viz.visualize_trajectories(gt_c2w, pred_c2w, point_cloud=gt_p)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    single = args.gt is not None and args.pred is not None
    batch = args.gt_dir is not None and args.pred_dir is not None
    if single == batch:
        print("Specify either single mode (--gt, --pred) or batch mode "
              "(--gt_dir, --pred_dir), not both.")
        return 2

    config = _config_from_args(args)

    if single:
        result = evaluate_pair(
            gt_ply=args.gt,
            pred_ply=args.pred,
            gt_traj=args.gt_traj,
            pred_traj=args.pred_traj,
            config=config,
            rendered_images=args.rendered,
            gt_images=args.gt_images,
            rce_intrinsics_json=args.rce_intrinsics,
            rce_video=args.rce_video,
            rce_image_dir=args.rce_image_dir,
        )
        print_summary(result["summary"])

        out_path = args.output
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            print(f"Wrote {out_path}")

        if args.visualize:
            _maybe_visualize(args.gt, args.pred, config, args.gt_traj, args.pred_traj)
        return 0

    out = evaluate_batch(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        gt_traj_dir=args.gt_traj_dir,
        pred_traj_dir=args.pred_traj_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print("\n=== Aggregate over", len(out["per_scene"]), "scenes ===")
    print(json.dumps(out["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
