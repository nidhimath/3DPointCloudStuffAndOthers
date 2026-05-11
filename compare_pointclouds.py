"""Compare each scene's baseline point cloud against trajectory variants.

For every scene `<base>` in `pointclouds/` we:
  baseline = pointclouds/<base>_standard.ply   trajectory: trajectories/<base>.txt
  variants = pointclouds/<base>_<suf>.ply     trajectory: trajectories/<base>_<suf>.txt

Default suffices are ``1``, ``2``, ``3``, ``4``. Override with ``--variants``,
e.g. ``--variants 45deg 90deg 135deg 180deg``.

For every (baseline, variant) pair we call `evaluation.evaluate.evaluate_pair`,
which computes:
  - Geometry         : Chamfer distance, F-score (@2 cm), Hausdorff
  - Registration     : ICP fitness, ICP RMSE, inlier ratio
  - Trajectory       : ATE, RPE-translation, RPE-rotation, Pose Consistency Score

Output:
  - pointclouds/comparison_metrics.txt           (human-readable report + table)
  - pointclouds/comparison_results/*.json        (full structured per-pair output)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PCS_DIR  = BASE_DIR / "pointclouds"
TRAJ_DIR = BASE_DIR / "trajectories"
OUT_TXT  = PCS_DIR / "comparison_metrics.txt"
JSON_DIR = PCS_DIR / "comparison_results"

if __package__ in (None, ""):
    sys.path.insert(0, str(BASE_DIR))

from evaluation.evaluate import evaluate_pair, EvalConfig  # noqa: E402

SCENES   = sorted({p.stem.rsplit("_", 1)[0]
                   for p in PCS_DIR.glob("*_standard.ply")})
DEFAULT_VARIANTS = ["1", "2", "3", "4"]

CONFIG = EvalConfig(
    voxel_size=0.02,            # our clouds are fused at 1.5 cm; 2 cm gives a
                                # fair, dense-region-uniform comparison
    f_score_threshold=0.05,     # 5 cm — interpretable indoor threshold
    icp_threshold_factor=5.0,   # ICP corresp. threshold = 5 * voxel = 10 cm
    icp_method="point_to_plane",
    icp_max_iterations=200,
    include_hausdorff=True,
    pose_with_scaling=True,
    rpe_delta=1,
    include_lpips=False,        # no rendered images involved here
)


def fmt(v, w=10, d=4):
    if v is None:
        return "n/a".rjust(w)
    if isinstance(v, float):
        return f"{v:>{w}.{d}f}"
    return str(v).rjust(w)


def write_report(per_scene: dict, txt_path: Path) -> None:
    """Render the collected results into a human-readable .txt report."""
    lines: list[str] = []
    push = lines.append

    push("=" * 96)
    push("POINT CLOUD COMPARISON  —  baseline (`_standard`) vs trajectory variant(s)")
    push("=" * 96)
    push(f"Voxel size (preprocess) : {CONFIG.voxel_size} m")
    push(f"F-score threshold       : {CONFIG.f_score_threshold} m")
    push(f"ICP method              : {CONFIG.icp_method}, max_iter={CONFIG.icp_max_iterations}")
    push(f"ICP corresp. threshold  : {CONFIG.icp_threshold_factor} * voxel = "
         f"{CONFIG.icp_threshold_factor * CONFIG.voxel_size} m")
    push(f"Trajectory alignment    : Sim(3) Umeyama (with_scaling=True)")
    push("")
    push("Notes:")
    push("  - 'Baseline' is each scene's fused `_standard` cloud. Each variant matches")
    push("    `pointclouds/<scene>_<suffix>.ply` and its trajectory `.txt`.")
    push("  - Lower is better for: Chamfer, ICP-RMSE, ATE, RPE-t, RPE-r.")
    push("  - Higher is better for: F-score, ICP-fitness, Inlier-ratio, PCS.")
    push("")

    flat_rows: list[tuple[str, str, dict]] = []   # (scene, variant, summary)

    for scene, comps in per_scene.items():
        push("=" * 96)
        push(f"SCENE  {scene}")
        push(f"  baseline  : pointclouds/{scene}_standard.ply  "
             f"(traj: trajectories/{scene}.txt)")
        push("=" * 96)
        for variant, result in comps.items():
            s = result["summary"]
            d = result.get("detailed", {})
            geom_extra = d.get("geometry", {}).get("hausdorff", {})
            haus = geom_extra.get("hausdorff")
            push("")
            push(f"  -- baseline  vs  {variant!r}   "
                 f"(pred: pointclouds/{scene}_{variant}.ply, "
                 f"pred_traj: trajectories/{scene}_{variant}.txt) --")
            push(f"     Chamfer distance        : {fmt(s['chamfer_distance'])}  m")
            push(f"     F-score @ {CONFIG.f_score_threshold} m         : {fmt(s['f_score'])}")
            if haus is not None:
                push(f"     Hausdorff distance      : {fmt(haus)}  m")
            push(f"     ICP fitness             : {fmt(s['icp_fitness'])}")
            push(f"     ICP RMSE                : {fmt(s['icp_rmse'])}  m")
            push(f"     Inlier ratio            : {fmt(s['inlier_ratio'])}")
            if s.get("ate") is not None:
                pcs = d.get("pose", {}).get("pose_consistency", {}).get(
                    "pose_consistency_score")
                push(f"     ATE (RMSE)              : {fmt(s['ate'])}  m")
                push(f"     RPE translation (RMSE)  : {fmt(s['rpe_translation'])}  m")
                push(f"     RPE rotation (RMSE)     : {fmt(s['rpe_rotation'])}  deg")
                push(f"     Pose Consistency Score  : {fmt(pcs)}")
            flat_rows.append((scene, variant, s, d))
        push("")

    push("")
    push("=" * 96)
    push("SUMMARY TABLE  (one row per [scene, variant] pair)")
    push("=" * 96)
    headers = [
        ("Scene", 18, "s"), ("Var", 9, "s"),
        ("Chamfer", 10, "f"), ("F@5cm", 10, "f"), ("Haus", 10, "f"),
        ("ICP-fit", 10, "f"), ("ICP-rmse", 10, "f"), ("Inlier", 10, "f"),
        ("ATE", 10, "f"), ("RPE-t", 10, "f"), ("RPE-r°", 10, "f"), ("PCS", 10, "f"),
    ]
    header_line = "  ".join(name.rjust(w) for name, w, _ in headers)
    push(header_line)
    push("-" * len(header_line))
    for scene, variant, s, d in flat_rows:
        haus = d.get("geometry", {}).get("hausdorff", {}).get("hausdorff")
        pcs = d.get("pose", {}).get("pose_consistency", {}).get(
            "pose_consistency_score")
        row = [
            scene[:18].rjust(18),
            str(variant)[:9].rjust(9),
            fmt(s["chamfer_distance"]),
            fmt(s["f_score"]),
            fmt(haus),
            fmt(s["icp_fitness"]),
            fmt(s["icp_rmse"]),
            fmt(s["inlier_ratio"]),
            fmt(s.get("ate")),
            fmt(s.get("rpe_translation")),
            fmt(s.get("rpe_rotation"), 10, 3),
            fmt(pcs),
        ]
        push("  ".join(row))

    push("")
    push("=" * 96)
    push("PER-SCENE AVERAGES  (mean over variant comparisons)")
    push("=" * 96)
    avg_headers = [
        ("Scene", 18), ("Chamfer", 10), ("F@5cm", 10), ("Haus", 10),
        ("ICP-fit", 10), ("ICP-rmse", 10), ("Inlier", 10),
        ("ATE", 10), ("RPE-t", 10), ("RPE-r°", 10), ("PCS", 10),
    ]
    push("  ".join(name.rjust(w) for name, w in avg_headers))
    push("-" * (sum(w for _, w in avg_headers) + 2 * (len(avg_headers) - 1)))

    def _mean(vals):
        vs = [v for v in vals if isinstance(v, (int, float))]
        return sum(vs) / len(vs) if vs else None

    for scene, comps in per_scene.items():
        rows = list(comps.values())
        sums = [r["summary"] for r in rows]
        dets = [r.get("detailed", {}) for r in rows]
        means = {
            "chamfer": _mean(s["chamfer_distance"] for s in sums),
            "f":       _mean(s["f_score"]          for s in sums),
            "haus":    _mean(d.get("geometry", {}).get("hausdorff", {})
                             .get("hausdorff") for d in dets),
            "icp_fit": _mean(s["icp_fitness"]      for s in sums),
            "icp_rms": _mean(s["icp_rmse"]         for s in sums),
            "inlier":  _mean(s["inlier_ratio"]     for s in sums),
            "ate":     _mean(s.get("ate")          for s in sums),
            "rpe_t":   _mean(s.get("rpe_translation") for s in sums),
            "rpe_r":   _mean(s.get("rpe_rotation") for s in sums),
            "pcs":     _mean(d.get("pose", {}).get("pose_consistency", {})
                             .get("pose_consistency_score") for d in dets),
        }
        push("  ".join([
            scene[:18].rjust(18),
            fmt(means["chamfer"]), fmt(means["f"]), fmt(means["haus"]),
            fmt(means["icp_fit"]), fmt(means["icp_rms"]), fmt(means["inlier"]),
            fmt(means["ate"]), fmt(means["rpe_t"]),
            fmt(means["rpe_r"], 10, 3), fmt(means["pcs"]),
        ]))

    txt_path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compare `<scene>_standard.ply` to named variant clouds.",
    )
    ap.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        metavar="SUF",
        help="Variant suffixes (default: 1 2 3 4), e.g. 45deg 90deg 135deg 180deg",
    )
    args = ap.parse_args()
    variants: list[str] = args.variants

    if not SCENES:
        print(f"No `<scene>_standard.ply` files found in {PCS_DIR}", file=sys.stderr)
        return 1
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(SCENES)} scene(s): {SCENES}")
    print(f"Variants: {variants}")
    print(f"Output report → {OUT_TXT}")
    print(f"Per-pair JSON → {JSON_DIR}/")

    per_scene: dict[str, dict[str, dict]] = {}
    t_total0 = time.time()

    for scene in SCENES:
        baseline_ply  = PCS_DIR / f"{scene}_standard.ply"
        baseline_traj = TRAJ_DIR / f"{scene}.txt"
        if not baseline_ply.exists():
            print(f"!! Missing baseline cloud for {scene}; skipping scene.")
            continue
        if not baseline_traj.exists():
            print(f"!! Missing baseline trajectory for {scene}; trajectory metrics will be skipped.")
            baseline_traj = None

        per_scene[scene] = {}
        for variant in variants:
            pred_ply  = PCS_DIR / f"{scene}_{variant}.ply"
            pred_traj = TRAJ_DIR / f"{scene}_{variant}.txt"
            if not pred_ply.exists():
                print(f"!! Missing variant cloud {pred_ply.name}; skipping.")
                continue

            print("\n" + "=" * 78)
            print(f"== {scene}   baseline  vs  {variant!r}")
            print("=" * 78)
            t0 = time.time()
            result = evaluate_pair(
                gt_ply=baseline_ply,
                pred_ply=pred_ply,
                gt_traj=baseline_traj,
                pred_traj=pred_traj if pred_traj.exists() else None,
                config=CONFIG,
                verbose=True,
            )
            print(f"   ({time.time() - t0:.1f}s total)")
            per_scene[scene][variant] = result

            json_path = JSON_DIR / f"{scene}__baseline_vs_{variant}.json"
            json_path.write_text(json.dumps(result, indent=2))

    write_report(per_scene, OUT_TXT)
    print(f"\nWrote {OUT_TXT}  ({OUT_TXT.stat().st_size / 1024:.1f} KB)")
    print(f"Total time: {time.time() - t_total0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
