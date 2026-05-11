"""Generate dense colored point clouds for every video in `generated_videos/`.

For each video `generated_videos/<base>(_suffix).mp4` we produce a fused cloud
that captures *what the camera saw along that trajectory*, so different
trajectories of the same scene yield visibly different point clouds.

Pipeline (per video):
  1. Find start image `start_images/<base>.jpg` and trajectory:
     `_standard` → `trajectories/<base>.txt`; `_<digits>` → `<base>_<digits>.txt`;
     `_*deg` → `trajectories/<full_video_stem>.txt`.
  2. Run Depth Anything V2 (Metric Indoor) on the full-res start image
     (cached per scene → only 4 inferences total for the anchor).
  3. Sample N video frames evenly, pair each with its trajectory pose
     (linspace mapping, same convention as the notebook), run depth on each.
  4. Unproject every kept pixel (start image + sampled video frames) to world
     coords using each frame's own pose + per-resolution intrinsics.
  5. Concatenate, voxel-downsample (duplicates merge, disagreement stays),
     remove outliers loosely, estimate normals.
  6. Write `pointclouds/<video_stem>.ply` (fused) and once per scene
     `pointclouds/<scene>_anchor.ply` (clean start-image-only reference).

Why fusion: a single-anchor cloud (start image only) is identical for every
trajectory of a scene because they share the start pose. Fusing per-frame
depths along the actual path is what makes trajectory variability visible —
both as new spatial coverage and as drift/disagreement where the generator is
inconsistent.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from frame_grid import compute_fusion_frame_picks, write_fusion_frame_grid

BASE_DIR        = Path(__file__).resolve().parent
START_IMAGES    = BASE_DIR / "start_images"
TRAJECTORIES    = BASE_DIR / "trajectories"
VIDEOS_DIR      = BASE_DIR / "generated_videos"
OUT_DIR         = BASE_DIR / "pointclouds"

METRIC_MODEL    = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"
RELATIVE_MODEL  = "depth-anything/Depth-Anything-V2-Large-hf"

# Plausible indoor depth range (metres). Pixels outside are dropped.
D_MIN, D_MAX           = 0.15, 6.0
# Number of video frames to sample per video for fusion.
N_FRAMES_PER_VIDEO     = 12
# Voxel size for the fused cloud (slightly looser than single-anchor to keep
# disagreement visible without exploding cloud size).
VOXEL_SIZE_FUSED       = 0.015
# Voxel size for the per-scene clean anchor cloud.
VOXEL_SIZE_ANCHOR      = 0.01
# Loose outlier rejection so per-trajectory drift is preserved.
OUTLIER_NB             = 30
OUTLIER_STD_FUSED      = 2.0
OUTLIER_STD_ANCHOR     = 1.5


def parse_re10k(path: Path):
    """Parse a RealEstate10K trajectory → (intr_norm, w2c, timestamps)."""
    intr, w2c, ts = [], [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("http"):
            continue
        parts = line.split()
        if len(parts) != 19:
            continue
        nums = list(map(float, parts))
        ts.append(int(nums[0]))
        intr.append(nums[1:5])
        rt = np.array(nums[7:19], dtype=np.float64).reshape(3, 4)
        m = np.eye(4)
        m[:3, :4] = rt
        w2c.append(m)
    return (
        np.asarray(intr, dtype=np.float64),
        np.asarray(w2c,  dtype=np.float64),
        np.asarray(ts,   dtype=np.int64),
    )


def video_to_paths(video_path: Path) -> tuple[Path, Path, str]:
    """Map a video filename to (start_image, trajectory, output_stem)."""
    stem = video_path.stem  # e.g. "..._standard", "..._4", "..._45deg"
    if stem.endswith("_standard"):
        base = stem[: -len("_standard")]
        image = START_IMAGES / f"{base}.jpg"
        traj = TRAJECTORIES / f"{base}.txt"
        return image, traj, stem
    # e.g. scene_45deg → trajectories/scene_45deg.txt (try before `_\\d+$`)
    if re.fullmatch(r".+_\d+deg", stem):
        base = stem.rsplit("_", 1)[0]
        image = START_IMAGES / f"{base}.jpg"
        traj = TRAJECTORIES / f"{stem}.txt"
        return image, traj, stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"Unrecognised video name: {video_path.name}")
    base, suffix = m.group(1), m.group(2)
    image = START_IMAGES / f"{base}.jpg"
    traj = TRAJECTORIES / f"{base}_{suffix}.txt"
    return image, traj, stem


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_depth_model(device: str):
    try:
        proc = AutoImageProcessor.from_pretrained(METRIC_MODEL)
        mdl  = AutoModelForDepthEstimation.from_pretrained(
            METRIC_MODEL, dtype=torch.float32
        ).to(device).eval()
        print(f"Loaded depth model: {METRIC_MODEL} (metric)")
        return proc, mdl, True
    except Exception as exc:
        print(f"Metric model unavailable ({exc!r}); falling back to relative model.")
        proc = AutoImageProcessor.from_pretrained(RELATIVE_MODEL)
        mdl  = AutoModelForDepthEstimation.from_pretrained(
            RELATIVE_MODEL, dtype=torch.float32
        ).to(device).eval()
        print(f"Loaded depth model: {RELATIVE_MODEL} (relative)")
        return proc, mdl, False


def infer_depth(proc, mdl, device, pil_img: PILImage.Image,
                out_hw: tuple[int, int]) -> np.ndarray:
    inputs = proc(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        pred = mdl(**inputs).predicted_depth
    d = pred.squeeze().float().cpu().numpy()
    d = cv2.resize(d, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(d, 0.0, None).astype(np.float32)


def read_video_frames_rgb(video_path: Path):
    """Decode a video → list of RGB uint8 frames + (H, W). Empty list on failure."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return [], 0, 0
    H, W = frames[0].shape[:2]
    return frames, H, W


def unproject(depth: np.ndarray, rgb_img: np.ndarray, c2w: np.ndarray,
              fl_x: float, fl_y: float, cx: float, cy: float):
    """Back-project a depth map + matching RGB image to world-space (pts, cols)."""
    H, W = depth.shape
    depth_clip = np.clip(depth, D_MIN, D_MAX)
    ys, xs = np.mgrid[0:H, 0:W]
    d_flat = depth_clip.ravel()
    keep = (d_flat > D_MIN) & (d_flat < D_MAX)
    if not np.any(keep):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)
    x_v = xs.ravel()[keep].astype(np.float64)
    y_v = ys.ravel()[keep].astype(np.float64)
    d_v = d_flat[keep].astype(np.float64)
    xc = (x_v - cx) / fl_x * d_v
    yc = (y_v - cy) / fl_y * d_v
    pts_cam   = np.stack([xc, yc, d_v, np.ones_like(d_v)], axis=1)
    pts_world = (c2w @ pts_cam.T).T[:, :3]
    cols = rgb_img.reshape(-1, 3)[keep].astype(np.float64) / 255.0
    return pts_world, cols


def finalize_pcd(pts: np.ndarray, cols: np.ndarray, voxel: float, std_ratio: float,
                 cam_center: np.ndarray) -> o3d.geometry.PointCloud | None:
    if len(pts) == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(cols, 0.0, 1.0))
    pcd = pcd.voxel_down_sample(voxel_size=voxel)
    if len(pcd.points) == 0:
        return None
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=OUTLIER_NB,
                                            std_ratio=std_ratio)
    if len(pcd.points) == 0:
        return None
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30)
    )
    pcd.orient_normals_towards_camera_location(cam_center)
    return pcd


def build_anchor_pointcloud(depth_hr: np.ndarray, hr_pil: PILImage.Image,
                            intr_n: np.ndarray, w2c0: np.ndarray):
    """Clean per-scene cloud from the start image alone (high quality reference)."""
    HR_W, HR_H = hr_pil.size
    fx_n, fy_n, cx_n, cy_n = intr_n.mean(axis=0)
    fl_x = fx_n * HR_W; fl_y = fy_n * HR_H
    cx_p = cx_n * HR_W; cy_p = cy_n * HR_H
    c2w0 = np.linalg.inv(w2c0)
    pts, cols = unproject(depth_hr, np.array(hr_pil), c2w0, fl_x, fl_y, cx_p, cy_p)
    cam_c0 = -w2c0[:3, :3].T @ w2c0[:3, 3]
    return finalize_pcd(pts, cols, VOXEL_SIZE_ANCHOR, OUTLIER_STD_ANCHOR, cam_c0)


def build_fused_pointcloud(video_path: Path, depth_hr: np.ndarray,
                           hr_pil: PILImage.Image, intr_n: np.ndarray,
                           w2c_all: np.ndarray, proc, mdl, device: str,
                           n_sample: int):
    """Fuse start image + N sampled video frames into one per-trajectory cloud."""
    if len(w2c_all) == 0:
        return None
    bgr_frames, H, W = read_video_frames_rgb(video_path)
    if H == 0:
        print(f"  Could not decode video {video_path.name}; skipping.")
        return None

    fx_n, fy_n, cx_n, cy_n = intr_n.mean(axis=0)

    # Pair frames ↔ poses via linspace (mirrors notebook), then sub-sample N.
    frame_picks, pose_picks = compute_fusion_frame_picks(
        len(bgr_frames), len(w2c_all), n_sample
    )

    all_pts, all_cols = [], []

    # Start image (high-res anchor) → unprojected at pose 0.
    HR_W, HR_H = hr_pil.size
    fl_x_h = fx_n * HR_W; fl_y_h = fy_n * HR_H
    cx_h   = cx_n * HR_W; cy_h   = cy_n * HR_H
    c2w0   = np.linalg.inv(w2c_all[0])
    pts_a, cols_a = unproject(depth_hr, np.array(hr_pil), c2w0,
                              fl_x_h, fl_y_h, cx_h, cy_h)
    all_pts.append(pts_a); all_cols.append(cols_a)

    # Sampled video frames at video resolution.
    fl_x_v = fx_n * W; fl_y_v = fy_n * H
    cx_v   = cx_n * W; cy_v   = cy_n * H

    for fi, pi in zip(frame_picks, pose_picks):
        rgb = bgr_frames[fi]
        pil = PILImage.fromarray(rgb)
        depth_f = infer_depth(proc, mdl, device, pil, (H, W))
        c2w = np.linalg.inv(w2c_all[pi])
        pts_f, cols_f = unproject(depth_f, rgb, c2w, fl_x_v, fl_y_v, cx_v, cy_v)
        all_pts.append(pts_f); all_cols.append(cols_f)

    pts_world  = np.concatenate(all_pts,  axis=0)
    cols_world = np.concatenate(all_cols, axis=0)
    cam_c0 = -w2c_all[0][:3, :3].T @ w2c_all[0][:3, 3]
    return finalize_pcd(pts_world, cols_world,
                        VOXEL_SIZE_FUSED, OUTLIER_STD_FUSED, cam_c0)


def resolve_video_paths(given: list[str]) -> list[Path]:
    """Resolve CLI paths; relative paths may be cwd or BASE_DIR."""
    out: list[Path] = []
    for raw in given:
        p = Path(raw).expanduser()
        if not p.is_absolute():
            cand = Path.cwd() / p
            if cand.is_file():
                p = cand.resolve()
            else:
                alt = (BASE_DIR / raw).resolve()
                p = alt if alt.is_file() else cand.resolve()
        else:
            p = p.resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Not a file: {raw}")
        out.append(p)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fuse Depth-Anything clouds per generated video trajectory.",
    )
    ap.add_argument(
        "videos",
        nargs="*",
        metavar="VIDEO",
        help="Optional paths to specific .mp4 files; default = all under generated_videos/",
    )
    args = ap.parse_args()

    if not VIDEOS_DIR.exists():
        print(f"ERROR: missing folder {VIDEOS_DIR}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = select_device()
    print(f"Device: {device}")
    proc, mdl, _is_metric = load_depth_model(device)

    if args.videos:
        try:
            videos = resolve_video_paths(args.videos)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            return 1
    else:
        videos = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not videos:
        print(f"No videos to process.", file=sys.stderr)
        return 1

    print(f"\nFound {len(videos)} videos. "
          f"Sampling {N_FRAMES_PER_VIDEO} frames per video for fusion.")
    print(f"Output → {OUT_DIR}\n" + "=" * 64)

    # Cached per-scene HR depth + saved anchor cloud filename.
    depth_cache: dict[str, tuple[np.ndarray, PILImage.Image]] = {}
    anchor_done: set[str] = set()

    n_ok = n_skip = 0
    t_total0 = time.time()

    for i, video in enumerate(videos, 1):
        print(f"\n[{i}/{len(videos)}] {video.name}")
        try:
            image_path, traj_path, stem = video_to_paths(video)
        except ValueError as e:
            print(f"  Skipping: {e}")
            n_skip += 1
            continue

        if not image_path.exists():
            print(f"  Missing start image {image_path}; skipping.")
            n_skip += 1
            continue
        if not traj_path.exists():
            print(f"  Missing trajectory  {traj_path}; skipping.")
            n_skip += 1
            continue

        # Cache HR depth per scene.
        scene_key = image_path.stem
        if scene_key not in depth_cache:
            print(f"  Running depth model on start image {image_path.name}…")
            t0 = time.time()
            hr_pil = PILImage.open(image_path).convert("RGB")
            HR_W, HR_H = hr_pil.size
            depth_hr = infer_depth(proc, mdl, device, hr_pil, (HR_H, HR_W))
            depth_cache[scene_key] = (depth_hr, hr_pil)
            print(f"    HR depth shape={depth_hr.shape} "
                  f"range=[{depth_hr.min():.2f},{depth_hr.max():.2f}] "
                  f"({time.time() - t0:.1f}s)")
        depth_hr, hr_pil = depth_cache[scene_key]

        intr_n, w2c_all, _ = parse_re10k(traj_path)
        if len(w2c_all) == 0:
            print(f"  Empty trajectory; skipping.")
            n_skip += 1
            continue

        # Per-scene clean anchor cloud (saved once).
        if scene_key not in anchor_done:
            anchor_path = OUT_DIR / f"{scene_key}_anchor.ply"
            print(f"  Building scene anchor → {anchor_path.name}")
            pcd_a = build_anchor_pointcloud(depth_hr, hr_pil, intr_n, w2c_all[0])
            if pcd_a is not None:
                o3d.io.write_point_cloud(str(anchor_path), pcd_a, write_ascii=False)
                print(f"    ✓ {anchor_path.name}  {len(pcd_a.points):,} pts, "
                      f"{anchor_path.stat().st_size / 1e6:.2f} MB")
            anchor_done.add(scene_key)

        # Per-video fused cloud.
        out_path = OUT_DIR / f"{stem}.ply"
        print(f"  Fusing start image + {N_FRAMES_PER_VIDEO} video frames "
              f"along {traj_path.name}…")
        t0 = time.time()
        pcd = build_fused_pointcloud(video, depth_hr, hr_pil, intr_n, w2c_all,
                                     proc, mdl, device, N_FRAMES_PER_VIDEO)
        if pcd is None:
            print("  Fusion produced no points; skipping.")
            n_skip += 1
            continue
        o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False)
        sz = out_path.stat().st_size / 1e6
        print(f"  ✓ {out_path.name}  {len(pcd.points):,} pts, {sz:.2f} MB "
              f"({time.time() - t0:.1f}s)")
        grid_dir = OUT_DIR / "frame_grids"
        grid_png = grid_dir / f"{stem}_frames.png"
        try:
            if write_fusion_frame_grid(
                video, len(w2c_all), grid_png, N_FRAMES_PER_VIDEO, rows=2, cols=6
            ):
                print(f"  ✓ frame grid → {grid_png.relative_to(BASE_DIR)}")
            else:
                print("  (frame grid not written — check video decode / frame count)")
        except Exception as exc:
            print(f"  (frame grid skipped: {exc!r})")
        n_ok += 1

    print("\n" + "=" * 64)
    print(f"Done. Wrote {n_ok} fused cloud(s) + {len(anchor_done)} scene anchor(s); "
          f"skipped {n_skip}.")
    print(f"Total time: {time.time() - t_total0:.1f}s")
    print(f"Output folder: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
