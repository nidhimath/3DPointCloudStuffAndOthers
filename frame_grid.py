"""2×6 contact-sheet PNGs of the exact video frames used for point-cloud fusion.

Uses the same linspace frame ↔ pose pairing as `generate_pointclouds.build_fused_pointcloud`.
Reads only the picked frames via OpenCV seek (no full-video decode).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont


def compute_fusion_frame_picks(
    num_bgr: int, num_poses: int, n_sample: int
) -> tuple[np.ndarray, np.ndarray]:
    """Same indices as `generate_pointclouds.build_fused_pointcloud` (video + pose rows)."""
    if num_bgr <= 0 or num_poses <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n_match = min(num_bgr, num_poses)
    frame_idx_full = np.linspace(0, num_bgr - 1, n_match).round().astype(int)
    pose_idx_full = np.linspace(0, num_poses - 1, n_match).round().astype(int)
    n_take = max(1, min(n_sample, n_match))
    sub = np.linspace(0, n_match - 1, n_take).round().astype(int)
    return frame_idx_full[sub], pose_idx_full[sub]


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, n)


def read_video_frame_rgb(video_path: Path, index: int) -> np.ndarray | None:
    """Decode a single frame by index (0-based). Returns RGB uint8 or None."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit_font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        p = Path(name)
        if p.is_file():
            try:
                return ImageFont.truetype(str(p), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def _letterbox_resize(rgb: np.ndarray, cell_w: int, cell_h: int) -> PILImage.Image:
    h, w = rgb.shape[:2]
    scale = min(cell_w / w, cell_h / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    im = PILImage.fromarray(rgb).resize((nw, nh), PILImage.Resampling.LANCZOS)
    canvas = PILImage.new("RGB", (cell_w, cell_h), (0, 0, 0))
    ox = (cell_w - nw) // 2
    oy = (cell_h - nh) // 2
    canvas.paste(im, (ox, oy))
    return canvas


def build_contact_sheet(
    frames_rgb: list[np.ndarray],
    labels: list[str],
    rows: int = 2,
    cols: int = 6,
    cell_h: int = 360,
    gutter: int = 2,
    margin: int = 4,
) -> PILImage.Image:
    if len(frames_rgb) != rows * cols:
        raise ValueError(f"Need {rows * cols} frames, got {len(frames_rgb)}")
    if len(labels) != len(frames_rgb):
        raise ValueError("labels must match frames")

    fh, fw = frames_rgb[0].shape[:2]
    cell_w = max(1, int(round(cell_h * fw / fh)))

    font = _fit_font(20)
    tiles: list[PILImage.Image] = []
    for rgb, lab in zip(frames_rgb, labels, strict=True):
        tile = _letterbox_resize(rgb, cell_w, cell_h)
        draw = ImageDraw.Draw(tile)
        pad = 6
        bbox = draw.textbbox((0, 0), lab, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cell_w - pad - tw
        ty = cell_h - pad - th
        for dx, dy in ((1, 1), (-1, -1), (1, -1), (-1, 1), (0, 1), (1, 0), (-1, 0), (0, -1)):
            draw.text((tx + dx, ty + dy), lab, font=font, fill=(0, 0, 0))
        draw.text((tx, ty), lab, font=font, fill=(255, 255, 255))
        tiles.append(tile)

    gw, gh = gutter, gutter
    sheet_w = cols * cell_w + (cols - 1) * gw + 2 * margin
    sheet_h = rows * cell_h + (rows - 1) * gh + 2 * margin
    sheet = PILImage.new("RGB", (sheet_w, sheet_h), (0, 0, 0))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = margin + c * (cell_w + gw)
        y = margin + r * (cell_h + gh)
        sheet.paste(tile, (x, y))
    return sheet


def write_fusion_frame_grid(
    video_path: Path,
    num_poses: int,
    out_png: Path,
    n_sample: int = 12,
    rows: int = 2,
    cols: int = 6,
) -> bool:
    """Write a rows×cols sheet of fusion-sampled frames. Returns False if skipped."""
    n_bgr = _video_frame_count(video_path)
    if n_bgr == 0:
        return False

    frame_picks, pose_picks = compute_fusion_frame_picks(n_bgr, num_poses, n_sample)
    if len(frame_picks) == 0:
        return False

    frames: list[np.ndarray] = []
    for fi in frame_picks:
        rgb = read_video_frame_rgb(video_path, int(fi))
        if rgb is None:
            # Seek can fail on some files; fall back sequential scan for this index
            rgb = _read_frame_sequential_fallback(video_path, int(fi))
        if rgb is None:
            return False
        frames.append(rgb)

    labels = [f"t={int(fi)}" for fi in frame_picks]
    sheet = build_contact_sheet(frames, labels, rows=rows, cols=cols)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, format="PNG", optimize=True)
    return True


def _read_frame_sequential_fallback(video_path: Path, target_index: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    i = 0
    while i <= target_index:
        ok, bgr = cap.read()
        if not ok:
            cap.release()
            return None
        if i == target_index:
            cap.release()
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()
    return None


def _parse_re10k_pose_count(path: Path) -> int:
    n = 0
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("http"):
            continue
        if len(line.split()) == 19:
            n += 1
    return n


def video_to_paths(video_path: Path, base: Path) -> tuple[Path, Path, str]:
    """Same rules as `generate_pointclouds.video_to_paths`."""
    start_images = base / "start_images"
    trajectories = base / "trajectories"
    stem = video_path.stem
    if stem.endswith("_standard"):
        base_name = stem[: -len("_standard")]
        image = start_images / f"{base_name}.jpg"
        traj = trajectories / f"{base_name}.txt"
        return image, traj, stem
    if re.fullmatch(r".+_\d+deg", stem):
        base_name = stem.rsplit("_", 1)[0]
        image = start_images / f"{base_name}.jpg"
        traj = trajectories / f"{stem}.txt"
        return image, traj, stem
    m = re.match(r"^(.+)_(\d+)$", stem)
    if not m:
        raise ValueError(f"Unrecognised video name: {video_path.name}")
    base_name, suffix = m.group(1), m.group(2)
    image = start_images / f"{base_name}.jpg"
    traj = trajectories / f"{base_name}_{suffix}.txt"
    return image, traj, stem


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Write 2×6 PNG contact sheets for fusion frames (per video).",
    )
    ap.add_argument(
        "videos",
        nargs="*",
        metavar="VIDEO",
        help="Video paths; default = all *.mp4 under generated_videos/",
    )
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: pointclouds/frame_grids next to this script)",
    )
    args = ap.parse_args()

    base = Path(__file__).resolve().parent
    videos_dir = base / "generated_videos"
    out_dir = args.out_dir or (base / "pointclouds" / "frame_grids")

    if args.videos:
        videos = [Path(p).expanduser().resolve() for p in args.videos]
    else:
        if not videos_dir.is_dir():
            print(f"ERROR: {videos_dir} not found", file=sys.stderr)
            return 1
        videos = sorted(videos_dir.glob("*.mp4"))

    if not videos:
        print("No videos to process.", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    ok = skip = 0
    for video in videos:
        if not video.is_file():
            print(f"Skip (not a file): {video}")
            skip += 1
            continue
        try:
            _img, traj_path, stem = video_to_paths(video, base)
        except ValueError as e:
            print(f"{video.name}: {e}")
            skip += 1
            continue
        if not traj_path.is_file():
            print(f"{video.name}: missing trajectory {traj_path}")
            skip += 1
            continue

        n_pose = _parse_re10k_pose_count(traj_path)
        out_png = out_dir / f"{stem}_frames.png"
        if write_fusion_frame_grid(video, n_pose, out_png):
            print(f"✓ {out_png.name}")
            ok += 1
        else:
            print(f"✗ {video.name} (decode / frame count)")
            skip += 1

    print(f"Done. {ok} written, {skip} skipped. → {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
