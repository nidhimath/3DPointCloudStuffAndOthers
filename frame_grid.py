"""Contact sheets of fusion-sampled video frames (same pairing as `generate_pointclouds`).

- Default: one 2×6 PNG per video.
- ``--combined``: per scene, one 5×12 PNG — rows are standard then each trajectory variant
  (numeric: _standard, _1…_4; degrees: _standard, _45deg…_180deg), 12 fusion columns.
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
    font_size: int = 20,
) -> PILImage.Image:
    if len(frames_rgb) != rows * cols:
        raise ValueError(f"Need {rows * cols} frames, got {len(frames_rgb)}")
    if len(labels) != len(frames_rgb):
        raise ValueError("labels must match frames")

    fh, fw = frames_rgb[0].shape[:2]
    cell_w = max(1, int(round(cell_h * fw / fh)))

    font = _fit_font(font_size)
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
    got = collect_twelve_fusion_frames(video_path, num_poses, n_sample=n_sample)
    if got is None:
        return False
    frames, idxs = got
    labels = [f"t={i}" for i in idxs]
    sheet = build_contact_sheet(frames, labels, rows=rows, cols=cols)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_png, format="PNG", optimize=True)
    return True


def collect_twelve_fusion_frames(
    video_path: Path, num_poses: int, n_sample: int = 12
) -> tuple[list[np.ndarray], list[int]] | None:
    """Decode the n_sample fusion frames; returns (frames, frame_indices) or None."""
    n_bgr = _video_frame_count(video_path)
    if n_bgr == 0:
        return None
    frame_picks, _ = compute_fusion_frame_picks(n_bgr, num_poses, n_sample)
    if len(frame_picks) != n_sample:
        return None
    frames: list[np.ndarray] = []
    for fi in frame_picks:
        rgb = read_video_frame_rgb(video_path, int(fi))
        if rgb is None:
            rgb = _read_frame_sequential_fallback(video_path, int(fi))
        if rgb is None:
            return None
        frames.append(rgb)
    return frames, [int(x) for x in frame_picks]


def write_combined_scene_grid(
    row_videos: list[Path],
    base_dir: Path,
    out_png: Path,
    n_sample: int = 12,
    rows: int = 5,
    cols: int = 12,
    cell_h: int = 160,
    font_size: int = 14,
) -> bool:
    """One sheet: each row is a trajectory video, each column a fusion sample (12)."""
    if len(row_videos) != rows:
        return False
    all_rgb: list[np.ndarray] = []
    all_labels: list[str] = []
    for video in row_videos:
        try:
            _img, traj_path, _stem = video_to_paths(video, base_dir)
        except ValueError:
            return False
        if not traj_path.is_file():
            return False
        n_pose = _parse_re10k_pose_count(traj_path)
        got = collect_twelve_fusion_frames(video, n_pose, n_sample=n_sample)
        if got is None:
            return False
        frames, idxs = got
        all_rgb.extend(frames)
        all_labels.extend(f"t={i}" for i in idxs)
    sheet = build_contact_sheet(
        all_rgb,
        all_labels,
        rows=rows,
        cols=cols,
        cell_h=cell_h,
        font_size=font_size,
    )
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


NUMERIC_ROW_ORDER = ["standard", "1", "2", "3", "4"]
DEGREE_ROW_ORDER = ["standard", "45deg", "90deg", "135deg", "180deg"]


def stem_for_scene_row(scene_base: str, row_suffix: str) -> str:
    return (
        f"{scene_base}_standard"
        if row_suffix == "standard"
        else f"{scene_base}_{row_suffix}"
    )


def stem_index(videos: list[Path]) -> dict[str, Path]:
    return {p.stem: p.resolve() for p in videos if p.is_file()}


def scene_bases_from_stems(idx: dict[str, Path]) -> set[str]:
    bases: set[str] = set()
    for stem in idx:
        if stem.endswith("_standard"):
            bases.add(stem[: -len("_standard")])
            continue
        m = re.fullmatch(r"(.+)_(\d+)deg", stem)
        if m:
            bases.add(m.group(1))
            continue
        m = re.fullmatch(r"(.+)_(\d+)$", stem)
        if m:
            bases.add(m.group(1))
    return bases


def numeric_row_paths(scene_base: str, idx: dict[str, Path]) -> list[Path] | None:
    out: list[Path] = []
    for suf in NUMERIC_ROW_ORDER:
        st = stem_for_scene_row(scene_base, suf)
        p = idx.get(st)
        if p is None:
            return None
        out.append(p)
    return out


def degree_row_paths(scene_base: str, idx: dict[str, Path]) -> list[Path] | None:
    out: list[Path] = []
    for suf in DEGREE_ROW_ORDER:
        st = stem_for_scene_row(scene_base, suf)
        p = idx.get(st)
        if p is None:
            return None
        out.append(p)
    return out


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
        description="PNG contact sheets for fusion frames (2×6 per video, or 5×12 per scene).",
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
    ap.add_argument(
        "--combined",
        action="store_true",
        help="Emit one 5×12 sheet per scene: rows standard + (_1…_4) or standard + (_45deg…_180deg).",
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

    if args.combined:
        idx = stem_index(videos)
        bases = sorted(scene_bases_from_stems(idx))
        ok = fail = 0
        for scene_base in bases:
            for family, row_fn in (
                ("numeric", numeric_row_paths),
                ("degrees", degree_row_paths),
            ):
                row_paths = row_fn(scene_base, idx)
                if row_paths is None:
                    continue
                out_png = out_dir / f"{scene_base}_fusion_{family}_5x12.png"
                if write_combined_scene_grid(row_paths, base, out_png):
                    print(f"✓ {out_png.name}")
                    ok += 1
                else:
                    print(
                        f"✗ {scene_base} ({family}): trajectories / decode / "
                        f"need 12 fusion frames per row",
                    )
                    fail += 1
        print(f"Done (combined). {ok} written, {fail} failed. → {out_dir}")
        return 0 if ok else 1

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
            print(f"✗ {video.name} (decode / frame count / need 12 fusion picks)")
            skip += 1

    print(f"Done. {ok} written, {skip} skipped. → {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
