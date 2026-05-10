"""View a .ply point cloud in an Open3D window.

Usage:
  .venv/bin/python view_ply.py path/to/cloud.ply

- Open3D-native coloured clouds (x,y,z,nx,ny,nz, red,green,blue) load directly.
- 3D Gaussian Splatting PLYs (f_dc_* SH coefficients) are decoded for colour.

If no window appears: you need a graphical session (macOS Terminal.app or iTerm,
not SSH without X11). From Cursor’s terminal it sometimes works; if not, open
the same command in Terminal.app or use MeshLab (see fusion_pipeline docs).

If the path contains spaces, quote it:
  .venv/bin/python view_ply.py "pointclouds/2cf1a544b179b1a7_standard.ply"
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData


def _load_gaussian_splat_rgb_ply(path: Path) -> o3d.geometry.PointCloud:
    data = PlyData.read(str(path))
    v = data["vertex"]
    props = [p.name for p in v.properties]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

    if "f_dc_0" in props:
        sh_c0 = 0.28209479177387814
        rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(
            np.float64
        )
        rgb = np.clip(sh_c0 * rgb + 0.5, 0.0, 1.0)
        kind = "Gaussian splat (SH f_dc)"
    elif "red" in props:
        rgb = (
            np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64)
            / 255.0
        )
        kind = "RGB vertex colours (plyfile)"
    else:
        rgb = np.ones((len(xyz), 3)) * 0.7
        kind = "no colour; grey"

    print(f"Loaded via plyfile: {kind}  |  {len(xyz):,} points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2

    path = Path(sys.argv[1]).expanduser().resolve()
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    # Prefer Open3D’s reader: handles our fused clouds and preserves normals.
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        print("Open3D read 0 points; trying plyfile (e.g. non-standard PLY)…")
        pcd = _load_gaussian_splat_rgb_ply(path)
    else:
        n = len(pcd.points)
        has_colour = pcd.has_colors()
        print(f"Loaded via Open3D: {n:,} points  |  colours={has_colour}")

    o3d.visualization.draw_geometries(
        [pcd],
        window_name=path.name,
        width=1280,
        height=800,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
