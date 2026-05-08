import sys
import numpy as np
import open3d as o3d
from plyfile import PlyData

PLY = sys.argv[1] if len(sys.argv) > 1 else (
    "/Users/nidhimathihalli/Downloads/drive-download-20260507T233133Z-3-001"
    "/workdir_gs/splats/bedroom_5000.ply"
)

data  = PlyData.read(PLY)
v     = data["vertex"]
props = [p.name for p in v.properties]
print("Properties:", props[:12], "...")

xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

# Gaussian Splatting PLY stores colour as SH DC coefficients.
# RGB = clamp(SH_C0 * f_dc + 0.5,  0, 1)  where SH_C0 = 1/(2*sqrt(pi))
if "f_dc_0" in props:
    SH_C0 = 0.28209479177387814
    rgb = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1).astype(np.float64)
    rgb = np.clip(SH_C0 * rgb + 0.5, 0.0, 1.0)
    print("Decoded SH colours (Gaussian Splatting PLY).")
elif "red" in props:
    rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
    print("Standard RGB point cloud.")
else:
    rgb = np.ones((len(xyz), 3)) * 0.7
    print("No colour data found; using grey.")

print(f"Points: {len(xyz):,}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries(
    [pcd],
    window_name=PLY.split("/")[-1],
    width=1280, height=800,
)
