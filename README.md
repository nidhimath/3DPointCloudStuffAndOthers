# 3D Point Cloud + Gaussian Splatting from RE10K

Generates a dense 3D point cloud and Gaussian Splat from a single bedroom image + camera trajectory from the [RealEstate10K](https://google.github.io/realestate10k/) dataset, with video frames from [DFoT](https://github.com/desaixie/gcd).

## How it works

1. **Depth estimation** — Runs [Depth Anything V2 Metric Indoor](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf) on the source image (`bedroom.jpg`) to get per-pixel metric depth
2. **Unprojection** — Back-projects the depth map into 3D using the RE10K camera intrinsics and pose, producing a dense point cloud (~470k points)
3. **Cleanup** — Voxel downsamples (1 cm) and removes outliers with Open3D
4. **GS init** — Converts the point cloud to a 3DGS-format PLY (SH coefficients, opacity, scale, rotation) for Brush initialization
5. **Training** — Trains a Gaussian Splat for 15k steps using [Brush](https://github.com/ArthurBrussee/brush) (Metal-native, macOS)

## Inputs (`project_data/`)

| File | Description |
|------|-------------|
| `bedroom.jpg` | Starting frame (1878×1050, real photo) |
| `fff9864727c42c80.txt` | RE10K camera trajectory (109 poses, world→cam OpenCV) |
| `output64.mp4` | DFoT-generated video (64 frames, 256×256) |

## Usage

```bash
pip install jupyter
jupyter notebook gaussian_splatting_from_image_trajectory.ipynb
```

Then **Kernel → Restart & Run All**. Outputs appear in `workdir_gs/splats/`.

## Viewing the output

```bash
# Point cloud
python view_ply.py workdir_gs/splats/bedroom_depth_cloud.ply

# Gaussian Splat (after training)
tools/brush-app-aarch64-apple-darwin/brush_app --with-viewer workdir_gs/splats/bedroom_15000.ply
```

## Requirements

- macOS (Apple Silicon recommended for MPS acceleration)
- Python 3.10+
- ~4 GB disk for model weights (auto-downloaded from HuggingFace on first run)
