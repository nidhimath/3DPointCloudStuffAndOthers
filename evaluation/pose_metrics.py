"""Camera-trajectory evaluation (TPS-inspired).

Different camera trajectories applied to the same image yield different
reconstructions; we therefore care not only about geometry but about
*how consistent the underlying camera trajectory is* with the reference.

Metrics implemented
-------------------
* ``umeyama_alignment``  - similarity (sR, t) alignment of two trajectories
  (also exposes a strict SE(3) variant).
* ``absolute_trajectory_error`` - RMS of per-frame translation error after
  alignment ("ATE", as in TUM-RGBD).
* ``relative_pose_error`` - drift between consecutive frames; reports both
  translation and rotation components.
* ``pose_consistency_score`` - a TPS-*inspired* combined score.

Note on "TPS"
-------------
In the literature on trajectory-conditioned novel-view synthesis, "TPS" /
"True Pose Similarity" is sometimes used to describe the cross-scene
transferability of a camera-trajectory representation. We do **not**
implement that full notion. ``pose_consistency_score`` here is a
*task-specific surrogate* combining well-understood SLAM-style metrics
(ATE + RPE) into a single scalar that is easy to compare across runs.

Trajectory file formats
-----------------------
We accept two common formats automatically:

* Nerfstudio / Instant-NGP ``transforms.json`` - dict with a ``frames``
  list, each frame having ``transform_matrix`` (a 4x4 *camera-to-world*
  matrix in OpenGL convention).
* RealEstate10K ``.txt`` - whitespace-separated rows where the last 12
  numbers form the world-to-camera 3x4 extrinsics in OpenCV convention.

Both are converted to a canonical ``(N, 4, 4)`` array of camera-to-world
matrices in OpenCV convention (z-forward, y-down). This is the only
convention used inside the metric functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


PathLike = Union[str, Path]


# OpenGL has +x right, +y up, +z back. OpenCV has +x right, +y down, +z forward.
# Going one way or the other is a flip of the y- and z-axes of the camera frame.
_OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_transforms_json(path: Path) -> np.ndarray:
    """Load a Nerfstudio-style ``transforms.json`` and return c2w (OpenCV)."""
    with open(path, "r") as f:
        data = json.load(f)

    frames = data.get("frames")
    if not frames:
        raise ValueError(f"{path} contains no 'frames' list")

    mats = []
    for frame in frames:
        m = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(
                f"transform_matrix in {path} has shape {m.shape}, expected (4, 4)"
            )
        # Nerfstudio stores OpenGL c2w; convert to OpenCV by flipping the
        # camera y- and z-axes (post-multiplication on the c2w matrix).
        mats.append(m @ _OPENGL_TO_OPENCV)
    return np.stack(mats, axis=0)


def _load_re10k_txt(path: Path) -> np.ndarray:
    """Load a RealEstate10K trajectory file and return c2w (OpenCV).

    The first line is a YouTube URL; subsequent lines have the form

        timestamp fx fy cx cy k1 k2 r11 r12 r13 t1 r21 r22 r23 t2 r31 r32 r33 t3

    which is a 3x4 world-to-camera matrix. We invert it to obtain c2w.
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if not lines:
        raise ValueError(f"Empty trajectory file: {path}")

    # Skip the URL header if present.
    if not lines[0].split()[0].replace(".", "").lstrip("-").isdigit():
        lines = lines[1:]

    c2w_list = []
    for ln in lines:
        toks = ln.split()
        if len(toks) < 19:
            continue
        nums = np.array(toks[-12:], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = nums.reshape(3, 4)
        c2w_list.append(np.linalg.inv(w2c))
    if not c2w_list:
        raise ValueError(f"No pose rows parsed from {path}")
    return np.stack(c2w_list, axis=0)


def load_trajectory(path: PathLike) -> np.ndarray:
    """Load a trajectory from disk.

    Returns
    -------
    np.ndarray of shape ``(N, 4, 4)`` containing camera-to-world matrices
    in OpenCV convention.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory not found: {path}")

    if path.suffix.lower() == ".json":
        return _load_transforms_json(path)
    if path.suffix.lower() == ".txt":
        return _load_re10k_txt(path)
    raise ValueError(f"Unsupported trajectory format: {path.suffix!r}")


def trajectory_translations(c2w: np.ndarray) -> np.ndarray:
    """Return the ``(N, 3)`` array of camera *centres* from c2w matrices."""
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N, 4, 4), got {c2w.shape}")
    return c2w[:, :3, 3].astype(np.float64, copy=False)


def trajectory_rotations(c2w: np.ndarray) -> np.ndarray:
    """Return the ``(N, 3, 3)`` rotation block of c2w matrices."""
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError(f"Expected (N, 4, 4), got {c2w.shape}")
    return c2w[:, :3, :3].astype(np.float64, copy=False)


def match_trajectory_lengths(
    pred: np.ndarray, gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample the longer trajectory to the length of the shorter one.

    Predicted and GT trajectories often have different frame counts (e.g.
    the GT contains 109 RE10K poses while the generated video has 64).
    We match them by *integer index sub-sampling*, which preserves rotation
    matrices exactly (no SLERP is needed). For dense trajectories this is
    a good approximation; for very sparse ones a SLERP-based matcher would
    be a future improvement.
    """
    n = min(len(pred), len(gt))
    if n < 2:
        raise ValueError("Need at least 2 frames per trajectory")

    if len(pred) != n:
        idx = np.linspace(0, len(pred) - 1, n).round().astype(int)
        pred = pred[idx]
    if len(gt) != n:
        idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
        gt = gt[idx]
    return pred, gt


# ---------------------------------------------------------------------------
# SE(3) / Sim(3) alignment (Umeyama)
# ---------------------------------------------------------------------------


@dataclass
class AlignmentResult:
    """Result of trajectory alignment.

    The estimated transform takes a *predicted* camera centre into the GT
    frame:  ``t_gt ~= scale * R @ t_pred + translation``.
    """

    rotation: np.ndarray         # (3, 3)
    translation: np.ndarray      # (3,)
    scale: float
    with_scaling: bool

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        return (self.scale * (self.rotation @ points.T)).T + self.translation

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "scale": float(self.scale),
            "with_scaling": bool(self.with_scaling),
        }


def umeyama_alignment(
    src: np.ndarray,
    dst: np.ndarray,
    with_scaling: bool = True,
) -> AlignmentResult:
    """Closed-form least-squares similarity alignment (Umeyama 1991).

    Solves for ``(s, R, t)`` minimising
        sum_i || dst_i - (s R src_i + t) ||^2
    via SVD of the cross-covariance matrix. With ``with_scaling=False``
    this collapses to the standard SE(3) Procrustes solution used in the
    SLAM literature for ATE.

    Notes
    -----
    Monocular reconstructions are inherently up-to-scale, so we default to
    similarity alignment. Use the strict SE(3) variant when the predicted
    trajectory is already in metric units (e.g. when fed by a depth-anchored
    pipeline) and you specifically want to penalise scale error.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(
            f"Expected matching (N, 3) arrays, got {src.shape} and {dst.shape}"
        )

    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance.
    sigma = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(sigma)

    # Reflection correction: ensure a proper rotation (det = +1).
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    R = U @ S @ Vt

    if with_scaling:
        var_src = (src_c ** 2).sum() / n
        scale = float((D * np.diag(S)).sum() / max(var_src, 1e-12))
    else:
        scale = 1.0

    t = mu_dst - scale * R @ mu_src
    return AlignmentResult(rotation=R, translation=t, scale=scale,
                           with_scaling=with_scaling)


# ---------------------------------------------------------------------------
# ATE / RPE
# ---------------------------------------------------------------------------


@dataclass
class ATEResult:
    ate_rmse: float                 # primary number reported as "ATE"
    ate_mean: float
    ate_median: float
    ate_max: float
    aligned_with_scale: bool
    n_frames: int

    def to_dict(self) -> dict:
        return asdict(self)


def absolute_trajectory_error(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    with_scaling: bool = True,
) -> Tuple[ATEResult, AlignmentResult]:
    """Compute ATE between predicted and GT camera-centre trajectories.

    Steps:
      1. Match trajectory lengths.
      2. Solve Umeyama alignment on the camera centres.
      3. Apply the alignment to the predicted centres.
      4. Report L2 error statistics. RMSE is the canonical ATE.
    """
    pred_c2w, gt_c2w = match_trajectory_lengths(pred_c2w, gt_c2w)

    pred_t = trajectory_translations(pred_c2w)
    gt_t = trajectory_translations(gt_c2w)

    align = umeyama_alignment(pred_t, gt_t, with_scaling=with_scaling)
    pred_t_aligned = align.transform_points(pred_t)

    err = np.linalg.norm(pred_t_aligned - gt_t, axis=1)
    ate = ATEResult(
        ate_rmse=float(np.sqrt(np.mean(err ** 2))),
        ate_mean=float(err.mean()),
        ate_median=float(np.median(err)),
        ate_max=float(err.max()),
        aligned_with_scale=bool(with_scaling),
        n_frames=int(err.shape[0]),
    )
    return ate, align


@dataclass
class RPEResult:
    rpe_translation_rmse: float     # metres (or scene units)
    rpe_translation_mean: float
    rpe_rotation_rmse_deg: float    # degrees
    rpe_rotation_mean_deg: float
    delta: int                      # frame step used for relative motion
    n_pairs: int

    def to_dict(self) -> dict:
        return asdict(self)


def _rotation_angle_deg(R: np.ndarray) -> float:
    """Return the rotation angle of ``R`` in degrees."""
    # Standard formula:  theta = arccos((trace(R) - 1) / 2)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def relative_pose_error(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    delta: int = 1,
    align: Optional[AlignmentResult] = None,
) -> RPEResult:
    """RPE: per-frame drift in relative motion.

    For frame pair ``(i, i+delta)`` we compute the *relative* motion in each
    trajectory and the residual between them:

        E_i = (gt[i]^-1 @ gt[i+d]) ^ -1  @  (pred[i]^-1 @ pred[i+d])

    The translation norm and rotation angle of ``E_i`` are accumulated.

    Notes
    -----
    * RPE is sensitive to *local* drift but invariant to a global SE(3)
      mismatch, so it is the right complement to ATE (which captures
      global error).
    * If the prediction is up-to-scale, supply the Umeyama alignment (with
      its scale) so the predicted translations are rescaled before the
      relative motion is computed. We do *not* rotate predictions here -
      RPE is invariant to any global rotation.
    """
    if delta < 1:
        raise ValueError("delta must be >= 1")

    pred_c2w, gt_c2w = match_trajectory_lengths(pred_c2w.copy(), gt_c2w.copy())
    n = len(pred_c2w)
    if n <= delta:
        raise ValueError(f"Trajectory too short for delta={delta} (n={n})")

    if align is not None and align.scale != 1.0:
        # Rescale predicted translations so that translation magnitudes
        # are comparable. Rotation block is unchanged.
        pred_c2w = pred_c2w.copy()
        pred_c2w[:, :3, 3] *= align.scale

    trans_err = []
    rot_err_deg = []
    for i in range(n - delta):
        rel_pred = np.linalg.inv(pred_c2w[i]) @ pred_c2w[i + delta]
        rel_gt = np.linalg.inv(gt_c2w[i]) @ gt_c2w[i + delta]
        E = np.linalg.inv(rel_gt) @ rel_pred

        trans_err.append(float(np.linalg.norm(E[:3, 3])))
        rot_err_deg.append(_rotation_angle_deg(E[:3, :3]))

    trans_err = np.asarray(trans_err)
    rot_err_deg = np.asarray(rot_err_deg)

    return RPEResult(
        rpe_translation_rmse=float(np.sqrt(np.mean(trans_err ** 2))),
        rpe_translation_mean=float(trans_err.mean()),
        rpe_rotation_rmse_deg=float(np.sqrt(np.mean(rot_err_deg ** 2))),
        rpe_rotation_mean_deg=float(rot_err_deg.mean()),
        delta=int(delta),
        n_pairs=int(trans_err.shape[0]),
    )


# ---------------------------------------------------------------------------
# Pose Consistency Score (TPS-inspired)
# ---------------------------------------------------------------------------


@dataclass
class PoseConsistencyResult:
    pose_consistency_score: float   # in [0, 1], higher is better
    ate_term: float
    rpe_translation_term: float
    rpe_rotation_term: float
    ate_scale: float
    rpe_translation_scale: float
    rpe_rotation_scale_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


def pose_consistency_score(
    ate: ATEResult,
    rpe: RPEResult,
    ate_scale: float = 0.10,
    rpe_translation_scale: float = 0.05,
    rpe_rotation_scale_deg: float = 5.0,
) -> PoseConsistencyResult:
    """A bounded scalar combining ATE and RPE.

    Definition
    ----------
        PCS = exp(-ATE/sigma_a) * exp(-RPE_t/sigma_t) * exp(-RPE_r/sigma_r)

    where the ``sigma`` constants are characteristic error scales. Each
    factor lives in (0, 1]; the product is therefore also in (0, 1] and is
    monotonically decreasing in every error component.

    Caveat
    ------
    This is *not* the "True Pose Similarity" of the trajectory-conditioned
    novel-view synthesis literature, which is a cross-scene transferability
    notion. We use the same TPS spelling because the spirit is the same -
    measure how well a predicted trajectory matches a reference pose
    sequence - while keeping the implementation simple and reproducible.
    """
    ate_term = float(np.exp(-ate.ate_rmse / max(ate_scale, 1e-12)))
    rpe_t_term = float(np.exp(-rpe.rpe_translation_rmse / max(rpe_translation_scale, 1e-12)))
    rpe_r_term = float(np.exp(-rpe.rpe_rotation_rmse_deg / max(rpe_rotation_scale_deg, 1e-12)))

    return PoseConsistencyResult(
        pose_consistency_score=ate_term * rpe_t_term * rpe_r_term,
        ate_term=ate_term,
        rpe_translation_term=rpe_t_term,
        rpe_rotation_term=rpe_r_term,
        ate_scale=float(ate_scale),
        rpe_translation_scale=float(rpe_translation_scale),
        rpe_rotation_scale_deg=float(rpe_rotation_scale_deg),
    )


# ---------------------------------------------------------------------------
# One-shot bundle
# ---------------------------------------------------------------------------


def compute_pose_metrics(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    with_scaling: bool = True,
    rpe_delta: int = 1,
    pcs_kwargs: Optional[dict] = None,
) -> dict:
    """Compute ATE, RPE and PCS in one call. Returns plain dicts for JSON."""
    ate, align = absolute_trajectory_error(
        pred_c2w, gt_c2w, with_scaling=with_scaling
    )
    rpe = relative_pose_error(pred_c2w, gt_c2w, delta=rpe_delta, align=align)
    pcs = pose_consistency_score(ate, rpe, **(pcs_kwargs or {}))

    return {
        "ate": ate.to_dict(),
        "rpe": rpe.to_dict(),
        "alignment": align.to_dict(),
        "pose_consistency": pcs.to_dict(),
    }
