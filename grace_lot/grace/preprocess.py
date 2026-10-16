"""Preprocess GRACE videos into pose and trajectory caches.

    python -m grace_lot.grace.preprocess              # the 26 Study-3 exemplars
    python -m grace_lot.grace.preprocess --all        # every locomotion video
    python -m grace_lot.grace.preprocess --preview    # also write stick-figure PNGs

Outputs (gitignored) in ``data/processed/``:
    pose/<stem>.npz    raw estimator output (image + world landmarks, visibility)
    traj/<stem>.npz    normalised Trajectory (joints, centre, yaw, fields)
    preview/<stem>.png canonical stick figures and fields, for inspection
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from grace_lot.config import load_config
from grace_lot.grace.dataset import Video, locomotion_videos, study3_exemplars
from grace_lot.grace.pose import MediaPipePose, PoseSequence
from grace_lot.grace.trajectories import Trajectory, preview, trajectory_from_pose


def processed_paths(processed_dir: Path, video: Video) -> Dict[str, Path]:
    return {k: Path(processed_dir) / k / f"{video.stem}.{ext}"
            for k, ext in (("pose", "npz"), ("traj", "npz"), ("preview", "png"))}


def load_trajectory(cfg: dict, video: Video) -> Trajectory:
    return Trajectory.load(processed_paths(cfg["data"]["processed_dir"], video)["traj"])


_estimator = None


def process_one(job) -> str:
    """Pose (cached unless ``redo``) -> trajectory -> optional preview, for one video."""
    global _estimator
    cfg, video, make_preview, redo = job
    paths = processed_paths(cfg["data"]["processed_dir"], video)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    if redo or not paths["pose"].exists():
        _estimator = _estimator or MediaPipePose(cfg["data"]["model_path"])
        _estimator(video.path).save(paths["pose"])
    try:
        traj = trajectory_from_pose(PoseSequence.load(paths["pose"]), cfg["preprocess"])
    except ValueError as e:
        return f"{video.stem}: SKIPPED ({e})"
    traj.save(paths["traj"])
    if make_preview:
        preview(traj, paths["preview"], f"{video.stem}  frames {traj.segment[0]}-{traj.segment[1]}")
    return f"{video.stem}: segment {traj.segment}, body {traj.scale_px:.0f}px"


def run(cfg: dict, videos: List[Video], make_preview: bool = False, redo: bool = False,
        workers: int = 1) -> None:
    jobs = [(cfg, v, make_preview, redo) for v in videos]
    if workers <= 1:
        results = map(process_one, jobs)
    else:
        from multiprocessing import Pool
        pool = Pool(workers)
        results = pool.imap_unordered(process_one, jobs)
    for i, line in enumerate(results):
        print(f"[{i + 1}/{len(videos)}] {line}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true", help="every locomotion video, not only the exemplars")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--redo", action="store_true", help="rerun pose estimation even if it is cached")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    videos = (locomotion_videos(d["raw_dir"]) if args.all
              else list(study3_exemplars(d["similarity_csv"], d["raw_dir"], d["substitutes"]).values()))
    run(cfg, videos, args.preview, args.redo, args.workers)


if __name__ == "__main__":
    main()
