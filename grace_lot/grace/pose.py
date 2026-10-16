"""Pose estimation front end.

Any estimator that returns a ``PoseSequence`` can be plugged in. The
default is MediaPipe Pose Landmarker (heavy): a pretrained, single-person
model with 33 landmarks, 2-D image coordinates, 3-D world coordinates (in
metres, hip-centred) and per-landmark visibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np


@dataclass
class PoseSequence:
    image: np.ndarray        # (F, J, 2) pixel coordinates (x right, y down); NaN where nothing was detected
    world: np.ndarray        # (F, J, 3) metres, hip-centred; NaN where nothing was detected
    visibility: np.ndarray   # (F, J) in [0, 1]; 0 where nothing was detected
    fps: float
    size: Tuple[int, int]    # (width, height)
    connections: List[Tuple[int, int]]   # the estimator's own skeleton edges
    joint_names: List[str]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, image=self.image, world=self.world, visibility=self.visibility,
                            fps=self.fps, size=np.array(self.size),
                            connections=np.array(self.connections), joint_names=np.array(self.joint_names))

    @classmethod
    def load(cls, path: Path) -> "PoseSequence":
        d = np.load(path)
        return cls(d["image"], d["world"], d["visibility"], float(d["fps"]), tuple(d["size"]),
                   [tuple(c) for c in d["connections"]], list(d["joint_names"]))


class MediaPipePose:
    """MediaPipe Pose Landmarker in VIDEO mode, so it tracks the person across frames."""

    def __init__(self, model_path: Path) -> None:
        from mediapipe.tasks.python import BaseOptions, vision
        self._vision = vision
        self._options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1)

    def __call__(self, video_path: Path) -> PoseSequence:
        import cv2
        import mediapipe as mp
        vision = self._vision
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        J = len(vision.PoseLandmark)
        image, world, vis = [], [], []
        with vision.PoseLandmarker.create_from_options(self._options) as landmarker:
            n = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                res = landmarker.detect_for_video(rgb, int(round(n * 1000 / fps)))
                n += 1
                if res.pose_landmarks:
                    lm, wl = res.pose_landmarks[0], res.pose_world_landmarks[0]
                    image.append([[p.x * w, p.y * h] for p in lm])
                    world.append([[p.x, p.y, p.z] for p in wl])
                    vis.append([p.visibility for p in lm])
                else:
                    image.append(np.full((J, 2), np.nan))
                    world.append(np.full((J, 3), np.nan))
                    vis.append(np.zeros(J))
        cap.release()
        connections = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
        names = [p.name.lower() for p in vision.PoseLandmark]
        return PoseSequence(np.array(image, float), np.array(world, float), np.array(vis, float),
                            float(fps), (w, h), connections, names)
