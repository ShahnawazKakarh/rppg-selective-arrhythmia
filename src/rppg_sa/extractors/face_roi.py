"""Face tracking and ROI extraction via MediaPipe FaceMesh.

Returns mean RGB time-series from the forehead and bilateral cheek regions for
downstream rPPG extraction (CHROM, POS, PhysNet).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices for the canonical 468-point topology.
# These are the standard reference landmarks used in the rPPG literature.
FOREHEAD_LANDMARKS = [10, 67, 69, 109, 108, 151, 337, 338, 297, 299]
LEFT_CHEEK_LANDMARKS = [50, 101, 118, 119, 120, 100, 36, 47]
RIGHT_CHEEK_LANDMARKS = [280, 330, 347, 348, 349, 329, 266, 277]


@dataclass
class ROIMeanRGB:
    """Mean RGB time-series for a set of facial ROIs.

    Attributes:
        forehead: (T, 3) array, mean RGB per frame for the forehead ROI.
        left_cheek: (T, 3) array, mean RGB per frame for the left cheek ROI.
        right_cheek: (T, 3) array, mean RGB per frame for the right cheek ROI.
        fps: Source video frame rate.
        valid_frames: Boolean mask (T,) flagging frames where the face was detected.
    """

    forehead: np.ndarray
    left_cheek: np.ndarray
    right_cheek: np.ndarray
    fps: float
    valid_frames: np.ndarray


def _polygon_mean_rgb(frame: np.ndarray, landmarks_xy: np.ndarray) -> np.ndarray:
    """Mean RGB inside a convex polygon defined by pixel-space landmarks.

    Args:
        frame: (H, W, 3) BGR uint8 array as returned by OpenCV.
        landmarks_xy: (K, 2) float array of pixel coordinates.

    Returns:
        (3,) float array of mean RGB values (in RGB order, not BGR).
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = landmarks_xy.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
    mean_bgr = cv2.mean(frame, mask=mask)[:3]
    # OpenCV returns BGR; flip to RGB for downstream convention.
    return np.array([mean_bgr[2], mean_bgr[1], mean_bgr[0]], dtype=np.float64)


def extract_roi_signals(video_path: str, max_frames: int | None = None) -> ROIMeanRGB:
    """Extract per-frame mean RGB time-series from forehead and cheek ROIs.

    Args:
        video_path: Path to a video file readable by OpenCV.
        max_frames: Optional cap on number of frames processed.

    Returns:
        ROIMeanRGB container with three (T, 3) arrays and metadata.
    """
    # Lazy import: mediapipe is heavy and not needed for every code path.
    # Defensive against multiple mediapipe layouts:
    #   - 0.10.x exposes `mediapipe.solutions.face_mesh`
    #   - some 0.11.x builds expose it as `mediapipe.python.solutions.face_mesh`
    #   - 0.11+ Tasks API moves the recommended path to `mediapipe.tasks.python.vision`
    # We try the classic paths first since the rest of this function uses the
    # FaceMesh solution API. If both fail, we surface a clear actionable error.
    try:
        from mediapipe.solutions import face_mesh as mp_face_mesh
    except ImportError:
        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh  # type: ignore[no-redef]
        except ImportError as e:
            raise ImportError(
                "Could not import MediaPipe FaceMesh. The installed mediapipe "
                "version may have dropped the legacy solutions API. "
                "Pin a compatible version with: pip install 'mediapipe==0.10.18'."
            ) from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # cv2.CAP_PROP_FRAME_COUNT is unreliable for many AVI codecs (often 0 or -1).
    # Accumulate frames dynamically instead of preallocating.
    reported_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    hard_cap = max_frames if max_frames is not None else 10**9

    forehead_list: list[np.ndarray] = []
    left_cheek_list: list[np.ndarray] = []
    right_cheek_list: list[np.ndarray] = []
    valid_list: list[bool] = []

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as mesh:
        while len(valid_list) < hard_cap:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                # Preserve frame index alignment with zero rows.
                forehead_list.append(np.zeros(3, dtype=np.float64))
                left_cheek_list.append(np.zeros(3, dtype=np.float64))
                right_cheek_list.append(np.zeros(3, dtype=np.float64))
                valid_list.append(False)
                continue
            lms = result.multi_face_landmarks[0].landmark
            pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float64)

            forehead_list.append(_polygon_mean_rgb(frame, pts[FOREHEAD_LANDMARKS]))
            left_cheek_list.append(_polygon_mean_rgb(frame, pts[LEFT_CHEEK_LANDMARKS]))
            right_cheek_list.append(_polygon_mean_rgb(frame, pts[RIGHT_CHEEK_LANDMARKS]))
            valid_list.append(True)

    cap.release()

    if not valid_list:
        raise RuntimeError(
            f"Read zero frames from {video_path}. "
            f"OpenCV reported fps={fps}, frame_count={reported_n}. "
            "The container or codec may be unsupported by your OpenCV build."
        )

    forehead = np.stack(forehead_list, axis=0)
    left_cheek = np.stack(left_cheek_list, axis=0)
    right_cheek = np.stack(right_cheek_list, axis=0)
    valid = np.asarray(valid_list, dtype=bool)

    return ROIMeanRGB(
        forehead=forehead,
        left_cheek=left_cheek,
        right_cheek=right_cheek,
        fps=float(fps),
        valid_frames=valid,
    )


def merge_rois(rois: ROIMeanRGB, weights: Iterable[float] = (0.5, 0.25, 0.25)) -> np.ndarray:
    """Combine the three ROI mean-RGB streams into a single (T, 3) trace.

    Default weighting (0.5 forehead, 0.25 each cheek) follows common practice
    in the rPPG literature, which favors the forehead's stable perfusion.
    """
    w_fh, w_lc, w_rc = weights
    return w_fh * rois.forehead + w_lc * rois.left_cheek + w_rc * rois.right_cheek
