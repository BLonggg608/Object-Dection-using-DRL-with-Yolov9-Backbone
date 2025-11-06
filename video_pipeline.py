"""Offline video annotation helpers built on top of the DRL inference stack."""

import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from drl_inference import predict_mask_and_class

MaskResult = Tuple[np.ndarray, Sequence[str], Sequence[float]]


def _safe_bbox(mask: np.ndarray, width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    mask_arr = np.asarray(mask, dtype=np.float32)
    x1, y1, x2, y2 = mask_arr.astype(int)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _draw_results(frame_bgr: np.ndarray, detections: Iterable[MaskResult]) -> np.ndarray:
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    for mask, class_names, class_probs in detections:
        bbox = _safe_bbox(mask, width, height)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        label = f"{class_names[0]} {class_probs[0]:.2f}" if class_names else "object"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def annotate_video_with_drl(
    feature_model,
    deep_q,
    classification_model,
    video_path,
    output_path,
    frame_stride: int = 10,
    progress_interval: int = 25,
) -> Path:
    """Run the DRL pipeline on a recorded video and save the annotated output."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(output_path)
    if output_path.suffix == "":
        raise ValueError("output_path must include a file extension")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_idx = 0
    last_mask_class_prob: Optional[List[MaskResult]] = None
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if frame_idx % frame_stride == 0:
                mask_class_prob, _ = predict_mask_and_class(
                    feature_model, deep_q, classification_model, frame_rgb
                )
                last_mask_class_prob = mask_class_prob

            mask_class_prob = last_mask_class_prob or []
            annotated = _draw_results(frame_bgr, mask_class_prob)
            writer.write(annotated)
            frame_idx += 1

            if progress_interval and frame_idx % progress_interval == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        raise RuntimeError("No frames were processed from the video.")

    return output_path


def snapshot_frames_with_drl(
    feature_model,
    deep_q,
    classification_model,
    video_path,
    output_dir,
    frame_stride: int = 10,
    progress_interval: int = 25,
    on_snapshot: Optional[Callable[[Path], None]] = None,
) -> int:
    """Process video and save annotated frames at stride intervals."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear previous snapshots to avoid mixing results across runs.
    for existing in output_dir.iterdir():
        if existing.is_dir():
            shutil.rmtree(existing)
        else:
            existing.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_idx = 0
    snapshot_idx = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_stride and frame_idx % frame_stride == 0:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mask_class_prob, _ = predict_mask_and_class(
                    feature_model, deep_q, classification_model, frame_rgb
                )
                if mask_class_prob:
                    annotated = _draw_results(frame_bgr, mask_class_prob)
                    snapshot_idx += 1
                    file_path = output_dir / f"frame_{snapshot_idx:05d}.png"
                    cv2.imwrite(str(file_path), annotated)
                    if on_snapshot is not None:
                        on_snapshot(file_path)

            frame_idx += 1
            if progress_interval and frame_idx % progress_interval == 0:
                print(f"Processed {frame_idx}/{total_frames} frames")
    finally:
        cap.release()

    return snapshot_idx
