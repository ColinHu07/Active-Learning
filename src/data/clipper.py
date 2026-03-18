"""
Video clip sampler using OpenCV.
Extracts a centre-crop 8-second clip and returns uniformly-sampled frames.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path

from src.config import CLIP_LENGTH_SEC, CLIP_FPS, FRAMES_PER_CLIP, FRAME_SIZE
from src.utils.logging import get_logger

log = get_logger(__name__)


def extract_frames(
    video_path: str | Path,
    clip_length_sec: float = CLIP_LENGTH_SEC,
    clip_fps: int = CLIP_FPS,
    frame_size: int = FRAME_SIZE,
) -> torch.Tensor | None:
    """
    Extract frames from the centre clip of a video.

    Parameters
    ----------
    video_path : path to an .mp4 / .avi file
    clip_length_sec : duration of the clip in seconds
    clip_fps : target frames per second inside the clip
    frame_size : resize shortest edge then centre-crop to this size

    Returns
    -------
    Tensor of shape [T, 3, H, W]  (T = clip_length_sec * clip_fps)
    or None if the video cannot be opened / is too short.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return None

    fps_native = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps_native <= 0 or total_frames <= 0:
        log.warning(f"Invalid video metadata: {video_path}")
        cap.release()
        return None

    duration = total_frames / fps_native
    num_target_frames = int(clip_length_sec * clip_fps)  # 16

    # If video is shorter than requested clip, use entire video
    if duration < clip_length_sec:
        start_sec = 0.0
        end_sec = duration
    else:
        # Centre clip
        mid = duration / 2.0
        start_sec = mid - clip_length_sec / 2.0
        end_sec = mid + clip_length_sec / 2.0

    # Compute frame indices to grab (uniformly spaced in the clip window)
    start_frame = int(start_sec * fps_native)
    end_frame = int(end_sec * fps_native)
    end_frame = min(end_frame, total_frames - 1)

    if end_frame <= start_frame:
        log.warning(f"Degenerate range for {video_path}")
        cap.release()
        return None

    frame_indices = np.linspace(start_frame, end_frame, num=num_target_frames, dtype=int)

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat last good frame if read fails
            if frames:
                frames.append(frames[-1].copy())
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = _resize_and_center_crop(frame, frame_size)
        frames.append(frame)

    cap.release()

    if len(frames) < num_target_frames:
        # Pad by repeating last frame
        while len(frames) < num_target_frames:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((frame_size, frame_size, 3), dtype=np.uint8))

    # Stack → [T, H, W, 3] then to [T, 3, H, W] float32 [0,1]
    arr = np.stack(frames[:num_target_frames], axis=0)  # (T, H, W, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return tensor


def _resize_and_center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Resize shortest edge to *size* and then centre-crop to size×size."""
    h, w = img.shape[:2]
    if h < w:
        new_h = size
        new_w = int(w * size / h)
    else:
        new_w = size
        new_h = int(h * size / w)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Centre crop
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]
