"""Utility functions for feature extraction and geometric helpers."""

import os
from io import BytesIO
from typing import Iterable

import cv2
import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image

from yolov9.utils.augmentations import letterbox

from config import ACTION_OPTION, DEVICE, HISTORY_SIZE


def load_image(img_path: str, target_size: tuple[int, int], url: bool = False) -> np.ndarray:
    """Load an RGB image either from disk or an HTTP(S) URL."""

    if url:
        response = requests.get(img_path, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image = np.array(image)
    else:
        abs_path = os.path.abspath(img_path)
        image = cv2.imread(abs_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image at {abs_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return image.astype(np.uint8)


def _encode_history(history: Iterable[int]) -> np.ndarray:
    history_feature = np.zeros(ACTION_OPTION * HISTORY_SIZE, dtype=np.float32)
    for idx, action in enumerate(history):
        if 0 <= action < ACTION_OPTION:
            history_feature[idx * ACTION_OPTION + action] = 1.0
    return history_feature


def extract_feature(image: np.ndarray, history: list[int], feature_model: nn.Module) -> np.ndarray:
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    img_letterbox, _, _ = letterbox(
        image.copy(),
        new_shape=(640, 640),
        auto=False,
        scaleFill=False,
        scaleup=False,
    )

    img_letterbox = img_letterbox.astype(np.float32) / 255.0
    img_letterbox = np.ascontiguousarray(img_letterbox.transpose(2, 0, 1))
    input_tensor = torch.from_numpy(img_letterbox).unsqueeze(0).to(DEVICE)

    requires_grad = feature_model.training and any(
        param.requires_grad for param in feature_model.parameters())
    with torch.set_grad_enabled(requires_grad):
        feats = feature_model(input_tensor)

    pool = nn.AdaptiveAvgPool2d((1, 1))
    feats = pool(feats)
    feats = feats.view(feats.size(0), -1)

    image_feature = feats.detach().cpu().numpy().flatten().astype(np.float32)
    history_feature = _encode_history(history)
    feature = np.concatenate((image_feature, history_feature))
    return feature[np.newaxis, :]


def compute_q(feature: np.ndarray, deep_q_model: nn.Module) -> np.ndarray:
    feature_tensor = torch.from_numpy(feature).float().to(DEVICE)
    with torch.no_grad():
        output = deep_q_model(feature_tensor)
    return output.cpu().numpy().flatten()


def compute_mask(action: int, current_mask: np.ndarray) -> np.ndarray:
    image_rate = 0.1
    delta_width = image_rate * (current_mask[2] - current_mask[0])
    delta_height = image_rate * (current_mask[3] - current_mask[1])

    dx1 = dy1 = dx2 = dy2 = 0.0

    if action == 0:
        dx1 = delta_width
        dx2 = delta_width
    elif action == 1:
        dx1 = -delta_width
        dx2 = -delta_width
    elif action == 2:
        dy1 = delta_height
        dy2 = delta_height
    elif action == 3:
        dy1 = -delta_height
        dy2 = -delta_height
    elif action == 4:
        dx1 = -delta_width
        dx2 = delta_width
        dy1 = -delta_height
        dy2 = delta_height
    elif action == 5:
        dx1 = delta_width
        dx2 = -delta_width
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 6:
        dy1 = delta_height
        dy2 = -delta_height
    elif action == 7:
        dx1 = delta_width
        dx2 = -delta_width

    new_mask_tmp = np.array(
        [
            current_mask[0] + dx1,
            current_mask[1] + dy1,
            current_mask[2] + dx2,
            current_mask[3] + dy2,
        ]
    )

    new_mask = np.array(
        [
            min(new_mask_tmp[0], new_mask_tmp[2]),
            min(new_mask_tmp[1], new_mask_tmp[3]),
            max(new_mask_tmp[0], new_mask_tmp[2]),
            max(new_mask_tmp[1], new_mask_tmp[3]),
        ]
    )

    return new_mask


def crop_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    new_mask = np.asarray(mask, dtype=int).copy()
    new_mask[0] = max(new_mask[0], 0)
    new_mask[1] = max(new_mask[1], 0)
    new_mask[2] = min(new_mask[2], width)
    new_mask[3] = min(new_mask[3], height)

    x1, y1, x2, y2 = new_mask
    if x2 <= x1 or y2 <= y1:
        return np.zeros((512, 512, 3), dtype=np.uint8)
    return image[y1:y2, x1:x2]
