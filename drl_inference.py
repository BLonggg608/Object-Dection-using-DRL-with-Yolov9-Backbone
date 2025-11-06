"""High-level inference helpers built on top of the DRL models."""

from typing import List, Sequence, Tuple

import numpy as np
from ultralytics import YOLO

from config import DEVICE, HISTORY_SIZE, MAX_STEPS
from drl_utils import compute_mask, compute_q, crop_image, extract_feature

MaskClassProb = Tuple[np.ndarray, Sequence[str], Sequence[float]]


def predict_mask(feature_model, deep_q, image: np.ndarray, verbose: bool = True) -> np.ndarray:
    if verbose:
        print("Processing Image...")

    history = [-1] * HISTORY_SIZE
    height, width = image.shape[:2]
    current_mask = np.array([0, 0, width, height], dtype=np.float32)

    feature = extract_feature(image, history, feature_model)
    masks: List[np.ndarray] = []
    step = 0

    while True:
        q_value = compute_q(feature, deep_q)
        action = int(np.argmax(q_value))

        history = history[1:]
        history.append(action)

        if action == 8 or step == MAX_STEPS:
            new_mask = current_mask
            masks.append(new_mask)
            break

        new_mask = compute_mask(action, current_mask)
        cropped_image = crop_image(image, new_mask)
        feature = extract_feature(cropped_image, history, feature_model)
        masks.append(new_mask)
        current_mask = new_mask
        step += 1

    mask = masks[-1] if masks else current_mask
    mask[0] = max(mask[0], 0)
    mask[1] = max(mask[1], 0)
    mask[2] = min(mask[2], width)
    mask[3] = min(mask[3], height)
    return mask


def predict_class(classification_model: YOLO, image: np.ndarray, mask: np.ndarray) -> Tuple[List[str], List[float]]:
    cropped_image = crop_image(image, mask)
    results = classification_model.predict(
        source=cropped_image,
        conf=0.25,
        device=str(DEVICE),
        verbose=False,
    )
    filter_probs = [prob.item()
                    for prob in results[0].probs.top5conf if prob.item() > 0.75]
    filter_classes = results[0].probs.top5[: len(filter_probs)]
    filter_classes_names = [results[0].names[idx] for idx in filter_classes]
    return filter_classes_names, filter_probs


def mask_image(image: np.ndarray, mask: np.ndarray, mean_color, noise_level: int = 10) -> np.ndarray:
    masked_image = image.copy()
    x1, y1, x2, y2 = map(int, mask)
    x1 = max(0, min(masked_image.shape[1], x1))
    x2 = max(0, min(masked_image.shape[1], x2))
    y1 = max(0, min(masked_image.shape[0], y1))
    y2 = max(0, min(masked_image.shape[0], y2))

    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return masked_image

    noise = np.random.randint(-noise_level, noise_level + 1,
                              size=(h, w, 3), dtype=np.int16)
    mean_arr = np.array(mean_color, dtype=np.int16).reshape(1, 1, 3)
    mask_value = np.clip(mean_arr + noise, 0, 255).astype(np.uint8)
    masked_image[y1:y2, x1:x2] = mask_value
    return masked_image


def predict_mask_and_class(
    feature_model,
    deep_q,
    classification_model: YOLO,
    image: np.ndarray,
    attempt: int = 3,
) -> Tuple[List[MaskClassProb], np.ndarray]:
    mask_class_prob: List[MaskClassProb] = []
    image_to_process = image.copy()

    for _ in range(attempt):
        mask = predict_mask(feature_model, deep_q,
                            image_to_process, verbose=False)
        class_names, class_probs = predict_class(
            classification_model, image_to_process, mask)

        mean_color = [int(np.mean(image[:, :, channel]))
                      for channel in range(3)]
        image_to_process = mask_image(image_to_process, mask, mean_color, 15)

        if any(item in class_names for item in ["blade", "gun", "knife", "shuriken"]):
            mask_class_prob.append((mask, class_names, class_probs))

    return mask_class_prob, image_to_process
