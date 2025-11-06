"""Helper to initialise all DRL components in one call."""

from typing import Tuple

import torch
from ultralytics import YOLO

from config import DEVICE
from drl_models import create_feature_extractor, load_deep_q_network, resolve_resource


def load_drl_components(
    feature_config_path: str = "yolov9/models/detect/gelan-c.yaml",
    feature_weights_path: str = "weights/best.pt",
    dqn_checkpoint_path: str = "weights_DQN_4layer_yolov9/best_model.pth",
    classification_weights_path: str = "classify_yolov8_weights/best_5_class.pt",
) -> Tuple[torch.nn.Module, torch.nn.Module, YOLO]:
    """Load feature extractor, DQN and classifier with default project weights."""

    feature_model = create_feature_extractor(
        config_path=str(resolve_resource(feature_config_path)),
        weights_path=str(resolve_resource(feature_weights_path)),
        finetune=False,
    ).to(str(DEVICE))
    feature_model.eval()

    deep_q = load_deep_q_network(
        checkpoint_path=str(resolve_resource(dqn_checkpoint_path)),
    ).to(str(DEVICE))
    deep_q.eval()

    classification_model = YOLO(
        str(resolve_resource(classification_weights_path)))
    try:
        classification_model.to(str(DEVICE))
    except AttributeError:
        if hasattr(classification_model, "model"):
            classification_model.model.to(DEVICE)
    try:
        classification_model.fuse()
    except AttributeError:
        pass
    try:
        classification_model.eval()
    except AttributeError:
        # YOLO objects in Ultralytics expose .model for low-level control.
        if hasattr(classification_model, "model"):
            classification_model.model.eval()

    return feature_model, deep_q, classification_model
