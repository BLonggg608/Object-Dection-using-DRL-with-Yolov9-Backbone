"""Model definitions and loaders for the DRL baggage inspection pipeline."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml

from yolov9.models.yolo import Model

from config import ACTION_OPTION, HISTORY_SIZE


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def resolve_resource(path_like: str | Path) -> Path:
    """Resolve resource paths relative to the package or project root."""

    path = Path(path_like)
    if path.is_absolute():
        return path

    for base in (PACKAGE_ROOT, PROJECT_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate

    # Fall back to the package path even if missing so callers surface a clear error.
    return PACKAGE_ROOT / path


class QNetwork(nn.Module):
    """Fully connected Q-network matching the training configuration."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def create_q_model(feature_size: int = 512) -> QNetwork:
    """Instantiate a Q-network using the project-wide defaults."""

    input_size = feature_size + ACTION_OPTION * HISTORY_SIZE
    output_size = ACTION_OPTION
    return QNetwork(input_size, output_size)


def create_feature_extractor(
    config_path: str = "yolov9/models/detect/gelan-c.yaml",
    weights_path: Optional[str] = "weights/best.pt",
    finetune: bool = False,
) -> nn.Module:
    """Create a YOLOv9 backbone feature extractor pre-trained on GDXray."""

    config_file = resolve_resource(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Cannot find YOLOv9 config at {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = Model(config, ch=3, nc=4)

    if weights_path:
        ckpt_file = resolve_resource(weights_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(
                f"Cannot find YOLOv9 weights at {ckpt_file}")
        checkpoint = torch.load(
            ckpt_file, map_location="cpu", weights_only=False)
        model.load_state_dict(
            checkpoint["model"].float().state_dict(), strict=False)

    feature_extractor = nn.Sequential(*model.model[:9])

    for param in feature_extractor.parameters():
        param.requires_grad = finetune

    feature_extractor.train(mode=finetune)
    return feature_extractor


def load_deep_q_network(
    checkpoint_path: str,
    feature_size: int = 512,
    key: str = "policy_net_state_dict",
) -> QNetwork:
    """Create a Q-network instance and load the trained weights."""

    checkpoint_file = resolve_resource(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Cannot find DQN checkpoint at {checkpoint_file}")

    model = create_q_model(feature_size=feature_size)
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict[key])
    model.eval()
    return model
