"""Configuration constants shared across the DRL pipeline modules."""

import torch

HISTORY_SIZE: int = 1
ACTION_OPTION: int = 9
MAX_STEPS: int = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
