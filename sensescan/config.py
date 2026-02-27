from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from loguru import logger


_BASE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = _BASE_DIR

MODELS_DIR = BASE_DIR / "models"
_DEFAULT_MODELS_DIR = MODELS_DIR / "HW"

DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"


EAST_MODEL_PATH: str = os.environ.get(
    "SENSESCAN_HW_EAST_MODEL_PATH",
    str(_DEFAULT_MODELS_DIR / "SENSESCAN_HW_EAST_MODEL.pth"),
)

WORD_MODEL_PATH: str = os.environ.get(
    "SENSESCAN_HW_WORD_MODEL_PATH",
    str(_DEFAULT_MODELS_DIR / "SENSESCAN_HW_WORD_MODEL.pth"),
)


_LOGGING_CONFIGURED = False


def setup_logging() -> None:
    """Configure a rotating file logger for SenseScan (idempotent)."""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "sensescan.log"

    logger.add(
        str(log_path),
        rotation="10 MB",
        retention="14 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    _LOGGING_CONFIGURED = True


@dataclass
class RuntimeConfig:
    """Runtime configuration for SenseScan models."""

    device: torch.device
    num_gpus: int


def get_runtime_config() -> RuntimeConfig:
    """Return device and GPU configuration, logging a concise summary."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        num_gpus = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        num_gpus = 0

    logger.info(
        f"SenseScan using device: {device}"
        + (f" with DataParallel over {num_gpus} GPUs" if num_gpus > 1 else "")
    )
    logger.info(f"SenseScan EAST model path: {EAST_MODEL_PATH}")
    logger.info(f"SenseScan handwritten word model path: {WORD_MODEL_PATH}")

    return RuntimeConfig(device=device, num_gpus=num_gpus)

