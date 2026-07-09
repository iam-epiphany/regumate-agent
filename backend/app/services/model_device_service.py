from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from backend.app.core.config import MODEL_DEVICE


@dataclass(frozen=True)
class ModelDeviceInfo:
    requested_device: str
    selected_device: str
    torch_version: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    fallback_reason: str | None = None

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "requested_device": self.requested_device,
            "selected_device": self.selected_device,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "fallback_reason": self.fallback_reason,
        }


@lru_cache(maxsize=1)
def get_model_device_info() -> ModelDeviceInfo:
    requested = MODEL_DEVICE if MODEL_DEVICE in {"auto", "cpu", "cuda"} else "auto"
    try:
        import torch
    except Exception as exc:
        return ModelDeviceInfo(
            requested_device=requested,
            selected_device="cpu",
            torch_version=None,
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_name=None,
            fallback_reason=f"torch unavailable: {exc}",
        )

    cuda_available = bool(torch.cuda.is_available())
    cuda_device_count = int(torch.cuda.device_count() or 0)
    cuda_device_name = torch.cuda.get_device_name(0) if cuda_available else None

    if requested == "cuda" and not cuda_available:
        selected = "cpu"
        fallback_reason = "MODEL_DEVICE=cuda requested but CUDA is unavailable"
    elif requested == "cpu":
        selected = "cpu"
        fallback_reason = None
    elif cuda_available:
        selected = "cuda"
        fallback_reason = None
    else:
        selected = "cpu"
        fallback_reason = "auto selected CPU because CUDA is unavailable"

    return ModelDeviceInfo(
        requested_device=requested,
        selected_device=selected,
        torch_version=str(getattr(torch, "__version__", "")) or None,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_device_name,
        fallback_reason=fallback_reason,
    )


def selected_model_device() -> str:
    return get_model_device_info().selected_device
