"""
Zensei Triton FP8 GEMM Kernels.

This module contains optimized Triton kernels for FP8 (8-bit floating point)
matrix multiplication, targeting NVIDIA Hopper (H100/H200) tensor cores.

FP8 GEMM is critical for efficient MoE inference and training: each expert's
forward pass is a small GEMM that benefits greatly from reduced precision.

Currently provides a fallback implementation that converts to float16 and
delegates to torch.matmul. The Triton FP8 kernel will be enabled once the
custom kernel is validated on target hardware.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triton FP8 availability check
# ---------------------------------------------------------------------------

_TRITON_FP8_AVAILABLE: bool = False

try:
    import triton  # noqa: F401
    import triton.language as tl  # noqa: F401

    # Check for FP8 support (requires Triton >= 2.1 and compatible GPU)
    if hasattr(tl, "float8e4m3fn"):
        # Additional runtime check would go here (e.g. CUDA capability >= 9.0)
        # For now, we leave the flag False until the kernel is fully validated.
        _TRITON_FP8_AVAILABLE = False
        logger.info("Triton is installed but FP8 kernel is not yet enabled; using fallback.")
    else:
        logger.info("Triton installed but FP8 dtypes not available in this build.")
except ImportError:
    logger.info("Triton is not installed; FP8 GEMM will use the torch.matmul fallback.")


def is_triton_fp8_available() -> bool:
    """Return True if the optimized Triton FP8 GEMM kernel is available."""
    return _TRITON_FP8_AVAILABLE


# ---------------------------------------------------------------------------
# FP8 GEMM — fallback implementation
# ---------------------------------------------------------------------------

def fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: Optional[torch.Tensor] = None,
    b_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute a matrix multiplication, optionally in FP8.

    When the Triton FP8 kernel is available this dispatches to a fused
    FP8 GEMM.  Otherwise it falls back to float16 torch.matmul with a
    one-time warning.

    Args:
        a: Left operand of shape (..., M, K).
        b: Right operand of shape (..., K, N).
        a_scale: Optional per-tensor or per-channel scale for *a*.
                 Used to dequantize FP8 inputs: ``a_fp16 = a_fp8 * a_scale``.
        b_scale: Optional per-tensor or per-channel scale for *b*.

    Returns:
        Result of shape (..., M, N) in the same dtype as the inputs (or float16
        when running in fallback mode).
    """
    if _TRITON_FP8_AVAILABLE:
        # Placeholder: dispatch to optimized Triton FP8 kernel
        # return _triton_fp8_gemm(a, b, a_scale, b_scale)
        raise NotImplementedError("Triton FP8 kernel dispatch not yet implemented.")

    # ---- Fallback path ----
    warnings.warn(
        "Triton FP8 GEMM kernel is not loaded; falling back to float16 torch.matmul. "
        "Performance will be significantly lower than the optimized kernel.",
        stacklevel=2,
    )

    # Dequantize if scales are provided (simulates FP8 → FP16 conversion)
    a_fp16 = a.to(torch.float16)
    b_fp16 = b.to(torch.float16)

    if a_scale is not None:
        a_fp16 = a_fp16 * a_scale.to(torch.float16)
    if b_scale is not None:
        b_fp16 = b_fp16 * b_scale.to(torch.float16)

    return torch.matmul(a_fp16, b_fp16)
