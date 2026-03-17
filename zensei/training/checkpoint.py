"""Checkpoint management utilities for Zensei training.

Provides DeepSpeed-aware save / load helpers, automatic cleanup of old
checkpoints, best-checkpoint tagging, and export to HuggingFace safetensors
format.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    step: int,
    path: str,
    *,
    extra_state: Optional[dict] = None,
    use_deepspeed: bool = True,
) -> str:
    """Save a training checkpoint.

    When ``use_deepspeed`` is *True* the model is assumed to be a DeepSpeed
    engine and ``model.save_checkpoint`` is used, which handles ZeRO-3
    parameter gathering automatically.  Otherwise a plain PyTorch checkpoint
    is written.

    Args:
        model: Model or DeepSpeed engine.
        optimizer: Optimizer (ignored when DeepSpeed handles it).
        scheduler: LR scheduler.
        step: Current global step number.
        path: Base directory for checkpoints.
        extra_state: Optional dict of extra metadata to persist.
        use_deepspeed: Whether to use DeepSpeed checkpoint utilities.

    Returns:
        Path to the saved checkpoint directory / file.
    """
    ckpt_dir = os.path.join(path, f"step_{step:08d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    client_state: dict[str, Any] = {
        "step": step,
    }
    if scheduler is not None:
        client_state["scheduler_state"] = (
            scheduler.state_dict()
            if hasattr(scheduler, "state_dict")
            else None
        )
    if extra_state is not None:
        client_state["extra_state"] = extra_state

    if use_deepspeed and hasattr(model, "save_checkpoint"):
        model.save_checkpoint(ckpt_dir, tag="deepspeed", client_state=client_state)
        logger.info("DeepSpeed checkpoint saved at step %d -> %s", step, ckpt_dir)
    else:
        state = {
            "model_state_dict": (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            ),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            **client_state,
        }
        torch.save(state, os.path.join(ckpt_dir, "checkpoint.pt"))
        logger.info("PyTorch checkpoint saved at step %d -> %s", step, ckpt_dir)

    return ckpt_dir


def load_checkpoint(
    model,
    optimizer,
    scheduler,
    path: str,
    *,
    use_deepspeed: bool = True,
    tag: Optional[str] = None,
) -> int:
    """Load a training checkpoint and return the global step.

    Args:
        model: Model or DeepSpeed engine.
        optimizer: Optimizer (ignored when DeepSpeed handles it).
        scheduler: LR scheduler.
        path: Path to the checkpoint directory.
        use_deepspeed: Whether to use DeepSpeed checkpoint utilities.
        tag: DeepSpeed checkpoint tag (default ``"deepspeed"``).

    Returns:
        The global step at which the checkpoint was saved.
    """
    if tag is None:
        tag = "deepspeed"

    if use_deepspeed and hasattr(model, "load_checkpoint"):
        _, client_state = model.load_checkpoint(path, tag=tag)
        client_state = client_state or {}
        step = client_state.get("step", 0)
        if scheduler is not None and "scheduler_state" in client_state:
            sched_state = client_state["scheduler_state"]
            if sched_state is not None:
                scheduler.load_state_dict(sched_state)
        logger.info("DeepSpeed checkpoint loaded from %s (step %d)", path, step)
    else:
        ckpt_file = os.path.join(path, "checkpoint.pt")
        state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state["model_state_dict"])
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        step = state.get("step", 0)
        logger.info("PyTorch checkpoint loaded from %s (step %d)", ckpt_file, step)

    return step


# ---------------------------------------------------------------------------
# Checkpoint cleanup (keep last N)
# ---------------------------------------------------------------------------


def cleanup_checkpoints(path: str, keep_last_n: int = 5) -> list[str]:
    """Remove old checkpoints, keeping only the most recent *N*.

    Checkpoint directories are expected to follow the naming pattern
    ``step_XXXXXXXX``.

    Args:
        path: Base checkpoint directory.
        keep_last_n: Number of most-recent checkpoints to keep.

    Returns:
        List of removed directory paths.
    """
    base = Path(path)
    if not base.exists():
        return []

    ckpt_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: d.name,
    )

    to_remove = ckpt_dirs[:-keep_last_n] if len(ckpt_dirs) > keep_last_n else []
    removed = []
    for d in to_remove:
        # Never remove a directory tagged as "best"
        if (d / "BEST").exists():
            continue
        shutil.rmtree(d)
        removed.append(str(d))
        logger.info("Removed old checkpoint: %s", d)

    return removed


# ---------------------------------------------------------------------------
# Best checkpoint tagging
# ---------------------------------------------------------------------------


def tag_best_checkpoint(
    path: str,
    step: int,
    metric_name: str,
    metric_value: float,
) -> str:
    """Tag a checkpoint directory as the current best.

    A sentinel file ``BEST`` is written inside the checkpoint directory
    together with the metric that qualified it.  Previous ``BEST`` tags
    are removed automatically.

    Args:
        path: Base checkpoint directory.
        step: Step number of the best checkpoint.
        metric_name: Name of the evaluation metric (e.g., ``"eval_loss"``).
        metric_value: Value of the evaluation metric.

    Returns:
        Path to the tagged checkpoint directory.
    """
    base = Path(path)

    # Remove old BEST tags
    for d in base.iterdir():
        best_file = d / "BEST"
        if best_file.exists():
            best_file.unlink()

    ckpt_dir = base / f"step_{step:08d}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    best_file = ckpt_dir / "BEST"
    best_file.write_text(
        json.dumps({"step": step, "metric": metric_name, "value": metric_value})
    )
    logger.info(
        "Tagged best checkpoint: step=%d, %s=%.6f -> %s",
        step,
        metric_name,
        metric_value,
        ckpt_dir,
    )
    return str(ckpt_dir)


# ---------------------------------------------------------------------------
# Export to HuggingFace safetensors
# ---------------------------------------------------------------------------


def export_to_huggingface(
    checkpoint_path: str,
    output_path: str,
    model_config: Optional[dict] = None,
) -> None:
    """Convert a Zensei checkpoint to HuggingFace safetensors format.

    The function loads the model state dict, converts it to safetensors, and
    writes the necessary HuggingFace metadata files (``config.json``,
    ``model.safetensors``).

    Args:
        checkpoint_path: Path to a Zensei checkpoint directory.
        output_path: Output directory for HuggingFace-format files.
        model_config: Optional model configuration dict. If provided it will
            be written as ``config.json``.
    """
    try:
        from safetensors.torch import save_file as save_safetensors
    except ImportError:
        raise ImportError(
            "safetensors is required for HuggingFace export. "
            "Install it with: pip install safetensors"
        )

    os.makedirs(output_path, exist_ok=True)

    # --- Load state dict ---
    ds_ckpt = os.path.join(checkpoint_path, "deepspeed")
    pt_ckpt = os.path.join(checkpoint_path, "checkpoint.pt")

    if os.path.isdir(ds_ckpt):
        # DeepSpeed checkpoint -- attempt to load the consolidated state
        # This requires zero_to_fp32.py or similar consolidation
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path, tag="deepspeed")
    elif os.path.isfile(pt_ckpt):
        raw = torch.load(pt_ckpt, map_location="cpu", weights_only=False)
        state_dict = raw.get("model_state_dict", raw)
    else:
        raise FileNotFoundError(
            f"No recognizable checkpoint found in {checkpoint_path}. "
            "Expected 'deepspeed/' directory or 'checkpoint.pt' file."
        )

    # --- Save as safetensors ---
    safetensors_path = os.path.join(output_path, "model.safetensors")
    save_safetensors(state_dict, safetensors_path)
    logger.info("Saved safetensors to %s", safetensors_path)

    # --- Write config.json ---
    if model_config is not None:
        config_path = os.path.join(output_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info("Saved config.json to %s", config_path)

    logger.info("HuggingFace export complete -> %s", output_path)
