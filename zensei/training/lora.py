"""LoRA (Low-Rank Adaptation) utilities for Zensei.

Provides helpers to apply, merge, save, and load LoRA adapters on top of
a Zensei model using the ``peft`` library.

Usage (standalone):
    # Apply LoRA to a model
    python -m zensei.training.lora apply \
        --model_config configs/model/zensei_671B.json \
        --output_path checkpoints/lora_adapter

    # Merge LoRA back into the base model
    python -m zensei.training.lora merge \
        --base_model_path checkpoints/base \
        --adapter_path checkpoints/lora_adapter \
        --output_path checkpoints/merged
"""

import json
import logging
import os
from typing import Optional

import fire
import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Default LoRA configuration for Zensei
DEFAULT_LORA_CONFIG = {
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}


# ---------------------------------------------------------------------------
# Apply LoRA
# ---------------------------------------------------------------------------


def apply_lora(
    model: nn.Module,
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    bias: str = "none",
    task_type: str = "CAUSAL_LM",
) -> nn.Module:
    """Wrap a Zensei model with LoRA adapters.

    Args:
        model: The base Zensei model.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: List of module name patterns to apply LoRA to.
            Defaults to attention and MLP projection layers.
        bias: Bias training mode (``"none"``, ``"all"``, or ``"lora_only"``).
        task_type: Task type for peft (``"CAUSAL_LM"``).

    Returns:
        The model wrapped with LoRA adapters (a ``PeftModel``).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    if target_modules is None:
        target_modules = DEFAULT_LORA_CONFIG["target_modules"]

    task_type_enum = getattr(TaskType, task_type, TaskType.CAUSAL_LM)

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=bias,
        task_type=task_type_enum,
    )

    peft_model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    logger.info(
        "LoRA applied: trainable=%d (%.2f%% of %d total)",
        trainable_params,
        100.0 * trainable_params / total_params,
        total_params,
    )

    return peft_model


def apply_lora_from_config(model: nn.Module, config: dict) -> nn.Module:
    """Apply LoRA using a configuration dictionary.

    Args:
        model: The base Zensei model.
        config: Dictionary with LoRA hyper-parameters. Keys correspond to
            :func:`apply_lora` arguments.

    Returns:
        The model wrapped with LoRA adapters.
    """
    lora_kwargs = {**DEFAULT_LORA_CONFIG, **config}
    return apply_lora(
        model,
        r=lora_kwargs["r"],
        lora_alpha=lora_kwargs["lora_alpha"],
        lora_dropout=lora_kwargs.get("lora_dropout", 0.05),
        target_modules=lora_kwargs["target_modules"],
        bias=lora_kwargs.get("bias", "none"),
        task_type=lora_kwargs.get("task_type", "CAUSAL_LM"),
    )


# ---------------------------------------------------------------------------
# Merge LoRA weights back into base model
# ---------------------------------------------------------------------------


def merge_lora(peft_model) -> nn.Module:
    """Merge LoRA adapter weights into the base model.

    After merging, the LoRA layers are removed and the model is equivalent
    to a standard ``nn.Module`` with the adapted weights baked in.

    Args:
        peft_model: A ``PeftModel`` with LoRA adapters.

    Returns:
        The base model with LoRA weights merged in.
    """
    merged_model = peft_model.merge_and_unload()
    logger.info("LoRA weights merged into base model.")
    return merged_model


# ---------------------------------------------------------------------------
# Save / Load LoRA adapters
# ---------------------------------------------------------------------------


def save_lora(peft_model, path: str) -> None:
    """Save LoRA adapter weights and configuration to disk.

    Args:
        peft_model: A ``PeftModel`` with LoRA adapters.
        path: Directory to save the adapter files.
    """
    os.makedirs(path, exist_ok=True)
    peft_model.save_pretrained(path)
    logger.info("LoRA adapter saved to %s", path)


def load_lora(model: nn.Module, adapter_path: str) -> nn.Module:
    """Load LoRA adapter weights onto a base model.

    Args:
        model: The base Zensei model (without LoRA).
        adapter_path: Directory containing saved LoRA adapter files.

    Returns:
        The model with LoRA adapters loaded.
    """
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    logger.info("LoRA adapter loaded from %s", adapter_path)
    return peft_model


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


def apply(
    model_config: str = "configs/model/zensei_671B.json",
    checkpoint_path: Optional[str] = None,
    output_path: str = "checkpoints/lora_adapter",
    r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
) -> None:
    """Apply LoRA to a Zensei model and save the adapter.

    Args:
        model_config: Path to the model config JSON.
        checkpoint_path: Optional base model checkpoint to load first.
        output_path: Where to save the LoRA adapter.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout probability.
    """
    with open(model_config, "r") as f:
        mcfg = json.load(f)

    try:
        from zensei.model import ZenseiModel

        model = ZenseiModel(mcfg)
    except (ImportError, AttributeError):
        raise RuntimeError("Could not import ZenseiModel from zensei.model.")

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Loaded base model from %s", checkpoint_path)

    peft_model = apply_lora(model, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    save_lora(peft_model, output_path)


def merge(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    model_config: str = "configs/model/zensei_671B.json",
) -> None:
    """Merge LoRA adapter into base model and save the result.

    Args:
        base_model_path: Path to base model checkpoint.
        adapter_path: Path to LoRA adapter directory.
        output_path: Where to save the merged model.
        model_config: Path to the model config JSON.
    """
    with open(model_config, "r") as f:
        mcfg = json.load(f)

    try:
        from zensei.model import ZenseiModel

        model = ZenseiModel(mcfg)
    except (ImportError, AttributeError):
        raise RuntimeError("Could not import ZenseiModel from zensei.model.")

    state_dict = torch.load(base_model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)

    peft_model = load_lora(model, adapter_path)
    merged = merge_lora(peft_model)

    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(output_path, "model.pt")
    torch.save(merged.state_dict(), save_path)
    logger.info("Merged model saved to %s", save_path)


if __name__ == "__main__":
    fire.Fire({"apply": apply, "merge": merge})
