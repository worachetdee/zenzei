"""Zensei training loop with DeepSpeed ZeRO-3 support.

Implements a two-stage continued-pretraining recipe:
  - Stage 1: freeze backbone, train only embeddings + LM head
  - Stage 2: unfreeze everything for full continued pretraining

Usage:
    Typically invoked via ``zensei.training.pretrain`` rather than directly.
"""

import logging
import math
import os
import time
from typing import Any, Optional

import torch
import torch.nn as nn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepSpeed config generation
# ---------------------------------------------------------------------------


def generate_deepspeed_config(
    zero_stage: int = 3,
    bf16: bool = True,
    gradient_accumulation_steps: int = 8,
    train_micro_batch_size_per_gpu: int = 1,
    gradient_clipping: float = 1.0,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> dict:
    """Generate a DeepSpeed configuration dictionary.

    Args:
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3).
        bf16: Enable bf16 mixed precision.
        gradient_accumulation_steps: Gradient accumulation steps.
        train_micro_batch_size_per_gpu: Micro batch size per GPU.
        gradient_clipping: Maximum gradient norm.
        offload_optimizer: Offload optimizer state to CPU (ZeRO-2/3).
        offload_param: Offload parameters to CPU (ZeRO-3 only).

    Returns:
        DeepSpeed config dict ready for ``deepspeed.initialize``.
    """
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": train_micro_batch_size_per_gpu,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": gradient_clipping,
        "steps_per_print": 100,
        "wall_clock_breakdown": False,
    }

    # Mixed precision
    if bf16:
        ds_config["bf16"] = {"enabled": True}
    else:
        ds_config["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 16}

    # ZeRO configuration
    zero_config: dict[str, Any] = {
        "stage": zero_stage,
        "contiguous_gradients": True,
        "overlap_comm": True,
    }

    if zero_stage >= 2:
        zero_config["allgather_partitions"] = True
        zero_config["allgather_bucket_size"] = 5e8
        zero_config["reduce_scatter"] = True
        zero_config["reduce_bucket_size"] = 5e8

    if zero_stage == 3:
        zero_config["stage3_max_live_parameters"] = 1e9
        zero_config["stage3_max_reuse_distance"] = 1e9
        zero_config["stage3_prefetch_bucket_size"] = 5e8
        zero_config["stage3_param_persistence_threshold"] = 1e6
        zero_config["stage3_gather_16bit_weights_on_model_save"] = True

    if offload_optimizer and zero_stage >= 2:
        zero_config["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    if offload_param and zero_stage == 3:
        zero_config["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    ds_config["zero_optimization"] = zero_config

    return ds_config


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class ZenseiTrainer:
    """Training loop for Zensei with DeepSpeed ZeRO-3.

    Args:
        model: The Zensei model (``nn.Module``).
        train_dataset: A ``torch.utils.data.Dataset`` yielding tokenized
            sequences.
        config: Training configuration dictionary with keys such as
            ``learning_rate``, ``max_steps``, ``batch_size``, etc.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        config: dict,
        eval_dataset=None,
    ) -> None:
        import deepspeed

        self.config = config
        self.eval_dataset = eval_dataset

        # ---- Training hyper-parameters ----
        self.stage: int = config.get("stage", 1)
        self.max_steps: int = config.get("max_steps", 100000)
        self.gradient_accumulation_steps: int = config.get("gradient_accumulation_steps", 8)
        self.max_seq_len: int = config.get("max_seq_len", 4096)
        self.gradient_clip: float = config.get("gradient_clip", 1.0)
        self.checkpoint_interval: int = config.get("checkpoint_interval", 5000)
        self.eval_interval: int = config.get("eval_interval", 2000)
        self.checkpoint_dir: str = config.get("checkpoint_dir", "checkpoints")
        self.keep_last_n_checkpoints: int = config.get("keep_last_n_checkpoints", 5)
        self.log_interval: int = config.get("log_interval", 10)
        self.aux_loss_weight: float = config.get("aux_loss_weight", 0.01)

        # ---- Freeze / unfreeze for two-stage training ----
        freeze_backbone: bool = config.get("freeze_backbone", False)
        if freeze_backbone:
            self._freeze_backbone(model)

        # ---- DeepSpeed initialization ----
        micro_batch = config.get("batch_size", 256) // (
            self.gradient_accumulation_steps * max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        )
        micro_batch = max(micro_batch, 1)

        ds_config = generate_deepspeed_config(
            zero_stage=config.get("zero_stage", 3),
            bf16=config.get("bf16", True),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            train_micro_batch_size_per_gpu=micro_batch,
            gradient_clipping=self.gradient_clip,
            offload_optimizer=config.get("offload_optimizer", False),
            offload_param=config.get("offload_param", False),
        )

        # ---- Optimizer ----
        optimizer_params = self._get_optimizer_params(model, config)

        self.model_engine, self.optimizer, self.train_dataloader, self.lr_scheduler = (
            deepspeed.initialize(
                model=model,
                model_parameters=optimizer_params,
                training_data=train_dataset,
                config=ds_config,
            )
        )

        # ---- Learning rate scheduler ----
        from zensei.training.lr_schedule import WarmupCosineScheduler

        self.lr_scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            peak_lr=config.get("learning_rate", 1e-5),
            min_lr=config.get("min_lr", 1e-6),
            warmup_steps=config.get("warmup_steps", 1000),
            max_steps=self.max_steps,
        )

        # ---- Wandb ----
        self.use_wandb: bool = config.get("use_wandb", True)
        self._wandb_initialized = False
        if self.use_wandb and self._is_main_process():
            try:
                import wandb

                wandb.init(
                    project=config.get("wandb_project", "zensei"),
                    name=config.get("wandb_run_name", f"stage{self.stage}"),
                    config=config,
                    resume="allow",
                )
                self._wandb_initialized = True
            except ImportError:
                logger.warning("wandb not installed; disabling logging.")
                self.use_wandb = False

        self.global_step = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_main_process() -> bool:
        return int(os.environ.get("LOCAL_RANK", "0")) == 0

    @staticmethod
    def _freeze_backbone(model: nn.Module) -> None:
        """Freeze all parameters except embed and head layers."""
        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze embedding and LM head
        for name, param in model.named_parameters():
            if "embed" in name or "head" in name:
                param.requires_grad = True
                logger.info("Trainable (unfrozen): %s", name)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Froze backbone: %d / %d params trainable (%.2f%%)",
            trainable,
            total,
            100.0 * trainable / total if total > 0 else 0,
        )

    @staticmethod
    def _get_optimizer_params(model: nn.Module, config: dict) -> list[dict]:
        """Separate parameters into weight-decay and no-weight-decay groups."""
        weight_decay = config.get("weight_decay", 0.1)
        no_decay = {"bias", "layernorm", "rmsnorm", "ln_"}

        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name.lower() for nd in no_decay):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #

    def train(self, resume_step: int = 0) -> None:
        """Run the training loop.

        Args:
            resume_step: Global step to resume from (e.g., after loading a
                checkpoint).
        """
        from zensei.training.checkpoint import (
            cleanup_checkpoints,
            save_checkpoint,
        )

        self.global_step = resume_step
        self.model_engine.train()

        logger.info("Starting training from step %d (max_steps=%d)", self.global_step, self.max_steps)

        data_iter = iter(self.train_dataloader)
        running_loss = 0.0
        running_aux_loss = 0.0
        t0 = time.time()

        while self.global_step < self.max_steps:
            # ---- Get batch ----
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(self.model_engine.device)
            labels = batch.get("labels", input_ids.clone()).to(self.model_engine.device)

            # ---- Forward pass ----
            outputs = self.model_engine(input_ids)

            # Compute language modeling loss
            if isinstance(outputs, dict):
                logits = outputs.get("logits", outputs.get("output"))
                aux_loss = outputs.get("aux_loss", torch.tensor(0.0, device=input_ids.device))
            elif isinstance(outputs, tuple):
                logits = outputs[0]
                aux_loss = outputs[1] if len(outputs) > 1 else torch.tensor(0.0, device=input_ids.device)
            else:
                logits = outputs
                aux_loss = torch.tensor(0.0, device=input_ids.device)

            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss()
            lm_loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # MoE auxiliary / load-balancing loss
            if aux_loss is None:
                aux_loss = torch.tensor(0.0, device=input_ids.device)
            total_loss = lm_loss + self.aux_loss_weight * aux_loss

            # ---- Backward pass ----
            self.model_engine.backward(total_loss)
            self.model_engine.step()

            # ---- LR scheduler step ----
            self.lr_scheduler.step()

            # ---- Logging ----
            running_loss += lm_loss.item()
            running_aux_loss += aux_loss.item()
            self.global_step += 1

            if self.global_step % self.log_interval == 0 and self._is_main_process():
                avg_loss = running_loss / self.log_interval
                avg_aux = running_aux_loss / self.log_interval
                elapsed = time.time() - t0
                tokens_per_sec = (
                    self.log_interval
                    * input_ids.shape[0]
                    * input_ids.shape[1]
                    / elapsed
                )
                current_lr = self.lr_scheduler.get_last_lr()[0]

                logger.info(
                    "step=%d  loss=%.4f  aux_loss=%.4f  lr=%.2e  tok/s=%.0f",
                    self.global_step,
                    avg_loss,
                    avg_aux,
                    current_lr,
                    tokens_per_sec,
                )

                if self._wandb_initialized:
                    import wandb

                    wandb.log(
                        {
                            "train/loss": avg_loss,
                            "train/aux_loss": avg_aux,
                            "train/lr": current_lr,
                            "train/tokens_per_sec": tokens_per_sec,
                            "train/step": self.global_step,
                        },
                        step=self.global_step,
                    )

                running_loss = 0.0
                running_aux_loss = 0.0
                t0 = time.time()

            # ---- Evaluation ----
            if (
                self.eval_dataset is not None
                and self.global_step % self.eval_interval == 0
            ):
                self._evaluate()

            # ---- Checkpoint ----
            if self.global_step % self.checkpoint_interval == 0:
                save_checkpoint(
                    model=self.model_engine,
                    optimizer=self.optimizer,
                    scheduler=self.lr_scheduler,
                    step=self.global_step,
                    path=self.checkpoint_dir,
                )
                cleanup_checkpoints(
                    self.checkpoint_dir,
                    keep_last_n=self.keep_last_n_checkpoints,
                )

        logger.info("Training complete at step %d.", self.global_step)

        if self._wandb_initialized:
            import wandb

            wandb.finish()

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    def _evaluate(self) -> Optional[float]:
        """Run evaluation on the eval dataset and return the eval loss."""
        if self.eval_dataset is None:
            return None

        self.model_engine.eval()
        total_loss = 0.0
        num_batches = 0
        max_eval_batches = self.config.get("max_eval_batches", 50)

        eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.get("eval_batch_size", 4),
            shuffle=False,
            num_workers=2,
        )

        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if i >= max_eval_batches:
                    break

                input_ids = batch["input_ids"].to(self.model_engine.device)
                labels = batch.get("labels", input_ids.clone()).to(self.model_engine.device)

                outputs = self.model_engine(input_ids)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                elif isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss = nn.CrossEntropyLoss()(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 20.0))

        if self._is_main_process():
            logger.info(
                "Eval step=%d  loss=%.4f  ppl=%.2f",
                self.global_step,
                avg_loss,
                ppl,
            )
            if self._wandb_initialized:
                import wandb

                wandb.log(
                    {
                        "eval/loss": avg_loss,
                        "eval/perplexity": ppl,
                        "eval/step": self.global_step,
                    },
                    step=self.global_step,
                )

        self.model_engine.train()
        return avg_loss
