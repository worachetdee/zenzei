"""Entry point for Zensei continued pretraining.

Launch with DeepSpeed:
    deepspeed --num_gpus 8 zensei/training/pretrain.py \
        --config configs/training/pretrain_stage1.yaml

Resume from checkpoint:
    deepspeed --num_gpus 8 zensei/training/pretrain.py \
        --config configs/training/pretrain_stage1.yaml \
        --resume_from checkpoints/step_00005000
"""

import json
import logging
import os
from typing import Optional

import fire
import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _load_json(path: str) -> dict:
    """Load a JSON configuration file."""
    with open(path, "r") as f:
        return json.load(f)


def _build_model(model_config: dict) -> torch.nn.Module:
    """Build a Zensei model from its configuration dictionary.

    Attempts to import the model class from ``zensei.model``. Falls back to a
    simple stub if the model module is not yet fully implemented.
    """
    try:
        from zensei.model import ZenseiModel

        model = ZenseiModel(model_config)
    except (ImportError, AttributeError):
        logger.warning(
            "Could not import ZenseiModel from zensei.model. "
            "Ensure the model module is implemented."
        )
        raise
    return model


def _load_deepseek_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    """Load a DeepSeek-V3 checkpoint into the Zensei model."""
    try:
        from zensei.model.convert import load_deepseek_weights

        load_deepseek_weights(model, checkpoint_path)
        logger.info("Loaded DeepSeek checkpoint from %s", checkpoint_path)
    except (ImportError, AttributeError):
        logger.warning(
            "zensei.model.convert.load_deepseek_weights not available. "
            "Loading raw state dict instead."
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(state_dict, strict=False)
        logger.info("Loaded raw state dict from %s", checkpoint_path)


def _load_tokenizer(tokenizer_path: str):
    """Load the Zensei tokenizer (SentencePiece-based)."""
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    logger.info("Loaded tokenizer from %s (vocab_size=%d)", tokenizer_path, sp.get_piece_size())
    return sp


def _load_dataset(data_path: str, max_seq_len: int):
    """Load a pre-tokenized dataset from disk.

    Expects the output format of ``zensei.data.prepare`` -- a directory of
    ``.pt`` or ``.bin`` shard files, or a memory-mapped numpy array.
    """
    try:
        from zensei.data.prepare import ZenseiDataset

        dataset = ZenseiDataset(data_path, max_seq_len=max_seq_len)
        logger.info("Loaded dataset from %s (%d samples)", data_path, len(dataset))
        return dataset
    except (ImportError, AttributeError):
        logger.warning(
            "zensei.data.prepare.ZenseiDataset not available. "
            "Falling back to simple shard-based dataset."
        )
        return _SimpleShardDataset(data_path, max_seq_len)


class _SimpleShardDataset(torch.utils.data.Dataset):
    """Minimal dataset that reads pre-tokenized ``.pt`` shard files."""

    def __init__(self, data_dir: str, max_seq_len: int = 4096) -> None:
        self.max_seq_len = max_seq_len
        self.data_dir = data_dir

        import glob

        self.shards = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if not self.shards:
            self.shards = sorted(glob.glob(os.path.join(data_dir, "*.bin")))
        if not self.shards:
            raise FileNotFoundError(f"No .pt or .bin shard files found in {data_dir}")

        # Load all token IDs into a flat tensor
        all_tokens = []
        for shard in self.shards:
            tokens = torch.load(shard, map_location="cpu", weights_only=True)
            if isinstance(tokens, dict):
                tokens = tokens.get("input_ids", tokens.get("tokens"))
            if isinstance(tokens, torch.Tensor):
                all_tokens.append(tokens.view(-1))
        self.tokens = torch.cat(all_tokens)
        self.n_samples = len(self.tokens) // max_seq_len
        logger.info(
            "SimpleShardDataset: %d tokens, %d samples (seq_len=%d)",
            len(self.tokens),
            self.n_samples,
            max_seq_len,
        )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        input_ids = self.tokens[start:end].long()
        return {"input_ids": input_ids, "labels": input_ids.clone()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(
    config: str = "configs/training/pretrain_stage1.yaml",
    resume_from: Optional[str] = None,
    deepseek_checkpoint: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    data_path: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Zensei continued pretraining entry point.

    Args:
        config: Path to a YAML training configuration file.
        resume_from: Path to a checkpoint directory to resume training from.
        deepseek_checkpoint: Path to a DeepSeek-V3 checkpoint to initialize
            the model weights (used with vocabulary expansion).
        tokenizer_path: Path to the SentencePiece ``.model`` file. Overrides
            the value in the YAML config.
        data_path: Path to the pre-tokenized dataset directory. Overrides the
            value in the YAML config.
        local_rank: Local rank for distributed training (set by DeepSpeed
            launcher automatically).
    """
    # ---- Load configuration ----
    cfg = _load_yaml(config)
    logger.info("Loaded training config from %s", config)

    # CLI overrides
    if tokenizer_path is not None:
        cfg["tokenizer_path"] = tokenizer_path
    if data_path is not None:
        cfg["data_path"] = data_path

    # ---- Load model config ----
    model_config_path = cfg.get("model_config", "configs/model/zensei_671B.json")
    model_config = _load_json(model_config_path)
    logger.info("Loaded model config from %s", model_config_path)

    # ---- Build model ----
    model = _build_model(model_config)

    # ---- Optionally load DeepSeek checkpoint and expand vocab ----
    if deepseek_checkpoint is not None:
        _load_deepseek_checkpoint(model, deepseek_checkpoint)

        # Vocabulary expansion
        old_vocab = model_config.get("original_vocab_size")
        new_vocab = model_config.get("vocab_size")
        if old_vocab is not None and new_vocab is not None and new_vocab > old_vocab:
            from zensei.training.embed_resize import resize_embeddings

            resize_embeddings(model, old_vocab, new_vocab)
            logger.info("Expanded vocabulary: %d -> %d", old_vocab, new_vocab)

    # ---- Load tokenizer ----
    tok_path = cfg.get("tokenizer_path", "data/tokenizer/zensei_ja_sp.model")
    tokenizer = _load_tokenizer(tok_path)

    # ---- Load dataset ----
    ds_path = cfg.get("data_path", "data/pretokenized")
    max_seq_len = cfg.get("max_seq_len", 4096)
    train_dataset = _load_dataset(ds_path, max_seq_len)

    # Optional eval dataset
    eval_dataset = None
    eval_path = cfg.get("eval_data_path")
    if eval_path is not None:
        eval_dataset = _load_dataset(eval_path, max_seq_len)

    # ---- Create trainer ----
    from zensei.training.trainer import ZenseiTrainer

    trainer = ZenseiTrainer(
        model=model,
        train_dataset=train_dataset,
        config=cfg,
        eval_dataset=eval_dataset,
    )

    # ---- Resume from checkpoint ----
    resume_step = 0
    if resume_from is not None:
        from zensei.training.checkpoint import load_checkpoint

        resume_step = load_checkpoint(
            model=trainer.model_engine,
            optimizer=trainer.optimizer,
            scheduler=trainer.lr_scheduler,
            path=resume_from,
        )
        logger.info("Resumed from checkpoint at step %d", resume_step)

    # ---- Train ----
    trainer.train(resume_step=resume_step)


if __name__ == "__main__":
    fire.Fire(main)
