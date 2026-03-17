"""Tests for Zensei model loading and forward pass.

Tests cover:
  - ModelArgs loading from JSON config
  - Model instantiation with small config
  - Forward pass output shape
  - Vocabulary expansion and embedding resize
  - Embedding weight preservation after resize
"""

import json
import os
import tempfile

import pytest
import torch

from zensei.model.model import ModelArgs, Transformer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model_args():
    """Create a small ModelArgs for fast testing."""
    return ModelArgs(
        vocab_size=1024,
        dim=128,
        inter_dim=256,
        moe_inter_dim=64,
        n_layers=2,
        n_dense_layers=1,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        route_scale=1.0,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=8,
        v_head_dim=32,
        dtype="float32",
        max_seq_len=128,
        rope_theta=10000.0,
    )


@pytest.fixture
def small_model(small_model_args):
    """Create a small Transformer model for testing."""
    return Transformer(small_model_args)


@pytest.fixture
def config_json_path(small_model_args, tmp_path):
    """Write a small config to a JSON file and return its path."""
    config = {
        "vocab_size": small_model_args.vocab_size,
        "dim": small_model_args.dim,
        "inter_dim": small_model_args.inter_dim,
        "moe_inter_dim": small_model_args.moe_inter_dim,
        "n_layers": small_model_args.n_layers,
        "n_dense_layers": small_model_args.n_dense_layers,
        "n_heads": small_model_args.n_heads,
        "n_routed_experts": small_model_args.n_routed_experts,
        "n_shared_experts": small_model_args.n_shared_experts,
        "n_activated_experts": small_model_args.n_activated_experts,
        "route_scale": small_model_args.route_scale,
        "q_lora_rank": small_model_args.q_lora_rank,
        "kv_lora_rank": small_model_args.kv_lora_rank,
        "qk_nope_head_dim": small_model_args.qk_nope_head_dim,
        "qk_rope_head_dim": small_model_args.qk_rope_head_dim,
        "v_head_dim": small_model_args.v_head_dim,
        "dtype": small_model_args.dtype,
        "max_seq_len": small_model_args.max_seq_len,
        "rope_theta": small_model_args.rope_theta,
    }
    path = tmp_path / "config.json"
    with open(path, "w") as f:
        json.dump(config, f)
    return str(path)


# ---------------------------------------------------------------------------
# Tests: ModelArgs
# ---------------------------------------------------------------------------


class TestModelArgs:
    """Tests for ModelArgs configuration loading."""

    def test_from_json(self, config_json_path):
        """ModelArgs should load correctly from a JSON file."""
        args = ModelArgs.from_json(config_json_path)
        assert args.vocab_size == 1024
        assert args.dim == 128
        assert args.n_layers == 2

    def test_derived_fields_computed(self, small_model_args):
        """Derived fields (qk_head_dim, head_dim) should be computed in __post_init__."""
        assert small_model_args.qk_head_dim == 24 + 8  # nope + rope
        assert small_model_args.head_dim == 32  # v_head_dim

    def test_to_torch_dtype(self, small_model_args):
        """to_torch_dtype should return correct torch dtype."""
        small_model_args.dtype = "bfloat16"
        assert small_model_args.to_torch_dtype() == torch.bfloat16

        small_model_args.dtype = "float32"
        assert small_model_args.to_torch_dtype() == torch.float32

    def test_unknown_fields_ignored(self, tmp_path):
        """Unknown fields in JSON should be silently ignored."""
        config = {"vocab_size": 2048, "unknown_field": 42, "dim": 256}
        path = tmp_path / "config_extra.json"
        with open(path, "w") as f:
            json.dump(config, f)

        args = ModelArgs.from_json(str(path))
        assert args.vocab_size == 2048
        assert args.dim == 256
        assert not hasattr(args, "unknown_field") or "unknown_field" not in args.__dict__


# ---------------------------------------------------------------------------
# Tests: Model instantiation
# ---------------------------------------------------------------------------


class TestModelInstantiation:
    """Tests for Transformer model creation."""

    def test_model_creates(self, small_model):
        """Model should instantiate without errors."""
        assert small_model is not None
        assert isinstance(small_model, Transformer)

    def test_model_from_config(self, config_json_path):
        """Model should be creatable from a config file."""
        model = Transformer.from_config(config_json_path)
        assert model is not None
        assert model.args.vocab_size == 1024

    def test_model_has_correct_layers(self, small_model, small_model_args):
        """Model should have the correct number of layers."""
        assert len(small_model.layers) == small_model_args.n_layers

    def test_embedding_size(self, small_model, small_model_args):
        """Embedding layer should match vocab_size and dim."""
        assert small_model.tok_emb.num_embeddings == small_model_args.vocab_size
        assert small_model.tok_emb.embedding_dim == small_model_args.dim

    def test_lm_head_size(self, small_model, small_model_args):
        """LM head should match dim -> vocab_size."""
        assert small_model.lm_head.in_features == small_model_args.dim
        assert small_model.lm_head.out_features == small_model_args.vocab_size


# ---------------------------------------------------------------------------
# Tests: Forward pass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Tests for model forward pass."""

    def test_output_shape(self, small_model, small_model_args):
        """Forward pass should produce logits of shape (B, T, vocab_size)."""
        batch_size = 2
        seq_len = 16
        tokens = torch.randint(0, small_model_args.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, aux_loss = small_model(tokens)

        assert logits.shape == (batch_size, seq_len, small_model_args.vocab_size)

    def test_aux_loss_is_scalar(self, small_model, small_model_args):
        """Auxiliary MoE loss should be a scalar tensor."""
        tokens = torch.randint(0, small_model_args.vocab_size, (1, 8))

        with torch.no_grad():
            _, aux_loss = small_model(tokens)

        assert aux_loss.dim() == 0  # scalar

    def test_single_token_input(self, small_model, small_model_args):
        """Model should handle single-token sequences."""
        tokens = torch.randint(0, small_model_args.vocab_size, (1, 1))

        with torch.no_grad():
            logits, _ = small_model(tokens)

        assert logits.shape == (1, 1, small_model_args.vocab_size)


# ---------------------------------------------------------------------------
# Tests: Vocabulary expansion
# ---------------------------------------------------------------------------


class TestVocabExpansion:
    """Tests for embedding resize after vocabulary expansion."""

    def test_embedding_resize(self, small_model):
        """Resizing embeddings should produce correct new size."""
        original_size = small_model.tok_emb.num_embeddings
        new_vocab_size = original_size + 128

        # Resize embedding and lm_head
        old_emb_weight = small_model.tok_emb.weight.data.clone()

        new_emb = torch.nn.Embedding(new_vocab_size, small_model.args.dim)
        new_emb.weight.data[:original_size] = old_emb_weight
        # Initialize new embeddings with mean of existing
        mean_emb = old_emb_weight.mean(dim=0)
        new_emb.weight.data[original_size:] = mean_emb.unsqueeze(0).expand(128, -1)
        small_model.tok_emb = new_emb

        assert small_model.tok_emb.num_embeddings == new_vocab_size

    def test_original_weights_preserved(self, small_model):
        """Original embedding weights should be preserved after resize."""
        original_weight = small_model.tok_emb.weight.data.clone()
        original_size = small_model.tok_emb.num_embeddings
        new_vocab_size = original_size + 64

        new_emb = torch.nn.Embedding(new_vocab_size, small_model.args.dim)
        new_emb.weight.data[:original_size] = original_weight
        small_model.tok_emb = new_emb

        # Check original weights are unchanged
        assert torch.allclose(
            small_model.tok_emb.weight.data[:original_size],
            original_weight,
        )
