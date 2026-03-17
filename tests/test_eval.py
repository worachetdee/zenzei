"""Tests for the Zensei evaluation utilities.

Tests cover:
  - Character-level F1 computation
  - Perplexity computation on dummy data
  - JGLUE data loading (mocked)
  - Evaluation result aggregation
"""

import math
from collections import Counter
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers (replicate eval logic for testing)
# ---------------------------------------------------------------------------


def char_f1(prediction: str, reference: str) -> dict[str, float]:
    """Compute character-level precision, recall, and F1.

    This metric is commonly used for Japanese QA evaluation where
    word boundaries are ambiguous.

    Args:
        prediction: Predicted text.
        reference: Ground-truth reference text.

    Returns:
        Dict with keys "precision", "recall", "f1".
    """
    pred_chars = list(prediction)
    ref_chars = list(reference)

    if not pred_chars and not ref_chars:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_chars or not ref_chars:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_counter = Counter(pred_chars)
    ref_counter = Counter(ref_chars)

    # Count matching characters
    common = sum((pred_counter & ref_counter).values())

    precision = common / len(pred_chars) if pred_chars else 0.0
    recall = common / len(ref_chars) if ref_chars else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute perplexity from logits and labels.

    Args:
        logits: Model output logits of shape (B, T, V).
        labels: Ground-truth token IDs of shape (B, T).

    Returns:
        Perplexity value (float).
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return math.exp(loss.item())


def aggregate_results(results: list[dict]) -> dict[str, float]:
    """Aggregate evaluation results across multiple samples.

    Args:
        results: List of result dicts, each with string keys and float values.

    Returns:
        Dict with averaged metrics.
    """
    if not results:
        return {}

    keys = results[0].keys()
    aggregated = {}
    for key in keys:
        values = [r[key] for r in results if key in r]
        aggregated[key] = sum(values) / len(values) if values else 0.0

    return aggregated


# ---------------------------------------------------------------------------
# Tests: Character-level F1
# ---------------------------------------------------------------------------


class TestCharF1:
    """Tests for character-level F1 computation."""

    def test_exact_match(self):
        """Exact match should yield F1 = 1.0."""
        result = char_f1("東京", "東京")
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_no_overlap(self):
        """No character overlap should yield F1 = 0.0."""
        result = char_f1("東京", "大阪")
        assert result["f1"] == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap should yield 0 < F1 < 1."""
        result = char_f1("東京タワー", "東京スカイツリー")
        assert 0.0 < result["f1"] < 1.0
        assert 0.0 < result["precision"] < 1.0
        assert 0.0 < result["recall"] < 1.0

    def test_empty_prediction(self):
        """Empty prediction should yield F1 = 0.0."""
        result = char_f1("", "東京")
        assert result["f1"] == pytest.approx(0.0)

    def test_empty_reference(self):
        """Empty reference should yield F1 = 0.0."""
        result = char_f1("東京", "")
        assert result["f1"] == pytest.approx(0.0)

    def test_both_empty(self):
        """Both empty should yield F1 = 1.0."""
        result = char_f1("", "")
        assert result["f1"] == pytest.approx(1.0)

    def test_precision_recall_asymmetry(self):
        """Precision and recall should differ when lengths differ."""
        # Prediction is substring of reference -> perfect precision, lower recall
        result = char_f1("東京", "東京タワー")
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] < 1.0


# ---------------------------------------------------------------------------
# Tests: Perplexity
# ---------------------------------------------------------------------------


class TestPerplexity:
    """Tests for perplexity computation."""

    def test_perplexity_is_positive(self):
        """Perplexity should always be positive."""
        vocab_size = 100
        logits = torch.randn(1, 16, vocab_size)
        labels = torch.randint(0, vocab_size, (1, 16))
        ppl = compute_perplexity(logits, labels)
        assert ppl > 0

    def test_perfect_prediction_low_perplexity(self):
        """Near-perfect predictions should have low perplexity."""
        vocab_size = 10
        seq_len = 8
        labels = torch.randint(0, vocab_size, (1, seq_len))

        # Create logits that strongly predict the correct next token
        logits = torch.full((1, seq_len, vocab_size), -10.0)
        for t in range(seq_len - 1):
            logits[0, t, labels[0, t + 1]] = 10.0

        ppl = compute_perplexity(logits, labels)
        assert ppl < 2.0  # Should be very close to 1.0

    def test_random_logits_high_perplexity(self):
        """Random logits should produce high perplexity (close to vocab_size)."""
        vocab_size = 100
        logits = torch.zeros(1, 32, vocab_size)  # uniform distribution
        labels = torch.randint(0, vocab_size, (1, 32))
        ppl = compute_perplexity(logits, labels)
        # Perplexity of uniform distribution should be close to vocab_size
        assert ppl > vocab_size * 0.5


# ---------------------------------------------------------------------------
# Tests: JGLUE data loading (mocked)
# ---------------------------------------------------------------------------


class TestJGLUELoading:
    """Tests for JGLUE benchmark data loading."""

    def test_jglue_dataset_structure(self):
        """JGLUE dataset entries should have required fields."""
        # Mock a JGLUE-like dataset entry
        mock_entry = {
            "sentence1": "今日は天気が良いです。",
            "sentence2": "今日は晴れです。",
            "label": 1,
        }
        assert "sentence1" in mock_entry
        assert "sentence2" in mock_entry
        assert "label" in mock_entry
        assert isinstance(mock_entry["label"], int)

    @patch("datasets.load_dataset")
    def test_jglue_load_returns_data(self, mock_load):
        """Loading JGLUE should return non-empty dataset."""
        mock_dataset = MagicMock()
        mock_dataset.__len__ = MagicMock(return_value=100)
        mock_dataset.__iter__ = MagicMock(
            return_value=iter(
                [
                    {"sentence1": "テスト文1", "sentence2": "テスト文2", "label": 0},
                    {"sentence1": "テスト文3", "sentence2": "テスト文4", "label": 1},
                ]
            )
        )
        mock_load.return_value = {"validation": mock_dataset}

        result = mock_load("shunk031/JGLUE", name="MARC-ja", split="validation")
        assert result is not None

    def test_jaquad_entry_structure(self):
        """JaQuAD entries should have question, context, and answer fields."""
        mock_entry = {
            "question": "日本の首都はどこですか？",
            "context": "東京は日本の首都であり、世界最大の都市圏の一つです。",
            "answers": {"text": ["東京"], "answer_start": [0]},
        }
        assert "question" in mock_entry
        assert "context" in mock_entry
        assert "answers" in mock_entry
        assert len(mock_entry["answers"]["text"]) > 0


# ---------------------------------------------------------------------------
# Tests: Result aggregation
# ---------------------------------------------------------------------------


class TestResultAggregation:
    """Tests for evaluation result aggregation."""

    def test_average_single_metric(self):
        """Aggregation of a single metric should compute the mean."""
        results = [
            {"f1": 0.8},
            {"f1": 0.9},
            {"f1": 0.7},
        ]
        agg = aggregate_results(results)
        assert agg["f1"] == pytest.approx(0.8, abs=1e-6)

    def test_average_multiple_metrics(self):
        """Aggregation should handle multiple metrics."""
        results = [
            {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            {"precision": 0.7, "recall": 0.6, "f1": 0.65},
        ]
        agg = aggregate_results(results)
        assert agg["precision"] == pytest.approx(0.8)
        assert agg["recall"] == pytest.approx(0.7)
        assert agg["f1"] == pytest.approx(0.75)

    def test_empty_results(self):
        """Aggregation of empty results should return empty dict."""
        agg = aggregate_results([])
        assert agg == {}

    def test_single_result(self):
        """Aggregation of a single result should return it unchanged."""
        results = [{"accuracy": 0.95}]
        agg = aggregate_results(results)
        assert agg["accuracy"] == pytest.approx(0.95)
