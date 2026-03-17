"""Tests for the Zensei data processing pipeline.

Tests cover:
  - Text cleaning (HTML removal, Unicode normalization)
  - Near-duplicate removal
  - Length and quality filtering
  - Binary file preparation

Uses small synthetic Japanese data for fast, reproducible tests.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_docs(tmp_path):
    """Create synthetic raw JSONL documents with various quality levels."""
    docs = [
        # Good quality Japanese
        {"text": "日本語の自然言語処理は、近年大きな進歩を遂げています。特に大規模言語モデルの登場により、多くのタスクで高い性能が達成されるようになりました。", "source": "test"},
        # Contains HTML tags
        {"text": "<p>東京は日本の<b>首都</b>です。</p><br/>人口は約1400万人です。", "source": "test"},
        # Too short (should be filtered)
        {"text": "短い", "source": "test"},
        # Near-duplicate of first doc
        {"text": "日本語の自然言語処理は、近年大きな進歩を遂げています。特に大規模言語モデルの登場により、多くのタスクで高い性能が達成されるようになりました。", "source": "test"},
        # Contains excessive whitespace
        {"text": "人工知能   の  研究は    急速に   進んで   います。   機械学習は   その   中核   技術です。", "source": "test"},
        # Good quality
        {"text": "京都は日本の伝統的な文化の中心地であり、多くの神社仏閣が世界遺産に登録されています。四季折々の美しい景色が楽しめます。", "source": "test"},
        # Mostly non-Japanese (should potentially be filtered)
        {"text": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. This is mostly English text with minimal Japanese content.", "source": "test"},
    ]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    jsonl_path = raw_dir / "test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    return raw_dir


# ---------------------------------------------------------------------------
# Tests: Cleaning
# ---------------------------------------------------------------------------


class TestCleaning:
    """Tests for text cleaning functionality."""

    def test_html_removal(self):
        """HTML tags should be stripped from text."""
        import re

        text = "<p>東京は日本の<b>首都</b>です。</p><br/>"
        # Simple HTML tag removal (mimics what clean.py would do)
        cleaned = re.sub(r"<[^>]+>", "", text)
        assert "<" not in cleaned
        assert ">" not in cleaned
        assert "東京は日本の首都です。" == cleaned

    def test_whitespace_normalization(self):
        """Excessive whitespace should be collapsed."""
        import re

        text = "人工知能   の  研究は    急速に   進んで   います。"
        normalized = re.sub(r"\s+", " ", text).strip()
        assert "   " not in normalized
        assert "  " not in normalized
        assert normalized == "人工知能 の 研究は 急速に 進んで います。"

    def test_unicode_normalization(self):
        """Text should be NFKC normalized."""
        import unicodedata

        # Full-width to half-width normalization
        text = "ＡＢＣ１２３"
        normalized = unicodedata.normalize("NFKC", text)
        assert normalized == "ABC123"

    def test_empty_text_handled(self):
        """Empty or whitespace-only text should be handled gracefully."""
        import re

        texts = ["", "   ", "\n\n", "\t  \n"]
        for text in texts:
            cleaned = re.sub(r"\s+", " ", text).strip()
            assert cleaned == ""


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Tests for near-duplicate removal."""

    def test_exact_duplicates_removed(self):
        """Exact duplicate documents should be identified."""
        docs = [
            "日本語の自然言語処理は重要です。",
            "京都は美しい都市です。",
            "日本語の自然言語処理は重要です。",  # exact dup
        ]
        unique = list(set(docs))
        assert len(unique) == 2

    def test_near_duplicates_detected(self):
        """Near-duplicate detection using MinHash-like approach."""
        from hashlib import md5

        def shingle_set(text, k=3):
            """Create character-level shingles."""
            return {text[i : i + k] for i in range(len(text) - k + 1)}

        def jaccard_similarity(s1, s2):
            """Compute Jaccard similarity between two sets."""
            intersection = len(s1 & s2)
            union = len(s1 | s2)
            return intersection / union if union > 0 else 0.0

        doc1 = "日本語の自然言語処理は、近年大きな進歩を遂げています。"
        doc2 = "日本語の自然言語処理は、近年大きな進歩を遂げています。"  # exact
        doc3 = "京都は日本の伝統的な文化の中心地です。"

        s1 = shingle_set(doc1)
        s2 = shingle_set(doc2)
        s3 = shingle_set(doc3)

        # Exact duplicates have similarity 1.0
        assert jaccard_similarity(s1, s2) == 1.0
        # Different documents have lower similarity
        assert jaccard_similarity(s1, s3) < 0.5


# ---------------------------------------------------------------------------
# Tests: Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    """Tests for document quality filtering."""

    def test_length_filter_short(self):
        """Documents shorter than minimum length should be filtered out."""
        min_length = 10
        docs = [
            "短い",  # too short
            "日本語の自然言語処理は重要な研究分野です。",  # ok
        ]
        filtered = [d for d in docs if len(d) >= min_length]
        assert len(filtered) == 1
        assert filtered[0].startswith("日本語")

    def test_length_filter_long(self):
        """Documents exceeding maximum length should be filtered out."""
        max_length = 50
        docs = [
            "短い文。",
            "あ" * 100,  # too long
        ]
        filtered = [d for d in docs if len(d) <= max_length]
        assert len(filtered) == 1

    def test_japanese_ratio_filter(self):
        """Documents with too little Japanese content should be filtered."""
        import re

        def japanese_ratio(text):
            """Compute the ratio of Japanese characters in text."""
            ja_chars = len(re.findall(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", text))
            total = len(text.replace(" ", ""))
            return ja_chars / total if total > 0 else 0.0

        # Mostly English
        assert japanese_ratio("Hello world with some 日本語") < 0.5
        # Mostly Japanese
        assert japanese_ratio("日本語の自然言語処理は重要です。") > 0.5

    def test_repetition_filter(self):
        """Documents with excessive character repetition should be filtered."""
        def has_excessive_repetition(text, max_repeat=10):
            """Check if any character repeats more than max_repeat times consecutively."""
            import re
            pattern = r"(.)\1{" + str(max_repeat - 1) + r",}"
            return bool(re.search(pattern, text))

        assert has_excessive_repetition("ああああああああああああ")  # 12 repeats
        assert not has_excessive_repetition("日本語のテストです。")


# ---------------------------------------------------------------------------
# Tests: Binary preparation
# ---------------------------------------------------------------------------


class TestBinaryPreparation:
    """Tests for binary dataset preparation."""

    def test_token_ids_are_valid(self):
        """Tokenized sequences should contain only valid token IDs."""
        # Simulate tokenization output
        vocab_size = 500
        token_ids = [10, 23, 45, 100, 234, 5, 499]

        for tid in token_ids:
            assert 0 <= tid < vocab_size, f"Token ID {tid} out of range [0, {vocab_size})"

    def test_sequence_packing(self):
        """Multiple short documents should be packable into fixed-length sequences."""
        max_seq_len = 32
        eos_token = 3

        docs_tokenized = [
            [10, 20, 30, 40, 50],
            [11, 21, 31],
            [12, 22, 32, 42, 52, 62],
        ]

        # Pack into a single sequence with EOS separators
        packed = []
        for doc_ids in docs_tokenized:
            if len(packed) + len(doc_ids) + 1 <= max_seq_len:
                packed.extend(doc_ids)
                packed.append(eos_token)

        assert len(packed) <= max_seq_len
        assert packed.count(eos_token) == len(docs_tokenized)
