"""Tests for the Zensei tokenizer pipeline.

Tests cover:
  - SentencePiece training produces valid model files
  - Merged tokenizer has correct vocab size
  - Roundtrip encode/decode on Japanese text
  - Fertility (tokens per character) is reasonable
  - Special tokens are preserved
"""

import os
import tempfile

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_japanese_corpus(tmp_path):
    """Create a small Japanese corpus for training tests."""
    texts = [
        "日本語のテストデータです。自然言語処理のために作られました。",
        "東京は日本の首都であり、世界最大の都市圏の一つです。",
        "桜の花が春に咲きます。日本の象徴的な花です。",
        "人工知能の研究は急速に進んでいます。",
        "深層学習モデルは大量のデータから学習します。",
        "日本語は漢字、ひらがな、カタカナの三種類の文字を使います。",
        "富士山は日本で最も高い山で、標高は3776メートルです。",
        "新幹線は日本の高速鉄道システムで、最高速度は時速320キロメートルです。",
    ]
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "test_corpus.txt"
    corpus_file.write_text("\n".join(texts * 50), encoding="utf-8")
    return corpus_dir


@pytest.fixture
def trained_sp_model(sample_japanese_corpus, tmp_path):
    """Train a small SentencePiece model for testing."""
    import sentencepiece as spm

    model_prefix = str(tmp_path / "test_sp")
    corpus_files = [str(sample_japanese_corpus / "test_corpus.txt")]

    spm.SentencePieceTrainer.train(
        input=",".join(corpus_files),
        model_prefix=model_prefix,
        vocab_size=500,
        model_type="bpe",
        character_coverage=0.9995,
        normalization_rule_name="nfkc",
        byte_fallback=True,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )
    return model_prefix


# ---------------------------------------------------------------------------
# Tests: SentencePiece training
# ---------------------------------------------------------------------------


class TestSentencePieceTraining:
    """Tests for SentencePiece model training."""

    def test_training_produces_model_file(self, trained_sp_model):
        """Training should produce a .model file."""
        model_path = f"{trained_sp_model}.model"
        assert os.path.exists(model_path), f"Expected model file at {model_path}"
        assert os.path.getsize(model_path) > 0, "Model file should not be empty"

    def test_training_produces_vocab_file(self, trained_sp_model):
        """Training should produce a .vocab file."""
        vocab_path = f"{trained_sp_model}.vocab"
        assert os.path.exists(vocab_path), f"Expected vocab file at {vocab_path}"
        assert os.path.getsize(vocab_path) > 0, "Vocab file should not be empty"

    def test_trained_model_has_correct_vocab_size(self, trained_sp_model):
        """Trained model should have the requested vocab size."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")
        assert sp.get_piece_size() == 500


# ---------------------------------------------------------------------------
# Tests: Roundtrip encode/decode
# ---------------------------------------------------------------------------


class TestRoundtrip:
    """Tests for encode/decode roundtrip fidelity."""

    def test_roundtrip_japanese(self, trained_sp_model):
        """Encoding then decoding should recover the original Japanese text."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")

        texts = [
            "日本語のテストです。",
            "東京タワーは333メートルの高さです。",
            "深層学習は人工知能の一分野です。",
        ]
        for text in texts:
            ids = sp.encode(text, out_type=int)
            decoded = sp.decode(ids)
            assert decoded == text, f"Roundtrip failed: '{text}' -> '{decoded}'"

    def test_roundtrip_mixed_script(self, trained_sp_model):
        """Roundtrip should work with mixed Japanese/Latin text."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")

        text = "AIモデルはGPUで学習します。"
        ids = sp.encode(text, out_type=int)
        decoded = sp.decode(ids)
        assert decoded == text


# ---------------------------------------------------------------------------
# Tests: Fertility
# ---------------------------------------------------------------------------


class TestFertility:
    """Tests for tokenizer fertility (tokens per character)."""

    def test_fertility_is_reasonable(self, trained_sp_model):
        """Fertility for Japanese text should be between 0.3 and 3.0 tokens/char."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")

        text = "日本語の自然言語処理は重要な研究分野です。"
        ids = sp.encode(text, out_type=int)
        fertility = len(ids) / len(text)

        assert 0.3 < fertility < 3.0, (
            f"Fertility {fertility:.2f} is outside reasonable range "
            f"({len(ids)} tokens / {len(text)} chars)"
        )


# ---------------------------------------------------------------------------
# Tests: Special tokens
# ---------------------------------------------------------------------------


class TestSpecialTokens:
    """Tests for special token handling."""

    def test_special_tokens_exist(self, trained_sp_model):
        """The trained model should have standard special tokens."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")

        # Check that special token IDs are valid
        assert sp.pad_id() == 0
        assert sp.unk_id() == 1
        assert sp.bos_id() == 2
        assert sp.eos_id() == 3

    def test_special_tokens_decode(self, trained_sp_model):
        """Special token pieces should be accessible."""
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(f"{trained_sp_model}.model")

        assert sp.id_to_piece(0) == "<pad>"
        assert sp.id_to_piece(1) == "<unk>"
        assert sp.id_to_piece(2) == "<s>"  # bos
        assert sp.id_to_piece(3) == "</s>"  # eos
