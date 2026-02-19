import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.embedding import randomEmbedder


def test_empty_string_input():
    """🕳️ Edge Case: Test empty string input"""
    embedder = randomEmbedder(dim=32)
    embeddings = embedder.embed([""])
    assert embeddings.shape == (1, 32), "🚨 Empty string should still return valid embedding shape."


def test_very_long_input():
    """🧵 Stress Test: 10,000 characters - Because users never stop typing"""
    long_text = "a" * 10000
    embedder = randomEmbedder(dim=32)
    embedding = embedder.embed([long_text])
    assert embedding.shape == (1, 32), "💥 Long input broke the embedder. Shape mismatch."


def test_unicode_and_emoji_input():
    """🌍 Emoji/Unicode Sanity: Can it handle 😎 and 世界?"""
    texts = ["hello 🌍", "emoji 😎", "unicode 世界"]
    embedder = randomEmbedder(dim=32)
    embeddings = embedder.embed(texts)
    assert embeddings.shape == (3, 32), "📛 Unicode handling failed."


def test_case_sensitivity():
    """🔡 Case Check: 'Text' vs 'text' should still yield consistent shape"""
    texts = ["Text", "text"]
    embedder = randomEmbedder(dim=32)
    embeddings = embedder.embed(texts)
    assert embeddings.shape == (2, 32), "❌ Case confusion: Shape mismatch on case variant inputs."


def test_dim_zero():
    """🧨 Dim=0: Should raise ValueError or handle gracefully"""
    with pytest.raises(ValueError):
        randomEmbedder(dim=0)


def test_no_normalization_norm():
    """⚖️ Check Norm Without Normalization"""
    embedder = randomEmbedder(dim=32, normalize=False)
    embedding = embedder.embed(["test"])
    norm = np.linalg.norm(embedding[0])
    assert not np.isclose(norm, 1.0), f"🧘 Too chill: Norm {norm} ≈ 1 despite normalize=False"


def test_with_normalization_norm():
    """🧘 With Normalization: Should be unit norm"""
    embedder = randomEmbedder(dim=32, normalize=True)
    embedding = embedder.embed(["test"])
    norm = np.linalg.norm(embedding[0])
    assert np.isclose(norm, 1.0), f"⚠️ Not normalized: Norm is {norm} instead of ≈1"


def test_seed_consistency():
    """🔐 Seed Check: Same seed, same output"""
    text = ["consistency"]
    embedder1 = randomEmbedder(dim=32, deterministic=False, seed=123)
    embedder2 = randomEmbedder(dim=32, deterministic=False, seed=123)
    emb1 = embedder1.embed(text)
    emb2 = embedder2.embed(text)
    assert np.allclose(emb1, emb2), "🔁 Same seed gave different results. That's chaos."


def test_different_seed_variation():
    """🔀 Different seeds, different outputs"""
    text = ["variation"]
    emb1 = randomEmbedder(dim=32, deterministic=False, seed=1).embed(text)
    emb2 = randomEmbedder(dim=32, deterministic=False, seed=2).embed(text)
    assert not np.allclose(emb1, emb2), "🎲 Random mode forgot how to be random."


def test_non_string_input_handling():
    """🤖 Type Agnosticism: Numbers, None, etc."""
    embedder = randomEmbedder(dim=32)
    inputs = [None, 123, 45.6, True]
    with pytest.raises(TypeError):
        embedder.embed(inputs)

