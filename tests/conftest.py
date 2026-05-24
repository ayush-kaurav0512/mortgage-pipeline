"""
conftest.py

Pytest configuration shared across the tests/ directory.

  * Puts the project root on sys.path so tests can write
    `from src.X import Y` regardless of where pytest is invoked from.
  * Provides the `isolated_project` fixture which monkeypatches every
    filesystem constant in src.paths to a per-test tmp_path. Use it
    in any test that calls into code which writes under loans/,
    pools/, runs/, or inbox/ so the test can't pollute the real
    project tree.
"""

import hashlib
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------- helpers reused across test modules ----------

class FakeEmbedder:
    """Deterministic test embedder so VectorStore tests don't download a model.

    Same input text -> same vector across runs and across instances, so
    a chunk indexed in one call can be retrieved by querying with the
    exact same text. Different texts produce visibly different vectors
    so ChromaDB's ordering by distance still behaves sensibly.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts):
        # sentence-transformers accepts a single string or a list — mirror that.
        if isinstance(texts, str):
            texts = [texts]
        import numpy as np
        vecs = []
        for text in texts:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "big", signed=False)
            rng = np.random.RandomState(seed % (2**32))
            v = rng.randn(self.dim).astype("float32")
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.array(vecs)


def make_groq_mock(response_text: str = "unknown") -> MagicMock:
    """Build a MagicMock that mimics Groq's chat.completions.create() return shape."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


# ---------- fixtures ----------


@pytest.fixture
def isolated_project(tmp_path, monkeypatch):
    """Redirect every filesystem constant in src.paths to a tmp_path tree.

    The functions in src.paths look up module-level constants
    (LOANS_DIR, POOLS_DIR, etc.) at call time, so patching them on
    the imported module is enough — no need to touch any caller.

    Yields the tmp_path root so tests can write fixture files there.
    """
    import src.paths as paths_mod

    monkeypatch.setattr(paths_mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(paths_mod, "LOANS_DIR", tmp_path / "loans")
    monkeypatch.setattr(paths_mod, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(paths_mod, "POOLS_DIR", tmp_path / "pools")
    monkeypatch.setattr(paths_mod, "INBOX_DIR", tmp_path / "inbox")
    monkeypatch.setattr(paths_mod, "INBOX_PROCESSED_DIR", tmp_path / "inbox" / "processed")
    monkeypatch.setattr(paths_mod, "INBOX_FAILED_DIR", tmp_path / "inbox" / "failed")
    monkeypatch.setattr(paths_mod, "INBOX_UNMATCHED_DIR", tmp_path / "inbox" / "unmatched")

    return tmp_path
