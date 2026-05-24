"""
vector_store.py

Thin ChromaDB wrapper. One persistent collection per loan_id, rooted at
loans/<loan_id>/vectors/. Every chunk carries an entity_type in its
metadata so the RAG layer can do entity-scoped lookups across documents
(e.g. "show me every chunk tagged borrower_income for this loan").

The embedder defaults to sentence-transformers' all-MiniLM-L6-v2 (local,
no API key). For tests that don't want to download the model an
alternate embedder can be injected via the `embedder` kwarg — anything
implementing `.encode(texts) -> array` works.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Quiet ChromaDB's telemetry pings; nothing we want phoning home from a dev box.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.paths import loan_vectors_dir


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Per-chunk metadata keys we always populate. ChromaDB requires every
# metadata value to be a primitive (str / int / float / bool); never
# pass None or nested containers.
_REQUIRED_META_KEYS = ("source_file", "doc_type", "entity_type", "page_number", "chunk_index")


def _default_embedder():
    """Build the real sentence-transformers embedder.

    Wrapped in a function so the heavy `SentenceTransformer` import only
    runs when a real embedder is actually needed (tests inject a fake).
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


def _to_list(embeddings):
    """Coerce a numpy array (or list) of embeddings to ChromaDB's list-of-lists shape."""
    if hasattr(embeddings, "tolist"):
        return embeddings.tolist()
    return [list(e) for e in embeddings]


class VectorStore:
    """Per-loan ChromaDB collection with entity-aware retrieval.

    Each instance binds to a single loan_id and stores its index at
    loans/<loan_id>/vectors/. Re-instantiating with the same loan_id
    reopens the existing collection (idempotent across runs).
    """

    def __init__(self, loan_id: str, embedder=None):
        self.loan_id = loan_id
        self.collection_name = f"loan_{loan_id}"

        vectors_dir = loan_vectors_dir(loan_id)
        vectors_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(vectors_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.embedder = embedder if embedder is not None else _default_embedder()
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- ingestion ----------

    def add_document(self, doc_id: str, chunks: List[dict]) -> int:
        """Index chunks for one document. Returns the count of NEW chunks added.

        Each chunk dict must have:
            text: str
            metadata: dict with at least
                source_file, doc_type, entity_type, page_number, chunk_index

        The ChromaDB id for each chunk is "<doc_id>#<chunk_index>", so
        re-indexing the same (doc_id, chunk_index) pair is a no-op
        (idempotent). Use delete_document(doc_id) first if you want to
        force a re-embed (the ingestion layer does this on force=True).
        """
        if not chunks:
            return 0

        candidate_ids = [f"{doc_id}#{c['metadata']['chunk_index']}" for c in chunks]

        # Find which candidate IDs are already present so we can skip them.
        existing = set()
        if self.collection.count() > 0:
            existing = set(self.collection.get(ids=candidate_ids).get("ids", []))

        new_indices = [i for i, cid in enumerate(candidate_ids) if cid not in existing]
        if not new_indices:
            return 0

        new_ids = [candidate_ids[i] for i in new_indices]
        new_texts = [chunks[i]["text"] for i in new_indices]
        new_metas = [self._clean_metadata(chunks[i]["metadata"], doc_id) for i in new_indices]
        new_embeds = _to_list(self.embedder.encode(new_texts))

        self.collection.add(
            ids=new_ids,
            embeddings=new_embeds,
            documents=new_texts,
            metadatas=new_metas,
        )
        return len(new_ids)

    @staticmethod
    def _clean_metadata(meta: dict, doc_id: str) -> dict:
        """Strip Nones / coerce to ChromaDB-acceptable primitives, add doc_id."""
        out = {"doc_id": str(doc_id)}
        for k in _REQUIRED_META_KEYS:
            v = meta.get(k)
            if v is None:
                # ChromaDB rejects None; substitute a sentinel string for missing string fields
                # and 0 for missing numeric ones.
                v = 0 if k in ("page_number", "chunk_index") else ""
            if isinstance(v, (str, int, float, bool)):
                out[k] = v
            else:
                out[k] = str(v)
        return out

    # ---------- retrieval ----------

    def query(self, question: str, n_results: int = 5,
              entity_type_filter: Optional[str] = None) -> List[dict]:
        """Semantic search: embed `question`, return top n_results chunks.

        If `entity_type_filter` is set, results are restricted to chunks
        carrying that entity_type. Returns [] on an empty collection
        (rather than raising) so callers can degrade gracefully.
        Results are sorted by distance ascending (most relevant first).
        """
        if self.collection.count() == 0:
            return []

        embed = _to_list(self.embedder.encode([question]))
        where = {"entity_type": entity_type_filter} if entity_type_filter else None

        raw = self.collection.query(
            query_embeddings=embed,
            n_results=n_results,
            where=where,
        )

        return self._unpack_query_result(raw)

    def query_by_entity(self, entity_type: str, n_results: int = 10) -> List[dict]:
        """Return up to n_results chunks of the given entity_type, no semantic ranking.

        Used by the RAG layer for cross-document comparison ("give me
        every borrower_income chunk across all this loan's documents").
        Order is whatever ChromaDB returns from `.get()` — not
        similarity-sorted, since there's no query string here.
        """
        if self.collection.count() == 0:
            return []

        raw = self.collection.get(
            where={"entity_type": entity_type},
            limit=n_results,
        )
        return self._unpack_get_result(raw)

    @staticmethod
    def _unpack_query_result(raw: dict) -> List[dict]:
        """Flatten ChromaDB's nested .query() response into a list of chunk dicts."""
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return []
        ids = raw["ids"][0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        out = []
        for i in range(len(ids)):
            m = metas[i] if i < len(metas) else {}
            out.append({
                "text": docs[i] if i < len(docs) else "",
                "source_file": m.get("source_file", ""),
                "doc_type": m.get("doc_type", ""),
                "entity_type": m.get("entity_type", ""),
                "page_number": m.get("page_number", 0),
                "distance": dists[i] if i < len(dists) else None,
            })
        return out

    @staticmethod
    def _unpack_get_result(raw: dict) -> List[dict]:
        """Flatten ChromaDB's flat .get() response (note: not nested like .query())."""
        if not raw or not raw.get("ids"):
            return []
        ids = raw["ids"]
        docs = raw.get("documents") or [""] * len(ids)
        metas = raw.get("metadatas") or [{}] * len(ids)
        out = []
        for i in range(len(ids)):
            m = metas[i] if i < len(metas) else {}
            out.append({
                "text": docs[i] if i < len(docs) else "",
                "source_file": m.get("source_file", ""),
                "doc_type": m.get("doc_type", ""),
                "entity_type": m.get("entity_type", ""),
                "page_number": m.get("page_number", 0),
                "distance": None,
            })
        return out

    # ---------- maintenance ----------

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to `doc_id`. Returns the count deleted."""
        existing = self.collection.get(where={"doc_id": str(doc_id)})
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self.collection.delete(ids=ids)
        return len(ids)

    def document_ids(self) -> List[str]:
        """Return the sorted, deduplicated set of doc_ids currently indexed."""
        if self.collection.count() == 0:
            return []
        metas = self.collection.get().get("metadatas") or []
        return sorted({m.get("doc_id") for m in metas if m and m.get("doc_id")})

    def chunk_count(self) -> int:
        """Total number of chunks in this collection."""
        return self.collection.count()
