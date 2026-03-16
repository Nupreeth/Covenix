import json
import os
from typing import List, Dict, Optional, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision
from endee.schema import VectorItem


if not hasattr(VectorItem, "get"):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)


CHUNKS_PATH = "data/chunks/clause_chunks.json"
SAMPLE_PATH = "sample_data/clause_chunks.json"

DEFAULT_INDEX_NAME = "covenix_clauses"
MAX_UPSERT_BATCH = 1000


class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_name: Optional[str] = None,
        base_url: Optional[str] = None,
        auth_token: Optional[str] = None,
    ):
        self.model = SentenceTransformer(model_name)

        token = auth_token or os.getenv("ENDEE_AUTH_TOKEN")
        self.client = Endee(token=token) if token else Endee()

        base_url = base_url or os.getenv("ENDEE_BASE_URL")
        if base_url:
            self.client.set_base_url(base_url)

        raw_name = index_name or os.getenv("ENDEE_INDEX_NAME", DEFAULT_INDEX_NAME)
        self.index_name = raw_name.replace("-", "_")
        self.index = None

    # -----------------------------
    # Load Clause Data
    # -----------------------------
    def load_clauses(self) -> List[Dict]:
        if os.path.exists(CHUNKS_PATH):
            path = CHUNKS_PATH
        elif os.path.exists(SAMPLE_PATH):
            path = SAMPLE_PATH
        else:
            raise FileNotFoundError(
                "No clause data found. Generate with agents/document_agent.py "
                "or use sample_data/clause_chunks.json."
            )

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------
    # Index Helpers
    # -----------------------------
    def _get_index(self):
        if self.index is None:
            self.index = self.client.get_index(name=self.index_name)
        return self.index

    def _ensure_index(self, dimension: int):
        if self.index is not None:
            return self.index

        try:
            self.index = self.client.get_index(name=self.index_name)
        except Exception:
            try:
                self.client.delete_index(name=self.index_name)
            except Exception:
                pass
            precision = (
                getattr(Precision, "INT8D", None)
                or getattr(Precision, "INT8", None)
                or Precision.FLOAT32
            )
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                space_type="cosine",
                precision=precision,
            )
            self.index = self.client.get_index(name=self.index_name)

        return self.index

    def _chunk(self, items: List[Dict], size: int) -> Iterable[List[Dict]]:
        for i in range(0, len(items), size):
            yield items[i:i + size]

    # -----------------------------
    # Build Vector Index
    # -----------------------------
    def build_index(self):
        clauses = self.load_clauses()
        texts = [c["text"] for c in clauses]

        print("Embedding clauses...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.asarray(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = self._ensure_index(dim)

        vectors = []
        for i, clause in enumerate(clauses):
            vectors.append(
                {
                    "id": f"{clause['document']}:{i}",
                    "vector": embeddings[i].tolist(),
                    "meta": clause,
                    "filter": {"clause_type": clause["clause_type"]},
                }
            )

        print(f"Upserting {len(vectors)} vectors to Endee...")
        for batch in self._chunk(vectors, MAX_UPSERT_BATCH):
            index.upsert(batch)

        print(f"Vector index ready with {len(vectors)} clauses")

    # -----------------------------
    # Detect Query Intent
    # -----------------------------
    def detect_query_type(self, query: str) -> str:
        q = query.lower()

        if "rent" in q:
            return "rent"
        if "deposit" in q:
            return "deposit"
        if "lock-in" in q or "lock in" in q:
            return "lock_in"
        if "terminate" in q or "termination" in q:
            return "termination"
        if "maintenance" in q:
            return "maintenance"
        if "notice" in q:
            return "notice"

        return "other"

    # -----------------------------
    # Semantic Search
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_type = self.detect_query_type(query)

        query_embedding = self.model.encode([query]).astype("float32")[0].tolist()
        index = self._get_index()

        search_kwargs = {
            "vector": query_embedding,
            "top_k": top_k,
        }

        if query_type != "other":
            search_kwargs["filter"] = [{"clause_type": {"$eq": query_type}}]

        results = index.query(**search_kwargs)

        formatted = []
        for item in results:
            meta = item.get("meta", {})
            meta = dict(meta)
            meta["similarity"] = item.get("similarity")
            formatted.append(meta)

        return formatted


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":
    store = VectorStore()

    try:
        store._get_index()
    except Exception:
        print("Index not found. Building a new index in Endee...")
        store.build_index()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = store.search(query)

        print("\nTop Results:")
        for r in results:
            print(f"- [{r.get('clause_type')}] {r.get('text', '')[:200]}...\n")
