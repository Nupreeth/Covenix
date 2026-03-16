from storage.vector_store import VectorStore


class RetrievalAgent:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self._ensure_ready()

    def _ensure_ready(self):
        try:
            self.store._get_index()
        except Exception:
            self.store.build_index()

    def retrieve(self, query: str, top_k: int = 5):
        return self.store.search(query, top_k)
