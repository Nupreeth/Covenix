from storage.vector_store import VectorStore


class RetrievalAgent:
    def __init__(self, store: VectorStore | None = None):
        self.store = store or VectorStore()
        self._ensure_ready()

    def _ensure_ready(self):
        try:
            index = self.store._get_index()
            test_vec = (
                self.store.model.encode(["rent"]).astype("float32")[0].tolist()
            )
            results = index.query(vector=test_vec, top_k=1)
            if not results:
                self.store.build_index()
        except Exception:
            self.store.build_index()

    def retrieve(self, query: str, top_k: int = 5):
        return self.store.search(query, top_k)
