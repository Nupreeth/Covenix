import json
import os
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_PATH = "data/chunks/clause_chunks.json"
INDEX_PATH = "storage/faiss.index"
META_PATH = "storage/metadata.json"

os.makedirs("storage", exist_ok=True)


class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    # -----------------------------
    # Load Clause Data
    # -----------------------------
    def load_clauses(self) -> List[Dict]:
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    # -----------------------------
    # Build Vector Index
    # -----------------------------
    def build_index(self):
        clauses = self.load_clauses()
        texts = [c["text"] for c in clauses]

        print("Embedding clauses...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        self.metadata = clauses

        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        print(f"Vector index built with {len(clauses)} clauses")

    # -----------------------------
    # Load Existing Index
    # -----------------------------
    def load_index(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError("Vector index not found. Run build_index() first.")

        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

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
    # Hybrid Search
    # -----------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict]:

        if self.index is None:
            self.load_index()

        query_type = self.detect_query_type(query)

        # Step 1: Filter clauses by detected type
        if query_type != "other":
            filtered_indices = [
                i for i, m in enumerate(self.metadata)
                if m["clause_type"] == query_type
            ]
        else:
            filtered_indices = list(range(len(self.metadata)))

        # Fallback if no match
        if not filtered_indices:
            filtered_indices = list(range(len(self.metadata)))

        # Step 2: Embed only filtered clauses
        filtered_texts = [self.metadata[i]["text"] for i in filtered_indices]
        filtered_embeddings = self.model.encode(filtered_texts).astype("float32")

        # Step 3: Embed query
        query_embedding = self.model.encode([query]).astype("float32")

        # Step 4: Temporary FAISS index
        temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
        temp_index.add(filtered_embeddings)

        distances, indices = temp_index.search(query_embedding, top_k)

        # Step 5: Map back to original metadata
        results = []
        for idx in indices[0]:
            original_index = filtered_indices[idx]
            results.append(self.metadata[original_index])

        return results


# -----------------------------
# CLI Test
# -----------------------------
if __name__ == "__main__":

    store = VectorStore()

    # Build only if index doesn't exist
    if not os.path.exists(INDEX_PATH):
        store.build_index()
    else:
        store.load_index()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        results = store.search(query)

        print("\nTop Results:")
        for r in results:
            print(f"- [{r['clause_type']}] {r['text'][:200]}...\n")