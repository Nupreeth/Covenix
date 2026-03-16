from typing import List, Dict


class ReasoningAgent:
    def synthesize(self, query: str, results: List[Dict]):
        if not results:
            return {
                "query": query,
                "answer": "No relevant clauses found.",
                "evidence": [],
            }

        top = results[0]
        answer = top.get("text", "No clause text available.")
        evidence = [
            {
                "clause_type": r.get("clause_type"),
                "document": r.get("document"),
                "text": r.get("text"),
                "similarity": r.get("similarity"),
            }
            for r in results
        ]

        return {
            "query": query,
            "answer": answer,
            "evidence": evidence,
        }
