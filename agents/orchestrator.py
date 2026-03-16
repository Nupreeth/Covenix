import os
import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.safety_agent import SafetyAgent


class Orchestrator:
    def __init__(self):
        self.safety = SafetyAgent()
        self.retriever = RetrievalAgent()
        self.reasoner = ReasoningAgent()

    def answer(self, query: str, top_k: int = 5):
        cleaned = self.safety.validate(query)
        results = self.retriever.retrieve(cleaned, top_k=top_k)
        return self.reasoner.synthesize(cleaned, results)


if __name__ == "__main__":
    orchestrator = Orchestrator()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")
        if query.lower() == "exit":
            break

        response = orchestrator.answer(query)
        print("\nAnswer:")
        print(response["answer"])
        print("\nEvidence:")
        for r in response["evidence"]:
            snippet = (r.get("text") or "")[:200]
            print(f"- [{r.get('clause_type')}] {snippet}...\n")
