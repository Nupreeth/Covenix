from fastapi import FastAPI
from pydantic import BaseModel

from agents.orchestrator import Orchestrator


app = FastAPI(title="Covenix")
orchestrator = Orchestrator()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(req: SearchRequest):
    response = orchestrator.answer(req.query, req.top_k)
    return response
