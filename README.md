# Covenix

Endee-powered semantic retrieval for rental agreement clauses.

## Project Overview
Legal agreements are long and dense. Covenix builds a fast semantic search layer on top of rental agreements so users can ask questions and retrieve the most relevant clauses with high precision.

## Problem Statement
Given a rental agreement, find the clauses that best answer a user's query (e.g., deposit, rent, notice period) without manual scanning.

## System Design
1. Document processing
   - Parse DOCX/PDF files.
   - Split into clause-sized chunks.
   - Assign a coarse `clause_type` label (rent, deposit, notice, etc.).
2. Embedding
   - Encode each clause with `sentence-transformers`.
3. Vector storage and retrieval (Endee)
   - Store embeddings in Endee with metadata and filter fields.
   - Use clause-type filters for targeted search.
4. Query pipeline
   - Detect query intent.
   - Query Endee with optional filters.
   - Return top matches with metadata.

```mermaid
flowchart LR
    U[User Query] --> S[SafetyAgent]
    S --> R[RetrievalAgent]
    R --> E[(Endee Vector DB)]
    E --> G[ReasoningAgent]
    G --> O[Response]
```

## Multi-Agent Flow
The system uses a simple multi-agent pipeline:
- `SafetyAgent` validates the query.
- `RetrievalAgent` fetches top-k clauses from Endee.
- `ReasoningAgent` composes a concise answer with evidence.
- `Orchestrator` coordinates the flow end-to-end.

## How Endee Is Used
Endee is the vector database for this project. Each clause is stored as a vector with:
- `meta`: original text, document name, and `clause_type`.
- `filter`: `clause_type`, used for filtered semantic search.

The search pipeline queries Endee with a query embedding and applies filters to improve precision.

## Setup
### Prerequisites
- Python 3.9+
- Docker + Docker Compose (for local Endee server)

### 1. Start Endee (local)
Use the official Endee Docker Compose configuration as a local server:

```yaml
services:
  endee-oss:
    image: endee-oss:latest
    container_name: endee-oss
    ports:
      - "8080:8080"
    ulimits:
      nofile: 100000
    logging:
      driver: "json-file"
      options:
        max-size: "200m"
        max-file: "5"
    environment:
      NDD_NUM_THREADS: 0
      NDD_AUTH_TOKEN: ""
    volumes:
      - endee-data:/data
    restart: unless-stopped
volumes:
  endee-data:
```

Then run:

```bash
docker compose up -d
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare clause chunks
Option A: Use the sample dataset:

```bash
python storage/vector_store.py
```

Option B: Use your own documents:
1. Put DOCX/PDF files in `data/raw/`.
2. Run:

```bash
python agents/document_agent.py
```

### 4. Build index and search

```bash
python storage/vector_store.py
```

The script will build the Endee index if it doesn't exist, then open an interactive search loop.

### 5. Optional API demo

```bash
uvicorn api.main:app --reload --port 8000
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d "{\"query\": \"What is the security deposit?\", \"top_k\": 3}"
```

### 6. Optional multi-agent CLI

```bash
python agents/orchestrator.py
```

## Screenshot
Add a screenshot of the CLI or API response.
Recommended path: `docs/screenshot.png` (place the file there).

![Covenix Screenshot](docs/screenshot.png)

## Results
Example output using the bundled sample data:

Query:
```
security deposit
```

Top matches (truncated):
1. `deposit` — "A security deposit equivalent to two months of rent is due at signing..."
2. `deposit` — "If the tenant vacates early, the security deposit may be adjusted..."

## Configuration
Optional environment variables:
- `ENDEE_AUTH_TOKEN`: authentication token if enabled on the server.
- `ENDEE_BASE_URL`: custom base URL (include `/api/v1` if you change the port).
- `ENDEE_INDEX_NAME`: override the index name (default: `covenix_clauses`).

## Notes
- If you change the dataset and want a clean re-ingest, delete the index in Endee or use a new `ENDEE_INDEX_NAME`.
