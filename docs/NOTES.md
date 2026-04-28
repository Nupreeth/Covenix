# Covenix Notes

This folder is intentionally lightweight: quick notes, experiments, and checklists that
help keep momentum without changing runtime behavior.

## Query Set (Smoke Test)

Use these to validate retrieval quality after any changes to chunking, embeddings, or
index configuration.

- Deposit: "What is the security deposit and when is it returned?"
- Notice: "What is the notice period for termination?"
- Rent: "When is rent due and what happens if it's late?"
- Maintenance: "Who pays for repairs and maintenance?"
- Lock-in: "Is there a lock-in period? What are the penalties?"
- Access/Entry: "When can the landlord enter the premises?"

## Evaluation Checklist

- `top_k=3/5/10`: does at least one returned clause directly answer the question?
- Are clause types consistent (`deposit`, `notice`, `termination`, etc.)?
- Do returned snippets contain enough context (avoid mid-sentence cuts)?
- Do queries with synonyms work (e.g., "advance" vs "deposit")?

## Future Work (Safe Ideas)

- Add a small offline evaluation script to compute recall@k on a curated query set.
- Improve `detect_query_type` with synonyms + phrase matches.
- Add metadata fields for `section_heading` and `page_number` during parsing.
- Add a lightweight observability log format (query, latency, clause types) gated by an env var.
