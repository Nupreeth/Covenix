import os
import json
import re
from typing import List, Dict
from collections import Counter

from docx import Document
import pdfplumber


INPUT_FOLDER = "data/raw"
OUTPUT_PATH = "data/chunks/clause_chunks.json"

os.makedirs("data/chunks", exist_ok=True)


class DocumentAgent:

    def __init__(self):
        pass

    # -----------------------------
    # TEXT EXTRACTION
    # -----------------------------
    def extract_text_from_docx(self, path: str) -> str:
        doc = Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)

    def extract_text_from_pdf(self, path: str) -> str:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text

    # -----------------------------
    # CLAUSE SPLITTING
    # -----------------------------
    def split_into_clauses(self, text: str) -> List[str]:
        # Basic legal-style splitting
        clauses = re.split(r'\n|\.\s+', text)
        cleaned = []

        for clause in clauses:
            clause = clause.strip()
            if len(clause) > 40:  # ignore tiny fragments
                cleaned.append(clause)

        return cleaned

    # -----------------------------
    # CLAUSE TYPE DETECTION
    # -----------------------------
    def detect_clause_type(self, clause_text: str) -> str:
        text = clause_text.lower()

        # RENT
        if "rent" in text:
            return "rent"

        # SECURITY DEPOSIT
        if "security deposit" in text or "deposit" in text:
            return "deposit"

        # LOCK-IN / TERM
        if (
            "lock-in" in text
            or "lock in" in text
            or "minimum period" in text
            or "non-cancellable" in text
            or "fixed term" in text
            or "initial term" in text
            or "valid for" in text
            or "duration of this agreement" in text
        ):
            return "lock_in"

        # TERMINATION
        if "terminate" in text or "termination" in text:
            return "termination"

        # MAINTENANCE
        if "maintenance" in text:
            return "maintenance"

        # NOTICE
        if "notice period" in text or "notice" in text:
            return "notice"

        return "other"

    # -----------------------------
    # PROCESS DOCUMENTS
    # -----------------------------
    def process_documents(self):
        clauses = []

        for filename in os.listdir(INPUT_FOLDER):
            path = os.path.join(INPUT_FOLDER, filename)

            print(f"Processing: {filename}")

            if filename.endswith(".docx"):
                text = self.extract_text_from_docx(path)
            elif filename.endswith(".pdf"):
                text = self.extract_text_from_pdf(path)
            else:
                continue

            split_clauses = self.split_into_clauses(text)

            for clause in split_clauses:
                clause_entry = {
                    "document": filename,
                    "text": clause,
                    "clause_type": self.detect_clause_type(clause)
                }
                clauses.append(clause_entry)

        # Save
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(clauses, f, indent=2)

        print(f"\nSaved {len(clauses)} clauses to {OUTPUT_PATH}")

        # Distribution Debug
        types = [c["clause_type"] for c in clauses]
        print("\nClause Type Distribution:")
        print(Counter(types))


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    agent = DocumentAgent()
    agent.process_documents()