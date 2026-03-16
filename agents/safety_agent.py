class SafetyAgent:
    def validate(self, query: str):
        if not isinstance(query, str):
            raise ValueError("Query must be a string.")
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty.")
        if len(cleaned) > 500:
            raise ValueError("Query too long (max 500 characters).")
        return cleaned
