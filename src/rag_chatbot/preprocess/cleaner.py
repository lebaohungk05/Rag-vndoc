import re


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()

