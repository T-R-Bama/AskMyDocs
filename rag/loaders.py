"""Extraction du texte des documents (PDF, texte/Markdown, Word, pages web)."""

import io
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx"}


def load_file(name: str, data: bytes) -> list[dict]:
    """Renvoie une liste de sections {"text", "source", "page"} pour un fichier."""
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        reader = PdfReader(io.BytesIO(data))
        return [
            {"text": page.extract_text() or "", "source": name, "page": i + 1}
            for i, page in enumerate(reader.pages)
        ]
    if ext in {".txt", ".md", ".markdown"}:
        return [{"text": data.decode("utf-8", errors="replace"), "source": name, "page": None}]
    if ext == ".docx":
        doc = DocxDocument(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
        return [{"text": text, "source": name, "page": None}]
    raise ValueError(f"Format non supporté : {ext}")


def load_url(url: str) -> list[dict]:
    """Télécharge une page web et en extrait le texte principal."""
    resp = requests.get(url, timeout=20, headers={"User-Agent": "AskMyDocs/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = "\n".join(line.strip() for line in main.get_text("\n").splitlines() if line.strip())
    return [{"text": text, "source": url, "page": None}]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> list[str]:
    """Découpe le texte en morceaux d'environ `size` caractères qui se chevauchent."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        # Coupe de préférence en fin de phrase ou sur un espace
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return chunks
