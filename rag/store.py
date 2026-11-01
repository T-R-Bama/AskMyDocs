"""Base vectorielle (ChromaDB) avec embeddings multilingues locaux (fastembed / ONNX)."""

import hashlib
import uuid
from functools import lru_cache

import chromadb
from fastembed import TextEmbedding

from .loaders import chunk_text

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    """Modèle d'embeddings partagé (~250 Mo, téléchargé au premier appel)."""
    return TextEmbedding(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in get_embedder().embed(texts)]


class VectorStore:
    """Avec `path`, la base est enregistrée sur disque ; sans, elle vit en mémoire
    (une collection distincte par instance, utile pour isoler les visiteurs d'une démo)."""

    def __init__(self, path: str | None = "data/chroma"):
        if path:
            client, name = chromadb.PersistentClient(path=path), "documents"
        else:
            client, name = chromadb.EphemeralClient(), f"session-{uuid.uuid4().hex}"
        self.collection = client.get_or_create_collection(
            name, embedding_function=None, metadata={"hnsw:space": "cosine"}
        )

    def add(self, sections: list[dict]) -> int:
        """Découpe et indexe des sections. Renvoie le nombre de morceaux ajoutés."""
        ids, docs, metas = [], [], []
        for section in sections:
            for i, chunk in enumerate(chunk_text(section["text"])):
                key = f"{section['source']}|{section['page']}|{i}"
                ids.append(hashlib.sha1(key.encode()).hexdigest())
                docs.append(chunk)
                metas.append({"source": section["source"], "page": section["page"] or 0})
        if ids:
            # upsert : réindexer un même document remplace ses morceaux
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
        return len(ids)

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_embeddings=embed([query]), n_results=k)
        return [
            {"text": doc, "source": meta["source"], "page": meta["page"] or None, "score": 1 - dist}
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
