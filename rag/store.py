"""Base vectorielle (ChromaDB) avec embeddings multilingues locaux (fastembed / ONNX)
et recherche hybride optionnelle (sens + mots-clés)."""

import hashlib
import math
import re
import unicodedata
import uuid
from collections import Counter
from functools import lru_cache

import chromadb
from fastembed import TextEmbedding

from .loaders import chunk_text

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Taille des passages en caractères. Le modèle d'embeddings ne lit que les 128 premiers tokens
# (~400 caractères) : la fin des passages longs n'est vue que par BM25, qui lit tout le texte.
# Des passages plus courts ont pourtant donné de moins bons résultats (voir evaluation/).
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
HYBRID = True
DENSE_CANDIDATES = 50  # passages retenus par la recherche sémantique avant fusion
RRF_K = 60  # constante usuelle de la fusion par rangs réciproques

STOPWORDS = set(
    "a au aux avec ce ces cette dans de des du elle en est et il ils je la le les leur lui ma "
    "mais me mes mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sont sur ta "
    "te tes ton tu un une vos votre vous y ai as avez avons etre faut peut puis quel quelle "
    "quels quelles combien comment pourquoi quoi est-ce chez".split()
)


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    """Modèle d'embeddings partagé (~250 Mo, téléchargé au premier appel)."""
    return TextEmbedding(EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    return [vec.tolist() for vec in get_embedder().embed(texts)]


def tokenize(text: str) -> list[str]:
    """Mots en minuscules, sans accents ni mots vides, avec un pluriel simplifié."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+(?:[,.][0-9]+)?", text)
    return [w[:-1] if len(w) > 3 and w[-1] in "sx" else w for w in words if w not in STOPWORDS]


def bm25_ranking(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices des documents classés par score BM25 décroissant (ceux sans mot commun exclus)."""
    tokenized = [tokenize(d) for d in docs]
    avg_len = sum(map(len, tokenized)) / max(len(tokenized), 1)
    df = Counter(t for doc in tokenized for t in set(doc))
    n = len(docs)
    scores = []
    for i, doc in enumerate(tokenized):
        tf = Counter(doc)
        score = 0.0
        for term in set(tokenize(query)):
            if term in tf:
                idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
                score += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(doc) / avg_len))
        if score > 0:
            scores.append((score, i))
    return [i for _, i in sorted(scores, reverse=True)]


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
        for source in {s["source"] for s in sections}:
            # Réindexer un document remplace tous ses anciens morceaux
            self.delete_source(source)
        for section in sections:
            for i, chunk in enumerate(chunk_text(section["text"], CHUNK_SIZE, CHUNK_OVERLAP)):
                key = f"{section['source']}|{section['page']}|{i}"
                ids.append(hashlib.sha1(key.encode()).hexdigest())
                docs.append(chunk)
                metas.append({"source": section["source"], "page": section["page"] or 0})
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
        return len(ids)

    def search(self, query: str, k: int = 5) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_embeddings=embed([query]), n_results=min(DENSE_CANDIDATES, count))
        dense = {
            id_: {"text": doc, "source": meta["source"], "page": meta["page"] or None, "score": 1 - dist}
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        }
        if not HYBRID:
            return list(dense.values())[:k]

        # Recherche hybride : on fusionne le classement sémantique et le classement BM25
        # par rangs réciproques (RRF), ce qui récupère les correspondances exactes de mots
        # (« E1 », « vélo », « garanti ») que les embeddings ratent parfois.
        everything = self.collection.get(include=["documents", "metadatas"])
        fused = Counter()
        for rank, id_ in enumerate(dense):
            fused[id_] += 1 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranking(query, everything["documents"])):
            fused[everything["ids"][idx]] += 1 / (RRF_K + rank + 1)
        by_id = {
            id_: {"text": doc, "source": meta["source"], "page": meta["page"] or None, "score": 0.0}
            for id_, doc, meta in zip(everything["ids"], everything["documents"], everything["metadatas"])
        }
        # Le score affiché reste la similarité sémantique, plus parlante qu'un score RRF
        return [dense.get(id_, by_id[id_]) for id_, _ in fused.most_common(k)]

    def sources(self) -> list[str]:
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})
