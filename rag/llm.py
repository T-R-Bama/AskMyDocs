"""Génération des réponses à partir des passages retrouvés.

Trois fournisseurs possibles :
- Ollama : modèle local, gratuit ;
- Groq : API dans le cloud, avec une offre gratuite (utilisée par la démo en ligne) ;
- Claude : API Anthropic, payante à l'usage.
"""

from collections.abc import Iterator

import anthropic
import groq
import ollama

CLAUDE_MODEL = "claude-opus-5-5"
# Modèles Groq proposés en premier, par ordre de préférence (le catalogue évolue)
GROQ_PREFERRED_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
OLLAMA_CONTEXT = 8192  # taille de contexte demandée à Ollama (tokens)

SYSTEM_PROMPT = """Tu es un assistant qui répond aux questions en s'appuyant sur les documents fournis par l'utilisateur.

Les passages pertinents sont donnés dans des balises <document>, chacun avec un numéro. Règles :
- Réponds en t'appuyant sur ces passages et cite tes sources avec leur numéro, par exemple [1] ou [2, 3].
- Si les passages ne contiennent pas l'information, dis-le clairement plutôt que d'inventer. Tu peux alors proposer une réponse générale en précisant qu'elle ne vient pas des documents.
- Réponds dans la langue de la question."""


def format_context(passages: list[dict]) -> str:
    parts = []
    for i, p in enumerate(passages, start=1):
        loc = f", page {p['page']}" if p["page"] else ""
        parts.append(f'<document index="{i}" source="{p["source"]}{loc}">\n{p["text"]}\n</document>')
    return "\n\n".join(parts)


def build_messages(history: list[dict], question: str, passages: list[dict]) -> list[dict]:
    """`history` contient les tours précédents ({"role", "content"} en texte simple) ;
    les passages ne sont joints qu'à la question courante."""
    context = format_context(passages) if passages else "(aucun passage trouvé)"
    return history + [
        {"role": "user", "content": f"<documents>\n{context}\n</documents>\n\nQuestion : {question}"}
    ]


def list_ollama_models() -> list[str]:
    """Modèles installés localement ; lève ConnectionError si Ollama ne tourne pas."""
    return [m.model for m in ollama.list().models]


def stream_ollama(model: str, history: list[dict], question: str, passages: list[dict]) -> Iterator[str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_messages(history, question, passages)
    for chunk in ollama.chat(
        model=model, messages=messages, stream=True, options={"num_ctx": OLLAMA_CONTEXT}
    ):
        yield chunk.message.content or ""


def list_groq_models(client: groq.Groq) -> list[str]:
    """Modèles de chat disponibles sur Groq, les modèles préférés en premier."""
    # Modèles audio, de modération, ou spécialisés dans une autre langue (allam : arabe)
    skip = ("whisper", "tts", "guard", "playai", "orpheus", "compound", "allam")
    models = sorted(
        m.id for m in client.models.list().data
        if getattr(m, "active", True) and not any(s in m.id.lower() for s in skip)
    )
    preferred = [m for m in GROQ_PREFERRED_MODELS if m in models]
    return preferred + [m for m in models if m not in preferred]


def stream_groq(
    client: groq.Groq, model: str, history: list[dict], question: str, passages: list[dict]
) -> Iterator[str]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + build_messages(history, question, passages)
    for chunk in client.chat.completions.create(model=model, messages=messages, stream=True):
        if chunk.choices and chunk.choices[0].delta.content:
            # Les modèles gpt-oss citent parfois avec des crochets 【1】 : on les uniformise
            yield chunk.choices[0].delta.content.replace("【", "[").replace("】", "]")


def stream_claude(
    client: anthropic.Anthropic, history: list[dict], question: str, passages: list[dict]
) -> Iterator[str]:
    with client.beta.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=build_messages(history, question, passages),
        output_config={"effort": "medium"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n_La demande a été refusée par le modèle._"
    elif final.stop_reason == "max_tokens":
        yield "\n\n_(réponse tronquée : limite de longueur atteinte)_"
