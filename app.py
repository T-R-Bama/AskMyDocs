"""Interface Streamlit d'AskMyDocs, chatbot RAG. Lancer avec : streamlit run app.py"""

import os
from pathlib import Path

import anthropic
import groq
import ollama
import streamlit as st

from rag.llm import (
    list_groq_models,
    list_ollama_models,
    stream_claude,
    stream_groq,
    stream_ollama,
)
from rag.loaders import SUPPORTED_EXTENSIONS, load_file, load_url
from rag.store import VectorStore, get_embedder

TOP_K = 5
HISTORY_TURNS = 6  # nombre de messages précédents renvoyés au modèle
EXAMPLES_DIR = Path(__file__).parent / "exemples"
SUGGESTIONS = [
    "Combien de jours de télétravail sont autorisés chez NovaTech ?",
    "Comment réinitialiser le thermostat Thermo One ?",
    "Quel est le plafond de remboursement d'un repas avec un client ?",
    "Qu'est-ce que le RAG et quelles sont ses limites ?",
]

GROQ, OLLAMA, CLAUDE = "Groq", "Ollama", "Claude"
PROVIDER_HELP = {
    GROQ: "API cloud avec offre gratuite",
    OLLAMA: "Modèle local, 100 % gratuit et privé",
    CLAUDE: "API Anthropic, payante à l'usage",
}

st.set_page_config(page_title="AskMyDocs", page_icon=":material/menu_book:", layout="wide")


def setting(name: str) -> str:
    """Lit un réglage dans .streamlit/secrets.toml, sinon dans les variables d'environnement."""
    try:
        value = st.secrets.get(name)
    except Exception:  # aucun fichier secrets.toml
        value = None
    return str(value or os.environ.get(name, "")).strip()


DEMO_MODE = setting("DEMO_MODE").lower() in {"1", "true", "yes"}


def load_examples(store: VectorStore) -> int:
    total = 0
    for path in sorted(EXAMPLES_DIR.iterdir()):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            total += store.add(load_file(path.name, path.read_bytes()))
    return total


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings…")
def get_persistent_store() -> VectorStore:
    get_embedder()
    return VectorStore()


def get_store() -> VectorStore:
    """En démo, chaque visiteur a sa propre base en mémoire, préremplie avec les exemples."""
    if not DEMO_MODE:
        return get_persistent_store()
    if "store" not in st.session_state:
        with st.spinner("Préparation des documents d'exemple…"):
            get_embedder()
            store = VectorStore(path=None)
            load_examples(store)
        st.session_state.store = store
    return st.session_state.store


@st.cache_resource
def get_groq_client(api_key: str) -> groq.Groq:
    return groq.Groq(api_key=api_key)


@st.cache_resource
def get_claude_client(api_key: str) -> anthropic.Anthropic:
    # Clé vide : le SDK la cherche dans l'environnement (ANTHROPIC_API_KEY)
    return anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()


@st.cache_data(ttl=3600, show_spinner=False)
def groq_models(api_key: str) -> list[str]:
    return list_groq_models(get_groq_client(api_key))


def friendly_error(e: Exception) -> str:
    """Traduit les erreurs des fournisseurs en messages compréhensibles."""
    if isinstance(e, (groq.AuthenticationError, anthropic.AuthenticationError)):
        return "Clé API invalide. Vérifie la clé saisie dans la barre latérale."
    if isinstance(e, (groq.RateLimitError, anthropic.RateLimitError)):
        return "Limite de requêtes atteinte (offre gratuite). Réessaie dans une minute."
    if isinstance(e, (groq.APIConnectionError, anthropic.APIConnectionError)):
        return "Impossible de joindre le service. Vérifie ta connexion internet."
    if isinstance(e, (groq.APIStatusError, anthropic.APIStatusError)):
        return f"Erreur du service ({e.status_code}) : {e.message}"
    if isinstance(e, ConnectionError):
        return "Ollama ne répond pas. Lance l'application Ollama puis réessaie."
    if isinstance(e, ollama.ResponseError):
        return f"Erreur d'Ollama : {e.error}"
    return f"Erreur inattendue : {e}"


store = get_store()

# --- Barre latérale : choix du modèle et gestion des documents ---
with st.sidebar:
    st.header(":material/smart_toy: Modèle")
    providers = [GROQ, CLAUDE] if DEMO_MODE else [OLLAMA, GROQ, CLAUDE]
    default = GROQ if DEMO_MODE or setting("GROQ_API_KEY") else OLLAMA
    provider = st.segmented_control(
        "Fournisseur", providers, default=default, required=True,
        label_visibility="collapsed", width="stretch",
    )
    st.caption(PROVIDER_HELP[provider])

    model, api_key = None, ""
    if provider == OLLAMA:
        try:
            models = list_ollama_models()
            if models:
                model = st.selectbox("Modèle installé", models)
            else:
                st.info("Aucun modèle installé. Dans un terminal : `ollama pull gemma3:12b`")
        except Exception:
            st.error("Ollama ne répond pas. Lance l'application Ollama puis recharge la page.")
    elif provider == GROQ:
        api_key = setting("GROQ_API_KEY")
        if not api_key:
            api_key = st.text_input(
                "Clé API Groq",
                type="password",
                placeholder="gsk_…",
                help="Clé gratuite sur console.groq.com/keys. Elle n'est pas enregistrée.",
            ).strip()
        if api_key:
            try:
                model = st.selectbox("Modèle", groq_models(api_key))
            except Exception as e:
                st.error(friendly_error(e))
    else:
        api_key = st.text_input(
            "Clé API Anthropic",
            type="password",
            placeholder="sk-ant-api…",
            help="Crée une clé sur console.anthropic.com/settings/keys. Elle n'est pas enregistrée.",
        ).strip()

    st.header(":material/folder_open: Documents")
    files = st.file_uploader(
        "Ajouter des fichiers",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if files and st.button("Indexer les fichiers", type="primary", icon=":material/upload:"):
        for f in files:
            try:
                with st.spinner(f"Indexation de {f.name}…"):
                    n = store.add(load_file(f.name, f.getvalue()))
                st.success(f"{f.name} : {n} morceaux indexés")
            except Exception as e:
                st.error(f"{f.name} : {e}")

    url = st.text_input("Ajouter une page web", placeholder="https://…")
    if url and st.button("Indexer la page", icon=":material/link:"):
        try:
            with st.spinner("Téléchargement…"):
                n = store.add(load_url(url))
            st.success(f"{n} morceaux indexés")
        except Exception as e:
            st.error(f"Impossible de charger la page : {e}")

    sources = store.sources()
    st.subheader(f"Indexés ({len(sources)})")
    for src in sources:
        with st.container(horizontal=True, vertical_alignment="center"):
            st.caption(src, width="stretch")
            if st.button("", key=f"del-{src}", icon=":material/delete:", type="tertiary",
                         help="Retirer ce document"):
                store.delete_source(src)
                st.rerun()
    if not DEMO_MODE and st.button("Charger les documents d'exemple", icon=":material/library_add:"):
        with st.spinner("Indexation des exemples…"):
            load_examples(store)
        st.rerun()

    st.divider()
    if st.button("Effacer la conversation", icon=":material/restart_alt:"):
        st.session_state.messages = []
        st.rerun()

# --- Zone de chat ---
st.title(":material/menu_book: AskMyDocs")
st.caption(
    "Pose des questions sur tes documents : le chatbot retrouve les passages pertinents "
    "et répond en citant ses sources."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def show_sources(passages: list[dict]) -> None:
    if not passages:
        return
    with st.expander(f"Sources ({len(passages)})", icon=":material/description:"):
        for i, p in enumerate(passages, start=1):
            loc = f" — page {p['page']}" if p["page"] else ""
            st.markdown(f"**[{i}] {p['source']}{loc}** · pertinence {p['score']:.2f}")
            st.caption(p["text"][:500] + ("…" if len(p["text"]) > 500 else ""))


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

def pick_suggestion() -> None:
    # On garde la question choisie puis on désélectionne la pastille
    st.session_state.pending_question = st.session_state.suggestion
    st.session_state.suggestion = None


question = st.chat_input("Ta question…") or st.session_state.pop("pending_question", None)

# Questions suggérées tant que la conversation est vide et que les exemples sont indexés
if not st.session_state.messages and not question and "guide_salarie_novatech.pdf" in sources:
    st.markdown("**Essaie une question sur les documents d'exemple :**")
    st.pills(
        "Suggestions", SUGGESTIONS, key="suggestion",
        on_change=pick_suggestion, label_visibility="collapsed",
    )

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    passages = store.search(question, k=TOP_K)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1][-HISTORY_TURNS:]
    ]

    with st.chat_message("assistant"):
        try:
            if not model and provider != CLAUDE:
                raise RuntimeError("Choisis un modèle dans la barre latérale (clé API ou Ollama manquant).")
            if provider == OLLAMA:
                stream = stream_ollama(model, history, question, passages)
            elif provider == GROQ:
                stream = stream_groq(get_groq_client(api_key), model, history, question, passages)
            else:
                stream = stream_claude(get_claude_client(api_key), history, question, passages)
            answer = st.write_stream(stream)
        except RuntimeError as e:
            answer = str(e)
            st.warning(answer, icon=":material/warning:")
        # Sans aucun identifiant, le SDK Anthropic lève une TypeError avant même d'appeler l'API
        except TypeError:
            answer = "Clé API absente. Colle ta clé dans la barre latérale."
            st.error(answer, icon=":material/key:")
        except Exception as e:
            answer = friendly_error(e)
            st.error(answer, icon=":material/error:")
        show_sources(passages)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": passages})
    st.rerun()
