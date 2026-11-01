"""Évaluation du pipeline RAG sur les documents d'exemple.

Mesure deux choses pour chaque question de `questions.json` :
- la recherche : le passage qui contient la réponse est-il retrouvé, et à quel rang ?
- la génération : la réponse est-elle correcte ? Vérifié de deux façons, par la présence
  des faits attendus (déterministe) et par un LLM juge qui compare à la réponse de référence.

Exemples :
    python evaluation/run_eval.py --retrieval-only
    python evaluation/run_eval.py --retrieval-only --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
    python evaluation/run_eval.py --provider groq --model openai/gpt-oss-120b
    python evaluation/run_eval.py --provider ollama --model gemma3:12b
"""

import argparse
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import groq  # noqa: E402

import rag.store  # noqa: E402
from rag.llm import stream_groq, stream_ollama  # noqa: E402
from rag.loaders import load_file  # noqa: E402
from rag.store import VectorStore  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
TOP_K = 5
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = """Tu évalues la réponse d'un chatbot qui répond à partir de documents.

Question : {question}
Réponse de référence : {reference}
Réponse du chatbot : {answer}

La réponse du chatbot est-elle correcte ? Elle doit contenir la partie « Essentiel » de la
référence et ne rien affirmer de contradictoire. La partie « Facultatif » peut être omise ;
la reformulation, les détails supplémentaires exacts et le style ne comptent pas. Si la référence indique que l'information n'est pas
dans les documents, la réponse est correcte seulement si le chatbot le dit sans inventer.

Réponds uniquement en JSON : {{"correct": true ou false, "raison": "une phrase"}}"""


def normalize(text: str) -> str:
    """Minuscules, sans espaces (y compris insécables) ni apostrophes typographiques."""
    return re.sub(r"[\s  ]+", "", text.lower()).replace("’", "'")


def groq_key() -> str:
    key = os.environ.get("GROQ_API_KEY", "")
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if not key and secrets.exists():
        key = tomllib.loads(secrets.read_text()).get("GROQ_API_KEY", "")
    return key


def build_store() -> VectorStore:
    store = VectorStore(path=None)
    for path in sorted((ROOT / "exemples").iterdir()):
        if path.suffix.lower() in {".pdf", ".md", ".docx", ".txt"}:
            store.add(load_file(path.name, path.read_bytes()))
    return store


def evidence_rank(passages: list[dict], evidence: str) -> int | None:
    """Rang (à partir de 1) du premier passage qui contient la preuve attendue."""
    target = normalize(evidence)
    for rank, p in enumerate(passages, start=1):
        if target in normalize(p["text"]):
            return rank
    return None


def facts_ok(answer: str, facts: list[list[str]]) -> bool:
    """Chaque groupe de faits doit être présent (au moins une des variantes du groupe)."""
    text = normalize(answer)
    return all(any(normalize(v) in text for v in group) for group in facts)


def judge(client: groq.Groq, question: dict, answer: str) -> tuple[bool, str]:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(answer=answer, **question)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    verdict = json.loads(resp.choices[0].message.content)
    return bool(verdict.get("correct")), verdict.get("raison", "")


def pct(values: list[bool]) -> float:
    return round(100 * sum(values) / len(values), 1) if values else 0.0


def summarize(rows: list[dict], label: str, retrieval_only: bool) -> dict:
    answerable = [r for r in rows if r["evidence"]]
    ranks = [r["rank"] for r in answerable]
    summary = {
        "label": label,
        "questions": len(rows),
        "retrieval": {
            "hit@1": pct([r == 1 for r in ranks]),
            "hit@5": pct([r is not None for r in ranks]),
            "source@1": pct([r["top_source"] == r["source"] for r in answerable]),
            "mrr": round(sum(1 / r for r in ranks if r) / len(ranks), 3),
        },
    }
    if not retrieval_only:
        judged = [r["judge_correct"] for r in rows if r["judge_correct"] is not None]
        summary["generation"] = {
            "facts_ok": pct([r["facts_ok"] for r in answerable]),
            "judge_correct": pct(judged) if judged else None,
            "abstention_ok": pct([r["judge_correct"] for r in rows if not r["evidence"]])
            if judged else None,
            "cites_source": pct([bool(re.search(r"\[\d", r["answer"])) for r in answerable]),
            "avg_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        }
    return summary


def write_report() -> None:
    """Regroupe tous les résultats enregistrés dans evaluation/RESULTS.md."""
    runs = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    lines = [
        "# Résultats de l'évaluation",
        "",
        "Généré par `python evaluation/run_eval.py`. Voir `evaluation/README.md` pour la méthode.",
        "",
        "## Recherche (18 questions avec réponse dans les documents)",
        "",
        "| Configuration | Bon passage en 1ʳᵉ position | Bon passage dans le top 5 | Bon document en 1ʳᵉ position | MRR |",
        "|---|---|---|---|---|",
    ]
    seen = set()
    for run in runs:
        r = run["summary"]["retrieval"]
        if run["embedding_model"] in seen:  # la recherche ne dépend pas du LLM
            continue
        seen.add(run["embedding_model"])
        lines.append(f"| Embeddings `{run['embedding_model'].split('/')[-1]}` | {r['hit@1']} % | {r['hit@5']} % | {r['source@1']} % | {r['mrr']} |")
    lines += [
        "",
        "## Génération (20 questions, dont 2 sans réponse dans les documents)",
        "",
        "| LLM | Réponses correctes (juge) | Faits attendus présents | Refus quand l'info est absente | Cite ses sources | Temps moyen |",
        "|---|---|---|---|---|---|",
    ]
    for run in runs:
        g = run["summary"].get("generation")
        if g:
            judge_cell = f"{g['judge_correct']} %" if g["judge_correct"] is not None else "—"
            abst_cell = f"{g['abstention_ok']} %" if g["abstention_ok"] is not None else "—"
            lines.append(
                f"| {run['summary']['label']} (`{run['embedding_model'].split('/')[-1]}`) | {judge_cell} | {g['facts_ok']} % | "
                f"{abst_cell} | {g['cites_source']} % | {g['avg_seconds']} s |"
            )
    lines += ["", "## Points faibles (bon passage pas en 1ʳᵉ position, ou réponse jugée incorrecte)", ""]
    for run in runs:
        errors = [r for r in run["rows"] if r.get("judge_correct") is False or (r["evidence"] and r["rank"] != 1)]
        if run["summary"].get("generation") and errors:
            lines.append(f"### {run['summary']['label']}")
            lines.append("")
            for r in errors:
                problems = []
                if r["evidence"] and r["rank"] != 1:
                    problems.append(f"bon passage au rang {r['rank'] or 'non trouvé'}")
                if r.get("judge_correct") is False:
                    problems.append(f"réponse jugée incorrecte : {r['judge_reason']}")
                lines.append(f"- **{r['question']}** — {' ; '.join(problems)}")
            lines.append("")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["groq", "ollama"], default="groq")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--retrieval-only", action="store_true", help="évaluer seulement la recherche (gratuit, rapide)")
    parser.add_argument("--no-judge", action="store_true", help="ne pas utiliser le LLM juge")
    parser.add_argument("--embedding-model", default=rag.store.EMBEDDING_MODEL, help="modèle d'embeddings fastembed à tester")
    args = parser.parse_args()
    rag.store.EMBEDDING_MODEL = args.embedding_model

    questions = json.loads((HERE / "questions.json").read_text())
    store = build_store()
    key = groq_key()
    # max_retries élevé : l'offre gratuite de Groq limite le nombre de tokens par minute
    client = groq.Groq(api_key=key, max_retries=8) if key else None
    use_judge = not args.retrieval_only and not args.no_judge
    if (use_judge or args.provider == "groq") and not args.retrieval_only and not client:
        sys.exit("Clé Groq introuvable : définis GROQ_API_KEY ou .streamlit/secrets.toml.")

    rows = []
    for q in questions:
        passages = store.search(q["question"], k=TOP_K)
        row = {
            **q,
            "rank": evidence_rank(passages, q["evidence"]) if q["evidence"] else None,
            "top_source": passages[0]["source"] if passages else None,
            "answer": None, "facts_ok": None, "judge_correct": None, "judge_reason": None, "seconds": 0,
        }
        if not args.retrieval_only:
            start = time.time()
            if args.provider == "groq":
                answer = "".join(stream_groq(client, args.model, [], q["question"], passages))
            else:
                answer = "".join(stream_ollama(args.model, [], q["question"], passages))
            row["seconds"] = round(time.time() - start, 2)
            row["answer"] = answer
            if q["facts"]:
                row["facts_ok"] = facts_ok(answer, q["facts"])
            if use_judge:
                row["judge_correct"], row["judge_reason"] = judge(client, q, answer)
        rows.append(row)

        status = f"rang {row['rank']}" if q["evidence"] else "hors sujet"
        if row["answer"] is not None:
            status += f" · faits {'✓' if row['facts_ok'] else '✗' if row['facts_ok'] is False else '–'}"
            status += f" · juge {'✓' if row['judge_correct'] else '✗' if row['judge_correct'] is False else '–'}"
        print(f"{q['id']:<22} {status}")

    emb = args.embedding_model.split("/")[-1]
    label = f"Recherche seule ({emb})" if args.retrieval_only else f"{args.provider} · `{args.model}`"
    summary = summarize(rows, label, args.retrieval_only)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"retrieval-{emb}" if args.retrieval_only else f"{args.provider}-{args.model}-{emb}"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    (RESULTS_DIR / f"{slug}.json").write_text(json.dumps(
        {"summary": summary, "embedding_model": args.embedding_model, "rows": rows},
        ensure_ascii=False, indent=2,
    ))
    write_report()
    print(f"\nRésultats enregistrés dans evaluation/results/{slug}.json et evaluation/RESULTS.md")


if __name__ == "__main__":
    main()
