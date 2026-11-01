# Résultats de l'évaluation

Généré par `python evaluation/run_eval.py`. Voir `evaluation/README.md` pour la méthode.

## Recherche (18 questions avec réponse dans les documents)

| Configuration de la recherche | Bon passage en 1ʳᵉ position | Bon passage dans le top 5 | Bon document en 1ʳᵉ position | MRR |
|---|---|---|---|---|
| `MiniLM-L12-v2-300c-60o-semantique` | 38.9 % | 100.0 % | 100.0 % | 0.639 |
| `MiniLM-L12-v2-400c-80o-semantique` | 50.0 % | 94.4 % | 94.4 % | 0.669 |
| `MiniLM-L12-v2-500c-100o-semantique` | 44.4 % | 100.0 % | 100.0 % | 0.69 |
| `MiniLM-L12-v2-600c-120o-semantique` | 61.1 % | 100.0 % | 94.4 % | 0.738 |
| `MiniLM-L12-v2-1000c-200o-semantique` | 55.6 % | 100.0 % | 94.4 % | 0.75 |
| `MiniLM-L12-v2-300c-60o-hybride` | 61.1 % | 100.0 % | 100.0 % | 0.796 |
| `MiniLM-L12-v2-400c-80o-hybride` | 72.2 % | 100.0 % | 100.0 % | 0.838 |
| `mpnet-base-v2-1000c-200o-semantique` | 72.2 % | 100.0 % | 100.0 % | 0.861 |
| `MiniLM-L12-v2-1000c-200o-hybride` | 77.8 % | 100.0 % | 100.0 % | 0.88 |
| `mpnet-base-v2-1000c-200o-hybride` | 77.8 % | 100.0 % | 100.0 % | 0.889 |

## Génération (20 questions, dont 2 sans réponse dans les documents)

| LLM | Réponses correctes (juge) | Faits attendus présents | Refus quand l'info est absente | Cite ses sources | Temps moyen |
|---|---|---|---|---|---|
| groq · `openai/gpt-oss-120b` · `MiniLM-L12-v2-1000c-200o-hybride` | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 9.36 s |
| ollama · `gemma3:12b` · `MiniLM-L12-v2-1000c-200o-hybride` | 95.0 % | 94.4 % | 100.0 % | 100.0 % | 9.22 s |

## Points faibles (bon passage pas en 1ʳᵉ position, ou réponse jugée incorrecte)

### groq · `openai/gpt-oss-120b` · `MiniLM-L12-v2-1000c-200o-hybride`

- **Combien de temps dure la période d'essai d'un cadre ?** — bon passage au rang 2
- **Combien l'entreprise verse-t-elle pour le télétravail ?** — bon passage au rang 3
- **Combien de temps le Thermo One est-il garanti ?** — bon passage au rang 2
- **Le thermostat ne trouve pas mon Wi-Fi 5 GHz, pourquoi ?** — bon passage au rang 2

### ollama · `gemma3:12b` · `MiniLM-L12-v2-1000c-200o-hybride`

- **Combien de temps dure la période d'essai d'un cadre ?** — bon passage au rang 2
- **Combien l'entreprise verse-t-elle pour le télétravail ?** — bon passage au rang 3
- **Combien de temps le Thermo One est-il garanti ?** — bon passage au rang 2 ; réponse jugée incorrecte : Le chatbot ajoute une information non présente dans la référence, ce qui n’est pas autorisé.
- **Le thermostat ne trouve pas mon Wi-Fi 5 GHz, pourquoi ?** — bon passage au rang 2

