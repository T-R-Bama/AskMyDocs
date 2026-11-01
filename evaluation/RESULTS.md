# Résultats de l'évaluation

Généré par `python evaluation/run_eval.py`. Voir `evaluation/README.md` pour la méthode.

## Recherche (18 questions avec réponse dans les documents)

| Configuration | Bon passage en 1ʳᵉ position | Bon passage dans le top 5 | Bon document en 1ʳᵉ position | MRR |
|---|---|---|---|---|
| Embeddings `paraphrase-multilingual-MiniLM-L12-v2` | 55.6 % | 100.0 % | 94.4 % | 0.75 |
| Embeddings `paraphrase-multilingual-mpnet-base-v2` | 72.2 % | 100.0 % | 100.0 % | 0.861 |

## Génération (20 questions, dont 2 sans réponse dans les documents)

| LLM | Réponses correctes (juge) | Faits attendus présents | Refus quand l'info est absente | Cite ses sources | Temps moyen |
|---|---|---|---|---|---|
| groq · `openai/gpt-oss-120b` (`paraphrase-multilingual-MiniLM-L12-v2`) | 100.0 % | 100.0 % | 100.0 % | 100.0 % | 10.3 s |
| ollama · `gemma3:12b` (`paraphrase-multilingual-MiniLM-L12-v2`) | 95.0 % | 100.0 % | 100.0 % | 100.0 % | 10.13 s |

## Points faibles (bon passage pas en 1ʳᵉ position, ou réponse jugée incorrecte)

### groq · `openai/gpt-oss-120b`

- **Combien de temps dure la période d'essai d'un cadre ?** — bon passage au rang 2
- **J'ai droit à combien de vacances par an ?** — bon passage au rang 2
- **Combien l'entreprise verse-t-elle pour le télétravail ?** — bon passage au rang 3
- **Ai-je le droit de prendre l'avion pour un trajet de 3 heures ?** — bon passage au rang 3
- **Je viens au bureau à vélo, est-ce que j'ai droit à quelque chose ?** — bon passage au rang 3
- **Mon thermostat affiche E1, qu'est-ce que ça veut dire ?** — bon passage au rang 2
- **Combien de temps le Thermo One est-il garanti ?** — bon passage au rang 2
- **Le thermostat ne trouve pas mon Wi-Fi 5 GHz, pourquoi ?** — bon passage au rang 2

### ollama · `gemma3:12b`

- **Combien de temps dure la période d'essai d'un cadre ?** — bon passage au rang 2
- **J'ai droit à combien de vacances par an ?** — bon passage au rang 2
- **Combien l'entreprise verse-t-elle pour le télétravail ?** — bon passage au rang 3
- **Ai-je le droit de prendre l'avion pour un trajet de 3 heures ?** — bon passage au rang 3
- **Je viens au bureau à vélo, est-ce que j'ai droit à quelque chose ?** — bon passage au rang 3
- **Mon thermostat affiche E1, qu'est-ce que ça veut dire ?** — bon passage au rang 2
- **Combien de temps le Thermo One est-il garanti ?** — bon passage au rang 2 ; réponse jugée incorrecte : Le chatbot ajoute une information non présente dans la référence, ce qui n’est pas autorisé.
- **Le thermostat ne trouve pas mon Wi-Fi 5 GHz, pourquoi ?** — bon passage au rang 2

