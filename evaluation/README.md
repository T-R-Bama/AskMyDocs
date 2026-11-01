# Évaluation du RAG

Un chatbot RAG peut échouer à deux endroits : la **recherche** (le bon passage n'est pas retrouvé) et la **génération** (le passage est là, mais la réponse est fausse, incomplète ou inventée). Cette évaluation mesure les deux séparément, pour savoir où améliorer le système.

Résultats détaillés : [RESULTS.md](RESULTS.md).

## Le jeu de questions

[`questions.json`](questions.json) contient **20 questions** sur les trois documents d'exemple :

- **18 questions avec réponse dans les documents**, formulées comme le ferait un utilisateur, souvent avec d'autres mots que le document (« vacances » au lieu de « congés payés », « travailler de chez moi » au lieu de « télétravail »), et parfois avec un petit raisonnement à faire (« une nuit d'hôtel à Lyon » alors que le document parle des « autres villes » que Paris).
- **2 questions sans réponse dans les documents** (salaire, prix du produit), pour vérifier que le chatbot dit qu'il ne sait pas au lieu d'inventer.

Chaque question est accompagnée de :

| Champ | Rôle |
|---|---|
| `evidence` | Extrait exact du document qui contient la réponse : sert à vérifier que la recherche a retrouvé le bon passage |
| `reference` | Réponse attendue, séparée en partie « Essentiel » (obligatoire) et « Facultatif » |
| `facts` | Éléments qui doivent apparaître dans la réponse (avec variantes, comparés sans espaces ni majuscules) |

## Les métriques

**Recherche** (18 questions avec réponse) :
- **Bon passage en 1ʳᵉ position** (*hit@1*) : le passage qui contient `evidence` arrive en tête.
- **Bon passage dans le top 5** (*hit@5*) : il fait partie des 5 passages envoyés au LLM. C'est la condition pour que le LLM puisse répondre.
- **Bon document en 1ʳᵉ position** : le premier passage vient du bon fichier.
- **MRR** (*Mean Reciprocal Rank*) : moyenne de 1/rang du bon passage (1 = toujours premier).

**Génération** (20 questions) :
- **Réponses correctes (juge)** : un LLM juge (`openai/gpt-oss-120b` via Groq, température 0) compare la réponse à la référence et rend un verdict JSON avec sa raison.
- **Faits attendus présents** : vérification déterministe, sans LLM, que les éléments de `facts` sont dans la réponse.
- **Refus quand l'info est absente** : part des questions hors sujet où le chatbot dit ne pas savoir.
- **Cite ses sources** : la réponse contient au moins une référence `[n]`.

## Lancer l'évaluation

```bash
# Recherche seule : gratuit, quelques secondes, sans clé API
python evaluation/run_eval.py --retrieval-only

# Pipeline complet avec Groq (clé dans .streamlit/secrets.toml ou GROQ_API_KEY)
python evaluation/run_eval.py --provider groq --model openai/gpt-oss-120b

# Pipeline complet avec un modèle local Ollama (le juge reste sur Groq)
python evaluation/run_eval.py --provider ollama --model gemma3:12b

# Tester un autre modèle d'embeddings
python evaluation/run_eval.py --retrieval-only --embedding-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

Chaque exécution enregistre le détail dans `results/` et régénère `RESULTS.md`.

## Ce que l'évaluation a appris

1. **La recherche est le maillon à améliorer.** Le bon passage est toujours dans les 5 envoyés au LLM, mais il n'arrive en tête que dans 56 % des cas avec le modèle d'embeddings actuel. Le modèle plus gros `paraphrase-multilingual-mpnet-base-v2` monte à **72 %** (et 100 % de bon document en tête). Il n'est pas utilisé dans la démo, car il est environ 4 fois plus lourd (~1 Go), ce qui est risqué pour les ressources gratuites de Streamlit Cloud.
2. **Le LLM compense une recherche imparfaite.** Comme le bon passage est toujours dans le top 5, les deux LLM testés répondent correctement à presque toutes les questions et refusent d'inventer sur les questions hors sujet.
3. **Le juge doit être calibré.** Au premier essai, le juge pénalisait des réponses justes qui omettaient un détail secondaire (« 7 °C » sans préciser « pour les absences de plus de 48 heures »). Les références ont donc été séparées en « Essentiel » et « Facultatif ». Il reste une erreur du juge : pour `gemma3:12b`, la réponse sur la garantie est exacte mais a été refusée parce qu'elle ajoute un détail vrai. En relecture manuelle, **les 20 réponses des deux modèles sont correctes**. Le score brut du juge est tout de même conservé dans les résultats.

## Limites

- Le corpus est petit (3 documents, 12 passages) : le top 5 couvre une grande partie de la base, ce qui rend le *hit@5* facile. Le *hit@1* et le MRR sont plus parlants.
- Les documents et les questions ont été écrits par la même personne, ce qui peut favoriser des formulations proches.
- Un LLM juge peut se tromper ; c'est pourquoi la vérification déterministe des faits est calculée à côté.
- Le temps moyen de Groq inclut les attentes imposées par les limites de son offre gratuite (le modèle et le juge partagent le même quota) ; sans ces limites, une réponse prend moins d'une seconde.
