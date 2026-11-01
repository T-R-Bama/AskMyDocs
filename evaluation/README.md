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

### 1. La recherche était le maillon faible

Avec la configuration de départ (recherche sémantique seule, passages de 1 000 caractères), le bon passage était toujours dans les 5 envoyés au LLM, mais **en tête dans seulement 56 % des cas**.

### 2. Un bug caché : la fin des passages était invisible

En regardant les questions ratées, les informations manquées étaient souvent en fin de passage (indemnité de télétravail, forfait vélo, garantie). Le modèle d'embeddings `paraphrase-multilingual-MiniLM-L12-v2` **tronque tout texte à 128 tokens** : 11 passages sur 12 dépassaient cette limite, et en moyenne 43 % de leur texte était ignoré.

### 3. Réduire la taille des passages… n'a pas aidé

Hypothèse naturelle : des passages de 400 caractères tiennent dans la limite, donc la recherche devrait s'améliorer. La mesure dit le contraire (*hit@1* de 39 à 61 % selon la taille, contre 56 % au départ) : avec deux à trois fois plus de passages, courts et qui se chevauchent, le classement devient plus difficile. Sans évaluation, ce « correctif » évident aurait dégradé le système.

### 4. La recherche hybride est la vraie amélioration

La recherche hybride combine le classement sémantique et un classement par mots-clés **BM25** (implémenté en Python pur, sans dépendance), fusionnés par **rangs réciproques (RRF)**. BM25 lit tout le texte, y compris la partie tronquée pour les embeddings, et retrouve les correspondances exactes (« E1 », « vélo », « garanti »).

| Configuration | Bon passage en 1ʳᵉ position | Bon document en 1ʳᵉ position | MRR |
|---|---|---|---|
| Sémantique seule (départ) | 55,6 % | 94,4 % | 0,750 |
| Modèle 4× plus gros (`mpnet`), sémantique seule | 72,2 % | 100 % | 0,861 |
| **Hybride, modèle léger (retenu)** | **77,8 %** | **100 %** | **0,880** |
| Hybride, modèle 4× plus gros | 77,8 % | 100 % | 0,889 |

Le modèle léger avec la recherche hybride fait mieux que le gros modèle seul, et presque aussi bien que le gros modèle hybride : **la qualité augmente sans alourdir l'application**, ce qui compte pour les ressources gratuites de Streamlit Cloud. C'est la configuration utilisée par l'application. Les réglages n'ont pas été poussés plus loin (paramètres BM25 et RRF standards) pour éviter de sur-ajuster le système à 18 questions.

### 5. Le LLM compense une recherche imparfaite

Comme le bon passage est toujours dans le top 5, les deux LLM testés répondent correctement à toutes les questions et refusent d'inventer sur les questions hors sujet, avant comme après l'amélioration de la recherche. Un meilleur classement reste utile : il permet d'envoyer moins de passages au LLM, donc des requêtes plus rapides et moins chères, et rend le système plus robuste sur des corpus plus grands.

### 6. Le juge doit être calibré, et contrôlé

Au premier essai, le juge pénalisait des réponses justes qui omettaient un détail secondaire (« 7 °C » sans préciser « pour les absences de plus de 48 heures »). Les références ont donc été séparées en « Essentiel » et « Facultatif ». Il reste une erreur du juge : la réponse de `gemma3:12b` sur la garantie est exacte mais a été refusée parce qu'elle ajoute un détail vrai. À l'inverse, la vérification des faits a signalé une réponse de `gemma3:12b` juste (« vous ne pouvez pas prendre l'avion ») parce qu'elle ne mentionne pas le train. Les deux vérifications se contrôlent mutuellement ; en relecture manuelle, **les 20 réponses des deux modèles sont correctes**. Les scores bruts sont conservés dans les résultats.

## Limites

- Le corpus est petit (3 documents, 12 passages) : le top 5 couvre une grande partie de la base, ce qui rend le *hit@5* facile. Le *hit@1* et le MRR sont plus parlants.
- Un seul extrait est accepté comme preuve par question. Pour la question sur le Wi-Fi 5 GHz, le passage classé premier contient aussi la bonne information (« réseau 2,4 GHz ») mais pas l'extrait choisi : le *hit@1* est donc légèrement sous-estimé. Les questions n'ont pas été modifiées après avoir vu les résultats.
- La recherche hybride relit tous les passages à chaque question pour BM25 : parfait pour quelques documents, à remplacer par un index BM25 persistant pour un gros corpus.
- Les documents et les questions ont été écrits par la même personne, ce qui peut favoriser des formulations proches.
- Un LLM juge peut se tromper ; c'est pourquoi la vérification déterministe des faits est calculée à côté.
- Le temps moyen de Groq inclut les attentes imposées par les limites de son offre gratuite (le modèle et le juge partagent le même quota) ; sans ces limites, une réponse prend moins d'une seconde.
