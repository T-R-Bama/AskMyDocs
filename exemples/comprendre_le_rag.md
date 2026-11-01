# Comprendre le RAG (Retrieval-Augmented Generation)

## Le problème de départ

Un grand modèle de langage (LLM) ne connaît que ce qu'il a vu pendant son entraînement. Il ignore donc les documents internes d'une entreprise, les informations récentes et tout ce qui est privé. Lorsqu'on l'interroge sur ces sujets, il risque de répondre de façon vague ou, pire, d'inventer une réponse plausible mais fausse : on parle d'« hallucination ».

## L'idée du RAG

Le RAG, ou génération augmentée par la recherche, consiste à aller chercher les informations pertinentes dans une base de documents **avant** de demander au modèle de répondre. Le modèle ne répond plus de mémoire : il s'appuie sur des extraits fournis dans sa consigne, un peu comme un étudiant qui passe un examen à livre ouvert.

Le terme a été popularisé en 2020 par un article de chercheurs de Facebook AI Research (Lewis et al., « Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks »).

## Les étapes d'un système RAG

### 1. L'indexation (une seule fois par document)

- **Extraction** : on récupère le texte brut des fichiers (PDF, Word, pages web…).
- **Découpage** : le texte est coupé en morceaux (« chunks ») de quelques centaines de mots, avec un léger chevauchement pour ne pas couper une idée en deux.
- **Vectorisation** : chaque morceau est transformé en vecteur numérique par un modèle d'*embeddings*. Deux textes au sens proche obtiennent des vecteurs proches, même s'ils n'utilisent pas les mêmes mots.
- **Stockage** : les vecteurs sont rangés dans une base vectorielle (ChromaDB, FAISS, Qdrant, pgvector…).

### 2. La recherche (à chaque question)

La question de l'utilisateur est vectorisée avec le même modèle, puis la base renvoie les morceaux dont les vecteurs sont les plus proches, généralement mesurés par la similarité cosinus. C'est une recherche **sémantique** : « vacances » peut retrouver un passage qui parle de « congés payés ».

### 3. La génération

Les morceaux retrouvés sont insérés dans la consigne envoyée au LLM, avec des instructions : répondre uniquement à partir de ces extraits, citer ses sources et reconnaître quand l'information est absente.

## Avantages

- **Des réponses fondées et vérifiables** : chaque réponse peut citer les passages utilisés.
- **Des connaissances à jour** : il suffit d'ajouter un document à la base, sans réentraîner le modèle.
- **La confidentialité** : les documents restent dans la base et ne servent pas à entraîner le modèle.
- **Un coût réduit** : beaucoup moins cher que l'affinage (*fine-tuning*) d'un modèle.

## Limites et pistes d'amélioration

- La qualité dépend de la recherche : si le bon passage n'est pas retrouvé, le modèle ne peut pas bien répondre.
- La taille des morceaux est un compromis : trop petits, ils perdent le contexte ; trop grands, ils diluent l'information.
- Les tableaux, images et documents scannés demandent des traitements spécifiques (OCR, extraction de tableaux).

Pour aller plus loin, on peut combiner recherche sémantique et recherche par mots-clés (recherche hybride), ajouter une étape de reclassement (*reranking*) des passages, ou reformuler la question avant la recherche.

## Le RAG dans ce projet

Ce chatbot applique exactement ces étapes : les documents sont découpés en morceaux d'environ 1 000 caractères, vectorisés avec le modèle multilingue `paraphrase-multilingual-MiniLM-L12-v2`, stockés dans ChromaDB, puis les cinq passages les plus proches de chaque question sont transmis au modèle de langage choisi (Groq, Ollama ou Claude), qui répond en citant ses sources.
