# Compte-rendu — TP RAG Films TMDB

## Choix du sujet

**Sujet A — Recommandation de Films.** Ce sujet est particulièrement intéressant car il pose un problème de transformation de données tabulaires en texte sémantique, et parce que la recommandation est une tâche subjective (pas de "bonne" réponse unique), ce qui demande une conception du prompt différente d'un assistant factuel.

---

## Décisions de conception

### Transformation CSV → texte (Q1)

La principale difficulté du sujet A est que les données sont tabulaires, pas du texte libre. La fonction `convertir_film_en_texte()` reconstruit un texte structuré avec des labels explicites (`Title:`, `Genres:`, `Synopsis:`, etc.). Ce format aide l'embedding à comprendre le rôle de chaque information. Le synopsis est la colonne centrale car elle contient le contenu sémantique le plus riche. Les genres et mots-clés permettent de répondre aux requêtes par type de film.

### Parsing JSON imbriqué (Q2)

Les colonnes `genres`, `keywords`, `production_companies` contiennent du JSON sérialisé en string avec parfois des guillemets simples (format Python dict plutôt que JSON strict). `ast.literal_eval()` gère ces deux cas là où `json.loads()` échouerait. Un try/except protège contre les entrées malformées.

### Modèle d'embedding anglophone

Le dataset TMDB étant entièrement en anglais (titres, synopsis, mots-clés), le modèle `all-mpnet-base-v2` (anglophone) est préféré au modèle multilingue `paraphrase-multilingual-mpnet-base-v2`. Il offre de meilleures performances sur du contenu anglais pur.

### Chunking minimal pour les films

Contrairement aux notices médicales, une fiche film est déjà compacte (synopsis court + quelques métadonnées ≈ 400–600 chars). La grande majorité des films génèrent un seul chunk. Le chunker est présent pour les rares cas de synopses très longs, mais n'est quasiment jamais déclenché.

### Température 0.6 pour la recommandation

La recommandation de film est une tâche subjective : il n'y a pas une seule bonne réponse. Une température plus élevée (0.6 vs 0.2 pour les médicaments) permet au LLM d'argumenter avec plus de nuance et d'enthousiasme, comme un cinéphile qui partage ses avis.

---

## Difficultés rencontrées

**1. Installation des dépendances :** Erreur `ModuleNotFoundError: No module named 'numpy'` au premier lancement car les dépendances n'étaient pas installées dans l'environnement virtuel. Résolu avec `pip install -r requirements.txt`.

**2. Dataset TMDB introuvable :** Le zip Kaggle portait le nom `archive (1).zip`, pas évident à identifier. Extrait avec `tar -xf "archive (1).zip" -C data\`.

**3. Clé API mal configurée :** Le fichier `.env` contenait la clé brute sans le préfixe `GROQ_API_KEY=`, empêchant `python-dotenv` de la charger. Résolu en ajoutant le préfixe.

**4. Modèle Groq décommissionné :** Erreur `llama3-8b-8192 has been decommissioned` au premier test. Remplacé par `llama-3.1-8b-instant`.

**5. Clé API bloquée par GitHub :** GitHub Push Protection a bloqué le push car un fichier `.env.txt` avec la clé était dans l'historique Git. Solution : suppression de l'historique avec `Remove-Item -Recurse -Force .git`, recréation d'un repo propre et nouveau `git init`.

---

## Décisions de conception

- **Transformation CSV → texte :** `convertir_film_en_texte()` assemble titre, genres, synopsis, note, langue et mots-clés. Le synopsis est la colonne centrale car la plus riche sémantiquement.
- **Parsing JSON :** `ast.literal_eval()` utilisé à la place de `json.loads()` car certaines entrées TMDB utilisent des guillemets simples Python.
- **Persistance :** Vérification de l'existence de `index/films.index` au démarrage pour éviter une réindexation de 10 minutes à chaque test.
- **Modèle embedding :** `all-mpnet-base-v2` (anglophone) choisi car le dataset TMDB est entièrement en anglais.
- **Index FAISS :** `IndexFlatIP` + normalisation L2 = similarité cosinus. Score élevé = plus pertinent.
- **Température LLM :** 0.6 (plus élevée que pour un assistant factuel) car la recommandation est subjective.
- **Bonus implémentés :** Historique (A), score de confiance (B), reformulation automatique (C), mode comparaison (D).
