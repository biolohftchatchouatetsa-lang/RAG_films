# Compte-rendu — TP RAG Films

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

**1. Qualité variable des synopses :**  
Certains films du dataset ont des synopses très courts ("No overview available") ou absents. Le filtrage `dropna(subset=["overview"])` et `vote_count >= 50` élimine les films trop peu documentés, ce qui améliore la qualité des recommandations sans réduire significativement le corpus (on reste largement au-dessus des 500 films requis).

**2. Filtre langue sur un petit corpus :**  
Avec seulement ~500 films en VO française dans la base, activer `langue fr` peut retourner peu de résultats. La recherche est élargie (`k * 4`) avant application du filtre pour maximiser les chances de trouver des films correspondants. Si le filtre réduit trop les résultats, un message avertit l'utilisateur.

**3. Reformulation en anglais (Bonus C) :**  
Les questions des utilisateurs sont en français, mais la base est en anglais. La reformulation (Bonus C) est donc doublement utile : elle traduit implicitement en anglais ET extrait les mots-clés pertinents pour la recherche TMDB. Sans cette étape, une question comme "film d'animation familial" donnerait de moins bons résultats qu'avec les mots-clés "family animation children adventure".

**4. Correspondance titre exact (Bonus D) :**  
Pour le mode comparaison, si l'utilisateur saisit un titre avec une orthographe légèrement différente (majuscules, articles...), la recherche vectorielle retrouve quand même le bon film grâce à la similarité sémantique. Ce n'est pas une recherche exacte par titre mais une recherche par sens, ce qui est plus robuste.

---

## Ce qui fonctionnerait mieux avec plus de temps

- Intégrer les données de casting (`tmdb_5000_credits.csv`) dans le texte embeddé pour répondre à des requêtes comme "films avec Tom Hanks".
- Implémenter un filtre par note minimale (ex: "uniquement les films notés > 7") via les métadonnées post-recherche.
- Ajouter les affiches de films via l'API TMDB dans une interface Gradio/Streamlit.
