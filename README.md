# 🎬 Assistant Recommandation Films — RAG de bout en bout

Sujet A du TP — Construire un RAG avec Python et Groq  
**Dataset :** TMDB 5000 Movie Dataset (Kaggle)

---

## Démarrage rapide

```bash
# 1. Environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Dépendances
pip install -r requirements.txt

# 3. Télécharger le dataset TMDB
#    → kaggle.com/datasets/tmdb/tmdb-movie-metadata
#    → Placer tmdb_5000_movies.csv dans data/

# 4. Clé API Groq (console.groq.com)
echo "GROQ_API_KEY=votre_clé_ici" > .env

# 5. Indexation (une seule fois, ~5 min)
python indexation.py

# 6. Lancement
python rag.py
```

---

## Architecture

```
PHASE 1 — INDEXATION (indexation.py)
─────────────────────────────────────────
tmdb_5000_movies.csv
    │
    ▼  charger_films()
    │  • nettoyage (dropna, vote_count >= 50)
    │  • extraire_noms() → parse JSON genres/keywords
    │  • convertir_film_en_texte() → texte riche
    │
    ▼  chunker_documents()
    │  taille_max=800, overlap=100
    │  (la plupart des fiches < 800 chars → 1 chunk par film)
    │
    ▼  SentenceTransformer("all-mpnet-base-v2")
    │  → vecteurs 768 dimensions
    │
    ▼  faiss.IndexFlatIP + normalisation L2
    │
    ▼  index/films.index + films_meta.json

PHASE 2 — INTERROGATION (rag.py)
─────────────────────────────────────────
Question utilisateur
    │
    ▼  reformuler_question() [Bonus C]
    │  llama3-8b → mots-clés anglais TMDB
    │
    ▼  rechercher() → top-6 chunks + scores
    │  + filtre langue optionnel
    │
    ▼  verifier_pertinence() [Bonus B]
    │  seuil = 0.25
    │
    ▼  generer_reponse() — llama3-70b
    │  + historique [Bonus A]
    │
    ▼  Recommandations avec titre, année, note, justification
       + sources consultées
```

---

## Structure du projet

```
rag_films/
├── indexation.py        # Pipeline d'indexation
├── rag.py               # Système de recommandation
├── requirements.txt
├── README.md
├── compte_rendu.md
├── .env                 # Clé API (NON commité)
├── .gitignore
├── data/
│   └── tmdb_5000_movies.csv   # À télécharger sur Kaggle
└── index/               # Créé par indexation.py
    ├── films.index
    └── films_meta.json
```

---

## Choix techniques — Réponses aux questions de réflexion

**Q1 — Conversion CSV → texte :**  
La fonction `convertir_film_en_texte()` assemble les champs dans cet ordre de priorité : titre + année → genres → synopsis (overview) → note → durée → langue → mots-clés. Le synopsis est la colonne la plus riche sémantiquement. Les genres et mots-clés permettent les requêtes par type ("thriller", "animation"...). Budget et revenue ne sont pas inclus car rarement interrogés dans une reco.

**Q2 — Parsing JSON imbriqué :**  
La fonction `extraire_noms()` utilise `ast.literal_eval()` plutôt que `json.loads()` : certaines entrées TMDB utilisent des guillemets simples Python (`'`) non conformes JSON. Cette méthode est plus robuste sur ce dataset.

**Q3 — Éviter la réindexation :**  
Au démarrage de `indexation.py`, on vérifie si `index/films.index` existe déjà. Si oui, on s'arrête immédiatement. Le fichier n'est recréé que si on supprime manuellement le dossier `index/`.

**Q4 — Prompt pour recommandations subjectives :**  
La température est fixée à **0.6** (plus élevée que pour les médicaments) car la recommandation est subjective. Le prompt demande au LLM d'argumenter chaque choix en lien avec la demande spécifique, comme un ami cinéphile. Il impose le format titre + année + note + justification.

**Q5 — Films récents absents de la base :**  
Le dataset TMDB couvre jusqu'à 2017. Le prompt système instruit le LLM d'informer l'utilisateur si un film récent n'est pas dans la base et de proposer des alternatives similaires disponibles.

### Modèle d'embedding

`all-mpnet-base-v2` — modèle anglophone (le dataset TMDB est en anglais). Plus performant que le modèle multilingue pour du contenu 100% anglais.

### Index FAISS

`IndexFlatIP` + normalisation L2 = similarité cosinus. Un film peu connu mais parfaitement aligné thématiquement sera bien classé même si sa note est faible.

### Filtre par langue

Contrainte spécifique du sujet A. L'utilisateur peut restreindre aux films en VO française (`langue fr`) ou internationale (`langue inter`) via les métadonnées `original_language` du CSV.

---

## Commandes disponibles

| Commande | Action |
|----------|--------|
| *(demande quelconque)* | Recommandation de films |
| `sources` | Films consultés dans la base |
| `comparer` | Comparer deux films (Bonus D) |
| `langue fr` | Filtrer VO française uniquement |
| `langue inter` | Filtrer VO internationale |
| `langue reset` | Supprimer le filtre |
| `aide` | Afficher l'aide |
| `quit` | Quitter |

---

## Exemples de questions

```
Un thriller psychologique avec un retournement inattendu
Recommande-moi un film d'animation familial sorti après 2010
Un film comme Inception mais plus accessible
Science-fiction philosophique sur l'intelligence artificielle
Comédie romantique légère pour une soirée entre amis
```

---

## Bonus implémentés

- **Bonus A** — Historique de conversation (6 derniers échanges)
- **Bonus B** — Score de confiance avec avertissement si score < 0.25
- **Bonus C** — Reformulation automatique en mots-clés TMDB (llama3-8b)
- **Bonus D** — Mode comparaison entre deux films

---

## Ressources

- [Dataset TMDB Kaggle](https://kaggle.com/datasets/tmdb/tmdb-movie-metadata)
- [Documentation Groq](https://console.groq.com/docs)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [sentence-transformers](https://www.sbert.net)
