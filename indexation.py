"""
indexation.py - Script de création de la base vectorielle FAISS
Sujet A : Recommandation de Films
Source : TMDB 5000 Movie Dataset (Kaggle)
"""

import os
import json
import ast
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INDEX_DIR    = "index"
INDEX_BASE   = os.path.join(INDEX_DIR, "films")
CSV_PATH     = os.path.join("data", "tmdb_5000_movies.csv")
EMBEDDING_MODEL = "all-mpnet-base-v2"   # anglais car le dataset TMDB est en anglais
MIN_FILMS    = 500                       # contrainte du sujet
MIN_VOTES    = 50                        # filtrer les films peu connus


# ─────────────────────────────────────────────
# 1. Chargement et nettoyage du CSV
# ─────────────────────────────────────────────
def extraire_noms(json_str: str) -> list[str]:
    """
    Extrait les noms depuis une colonne JSON imbriquée TMDB.
    Ex: '[{"id": 18, "name": "Drama"}, ...]' → ["Drama", ...]

    Q2 : La colonne genres (et keywords, companies...) est du JSON sérialisé
    en string. On utilise ast.literal_eval (plus sûr que json.loads) car
    certaines entrées utilisent des guillemets simples Python.
    """
    if not json_str or pd.isna(json_str):
        return []
    try:
        data = ast.literal_eval(json_str)
        return [item["name"] for item in data if "name" in item]
    except (ValueError, SyntaxError):
        return []


def convertir_film_en_texte(row: pd.Series) -> str:
    """
    Q1 : Convertit une ligne du CSV en texte riche et cohérent pour l'embedding.
    On inclut en priorité :
    - titre + année (identifiant clair)
    - synopsis (overview) : contenu sémantique principal
    - genres : très utiles pour les requêtes de type "film d'horreur"
    - note + nombre de votes : utiles pour "bien noté"
    - durée + langue : filtres courants
    - mots-clés si disponibles

    On n'inclut PAS : budget, revenue (pas utiles pour la reco),
    production_companies (peu interrogés).
    """
    parties = []

    titre = row.get("title", "")
    annee = str(row.get("release_date", ""))[:4]
    parties.append(f"Title: {titre} ({annee})")

    genres = extraire_noms(row.get("genres", ""))
    if genres:
        parties.append(f"Genres: {', '.join(genres)}")

    overview = row.get("overview", "")
    if overview and not pd.isna(overview):
        parties.append(f"Synopsis: {overview}")

    note = row.get("vote_average", 0)
    votes = row.get("vote_count", 0)
    if note:
        parties.append(f"Rating: {note}/10 ({int(votes)} votes)")

    runtime = row.get("runtime", 0)
    if runtime and not pd.isna(runtime):
        parties.append(f"Runtime: {int(runtime)} minutes")

    langue = row.get("original_language", "")
    if langue:
        parties.append(f"Original language: {langue}")

    keywords = extraire_noms(row.get("keywords", ""))
    if keywords:
        parties.append(f"Keywords: {', '.join(keywords[:10])}")  # max 10

    return "\n".join(parties)


def charger_films(csv_path: str) -> list[dict]:
    """
    Charge le CSV TMDB, nettoie les données et construit la liste de documents.
    Q3 : On sauvegarde l'index FAISS sur disque pour ne pas réindexer à chaque test.
    """
    print(f"  Chargement : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Films bruts : {len(df)}")

    # Fusion avec tmdb_5000_credits.csv si disponible (optionnel)
    credits_path = os.path.join("data", "tmdb_5000_credits.csv")
    if os.path.exists(credits_path):
        credits = pd.read_csv(credits_path)
        # Renommer la colonne pour la jointure
        if "movie_id" in credits.columns:
            credits = credits.rename(columns={"movie_id": "id"})
        df = df.merge(credits[["id", "cast", "crew"]], on="id", how="left")
        print("  ✓ Données cast/crew fusionnées")

    # Nettoyage : supprimer les films sans synopsis ni note
    df = df.dropna(subset=["overview"])
    df = df[df["vote_count"] >= MIN_VOTES]
    df = df[df["vote_average"] > 0]
    print(f"  Films après nettoyage : {len(df)}")

    if len(df) < MIN_FILMS:
        print(f"  ⚠️  Moins de {MIN_FILMS} films disponibles.")

    documents = []
    for _, row in df.iterrows():
        texte = convertir_film_en_texte(row)
        genres = extraire_noms(row.get("genres", ""))
        documents.append({
            "id": f"film_{row.get('id', _)}",
            "contenu": texte,
            "metadata": {
                "titre": row.get("title", "Titre inconnu"),
                "annee": str(row.get("release_date", ""))[:4],
                "note": float(row.get("vote_average", 0)),
                "votes": int(row.get("vote_count", 0)),
                "genres": genres,
                "langue": row.get("original_language", ""),
                "runtime": int(row.get("runtime", 0)) if not pd.isna(row.get("runtime", 0)) else 0,
                "source": "TMDB 5000 Movie Dataset",
            }
        })

    return documents


# ─────────────────────────────────────────────
# 2. Chunking
# ─────────────────────────────────────────────
def chunker(texte: str, taille_max: int = 800, overlap: int = 100) -> list[str]:
    """
    Pour les films, chaque fiche est déjà compacte (< 800 chars en général).
    La plupart des films ne seront donc PAS découpés.
    On garde le chunker pour les rares films avec synopsis très long.
    """
    if len(texte) <= taille_max:
        return [texte]

    lignes = texte.split("\n")
    chunks, chunk_courant = [], ""
    for ligne in lignes:
        if len(chunk_courant) + len(ligne) + 1 > taille_max and chunk_courant:
            chunks.append(chunk_courant.strip())
            chunk_courant = chunk_courant[-overlap:] + "\n" + ligne
        else:
            chunk_courant += ("\n" if chunk_courant else "") + ligne
    if chunk_courant.strip():
        chunks.append(chunk_courant.strip())
    return [c for c in chunks if len(c.strip()) > 20]


def chunker_documents(documents: list[dict]) -> list[dict]:
    """Applique le chunker à tous les documents en conservant les métadonnées."""
    tous_les_chunks = []
    for doc in documents:
        chunks = chunker(doc["contenu"])
        for i, chunk in enumerate(chunks):
            tous_les_chunks.append({
                "contenu": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "doc_id": doc["id"],
                }
            })
    return tous_les_chunks


# ─────────────────────────────────────────────
# 3. Embeddings
# ─────────────────────────────────────────────
def embedder_chunks(chunks: list[dict], modele) -> np.ndarray:
    """Transforme les chunks en vecteurs numpy (shape: n x 768)."""
    textes = [c["contenu"] for c in chunks]
    print(f"  Encodage de {len(textes)} chunks (peut prendre quelques minutes)...")
    vecteurs = modele.encode(textes, show_progress_bar=True, batch_size=64)
    return np.array(vecteurs, dtype=np.float32)


# ─────────────────────────────────────────────
# 4. Index FAISS
# ─────────────────────────────────────────────
def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """
    IndexFlatIP + normalisation L2 = similarité cosinus.
    Score élevé (proche de 1.0) = plus pertinent.
    """
    dimension = vecteurs.shape[1]
    faiss.normalize_L2(vecteurs)
    index = faiss.IndexFlatIP(dimension)
    index.add(vecteurs)
    print(f"  Index FAISS : {index.ntotal} vecteurs, dim={dimension}")
    return index


def sauvegarder_index(index: faiss.Index, chunks: list[dict], chemin_base: str):
    """Sauvegarde l'index et les métadonnées (même ordre = correspondance garantie)."""
    os.makedirs(os.path.dirname(chemin_base) or ".", exist_ok=True)
    faiss.write_index(index, chemin_base + ".index")
    with open(chemin_base + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ Sauvegardé : {chemin_base}.index + _meta.json")


# ─────────────────────────────────────────────
# 5. Pipeline principal
# ─────────────────────────────────────────────
def main():
    # Q3 : vérifier si l'index existe déjà pour éviter la réindexation
    if os.path.exists(INDEX_BASE + ".index"):
        print("✓ Index FAISS déjà existant. Supprimez 'index/' pour réindexer.")
        return

    # Vérifier que le CSV est présent
    if not os.path.exists(CSV_PATH):
        print(f"❌ Fichier introuvable : {CSV_PATH}")
        print("   Téléchargez le dataset sur :")
        print("   kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("   Placez tmdb_5000_movies.csv dans le dossier data/")
        return

    print("=" * 60)
    print("PHASE 1 : INDEXATION — Films TMDB")
    print("=" * 60)

    print("\n[1/4] Chargement et nettoyage du CSV...")
    documents = charger_films(CSV_PATH)
    print(f"  ✓ {len(documents)} films prêts à indexer")

    print("\n[2/4] Découpage en chunks...")
    chunks = chunker_documents(documents)
    print(f"  ✓ {len(chunks)} chunks créés")
    print(f"\n  Exemple :")
    print(f"  {chunks[0]['metadata']['titre']} — {chunks[0]['contenu'][:150]}...")

    print(f"\n[3/4] Chargement du modèle : {EMBEDDING_MODEL}")
    modele = SentenceTransformer(EMBEDDING_MODEL)
    vecteurs = embedder_chunks(chunks, modele)
    print(f"  ✓ Vecteurs : shape={vecteurs.shape}")

    print("\n[4/4] Création et sauvegarde de l'index FAISS...")
    index = creer_index_faiss(vecteurs)
    sauvegarder_index(index, chunks, INDEX_BASE)

    print("\n" + "=" * 60)
    print("✓ INDEXATION TERMINÉE")
    print(f"  Films indexés : {len(documents)}")
    print(f"  Chunks        : {len(chunks)}")
    print(f"  Dimensions    : {vecteurs.shape[1]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
