"""
rag.py - Système de recommandation de films
Sujet A : RAG Films — TMDB 5000
"""

import os
import json
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
INDEX_BASE = os.path.join("index", "films")
EMBEDDING_MODEL = "all-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"
K_RESULTS = 6
SEUIL_SCORE = 0.25
# ─────────────────────────────────────────────
# 1. Chargement de l'index
# ─────────────────────────────────────────────
def charger_index(chemin_base: str):
    """Charge l'index FAISS et les métadonnées depuis le disque."""
    if not os.path.exists(chemin_base + ".index"):
        raise FileNotFoundError(
            f"Index introuvable : {chemin_base}.index\n"
            "Lancez d'abord : python indexation.py"
        )
    index = faiss.read_index(chemin_base + ".index")
    with open(chemin_base + "_meta.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks


# ─────────────────────────────────────────────
# 2. Recherche vectorielle
# ─────────────────────────────────────────────
def rechercher(
    question: str,
    modele,
    index: faiss.Index,
    chunks: list[dict],
    k: int = K_RESULTS,
    filtre_langue: str = None
) -> list[dict]:
    """
    Recherche les k chunks les plus pertinents.
    filtre_langue : "fr" pour VO française, "en" pour anglophone, None pour tout.

    Score IndexFlatIP : plus élevé = plus pertinent (similarité cosinus).
    """
    vecteur = modele.encode([question], show_progress_bar=False)
    vecteur = np.array(vecteur, dtype=np.float32)
    faiss.normalize_L2(vecteur)

    # On cherche plus large si un filtre est actif
    k_large = k * 4 if filtre_langue else k
    scores, indices = index.search(vecteur, min(k_large, index.ntotal))

    resultats = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = chunks[idx]
        meta = chunk["metadata"]

        # Filtre langue (contrainte spécifique sujet A)
        if filtre_langue:
            langue_film = meta.get("langue", "")
            if filtre_langue == "fr" and langue_film not in LANGUES_VO_FR:
                continue
            elif filtre_langue == "international" and langue_film not in LANGUES_INTER:
                continue

        resultats.append({
            "contenu": chunk["contenu"],
            "metadata": meta,
            "score": float(score),
        })

        if len(resultats) >= k:
            break

    return resultats


# ─────────────────────────────────────────────
# 3. Prompt système
# ─────────────────────────────────────────────
def construire_prompt_systeme() -> str:
    """
    Q4 : Prompt guidant le LLM pour des recommandations pertinentes et argumentées.
    Il ne s'agit pas d'une question factuelle — le LLM doit justifier ses choix.
    """
    return """Tu es un cinéphile expert et passionné qui recommande des films.
Tu te bases UNIQUEMENT sur les films présents dans le CONTEXTE fourni.

RÈGLES :
1. Pour chaque recommandation, cite OBLIGATOIREMENT le titre exact et la note (ex: 8.2/10).
2. Argumente chaque recommandation en expliquant pourquoi ce film correspond à la demande.
3. Si aucun film du contexte ne correspond bien à la demande, dis-le clairement plutôt qu'inventer.
4. Q5 : Si l'utilisateur demande un film très récent (après 2017) absent de ta base, réponds :
   "Ce film ne figure pas dans ma base de données (dataset TMDB 2017). Je te recommande des films similaires disponibles."
5. Structure ta réponse avec des numéros (1., 2., 3.) pour chaque recommandation.
6. Pour chaque film, mentionne : titre, année, note, genres, et une courte justification personnalisée.
7. Adopte un ton chaleureux et enthousiaste, comme un ami cinéphile qui partage ses coups de cœur.

FORMAT :
1. **Titre** (Année) — ⭐ Note/10
   Genres : ...
   Pourquoi : [justification personnalisée liée à la demande]"""


# ─────────────────────────────────────────────
# 4. Génération de la réponse
# ─────────────────────────────────────────────
def construire_contexte(chunks: list[dict]) -> str:
    """Formate les chunks pour le prompt LLM avec métadonnées visibles."""
    parties = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        parties.append(
            f"[Film {i} | Score: {chunk['score']:.2f}]\n"
            f"{chunk['contenu']}"
        )
    return "\n\n---\n\n".join(parties)


def generer_reponse(
    question: str,
    chunks_pertinents: list[dict],
    client: Groq,
    historique: list[dict] = None
) -> str:
    """
    Génère une réponse de recommandation via Groq.
    Inclut l'historique de conversation (Bonus A).
    """
    contexte = construire_contexte(chunks_pertinents)

    message_utilisateur = f"""FILMS DISPONIBLES (base TMDB) :

{contexte}

---

DEMANDE : {question}

Recommande les films les plus adaptés à cette demande en te basant UNIQUEMENT sur les films ci-dessus."""

    messages = []
    if historique:
        messages.extend(historique[-6:])
    messages.append({"role": "user", "content": message_utilisateur})

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            *messages
        ],
        max_tokens=1200,
        temperature=0.6,  # plus élevé que médicaments : la reco est subjective
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# 5. Bonus C — Reformulation de la question
# ─────────────────────────────────────────────
def reformuler_question(question: str, client: Groq) -> str:
    """
    Bonus C : transforme une question conversationnelle en mots-clés
    adaptés à la recherche dans une base de films TMDB.
    Ex: "un film comme Inception mais plus accessible" →
        "science fiction mind-bending thriller accessible mystery"
    """
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": (
                "Transforme cette demande de film en 8 à 12 mots-clés anglais "
                "pour une recherche dans une base de données de films "
                "(genres, thèmes, ambiance, mots-clés TMDB). "
                "Retourne UNIQUEMENT les mots-clés séparés par des espaces, sans ponctuation.\n\n"
                f"Demande : {question}"
            )
        }],
        max_tokens=60,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# 6. Bonus B — Score de confiance
# ─────────────────────────────────────────────
def verifier_pertinence(chunks: list[dict]) -> bool:
    """Retourne False si le meilleur score est sous le seuil."""
    if not chunks:
        return False
    return max(c["score"] for c in chunks) >= SEUIL_SCORE


# ─────────────────────────────────────────────
# 7. Affichage
# ─────────────────────────────────────────────
def afficher_sources(chunks: list[dict]):
    """Affiche les films utilisés comme sources."""
    print("\n🎬 Films consultés dans la base :")
    for i, chunk in enumerate(chunks, 1):
        meta = chunk["metadata"]
        titre = meta.get("titre", "?")
        annee = meta.get("annee", "?")
        note  = meta.get("note", 0)
        score = chunk["score"]
        print(f"   [{i}] {titre} ({annee}) — ⭐{note}/10 — similarité: {score:.2f}")


def afficher_aide():
    print("\n📋 Commandes disponibles :")
    print("   sources          → voir les films consultés")
    print("   comparer         → comparer deux films")
    print("   langue fr        → filtrer VO française uniquement")
    print("   langue inter     → filtrer VO internationale")
    print("   langue reset     → supprimer le filtre langue")
    print("   quit / exit / q  → quitter")


# ─────────────────────────────────────────────
# 8. Interface en ligne de commande
# ─────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  🎬 ASSISTANT RECOMMANDATION FILMS — RAG (TMDB 5000)")
    print("=" * 65)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY non définie dans .env")
        return

    print("\n⏳ Chargement de la base de films...")
    try:
        index, chunks = charger_index(INDEX_BASE)
        print(f"   ✓ {index.ntotal} films indexés")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print(f"⏳ Chargement du modèle ({EMBEDDING_MODEL})...")
    modele = SentenceTransformer(EMBEDDING_MODEL)
    print("   ✓ Prêt")

    client = Groq(api_key=api_key)

    print(f"\n✅ Système prêt. Modèle : {GROQ_MODEL}")
    print("   Tapez 'aide' pour voir les commandes disponibles.")
    print("-" * 65)

    historique   = []        # Bonus A
    derniers_chunks = []
    filtre_langue = None     # Contrainte sujet A : filtre par langue

    while True:
        try:
            entree = input("\n🎥 Votre demande : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nÀ bientôt !")
            break

        if not entree:
            continue

        # Commandes spéciales
        if entree.lower() in ["quit", "exit", "q"]:
            print("À bientôt !")
            break

        if entree.lower() == "aide":
            afficher_aide()
            continue

        if entree.lower() == "sources":
            if derniers_chunks:
                afficher_sources(derniers_chunks)
            else:
                print("Aucune recherche effectuée.")
            continue

        # Filtre langue (contrainte spécifique sujet A)
        if entree.lower().startswith("langue"):
            partie = entree.lower().replace("langue", "").strip()
            if partie == "fr":
                filtre_langue = "fr"
                print("✓ Filtre activé : VO française uniquement")
            elif partie in ["inter", "international"]:
                filtre_langue = "international"
                print("✓ Filtre activé : VO internationale")
            elif partie == "reset":
                filtre_langue = None
                print("✓ Filtre langue désactivé")
            continue

        # Bonus D : mode comparaison
        if entree.lower().startswith("comparer"):
            print("Entrez les deux films à comparer (séparés par une virgule) :")
            films_entree = input("  > ").strip()
            if "," in films_entree:
                f1, f2 = [f.strip() for f in films_entree.split(",", 1)]
                entree = f"Compare {f1} et {f2} : synopsis, genres, note, ambiance générale."
            else:
                entree = f"Compare {films_entree} à des films similaires dans la base."

        print("\n⏳ Analyse de votre demande...")

        # Bonus C : reformulation
        mots_cles = reformuler_question(entree, client)
        print(f"   🔍 Mots-clés : {mots_cles}")

        # Recherche vectorielle
        chunks_pertinents = rechercher(
            mots_cles, modele, index, chunks,
            k=K_RESULTS, filtre_langue=filtre_langue
        )

        # Fallback sur question originale si peu de résultats
        if not verifier_pertinence(chunks_pertinents):
            chunks_pertinents = rechercher(
                entree, modele, index, chunks,
                k=K_RESULTS, filtre_langue=filtre_langue
            )

        derniers_chunks = chunks_pertinents

        # Bonus B : avertissement si score faible
        if not verifier_pertinence(chunks_pertinents):
            print(
                f"\n⚠️  Peu de films très proches trouvés "
                f"(score max: {max(c['score'] for c in chunks_pertinents):.2f}). "
                "Les recommandations peuvent être moins précises."
            )

        if filtre_langue:
            print(f"   🌍 Filtre langue actif : {filtre_langue}")

        # Génération
        print("⏳ Génération des recommandations...\n")
        try:
            reponse = generer_reponse(entree, chunks_pertinents, client, historique)
        except Exception as e:
            print(f"❌ Erreur Groq : {e}")
            continue

        print("─" * 65)
        print(reponse)
        print("─" * 65)

        afficher_sources(chunks_pertinents)

        # Historique (Bonus A)
        historique.append({"role": "user", "content": entree})
        historique.append({"role": "assistant", "content": reponse})
        if len(historique) > 12:
            historique = historique[-12:]


if __name__ == "__main__":
    main()
