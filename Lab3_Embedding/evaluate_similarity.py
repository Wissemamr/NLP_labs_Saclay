"""
evaluate_similarity.py
----------------------
Evaluates semantic similarity for candidate words across 6 trained embedding models:
- Word2Vec CBOW (Med & Press)
- Word2Vec Skip-gram (Med & Press)
- FastText CBOW (Med & Press)

Methods used:
1. scipy.spatial.distance (cdist/cosine) on loaded embedding vectors
2. gensim's most_similar method on loaded models

Performs:
- Comparison across architectures (CBOW vs Skip-Gram vs FastText)
- Comparison across corpora (Medical vs Non-Medical Press)
- Detailed candidate words analysis: patient, treatment (traitement), disease (maladie), solution, yellow (jaune)
"""

import os
import sys
import numpy as np
from scipy.spatial.distance import cdist, cosine
from gensim.models import Word2Vec, FastText, KeyedVectors


def load_vectors_from_vec_file(vec_filepath):
    """
    Loads raw embeddings from a .vec file (Word2Vec text format).
    Returns:
        words: list of words
        word2idx: dict mapping word -> index
        vectors: numpy array of shape (vocab_size, dim)
    """
    words = []
    vectors = []
    with open(vec_filepath, "r", encoding="utf-8", errors="ignore") as f:
        first_line = f.readline().strip().split()
        if len(first_line) == 2:
            num_words, dim = int(first_line[0]), int(first_line[1])
        else:
            # Fallback if no header
            f.seek(0)

        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0]
            try:
                vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                words.append(word)
                vectors.append(vec)
            except ValueError:
                continue

    vectors = np.array(vectors)
    word2idx = {w: i for i, w in enumerate(words)}
    return words, word2idx, vectors


def find_top_k_scipy(target_word, words, word2idx, vectors, top_k=10):
    """
    Finds top_k closest words using scipy.spatial.distance (cosine metric).
    Note: Cosine Distance = 1 - Cosine Similarity.
    Smallest distance <=> highest similarity.
    """
    if target_word not in word2idx:
        return None

    target_idx = word2idx[target_word]
    target_vec = vectors[target_idx].reshape(1, -1)

    # Compute cosine distance between target_vec and all vectors in the vocabulary
    distances = cdist(target_vec, vectors, metric="cosine")[0]

    # Get top_k + 1 indices (excluding the target word itself whose distance is ~0)
    nearest_indices = np.argsort(distances)

    results = []
    for idx in nearest_indices:
        word = words[idx]
        if word.lower() == target_word.lower():
            continue
        dist = distances[idx]
        sim = 1.0 - dist
        results.append((word, float(sim)))
        if len(results) == top_k:
            break

    return results


def find_top_k_gensim(target_word, model_or_wv, top_k=10):
    """
    Finds top_k closest words using gensim's most_similar.
    """
    wv = model_or_wv.wv if hasattr(model_or_wv, "wv") else model_or_wv
    if target_word not in wv:
        return None
    return wv.most_similar(target_word, topn=top_k)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")

    # Candidate words: support French counterparts for the French corpora
    # patient -> patient, treatment -> traitement, disease -> maladie, solution -> solution, yellow -> jaune
    candidate_words_map = {
        "patient": ["patient", "patients"],
        "treatment": ["traitement", "traitements", "treatment"],
        "disease": ["maladie", "maladies", "affection", "disease"],
        "solution": ["solution", "solutions"],
        "yellow": ["jaune", "jaunes", "yellow"],
    }

    model_configs = [
        {
            "name": "W2V_CBOW_MED",
            "type": "w2v",
            "file": "w2v_cbow_med.model",
            "vec": "w2v_cbow_med.vec",
            "corpus": "Medical",
            "approach": "Word2Vec CBOW",
        },
        {
            "name": "W2V_SG_MED",
            "type": "w2v",
            "file": "w2v_sg_med.model",
            "vec": "w2v_sg_med.vec",
            "corpus": "Medical",
            "approach": "Word2Vec SG",
        },
        {
            "name": "FASTTEXT_CBOW_MED",
            "type": "ft",
            "file": "fasttext_cbow_med.model",
            "vec": "fasttext_cbow_med.vec",
            "corpus": "Medical",
            "approach": "FastText CBOW",
        },
        {
            "name": "W2V_CBOW_PRESS",
            "type": "w2v",
            "file": "w2v_cbow_press.model",
            "vec": "w2v_cbow_press.vec",
            "corpus": "Press",
            "approach": "Word2Vec CBOW",
        },
        {
            "name": "W2V_SG_PRESS",
            "type": "w2v",
            "file": "w2v_sg_press.model",
            "vec": "w2v_sg_press.vec",
            "corpus": "Press",
            "approach": "Word2Vec SG",
        },
        {
            "name": "FASTTEXT_CBOW_PRESS",
            "type": "ft",
            "file": "fasttext_cbow_press.model",
            "vec": "fasttext_cbow_press.vec",
            "corpus": "Press",
            "approach": "FastText CBOW",
        },
    ]

    print("=" * 80)
    print("STEP 2: SEMANTIC SIMILARITY EVALUATION")
    print("=" * 80)

    # 1. Verification of Scipy vs Gensim equivalence on sample words
    print(
        "\n--- [VERIFICATION] Comparing Scipy cdist Cosine vs Gensim most_similar ---"
    )
    test_model_path = os.path.join(models_dir, "w2v_cbow_med.model")
    test_vec_path = os.path.join(models_dir, "w2v_cbow_med.vec")

    test_model = Word2Vec.load(test_model_path)
    words_med, w2idx_med, vecs_med = load_vectors_from_vec_file(test_vec_path)

    sample_query = "patient"
    scipy_res = find_top_k_scipy(sample_query, words_med, w2idx_med, vecs_med, top_k=5)
    gensim_res = find_top_k_gensim(sample_query, test_model, top_k=5)

    print(f"Top-5 closest to '{sample_query}' in W2V CBOW (Med):")
    print(f"{'Rank':<5} {'Scipy (Cosine Sim)':<30} {'Gensim (most_similar)':<30}")
    print("-" * 65)
    for i in range(5):
        s_word, s_sim = scipy_res[i] if scipy_res else ("N/A", 0.0)
        g_word, g_sim = gensim_res[i] if gensim_res else ("N/A", 0.0)
        print(f"{i+1:<5} {s_word:<18} ({s_sim:.4f})     {g_word:<18} ({g_sim:.4f})")

    # Load all models and vector files
    loaded_models = {}
    loaded_vectors = {}
    print("\nLoading all 6 models and vector tables...")
    for cfg in model_configs:
        m_path = os.path.join(models_dir, cfg["file"])
        v_path = os.path.join(models_dir, cfg["vec"])
        if cfg["type"] == "w2v":
            loaded_models[cfg["name"]] = Word2Vec.load(m_path)
        else:
            loaded_models[cfg["name"]] = FastText.load(m_path)

        w, w2i, v = load_vectors_from_vec_file(v_path)
        loaded_vectors[cfg["name"]] = (w, w2i, v)
        print(f" - Loaded {cfg['name']}: {len(w):,} words in vocab")

    # Evaluate each candidate word
    primary_candidates = ["patient", "traitement", "maladie", "solution", "jaune"]

    summary_results = []

    for cand in primary_candidates:
        print("\n" + "=" * 90)
        print(f"EVALUATION FOR CANDIDATE WORD: '{cand.upper()}'")
        print("=" * 90)

        # Print table comparing all 6 models for this word
        all_results = {}
        for cfg in model_configs:
            m_name = cfg["name"]
            model = loaded_models[m_name]
            neighbors = find_top_k_gensim(cand, model, top_k=10)
            all_results[m_name] = neighbors

        # Check if word exists in all models
        print(f"\nTop 10 Nearest Neighbors for '{cand}':\n")
        header = f"{'Rank':<4} | " + " | ".join(
            f"{cfg['name']:<18}" for cfg in model_configs
        )
        print(header)
        print("-" * len(header))

        for rank in range(10):
            row_items = [f"{rank+1:<4}"]
            for cfg in model_configs:
                m_name = cfg["name"]
                res = all_results[m_name]
                if res and rank < len(res):
                    w, score = res[rank]
                    row_items.append(f"{w} ({score:.2f})")
                else:
                    row_items.append("OOV / None")
            print(
                " | ".join(
                    f"{item:<18}" if i > 0 else item for i, item in enumerate(row_items)
                )
            )


if __name__ == "__main__":
    main()
