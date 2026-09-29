"""
plot_tsne.py
------------
Performs 2D t-SNE dimensionality reduction and plotting for word embeddings
trained on QUAERO_FrenchMed and QUAERO_FrenchPress.

Visualizes:
1. Target candidate words ('patient', 'traitement', 'maladie', 'solution', 'jaune')
   and their top nearest neighbors across models.
2. Direct domain-shift comparison (Medical vs Press) for polysemous words (e.g., 'solution').
3. Architectural differences (CBOW vs Skip-Gram vs FastText).

Saves plots to the 'plots/' directory.
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless backend to avoid Tkinter dependency
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec, FastText


def get_cluster_words_and_vectors(model, candidate_words, top_n=8):
    """
    Extracts candidate words, their top_n nearest neighbors, and their vectors.
    """
    wv = model.wv if hasattr(model, "wv") else model
    words = []
    vectors = []
    word_labels = {}  # maps word -> category (target word it belongs to)

    for target in candidate_words:
        if target not in wv:
            continue
        words.append(target)
        vectors.append(wv[target])
        word_labels[target] = target

        # Get top_n neighbors
        neighbors = wv.most_similar(target, topn=top_n)
        for n_word, _ in neighbors:
            if n_word not in words:
                words.append(n_word)
                vectors.append(wv[n_word])
                word_labels[n_word] = target

    return words, np.array(vectors), word_labels


def plot_tsne_clusters(
    ax, words, vectors, word_labels, candidate_words, title, perplexity=10
):
    """
    Projects vectors to 2D using t-SNE and plots them on the given axis.
    """
    # Adjust perplexity if sample size is small
    n_samples = len(words)
    eff_perplexity = min(perplexity, max(2, n_samples // 3))

    tsne = TSNE(
        n_components=2, perplexity=eff_perplexity, random_state=42, max_iter=1000
    )
    coords_2d = tsne.fit_transform(vectors)

    # Distinct colors for each candidate cluster
    cmap = plt.get_cmap("tab10")
    colors = {cand: cmap(i % 10) for i, cand in enumerate(candidate_words)}

    for i, word in enumerate(words):
        cluster = word_labels[word]
        c = colors.get(cluster, "grey")
        is_target = word in candidate_words

        if is_target:
            ax.scatter(
                coords_2d[i, 0],
                coords_2d[i, 1],
                color=c,
                s=140,
                edgecolors="black",
                linewidth=1.5,
                zorder=5,
            )
            ax.annotate(
                f"★ {word.upper()}",
                (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=11,
                fontweight="bold",
                ha="center",
                va="bottom",
                xytext=(0, 7),
                textcoords="offset points",
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6, edgecolor=c
                ),
            )
        else:
            ax.scatter(
                coords_2d[i, 0], coords_2d[i, 1], color=c, s=50, alpha=0.75, zorder=3
            )
            ax.annotate(
                word,
                (coords_2d[i, 0], coords_2d[i, 1]),
                fontsize=8.5,
                alpha=0.9,
                ha="center",
                va="top",
                xytext=(0, -4),
                textcoords="offset points",
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    candidate_words = ["patient", "traitement", "maladie", "solution", "jaune"]

    print("=" * 70)
    print("STEP 3: GENERATING t-SNE VISUALIZATIONS")
    print("=" * 70)

    # 1. Load models
    models = {
        "W2V_CBOW_MED": Word2Vec.load(os.path.join(models_dir, "w2v_cbow_med.model")),
        "W2V_SG_MED": Word2Vec.load(os.path.join(models_dir, "w2v_sg_med.model")),
        "FASTTEXT_CBOW_MED": FastText.load(
            os.path.join(models_dir, "fasttext_cbow_med.model")
        ),
        "W2V_CBOW_PRESS": Word2Vec.load(
            os.path.join(models_dir, "w2v_cbow_press.model")
        ),
        "W2V_SG_PRESS": Word2Vec.load(os.path.join(models_dir, "w2v_sg_press.model")),
        "FASTTEXT_CBOW_PRESS": FastText.load(
            os.path.join(models_dir, "fasttext_cbow_press.model")
        ),
    }

    # Plot 1: 2x3 Grid comparing all 6 models
    print("\n[1/3] Generating 6-panel comparison grid (All models)...")
    fig, axes = plt.subplots(2, 3, figsize=(21, 13))

    grid_order = [
        ("W2V_CBOW_MED", "Word2Vec CBOW (Medical)", axes[0, 0]),
        ("W2V_SG_MED", "Word2Vec Skip-gram (Medical)", axes[0, 1]),
        ("FASTTEXT_CBOW_MED", "FastText CBOW (Medical)", axes[0, 2]),
        ("W2V_CBOW_PRESS", "Word2Vec CBOW (Press)", axes[1, 0]),
        ("W2V_SG_PRESS", "Word2Vec Skip-gram (Press)", axes[1, 1]),
        ("FASTTEXT_CBOW_PRESS", "FastText CBOW (Press)", axes[1, 2]),
    ]

    for model_key, title, ax in grid_order:
        model = models[model_key]
        words, vecs, labels = get_cluster_words_and_vectors(
            model, candidate_words, top_n=6
        )
        plot_tsne_clusters(
            ax, words, vecs, labels, candidate_words, title, perplexity=8
        )

    plt.suptitle(
        "t-SNE 2D Projections: Word Embeddings Across Architectures & Corpora\n(Candidate Words & Top Nearest Neighbors)",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot1_path = os.path.join(plots_dir, "tsne_comparison_all_models.png")
    plt.savefig(plot1_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot1_path}")

    # Plot 2: Direct Domain Shift Analysis for 'solution' & 'traitement' (Medical vs Press)
    print(
        "\n[2/3] Generating Domain Shift focus plot for 'solution' and 'traitement'..."
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (m_key, title) in enumerate(
        [
            ("W2V_SG_MED", "Skip-gram (Medical Corpus)"),
            ("W2V_SG_PRESS", "Skip-gram (Press Corpus)"),
        ]
    ):
        ax = axes[idx]
        focus_words = ["solution", "traitement", "patient"]
        words, vecs, labels = get_cluster_words_and_vectors(
            models[m_key], focus_words, top_n=10
        )
        plot_tsne_clusters(
            ax,
            words,
            vecs,
            labels,
            focus_words,
            f"Semantic Space in {title}",
            perplexity=6,
        )

    plt.suptitle(
        "Domain Shift Illustration: Medical Corpus vs General Press (Skip-Gram)\n"
        "Observe how 'solution' shifts from medicinal liquid in Medical to political/social resolution in Press",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "tsne_domain_shift_medical_vs_press.png")
    plt.savefig(plot2_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot2_path}")

    # Plot 3: Skip-Gram vs FastText on Medical Corpus (Morphology & Subword comparison)
    print(
        "\n[3/3] Generating Architecture comparison plot (Word2Vec vs FastText on Medical)..."
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for idx, (m_key, title) in enumerate(
        [
            ("W2V_SG_MED", "Word2Vec Skip-gram (Medical)"),
            ("FASTTEXT_CBOW_MED", "FastText CBOW (Medical)"),
        ]
    ):
        ax = axes[idx]
        words, vecs, labels = get_cluster_words_and_vectors(
            models[m_key], candidate_words, top_n=7
        )
        plot_tsne_clusters(
            ax, words, vecs, labels, candidate_words, title, perplexity=8
        )

    plt.suptitle(
        "Architecture Comparison on Small Medical Corpus: Word2Vec vs FastText\n"
        "FastText captures morphological variants (subwords) even on small corpora",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plot3_path = os.path.join(plots_dir, "tsne_architecture_comparison_med.png")
    plt.savefig(plot3_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot3_path}")

    print("\n" + "=" * 70)
    print("All t-SNE plots generated successfully in 'plots/' directory!")
    print("=" * 70)


if __name__ == "__main__":
    main()
