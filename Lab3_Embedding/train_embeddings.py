"""
train_embeddings.py
-------------------
Script to train Word2Vec (CBOW, Skip-Gram) and FastText (CBOW) embeddings
on medical (QUAERO_FrenchMed) and non-medical (QUAERO_FrenchPress) corpora.

Hyperparameters (as specified):
- Vector dimension: dim = 100
- Minimum count: min_count = 1
- Window size: window = 5
"""

import os
import sys
import time
import argparse
from gensim.models import Word2Vec, FastText


def read_corpus(filepath):
    """
    Reads a text file in 'one sentence per line' (.ospl) format
    where tokens are space-separated.
    Yields list of tokens per sentence.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Corpus file not found: {filepath}")

    sentences = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            if tokens:
                sentences.append(tokens)
    return sentences


class SentenceIterator:
    """Memory-efficient sentence iterator for large corpora."""

    def __init__(self, filepath):
        self.filepath = filepath

    def __iter__(self):
        with open(self.filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:
                    yield tokens


def train_model(
    corpus_type,
    model_type,
    data_path,
    output_dir,
    dim=100,
    min_count=1,
    window=5,
    epochs=30,
    workers=4,
):
    """
    Trains and saves a specified embedding model.
    """
    print(f"\n========================================================")
    print(f"Training: {model_type.upper()} on {corpus_type.upper()} corpus")
    print(f"Corpus file: {data_path}")
    print(
        f"Parameters: dim={dim}, min_count={min_count}, window={window}, epochs={epochs}"
    )
    print(f"========================================================")

    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()

    # Preload sentences into memory for fast multi-epoch training
    sentences = read_corpus(data_path)
    total_sentences = len(sentences)
    total_tokens = sum(len(s) for s in sentences)
    print(
        f"Loaded {total_sentences:,} sentences ({total_tokens:,} tokens) in {time.time() - t0:.2f}s"
    )

    t_train_start = time.time()
    if model_type == "w2v_cbow":
        # sg=0 specifies CBOW
        model = Word2Vec(
            sentences=sentences,
            vector_size=dim,
            window=window,
            min_count=min_count,
            sg=0,
            epochs=epochs,
            workers=workers,
            seed=42,
        )
    elif model_type == "w2v_sg":
        # sg=1 specifies Skip-gram
        model = Word2Vec(
            sentences=sentences,
            vector_size=dim,
            window=window,
            min_count=min_count,
            sg=1,
            epochs=epochs,
            workers=workers,
            seed=42,
        )
    elif model_type == "fasttext_cbow":
        # FastText CBOW (sg=0)
        model = FastText(
            sentences=sentences,
            vector_size=dim,
            window=window,
            min_count=min_count,
            sg=0,
            epochs=epochs,
            workers=workers,
            seed=42,
        )
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Expected 'w2v_cbow', 'w2v_sg', or 'fasttext_cbow'."
        )

    train_time = time.time() - t_train_start
    vocab_size = len(model.wv)
    print(
        f"Training completed in {train_time:.2f}s. Vocabulary size: {vocab_size:,} words."
    )

    # Save full model (contains model state, vocab, weights)
    model_filename = f"{model_type}_{corpus_type}.model"
    model_save_path = os.path.join(output_dir, model_filename)
    model.save(model_save_path)
    print(f"Saved full model to: {model_save_path}")

    # Save vector embeddings in standard word2vec text format (.vec)
    vec_filename = f"{model_type}_{corpus_type}.vec"
    vec_save_path = os.path.join(output_dir, vec_filename)
    model.wv.save_word2vec_format(vec_save_path, binary=False)
    print(f"Saved vector file to: {vec_save_path}")

    return model_save_path, vec_save_path


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec and FastText embeddings on QUAERO corpora."
    )
    parser.add_argument(
        "--corpus",
        choices=["med", "press", "all"],
        default="all",
        help="Corpus to train on ('med', 'press', or 'all')",
    )
    parser.add_argument(
        "--model",
        choices=["w2v_cbow", "w2v_sg", "fasttext_cbow", "all"],
        default="all",
        help="Embedding model type ('w2v_cbow', 'w2v_sg', 'fasttext_cbow', or 'all')",
    )
    parser.add_argument(
        "--dim", type=int, default=100, help="Embedding dimension (default: 100)"
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Minimum word count threshold (default: 1)",
    )
    parser.add_argument(
        "--window", type=int, default=5, help="Context window size (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs (default: 30)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Directory to save models and vectors",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )

    args = parser.parse_args()

    # Paths to the two corpora
    base_dir = os.path.dirname(os.path.abspath(__file__))
    corpora_paths = {
        "med": os.path.join(
            base_dir, "data", "QUAERO_FrenchMed", "QUAERO_FrenchMed_traindev.ospl"
        ),
        "press": os.path.join(
            base_dir, "data", "QUAERO_FrenchPress", "QUAERO_FrenchPress_traindev.ospl"
        ),
    }

    corpora_to_train = ["med", "press"] if args.corpus == "all" else [args.corpus]
    models_to_train = (
        ["w2v_cbow", "w2v_sg", "fasttext_cbow"] if args.model == "all" else [args.model]
    )

    print(f"Starting training pipeline...")
    print(f"Target corpora: {corpora_to_train}")
    print(f"Target models: {models_to_train}")

    start_all = time.time()
    trained_files = []

    for corp in corpora_to_train:
        data_path = corpora_paths[corp]
        for m_type in models_to_train:
            m_path, v_path = train_model(
                corpus_type=corp,
                model_type=m_type,
                data_path=data_path,
                output_dir=os.path.join(base_dir, args.output_dir),
                dim=args.dim,
                min_count=args.min_count,
                window=args.window,
                epochs=args.epochs,
                workers=args.workers,
            )
            trained_files.append((corp, m_type, m_path, v_path))

    print(f"\n========================================================")
    print(
        f"All {len(trained_files)} models successfully trained in {time.time() - start_all:.2f}s!"
    )
    print(f"========================================================")
    for corp, m_type, m_path, v_path in trained_files:
        print(f" - [{corp.upper()}] {m_type:15s} -> {m_path}")


if __name__ == "__main__":
    main()
