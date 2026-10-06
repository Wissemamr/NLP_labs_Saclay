"""
ner_data.py
-----------
CoNLL data loader for the QUAERO_FrenchMed NER dataset.
Handles the 5-column CoNLL format:  token_idx  token  start  end  tag
Sentences are separated by blank lines.
"""

import os
import numpy as np
from collections import Counter


def read_conll(filepath):
    """
    Read a CoNLL-format file and return a list of sentences.
    Each sentence is a list of (token, tag) tuples.

    Format: token_idx  token  start  end  tag
    Only token (col 1) and tag (last col) are used.
    """
    sentences = []
    current = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if current:
                    sentences.append(current)
                    current = []
            else:
                parts = line.split()
                if len(parts) >= 2:
                    token = parts[1]
                    tag = parts[-1] if len(parts) >= 5 else "O"
                    current.append((token, tag))
    if current:
        sentences.append(current)
    return sentences


def build_vocab(sentences_list, max_vocab=-1):
    """
    Build word2idx and tag2idx from a list of sentence lists.
    Each sentence is [(token, tag), ...].
    """
    word_counter = Counter()
    tag_set = set()
    for sentences in sentences_list:
        for sent in sentences:
            for token, tag in sent:
                word_counter[token.lower()] += 1
                tag_set.add(tag)

    vocab = sorted(word_counter, key=word_counter.get, reverse=True)
    if max_vocab != -1:
        vocab = vocab[:max_vocab]

    # Reserve 0=PAD, 1=UNK
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for w in vocab:
        word2idx[w] = len(word2idx)

    # Tag mapping -- keep O=0 for convenience
    tag_list = sorted(tag_set)
    if "O" in tag_list:
        tag_list.remove("O")
        tag_list = ["O"] + tag_list
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    return word2idx, tag2idx, idx2tag


def encode_sentences(sentences, word2idx, tag2idx, seq_length=128):
    """
    Encode a list of sentences into padded integer arrays.

    Returns:
        X  -- (N, seq_length) int64 array
        Y  -- (N, seq_length) int64 array  (-100 marks padding, ignored by loss)
    """
    PAD_ID  = word2idx["<PAD>"]
    UNK_ID  = word2idx["<UNK>"]
    PAD_TAG = -100  # ignored in CrossEntropyLoss

    X = np.full((len(sentences), seq_length), PAD_ID,  dtype=np.int64)
    Y = np.full((len(sentences), seq_length), PAD_TAG, dtype=np.int64)

    for i, sent in enumerate(sentences):
        for j, (token, tag) in enumerate(sent[:seq_length]):
            X[i, j] = word2idx.get(token.lower(), UNK_ID)
            Y[i, j] = tag2idx.get(tag, tag2idx["O"])

    return X, Y


def load_embedding_matrix(word2idx, vec_path, embedding_dim=100):
    """
    Load pre-trained word vectors and build an embedding matrix aligned with word2idx.
    Words not found in the pre-trained file get a small random initialisation.
    """
    vocab_size = len(word2idx)
    matrix = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
    # PAD row stays zeros
    matrix[0] = np.zeros(embedding_dim)

    found = 0
    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f):
            if line_no == 0:
                # header line: vocab_size dim
                continue
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word.lower() in word2idx:
                try:
                    vec = np.array(parts[1:], dtype=np.float32)
                    if len(vec) == embedding_dim:
                        matrix[word2idx[word.lower()]] = vec
                        found += 1
                except ValueError:
                    pass

    print("  -> Loaded {}/{} vectors from {}".format(
        found, vocab_size, os.path.basename(vec_path)))
    return matrix
