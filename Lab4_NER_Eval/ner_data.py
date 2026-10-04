"""Shared sentence-level data and entity-level evaluation for QUAERO NER."""

from pathlib import Path

from seqeval.metrics import f1_score, precision_score, recall_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Lab3_Embedding" / "data" / "QUAERO_FrenchMed"
SPLITS = {
    domain: {
        split: DATA / domain / f"{domain}{split}_layer1_ID.conll"
        for split in ("train", "dev", "test")
    }
    for domain in ("EMEA", "MEDLINE")
}


def read_conll(path):
    """Return [(tokens, tags)]; check token indices and preserve sentence boundaries."""
    sentences, tokens, tags = [], [], []
    with open(path, encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.strip().split()
            if not fields:
                if tokens:
                    sentences.append((tokens, tags))
                    tokens, tags = [], []
                continue
            if len(fields) != 5 or int(fields[0]) != len(tokens) + 1:
                raise ValueError(f"Invalid CoNLL row at {path}:{line_number}: {line!r}")
            tokens.append(fields[1])
            tags.append(fields[-1])
    if tokens:
        sentences.append((tokens, tags))
    return sentences


def load_domain(domain):
    if domain not in SPLITS:
        raise ValueError(f"Unknown domain: {domain}")
    return {split: read_conll(path) for split, path in SPLITS[domain].items()}


def tag_vocabulary(train):
    # Label mapping comes from training only; unseen dev/test labels are errors.
    return ["O"] + sorted({tag for _, tags in train for tag in tags} - {"O"})


def score_sequences(gold, predicted):
    if len(gold) != len(predicted) or any(len(a) != len(b) for a, b in zip(gold, predicted)):
        raise ValueError("Gold and prediction sentence/token counts differ")
    return {
        "precision": float(precision_score(gold, predicted, zero_division=0)),
        "recall": float(recall_score(gold, predicted, zero_division=0)),
        "f1": float(f1_score(gold, predicted, zero_division=0)),
    }
