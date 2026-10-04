"""Token-level CNN/BiLSTM NER with six TP1 embeddings or a random baseline."""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from gensim.models import FastText, Word2Vec
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from ner_data import ROOT, load_domain, score_sequences, tag_vocabulary

EMBEDDING_NAMES = [f"{kind}_{corpus}" for corpus in ("med", "press")
                   for kind in ("w2v_cbow", "w2v_sg", "fasttext_cbow")]


def make_vocab(train):
    counts = Counter(token for tokens, _ in train for token in tokens)
    return {token: i + 2 for i, token in enumerate(sorted(counts))}


def make_embedding_matrix(vocab, source, seed=42):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(0, 0.1, (len(vocab) + 2, 100)).astype("float32")
    matrix[0] = 0
    covered = 0
    if source != "random":
        path = ROOT / "Lab3_Embedding" / "models" / f"{source}.model"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run Lab3 train_embeddings.py first")
        model = FastText.load(str(path)) if source.startswith("fasttext") else Word2Vec.load(str(path))
        if model.vector_size != 100 or model.min_count != 1 or model.epochs < 30:
            raise ValueError(f"{path} must have dim=100, min_count=1, epochs>=30")
        for token, idx in vocab.items():
            if token in model.wv or isinstance(model, FastText):
                matrix[idx] = model.wv[token]
                covered += 1
        del model
    return torch.from_numpy(matrix), covered / max(len(vocab), 1)


class NERDataset(Dataset):
    def __init__(self, sentences, vocab, label_to_id):
        self.items = [
            (torch.tensor([vocab.get(t, 1) for t in tokens]),
             torch.tensor([label_to_id[t] for t in tags]))
            for tokens, tags in sentences
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    length = max(len(tokens) for tokens, _ in batch)
    x = torch.zeros(len(batch), length, dtype=torch.long)
    y = torch.full((len(batch), length), -100, dtype=torch.long)
    lengths = []
    for i, (tokens, tags) in enumerate(batch):
        x[i, :len(tokens)] = tokens
        y[i, :len(tags)] = tags
        lengths.append(len(tokens))
    return x, y, lengths


class TokenModel(nn.Module):
    def __init__(self, weights, n_labels, architecture):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(0.3)
        self.architecture = architecture
        if architecture == "lstm":
            self.encoder = nn.LSTM(100, 128, batch_first=True, bidirectional=True)
            output_dim = 256
        elif architecture == "cnn":
            self.encoder = nn.ModuleList([nn.Conv1d(100, 128, kernel_size=k, padding=k // 2)
                                          for k in (3, 5)])
            output_dim = 256
        else:
            raise ValueError(architecture)
        self.classifier = nn.Linear(output_dim, n_labels)

    def forward(self, ids, lengths):
        embeddings = self.dropout(self.embedding(ids))
        if self.architecture == "lstm":
            packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
            encoded, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=ids.size(1))
        else:
            channels = embeddings.transpose(1, 2)
            encoded = torch.cat([torch.relu(conv(channels)).transpose(1, 2)
                                 for conv in self.encoder], dim=-1)
        return self.classifier(self.dropout(encoded))


@torch.no_grad()
def evaluate(model, loader, labels, device):
    model.eval()
    gold, predicted = [], []
    for x, y, lengths in loader:
        logits = model(x.to(device), lengths).argmax(-1).cpu()
        for row, target, length in zip(logits, y, lengths):
            predicted.append([labels[int(i)] for i in row[:length]])
            gold.append([labels[int(i)] for i in target[:length]])
    return score_sequences(gold, predicted)


def run(domain, architecture, source, epochs=30, batch_size=32, lr=0.001, seed=42, output_dir=None):
    if epochs < 30:
        raise ValueError("Classical models require at least 30 epochs")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    splits = load_domain(domain)
    vocab = make_vocab(splits["train"])
    labels = tag_vocabulary(splits["train"])
    label_to_id = {tag: i for i, tag in enumerate(labels)}
    weights, coverage = make_embedding_matrix(vocab, source, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TokenModel(weights, len(labels), architecture).to(device)
    loaders = {split: DataLoader(NERDataset(data, vocab, label_to_id),
                                 batch_size=batch_size, shuffle=(split == "train"), collate_fn=collate)
               for split, data in splits.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    best_f1, best_state, best_epoch = -1, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for x, y, lengths in loaders["train"]:
            optimizer.zero_grad()
            logits = model(x.to(device), lengths)
            loss = criterion(logits.reshape(-1, len(labels)), y.to(device).reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item()
        dev = evaluate(model, loaders["dev"], labels, device)
        print(f"{domain} {architecture} {source} epoch={epoch}/{epochs} "
              f"loss={loss_total / len(loaders['train']):.4f} dev_f1={dev['f1']:.4f}", flush=True)
        if dev["f1"] > best_f1:
            best_f1, best_epoch = dev["f1"], epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    result = {
        "dataset": domain, "model": architecture, "embedding": source,
        "epochs": epochs, "best_epoch": best_epoch, "seed": seed,
        "embedding_coverage": coverage, "dev": evaluate(model, loaders["dev"], labels, device),
        "test": evaluate(model, loaders["test"], labels, device),
    }
    if output_dir:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / f"{domain}_{architecture}_{source}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["EMEA", "MEDLINE"], required=True)
    parser.add_argument("--model", choices=["cnn", "lstm"], required=True)
    parser.add_argument("--embedding", choices=["random"] + EMBEDDING_NAMES, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.model, args.embedding, args.epochs,
                         args.batch, args.lr, args.seed, args.output_dir), indent=2))
