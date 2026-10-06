"""
cnn_classification.py  (NER - token classification)
----------------------------------------------------
Trains LSTM or CNN models for Named Entity Recognition using token-level
sequence labelling. Supports random embeddings and 6 pre-trained embedding
types learned in Lab3 (TP1).

Usage examples:
  python cnn_classification.py --model lstm --corpus EMEA --emb random --epochs 30
  python cnn_classification.py --model cnn  --corpus MEDLINE --emb w2v_cbow_med --epochs 30

Arguments:
  --model   : lstm | cnn
  --corpus  : EMEA | MEDLINE
  --emb     : random | w2v_cbow_med | w2v_cbow_press | w2v_sg_med | w2v_sg_press
              | fasttext_cbow_med | fasttext_cbow_press
  --epochs  : int (default 30)
  --batch   : int (default 32)
  --seq_len : int (default 128)
  --lr      : float (default 1e-3)
  --data_dir: path to QUAERO_FrenchMed directory
  --emb_dir : path to Lab3 models directory containing .vec files
  --out_dir : directory for results JSON
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ner_data import read_conll, build_vocab, encode_sentences, load_embedding_matrix

from seqeval.metrics import classification_report as seq_classification_report
from seqeval.metrics import f1_score as seq_f1
from seqeval.metrics import precision_score as seq_prec
from seqeval.metrics import recall_score as seq_rec

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LAB3_MODELS = os.path.join(BASE_DIR, "..", "Lab3_Embedding", "models")
DATA_ROOT   = os.path.join(BASE_DIR, "..", "Lab3_Embedding", "data", "QUAERO_FrenchMed")

parser = argparse.ArgumentParser(description="NER with LSTM/CNN - token classification")
parser.add_argument("--model",   default="lstm", choices=["lstm", "cnn"])
parser.add_argument("--corpus",  default="EMEA", choices=["EMEA", "MEDLINE"])
parser.add_argument("--emb",     default="random",
                    choices=["random",
                             "w2v_cbow_med",   "w2v_cbow_press",
                             "w2v_sg_med",     "w2v_sg_press",
                             "fasttext_cbow_med", "fasttext_cbow_press"])
parser.add_argument("--epochs",  default=30, type=int)
parser.add_argument("--batch",   default=32, type=int)
parser.add_argument("--seq_len", default=128, type=int)
parser.add_argument("--lr",      default=1e-3, type=float)
parser.add_argument("--data_dir", default=DATA_ROOT)
parser.add_argument("--emb_dir",  default=LAB3_MODELS)
parser.add_argument("--out_dir",  default=os.path.join(BASE_DIR, "results"))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))
if device.type == "cuda":
    print("GPU: {}".format(torch.cuda.get_device_name(0)))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
corpus    = args.corpus
conll_dir = os.path.join(args.data_dir, corpus)

train_file = os.path.join(conll_dir, "{}train_layer1_ID.conll".format(corpus))
dev_file   = os.path.join(conll_dir, "{}dev_layer1_ID.conll".format(corpus))
test_file  = os.path.join(conll_dir, "{}test_layer1_ID.conll".format(corpus))

print("\nLoading CoNLL data from: {}".format(conll_dir))
train_sents = read_conll(train_file)
dev_sents   = read_conll(dev_file)
test_sents  = read_conll(test_file)
print("  train: {} | dev: {} | test: {} sentences".format(
    len(train_sents), len(dev_sents), len(test_sents)))

word2idx, tag2idx, idx2tag = build_vocab([train_sents, dev_sents, test_sents])
print("  Vocab size: {}  |  Tags: {}".format(len(word2idx), list(tag2idx.keys())))

# ---------------------------------------------------------------------------
# Encode data
# ---------------------------------------------------------------------------
SEQ_LEN = args.seq_len
train_X, train_Y = encode_sentences(train_sents, word2idx, tag2idx, SEQ_LEN)
dev_X,   dev_Y   = encode_sentences(dev_sents,   word2idx, tag2idx, SEQ_LEN)
test_X,  test_Y  = encode_sentences(test_sents,  word2idx, tag2idx, SEQ_LEN)

train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_Y))
dev_ds   = TensorDataset(torch.from_numpy(dev_X),   torch.from_numpy(dev_Y))
test_ds  = TensorDataset(torch.from_numpy(test_X),  torch.from_numpy(test_Y))

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
dev_loader   = DataLoader(dev_ds,   batch_size=args.batch, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

# ---------------------------------------------------------------------------
# Embedding matrix
# ---------------------------------------------------------------------------
EMB_DIM    = 100
vocab_size = len(word2idx)
num_tags   = len(tag2idx)

if args.emb == "random":
    print("Using random embeddings")
    pretrained_matrix = None
else:
    vec_path = os.path.join(args.emb_dir, "{}.vec".format(args.emb))
    print("Loading pre-trained vectors: {}".format(vec_path))
    pretrained_matrix = load_embedding_matrix(word2idx, vec_path, EMB_DIM)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class NERModelLSTM(nn.Module):
    """Bidirectional LSTM for token-level NER."""

    def __init__(self, vocab_size, num_tags, emb_dim=100, hidden_size=128,
                 n_layers=2, dropout=0.3, pretrained_matrix=None):
        super(NERModelLSTM, self).__init__()
        self.name = "lstm"

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_matrix, dtype=torch.float32))

        self.lstm = nn.LSTM(emb_dim, hidden_size, n_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_tags)  # bidirectional -> *2

    def forward(self, x):
        emb = self.embedding(x)       # (B, T, emb_dim)
        emb = self.dropout(emb)
        out, _ = self.lstm(emb)       # (B, T, 2*hidden)
        out = self.dropout(out)
        return self.fc(out)           # (B, T, num_tags)


class NERModelCNN(nn.Module):
    """1-D CNN over token embeddings for token-level NER."""

    def __init__(self, vocab_size, num_tags, emb_dim=100, num_filters=128,
                 kernel_sizes=(3, 5), dropout=0.3, pretrained_matrix=None):
        super(NERModelCNN, self).__init__()
        self.name = "cnn"

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        if pretrained_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.tensor(pretrained_matrix, dtype=torch.float32))

        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_tags)

    def forward(self, x):
        emb = self.embedding(x)            # (B, T, D)
        emb = self.dropout(emb)
        emb = emb.permute(0, 2, 1)        # (B, D, T)

        conv_outs = []
        for conv in self.convs:
            c = F.relu(conv(emb))          # (B, F, T')
            c = c[:, :, :x.size(1)]        # trim to original T
            conv_outs.append(c)

        cat = torch.cat(conv_outs, dim=1)  # (B, F*K, T)
        cat = self.dropout(cat)
        cat = cat.permute(0, 2, 1)         # (B, T, F*K)
        return self.fc(cat)                # (B, T, num_tags)


# ---------------------------------------------------------------------------
# Instantiate model
# ---------------------------------------------------------------------------
if args.model == "lstm":
    model = NERModelLSTM(vocab_size, num_tags, EMB_DIM,
                         hidden_size=128, n_layers=2, dropout=0.3,
                         pretrained_matrix=pretrained_matrix)
else:
    model = NERModelCNN(vocab_size, num_tags, EMB_DIM,
                        num_filters=128, kernel_sizes=(3, 5), dropout=0.3,
                        pretrained_matrix=pretrained_matrix)

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print("\nModel: {}  |  Embedding: {}  |  Params: {:,}".format(
    args.model.upper(), args.emb, total_params))

criterion = nn.CrossEntropyLoss(ignore_index=-100)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=5, factor=0.5)


# ---------------------------------------------------------------------------
# Helper: decode logits to seqeval format
# ---------------------------------------------------------------------------

def decode_predictions(logits_batch, labels_batch):
    """Convert (B, T, C) logits and (B, T) labels to seqeval list-of-lists."""
    pred_ids = logits_batch.argmax(dim=-1)
    preds_all, golds_all = [], []
    for pred_seq, gold_seq in zip(pred_ids, labels_batch):
        p_tags, g_tags = [], []
        for p, g in zip(pred_seq.tolist(), gold_seq.tolist()):
            if g == -100:
                continue
            p_tags.append(idx2tag[p])
            g_tags.append(idx2tag[g])
        preds_all.append(p_tags)
        golds_all.append(g_tags)
    return preds_all, golds_all


def evaluate(loader):
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for X_b, Y_b in loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            logits = model(X_b)
            p, g = decode_predictions(logits.cpu(), Y_b.cpu())
            all_preds.extend(p)
            all_golds.extend(g)
    f1 = seq_f1(all_golds, all_preds, average="weighted", zero_division=0)
    return f1, all_preds, all_golds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_f1    = -1.0  # start below 0 so epoch 1 always saves a checkpoint
best_state     = None
best_model_path = os.path.join(
    args.out_dir, "{}_{}_{}_best.pt".format(corpus, args.model, args.emb))

print("\n" + "="*60)
print("Training for {} epochs ...".format(args.epochs))
print("="*60)

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0

    for X_b, Y_b in tqdm(train_loader,
                          desc="Epoch {}/{}".format(epoch, args.epochs),
                          leave=False):
        X_b, Y_b = X_b.to(device), Y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)                        # (B, T, C)
        loss   = criterion(logits.view(-1, num_tags), Y_b.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1, _, _ = evaluate(dev_loader)
    scheduler.step(val_f1)

    print("Epoch {:3d} | loss={:.4f} | val_f1={:.4f}".format(epoch, avg_loss, val_f1))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        try:
            tmp_path = best_model_path + ".tmp"
            torch.save(best_state, tmp_path)
            if os.path.exists(best_model_path):
                os.remove(best_model_path)
            os.rename(tmp_path, best_model_path)
        except Exception as e:
            pass

# ---------------------------------------------------------------------------
# Test evaluation using best checkpoint
# ---------------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
elif os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
test_f1, test_preds, test_golds = evaluate(test_loader)

print("\n" + "="*60)
print("TEST RESULTS  corpus={}  model={}  emb={}".format(corpus, args.model, args.emb))
print("="*60)
report = seq_classification_report(test_golds, test_preds, zero_division=0)
print(report)
print("Weighted F1: {:.4f}".format(test_f1))

# ---------------------------------------------------------------------------
from sklearn.metrics import f1_score as sk_f1, precision_score as sk_prec, recall_score as sk_rec

g_flat = [t for s in test_golds for t in s]
p_flat = [t for s in test_preds for t in s]
entity_tags = [t for t in sorted(list(set(g_flat))) if t != 'O']

token_f1_weighted = round(sk_f1(g_flat, p_flat, average="weighted", zero_division=0), 4)
token_f1_entity   = round(sk_f1(g_flat, p_flat, labels=entity_tags, average="weighted", zero_division=0), 4)
strict_entity_f1  = round(seq_f1(test_golds, test_preds, zero_division=0), 4)

results = {
    "corpus":                corpus,
    "model":                 args.model,
    "embedding":             args.emb,
    "epochs":                args.epochs,
    "best_val_f1":           round(best_val_f1, 4),
    "test_f1":               round(test_f1, 4),
    "test_entity_strict_f1": strict_entity_f1,
    "test_token_f1":         token_f1_weighted,
    "test_token_entity_f1":  token_f1_entity,
    "test_precision":        round(seq_prec(test_golds, test_preds,
                                            average="weighted", zero_division=0), 4),
    "test_recall":           round(seq_rec(test_golds,  test_preds,
                                           average="weighted", zero_division=0), 4),
    "report":                report,
}

out_name = "{}_{}_{}.json".format(corpus, args.model, args.emb)
out_path = os.path.join(args.out_dir, out_name)
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print("\nResults saved to: {}".format(out_path))
