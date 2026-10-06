"""
transformers_classification.py  (NER - token classification with BERT)
-----------------------------------------------------------------------
Fine-tunes a Hugging Face transformer (AutoModelForTokenClassification)
on the QUAERO_FrenchMed NER dataset.

Usage examples:
  python transformers_classification.py --model bert-base-multilingual-uncased --corpus EMEA --epochs 5
  python transformers_classification.py --model camembert-base --corpus MEDLINE --epochs 5

Arguments:
  --model   : HuggingFace model id (default: bert-base-multilingual-uncased)
  --corpus  : EMEA | MEDLINE
  --epochs  : int (default 5)
  --batch   : int (default 16)
  --lr      : float (default 2e-5)
  --max_len : int (default 128)
  --data_dir: path to QUAERO_FrenchMed directory
  --out_dir : directory for results JSON
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score   as seq_f1,
    precision_score as seq_prec,
    recall_score    as seq_rec,
)

from ner_data import read_conll, build_vocab

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "..", "Lab3_Embedding", "data", "QUAERO_FrenchMed")

parser = argparse.ArgumentParser(description="NER fine-tuning with Transformer (BERT)")
parser.add_argument("--model",   default="bert-base-multilingual-uncased")
parser.add_argument("--corpus",  default="EMEA", choices=["EMEA", "MEDLINE"])
parser.add_argument("--epochs",  default=5,    type=int)
parser.add_argument("--batch",   default=16,   type=int)
parser.add_argument("--lr",      default=2e-5, type=float)
parser.add_argument("--max_len", default=128,  type=int)
parser.add_argument("--data_dir", default=DATA_ROOT)
parser.add_argument("--out_dir",  default=os.path.join(BASE_DIR, "results"))
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(device))
if device.type == "cuda":
    print("GPU: {}".format(torch.cuda.get_device_name(0)))

# ---------------------------------------------------------------------------
# Load CoNLL data
# ---------------------------------------------------------------------------
corpus    = args.corpus
conll_dir = os.path.join(args.data_dir, corpus)

train_sents = read_conll(os.path.join(conll_dir, "{}train_layer1_ID.conll".format(corpus)))
dev_sents   = read_conll(os.path.join(conll_dir, "{}dev_layer1_ID.conll".format(corpus)))
test_sents  = read_conll(os.path.join(conll_dir, "{}test_layer1_ID.conll".format(corpus)))

print("Loaded {} train | {} dev | {} test sentences".format(
    len(train_sents), len(dev_sents), len(test_sents)))

_, tag2idx, idx2tag = build_vocab([train_sents, dev_sents, test_sents])
num_labels = len(tag2idx)
print("Labels ({}): {}".format(num_labels, list(tag2idx.keys())))

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(args.model)

# ---------------------------------------------------------------------------
# Dataset with word-to-subtoken alignment
# ---------------------------------------------------------------------------

class NERDataset(torch.utils.data.Dataset):
    """
    Tokenises each sentence word-by-word (is_split_into_words=True).
    The NER label of a word is assigned only to its first sub-token;
    all subsequent sub-tokens and special tokens receive label -100
    (ignored by CrossEntropyLoss).
    """

    def __init__(self, sentences, tag2idx, tokenizer, max_len=128):
        self.samples = []
        for sent in sentences:
            words = [w for w, _ in sent]
            tags  = [t for _, t in sent]
            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_len,
                padding="max_length",
                return_tensors="pt",
            )
            word_ids = enc.word_ids(batch_index=0)
            labels = []
            prev_word_id = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)          # [CLS], [SEP], PAD
                elif wid != prev_word_id:
                    labels.append(tag2idx.get(tags[wid], 0))   # first sub-token
                else:
                    labels.append(-100)          # continuation sub-token
                prev_word_id = wid

            self.samples.append({
                "input_ids":      enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels":         torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


print("Building datasets ...")
train_ds = NERDataset(train_sents, tag2idx, tokenizer, args.max_len)
dev_ds   = NERDataset(dev_sents,   tag2idx, tokenizer, args.max_len)
test_ds  = NERDataset(test_sents,  tag2idx, tokenizer, args.max_len)

train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
dev_loader   = DataLoader(dev_ds,   batch_size=args.batch, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = AutoModelForTokenClassification.from_pretrained(
    args.model,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
)
model = model.to(device)
print("\nModel: {}  |  labels: {}".format(args.model, num_labels))

# ---------------------------------------------------------------------------
# Optimiser + linear warmup scheduler
# ---------------------------------------------------------------------------
optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total_steps  = len(train_loader) * args.epochs
scheduler    = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_batch(logits, labels):
    """Convert (B, T, C) logits and (B, T) label tensors to seqeval lists."""
    pred_ids = logits.argmax(dim=-1)
    preds_all, golds_all = [], []
    for pred_seq, gold_seq in zip(pred_ids, labels):
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
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            p, g = decode_batch(outputs.logits.cpu(), labels.cpu())
            all_preds.extend(p)
            all_golds.extend(g)
    f1 = seq_f1(all_golds, all_preds, average="weighted", zero_division=0)
    return f1, all_preds, all_golds


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_f1    = -1.0   # start below 0 so epoch 1 always saves a checkpoint
best_state     = None
model_tag      = args.model.replace("/", "_")
best_ckpt_path = os.path.join(args.out_dir, "{}_{}_best.pt".format(corpus, model_tag))

print("\n" + "="*60)
print("Fine-tuning for {} epochs ...".format(args.epochs))
print("="*60)

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0.0

    for batch in tqdm(train_loader,
                      desc="Epoch {}/{}".format(epoch, args.epochs),
                      leave=False):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    val_f1, _, _ = evaluate(dev_loader)
    print("Epoch {:3d} | loss={:.4f} | val_f1={:.4f}".format(epoch, avg_loss, val_f1))

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        try:
            tmp_path = best_ckpt_path + ".tmp"
            torch.save(best_state, tmp_path)
            if os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            os.rename(tmp_path, best_ckpt_path)
        except Exception as e:
            pass

# ---------------------------------------------------------------------------
# Test evaluation using best checkpoint
# ---------------------------------------------------------------------------
if best_state is not None:
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
elif os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
test_f1, test_preds, test_golds = evaluate(test_loader)

print("\n" + "="*60)
print("TEST RESULTS  corpus={}  model={}".format(corpus, args.model))
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
    "embedding":             "transformer",
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

out_name = "{}_{}.json".format(corpus, model_tag)
out_path = os.path.join(args.out_dir, out_name)
with open(out_path, "w", encoding="utf-8") as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print("\nResults saved to: {}".format(out_path))
