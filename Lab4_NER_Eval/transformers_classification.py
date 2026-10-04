"""Fine-tune a multilingual Transformer for QUAERO token classification."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer

from ner_data import load_domain, score_sequences, tag_vocabulary


class EncodedNER(Dataset):
    def __init__(self, sentences, tokenizer, label_to_id, max_length):
        self.items = []
        self.truncated_words = 0
        for words, tags in sentences:
            encoded = tokenizer(words, is_split_into_words=True, truncation=True,
                                max_length=max_length, return_attention_mask=True)
            word_ids = encoded.word_ids()
            observed = {i for i in word_ids if i is not None}
            if len(observed) != len(words):
                self.truncated_words += len(words) - len(observed)
            previous = None
            labels = []
            for word_id in word_ids:
                if word_id is None or word_id == previous:
                    labels.append(-100)
                else:
                    labels.append(label_to_id[tags[word_id]])
                previous = word_id
            self.items.append((encoded["input_ids"], encoded["attention_mask"], labels))
        if self.truncated_words:
            raise ValueError(f"{self.truncated_words} words truncated; increase --max-len")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch, pad_id):
    length = max(len(ids) for ids, _, _ in batch)
    result = {"input_ids": [], "attention_mask": [], "labels": []}
    for ids, mask, labels in batch:
        n = length - len(ids)
        result["input_ids"].append(ids + [pad_id] * n)
        result["attention_mask"].append(mask + [0] * n)
        result["labels"].append(labels + [-100] * n)
    return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


@torch.no_grad()
def evaluate(model, loader, labels, device):
    model.eval()
    gold, predicted = [], []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        predictions = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"]).logits.argmax(-1)
        for row, target in zip(predictions.cpu(), batch["labels"].cpu()):
            keep = target != -100
            predicted.append([labels[int(i)] for i in row[keep]])
            gold.append([labels[int(i)] for i in target[keep]])
    return score_sequences(gold, predicted)


def run(domain, model_name="bert-base-multilingual-cased", epochs=5, batch_size=8,
        lr=2e-5, max_length=512, seed=42, output_dir=None):
    if epochs < 5:
        raise ValueError("BERT requires at least 5 epochs")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    lower_name = model_name.lower()
    model_label = ("drbert" if "drbert" in lower_name else
                   "camembert" if "camembert" in lower_name else "bert")
    splits = load_domain(domain)
    labels = tag_vocabulary(splits["train"])
    label_to_id = {tag: i for i, tag in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for word-level label alignment")
    loaders = {}
    for split, sentences in splits.items():
        dataset = EncodedNER(sentences, tokenizer, label_to_id, max_length)
        loaders[split] = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(labels), id2label={i: tag for i, tag in enumerate(labels)},
        label2id=label_to_id, ignore_mismatched_sizes=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best_f1, best_epoch, best_path = -1, 0, None
    if output_dir:
        best_path = Path(output_dir) / f"{domain}_{model_label}_checkpoint.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loaders["train"]:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        dev = evaluate(model, loaders["dev"], labels, device)
        print(f"{domain} {model_label} epoch={epoch}/{epochs} "
              f"loss={total_loss / len(loaders['train']):.4f} dev_f1={dev['f1']:.4f}", flush=True)
        if dev["f1"] > best_f1:
            best_f1, best_epoch = dev["f1"], epoch
            if best_path:
                torch.save(model.state_dict(), best_path)
            else:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)
                          if best_path else best_state)
    result = {
        "dataset": domain, "model": model_label, "embedding": "contextual",
        "checkpoint": model_name, "epochs": epochs, "best_epoch": best_epoch,
        "seed": seed, "dev": evaluate(model, loaders["dev"], labels, device),
        "test": evaluate(model, loaders["test"], labels, device),
    }
    if output_dir:
        (Path(output_dir) / f"{domain}_{model_label}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["EMEA", "MEDLINE"], required=True)
    parser.add_argument("--model", default="bert-base-multilingual-cased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.model, args.epochs, args.batch,
                         args.lr, args.max_len, args.seed, args.output_dir), indent=2))
