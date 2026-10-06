"""
run_all.py
----------
Orchestrates all NER experiments for Lab4_NER_Eval_2.

Runs the following grid:
  Models      : lstm, cnn
  Corpora     : EMEA, MEDLINE
  Embeddings  : random (baseline) + 6 pre-trained from Lab3

  Transformers: bert-base-multilingual-uncased, camembert-base
  Corpora     : EMEA, MEDLINE

Results are saved as JSON files in the results/ directory.

Usage:
  python run_all.py                      # run everything
  python run_all.py --skip_transformer   # only classical models
  python run_all.py --skip_classical     # only transformers
  python run_all.py --corpus EMEA        # only EMEA corpus
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

BASE_DIR    = Path(__file__).parent
PYTHON      = sys.executable
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--skip_classical",     action="store_true")
parser.add_argument("--skip_transformer",   action="store_true")
parser.add_argument("--corpus",             default="all", choices=["all", "EMEA", "MEDLINE"])
parser.add_argument("--classical_epochs",   default=30, type=int)
parser.add_argument("--transformer_epochs", default=5,  type=int)
args = parser.parse_args()

corpora = ["EMEA", "MEDLINE"] if args.corpus == "all" else [args.corpus]

EMBEDDINGS = [
    "random",
    "w2v_cbow_med",
    "w2v_cbow_press",
    "w2v_sg_med",
    "w2v_sg_press",
    "fasttext_cbow_med",
    "fasttext_cbow_press",
]

TRANSFORMER_MODELS = [
    "bert-base-multilingual-uncased",
    "camembert-base",
    "Dr-BERT/DrBERT-7GB",
]


def run(cmd, label):
    print("\n" + "="*70)
    print("RUNNING: {}".format(label))
    print("="*70)
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    if result.returncode != 0:
        print("[WARNING] Command failed (return code {}): {}".format(
            result.returncode, label))
    return result.returncode


# ---------------------------------------------------------------------------
# 1. Classical models  (LSTM / CNN  x  7 embeddings  x  2 corpora)
# ---------------------------------------------------------------------------
if not args.skip_classical:
    for corpus in corpora:
        for model in ["lstm", "cnn"]:
            for emb in EMBEDDINGS:
                out_file = RESULTS_DIR / "{}_{}_{}.json".format(corpus, model, emb)
                if out_file.exists():
                    print("[SKIP - already done] {}".format(out_file.name))
                    continue
                cmd = [
                    PYTHON, "cnn_classification.py",
                    "--model",   model,
                    "--corpus",  corpus,
                    "--emb",     emb,
                    "--epochs",  str(args.classical_epochs),
                    "--out_dir", str(RESULTS_DIR),
                ]
                run(cmd, "{} | {} | {}".format(model.upper(), corpus, emb))

# ---------------------------------------------------------------------------
# 2. Transformer models
# ---------------------------------------------------------------------------
if not args.skip_transformer:
    for corpus in corpora:
        for tmodel in TRANSFORMER_MODELS:
            model_tag = tmodel.replace("/", "_")
            out_file = RESULTS_DIR / "{}_{}.json".format(corpus, model_tag)
            if out_file.exists():
                print("[SKIP - already done] {}".format(out_file.name))
                continue
            cmd = [
                PYTHON, "transformers_classification.py",
                "--model",   tmodel,
                "--corpus",  corpus,
                "--epochs",  str(args.transformer_epochs),
                "--out_dir", str(RESULTS_DIR),
            ]
            run(cmd, "TRANSFORMER | {} | {}".format(corpus, tmodel))

# ---------------------------------------------------------------------------
# 3. Print summary table
# ---------------------------------------------------------------------------
print("\n\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print("{:<10} {:<8} {:<28} {:>6} {:>6} {:>6}".format(
    "Corpus", "Model", "Embedding", "P", "R", "F1"))
print("-" * 70)

for json_file in sorted(RESULTS_DIR.glob("*.json")):
    try:
        with open(json_file, encoding="utf-8") as f:
            r = json.load(f)
        corpus = r.get("corpus", "?")
        model  = r.get("model",  "?")
        emb    = r.get("embedding", "?")
        p      = r.get("test_precision", 0)
        rec    = r.get("test_recall",    0)
        f1     = r.get("test_f1",        0)
        print("{:<10} {:<8} {:<28} {:>6.4f} {:>6.4f} {:>6.4f}".format(
            corpus, model, emb, p, rec, f1))
    except Exception as e:
        print("  Could not read {}: {}".format(json_file.name, e))

print("=" * 70)
