"""Create the Lab4 presentation notebook from reproducible cells."""

import json
from pathlib import Path

cells = []


def md(source):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)})


def code(source):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": source.splitlines(keepends=True)})


md("""# Medical named entity recognition: TP1 embeddings versus BERT

This notebook evaluates **EMEA** and **MEDLINE** separately on QUAERO FrenchMed layer 1. It compares CNN and BiLSTM token taggers initialized with each of the six TP1 embeddings, adds a randomly initialized baseline for each architecture, and fine tunes multilingual BERT. Additional CamemBERT and DrBERT runs test French and biomedical French pretraining. That is **17 configurations per dataset**, 34 total.

All reported numbers are **entity span precision, recall, and F1** from `seqeval`, not token accuracy. The train split fits model weights; dev selects the best epoch; test is scored once after selection. A saved JSON file records each result. Cells below run the complete grid when needed; they do not substitute guessed scores.
""")

# Include a readable saved snapshot when the complete grid has finished. The
# executable cells below always reload JSON and remain the source of truth.
result_dir = Path(__file__).with_name("results")
saved = []
for path in result_dir.glob("*.json"):
    saved.append(json.loads(path.read_text(encoding="utf-8")))
if len(saved) == 34:
    lines = ["## Completed result snapshot", "",
             "Scores are exact entity span test metrics. This snapshot was generated from the JSON result files.", "",
             "| Dataset | Model | Initialization | Precision | Recall | F1 |",
             "|---|---|---|---:|---:|---:|"]
    for result in sorted(saved, key=lambda r: (r["dataset"], r["model"], r["embedding"])):
        test = result["test"]
        lines.append(f"| {result['dataset']} | {result['model']} | {result['embedding']} | "
                     f"{test['precision']:.3f} | {test['recall']:.3f} | {test['f1']:.3f} |")
    lines += ["", "### Answers from the completed runs", ""]
    for dataset in ("EMEA", "MEDLINE"):
        subset = [r for r in saved if r["dataset"] == dataset]
        winner = max(subset, key=lambda r: r["test"]["f1"])
        classical = max((r for r in subset if r["model"] in ("cnn", "lstm")),
                        key=lambda r: r["test"]["f1"])
        bert = next(r for r in subset if r["model"] == "bert")
        camembert = next(r for r in subset if r["model"] == "camembert")
        drbert = next(r for r in subset if r["model"] == "drbert")
        pairs = [next(r for r in subset if r["model"] == model and r["embedding"] == kind + "_med")["test"]["f1"]
                 - next(r for r in subset if r["model"] == model and r["embedding"] == kind + "_press")["test"]["f1"]
                 for model in ("cnn", "lstm") for kind in ("w2v_cbow", "w2v_sg", "fasttext_cbow")]
        lines.append(f"- **{dataset}:** highest test F1 is {winner['test']['f1']:.3f} "
                     f"({winner['model']}). The best classical model is {classical['model']} + "
                     f"{classical['embedding']} at {classical['test']['f1']:.3f} F1. "
                     f"Multilingual BERT scores {bert['test']['f1']:.3f} "
                     f"(gap {bert['test']['f1'] - classical['test']['f1']:+.3f}); "
                     f"CamemBERT scores {camembert['test']['f1']:.3f}; "
                     f"DrBERT scores {drbert['test']['f1']:.3f}. "
                     f"Medical minus press pretraining averaged {sum(pairs)/len(pairs):+.3f} "
                     f"across six matched pairs.")
    lines += ["", "The professor's 0.50 classical and 0.75 BERT F1 values are reference targets; see the measured scores above. Press pretraining uses a much larger corpus than medical pretraining, so any press advantage is consistent with corpus size offsetting domain match; the single seeded runs do not prove the cause."]
    md("\n".join(lines))

md("""## Setup

Open this notebook from the repository, either locally or in Colab. Git ignores the corpus and TP1 model files, so a fresh Colab clone also needs the `Lab3_Embedding/data` folder uploaded. The notebook can regenerate the six models from that data if they are missing. In Colab choose **Runtime → Change runtime type → GPU** before Transformer training. The dataset and TP1 model files are already present in the local workspace. Running the install cell is safe in a fresh environment.
""")

code("""from pathlib import Path
import os, sys, subprocess, json
import pandas as pd
from IPython.display import display, Markdown

here = Path.cwd().resolve()
candidates = [Path(os.environ.get('NLP_LABS_ROOT', here)).resolve(), here, here.parent]
candidates += [p for p in here.iterdir() if p.is_dir()]
root = next((p for p in candidates if (p / 'Lab4_NER_Eval').exists()), None)
assert root is not None, 'Set NLP_LABS_ROOT to the cloned repository path'
lab = root / 'Lab4_NER_Eval'
results_dir = lab / 'results'
results_dir.mkdir(exist_ok=True)
print('Repository:', root)
""")

code("""# Installs missing packages in a fresh machine/Colab runtime.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', str(lab / 'requirements.txt')], check=True)
import torch, gensim, transformers, seqeval
print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('Gensim:', gensim.__version__, '| Transformers:', transformers.__version__)
""")

md("""## Data and TP1 embedding checks

The five CoNLL columns are read as index, token, ignored offsets, and entity tag. Blank lines delimit sentences. The label vocabulary comes only from training. The saved TP1 models must use `dim=100`, `min_count=1`, and at least 30 training epochs. The corpus used for pretraining is either FrenchMed (`med`) or FrenchPress (`press`).
""")

code("""sys.path.insert(0, str(lab))
from ner_data import load_domain, tag_vocabulary
from cnn_classification import EMBEDDING_NAMES
from gensim.models import Word2Vec, FastText

dataset_rows = []
for domain in ('EMEA', 'MEDLINE'):
    splits = load_domain(domain)
    for split, sentences in splits.items():
        dataset_rows.append({'dataset': domain, 'split': split, 'sentences': len(sentences),
                             'tokens': sum(len(tokens) for tokens, _ in sentences)})
display(pd.DataFrame(dataset_rows))

model_paths = [root / 'Lab3_Embedding' / 'models' / f'{name}.model' for name in EMBEDDING_NAMES]
needs_training = False
for name, path in zip(EMBEDDING_NAMES, model_paths):
    if not path.exists():
        needs_training = True
        break
    saved_model = FastText.load(str(path)) if name.startswith('fasttext') else Word2Vec.load(str(path))
    needs_training = (saved_model.vector_size != 100 or saved_model.min_count != 1
                      or saved_model.epochs < 30)
    del saved_model
    if needs_training:
        break
if needs_training:
    print('Training all six TP1 models for 30 epochs...')
    subprocess.run([sys.executable, str(root / 'Lab3_Embedding' / 'train_embeddings.py'),
                    '--corpus', 'all', '--model', 'all', '--dim', '100',
                    '--min_count', '1', '--epochs', '30'], check=True)

embedding_rows = []
for name in EMBEDDING_NAMES:
    path = root / 'Lab3_Embedding' / 'models' / f'{name}.model'
    assert path.exists(), f'Missing {path}; run Lab3_Embedding/train_embeddings.py first'
    model = FastText.load(str(path)) if name.startswith('fasttext') else Word2Vec.load(str(path))
    embedding_rows.append({'embedding': name, 'dimension': model.vector_size,
                           'min_count': model.min_count, 'epochs': model.epochs,
                           'vocabulary': len(model.wv)})
    assert model.vector_size == 100 and model.min_count == 1 and model.epochs >= 30
    del model
display(pd.DataFrame(embedding_rows))
""")

md("""If the six TP1 files are absent or were trained for fewer than 30 epochs, run the next cell. It trains all six with the required settings and overwrites the old model files. The checked files in this repository already meet the settings, so this step is usually unnecessary.
""")

code("""# Run only when the TP1 models need regeneration.
# subprocess.run([sys.executable, str(root / 'Lab3_Embedding' / 'train_embeddings.py'),
#                 '--corpus', 'all', '--model', 'all', '--dim', '100',
#                 '--min_count', '1', '--epochs', '30'], check=True)
""")

md("""## Training

For each EMEA and MEDLINE split, CNN and BiLSTM each run with random embeddings plus Word2Vec CBOW, Word2Vec Skip Gram, and FastText CBOW trained on both medical and press corpora. Embedding weights are trainable during NER fitting. Padding tokens have ignored labels; the BiLSTM packs variable length sequences. The CNN uses same length convolutions and a classifier at every token. All classical runs complete **30 epochs**; the best dev F1 checkpoint is used on test.

BERT uses `AutoModelForTokenClassification` with `bert-base-multilingual-cased`. Additional runs use `camembert-base` (French) and [`Dr-BERT/DrBERT-7GB`](https://huggingface.co/Dr-BERT/DrBERT-7GB) (French biomedical). Only the first subtoken receives each word label; all other subtokens and special tokens use `-100`. Each Transformer runs **5 epochs** and chooses the best dev F1. Tokenization raises an error if `max_length` would drop words. Full training is compute intensive, particularly without a GPU. Existing valid result files are skipped so the grid can resume.

The six 100-dimensional TP1 word embedding matrices initialize both CNN and BiLSTM. BERT-family models use their own pretrained subword embedding matrices, so the TP1 word vectors are not inserted into them.
""")

code("""# Full classical grid: 2 datasets × 2 architectures × (6 TP1 + random) = 28 runs.
# This cell can take substantial time. Run it once; reruns skip completed results.
subprocess.run([sys.executable, str(lab / 'run_all.py'), '--only', 'classical',
                '--output-dir', str(results_dir)], check=True)
""")

code("""# Two five-epoch BERT runs, one per dataset. A GPU is strongly recommended.
subprocess.run([sys.executable, str(lab / 'run_all.py'), '--only', 'bert',
                '--output-dir', str(results_dir)], check=True)
""")

code("""# Additional French Transformer comparison, also five epochs per dataset.
subprocess.run([sys.executable, str(lab / 'run_all.py'), '--only', 'bert',
                '--bert-model', 'camembert-base', '--output-dir', str(results_dir)], check=True)
""")

code("""# Additional biomedical French Transformer comparison, five epochs per dataset.
subprocess.run([sys.executable, str(lab / 'run_all.py'), '--only', 'bert',
                '--bert-model', 'Dr-BERT/DrBERT-7GB', '--output-dir', str(results_dir)], check=True)
""")

md("""## Results

The next cell reads only completed runs. A missing file is labeled as missing, never as a zero score. The coverage value is the fraction of the supervised training vocabulary that received pretrained vectors. FastText can synthesize vectors for unseen words using subword units.
""")

code("""expected = [(d, m, e) for d in ('EMEA', 'MEDLINE')
            for m in ('cnn', 'lstm') for e in ['random'] + EMBEDDING_NAMES]
expected += [(d, m, 'contextual') for d in ('EMEA', 'MEDLINE')
             for m in ('bert', 'camembert', 'drbert')]
rows = []
for dataset, model, embedding in expected:
    filename = (f'{dataset}_{model}_{embedding}.json' if model in ('cnn', 'lstm')
                else f'{dataset}_{model}.json')
    path = results_dir / filename
    if path.exists():
        result = json.loads(path.read_text(encoding='utf-8'))
        rows.append({'dataset': dataset, 'model': model, 'embedding': embedding,
                     'epochs': result['epochs'], 'best_epoch': result['best_epoch'],
                     'coverage': result.get('embedding_coverage'),
                     'dev_f1': result['dev']['f1'],
                     'precision': result['test']['precision'],
                     'recall': result['test']['recall'], 'f1': result['test']['f1'],
                     'status': 'complete'})
    else:
        rows.append({'dataset': dataset, 'model': model, 'embedding': embedding,
                     'epochs': None, 'best_epoch': None, 'coverage': None, 'dev_f1': None,
                     'precision': None, 'recall': None, 'f1': None, 'status': 'missing'})
results = pd.DataFrame(rows)
display(results.style.format({'coverage': '{:.1%}', 'dev_f1': '{:.3f}', 'precision': '{:.3f}',
                              'recall': '{:.3f}', 'f1': '{:.3f}'}, na_rep='—'))
print(f"Completed: {(results.status == 'complete').sum()} / {len(results)}")
""")

code("""# Test F1 overview; only completed runs are shown.
import matplotlib.pyplot as plt
plotted = results[results.status == 'complete'].copy()
if len(plotted):
    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, len(plotted) / 4)), sharex=True)
    for axis, dataset in zip(axes, ('EMEA', 'MEDLINE')):
        part = plotted[plotted.dataset == dataset].sort_values('f1')
        axis.barh(part.model + ' · ' + part.embedding, part.f1)
        axis.set_title(dataset)
        axis.set_xlim(0, 1)
        axis.axvline(.5, color='gray', linestyle='--', alpha=.6)
        axis.axvline(.75, color='gray', linestyle=':', alpha=.6)
        axis.set_xlabel('Entity span test F1')
    fig.tight_layout()
    plt.show()
""")

md("""## Answers to the lab questions

Interpret the comparisons only when the relevant runs are complete. The code below computes the winning model on each dataset, paired medical versus press differences within the same architecture and embedding method, and the BERT gap against the best classical model. The stated `0.50` classical and `0.75` BERT F1 targets are targets, not assumed outcomes.
""")

code("""done = results[results.status == 'complete'].copy()
for dataset in ('EMEA', 'MEDLINE'):
    subset = done[done.dataset == dataset]
    if len(subset) == 17:
        winner = subset.loc[subset.f1.idxmax()]
        display(Markdown(f"**{dataset}:** highest observed test F1 = {winner.f1:.3f} "
                         f"({winner.model}, {winner.embedding}); "
                         f"precision {winner.precision:.3f}, recall {winner.recall:.3f}."))
    else:
        display(Markdown(f"**{dataset}:** {len(subset)}/17 runs complete; final winner pending."))

pairs = []
for dataset in ('EMEA', 'MEDLINE'):
    for architecture in ('cnn', 'lstm'):
        for kind in ('w2v_cbow', 'w2v_sg', 'fasttext_cbow'):
            med = done[(done.dataset == dataset) & (done.model == architecture)
                       & (done.embedding == kind + '_med')]
            press = done[(done.dataset == dataset) & (done.model == architecture)
                         & (done.embedding == kind + '_press')]
            if len(med) and len(press):
                pairs.append({'dataset': dataset, 'model': architecture, 'method': kind,
                              'medical_f1': med.iloc[0].f1, 'press_f1': press.iloc[0].f1,
                              'medical_minus_press': med.iloc[0].f1 - press.iloc[0].f1})
display(Markdown('**TP1 corpus comparison** (positive difference favors medical pretraining)'))
display(pd.DataFrame(pairs).style.format(precision=3) if pairs else 'Awaiting paired runs')

comparison = []
for dataset in ('EMEA', 'MEDLINE'):
    subset = done[done.dataset == dataset]
    classical = subset[subset.model.isin(['cnn', 'lstm'])]
    bert = subset[subset.model == 'bert']
    camembert = subset[subset.model == 'camembert']
    drbert = subset[subset.model == 'drbert']
    if len(classical) == 14 and len(bert) == 1 and len(camembert) == 1 and len(drbert) == 1:
        best = classical.loc[classical.f1.idxmax()]
        comparison.append({'dataset': dataset, 'best_classical': f'{best.model} + {best.embedding}',
                           'classical_f1': best.f1, 'bert_f1': bert.iloc[0].f1,
                           'camembert_f1': camembert.iloc[0].f1,
                           'drbert_f1': drbert.iloc[0].f1,
                           'bert_minus_classical': bert.iloc[0].f1 - best.f1,
                           'classical_above_0.50': best.f1 >= .50,
                           'bert_above_0.75': bert.iloc[0].f1 >= .75,
                           'camembert_above_0.75': camembert.iloc[0].f1 >= .75,
                           'drbert_above_0.75': drbert.iloc[0].f1 >= .75})
display(Markdown('**Transformers versus best classical model**'))
display(pd.DataFrame(comparison).style.format(precision=3) if comparison else 'Awaiting complete runs')
""")

md("""### Interpretation guide

The two datasets differ in source and entity distribution, so compare winners **within each dataset** first. A positive medical minus press score supports an in-domain benefit for that fixed architecture and embedding method; a negative score favors the larger press corpus. Coverage helps distinguish vocabulary effects from embedding quality, but a single seeded run does not establish statistical significance. BERT uses contextual subword representations, while the CNN/BiLSTM start from static TP1 vectors. Report the measured gap from the table above, including any failure to reach the professor's F1 targets.
""")

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}
path = Path(__file__).with_name("Lab4_Medical_NER_Results.ipynb")
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(path)
