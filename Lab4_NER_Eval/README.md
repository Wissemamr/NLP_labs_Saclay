# Lab 4: medical named entity recognition

Open [Lab4_Medical_NER_Results.ipynb](Lab4_Medical_NER_Results.ipynb) to run the experiments and read the measured comparisons. The notebook evaluates the **EMEA** and **MEDLINE** QUAERO FrenchMed layer 1 train/dev/test splits separately. Metrics are exact entity span precision, recall, and F1 from `seqeval`.

## Setup

From the repository root, use the existing `.venv` or create a Python environment, then install dependencies:

```bash
python -m pip install -r Lab4_NER_Eval/requirements.txt
```

The local repository includes the QUAERO data and six TP1 `.model` files under `Lab3_Embedding`. The TP1 models have 100 dimensions, `min_count=1`, and 30 epochs. To regenerate them:

```bash
python Lab3_Embedding/train_embeddings.py --corpus all --model all --dim 100 --min_count 1 --epochs 30
```

For Colab, clone the repository and upload `Lab3_Embedding/data` separately; Git ignores the corpus and model folders. The notebook can regenerate missing TP1 models from the uploaded data. Install the requirements and select a GPU runtime before Transformer training.

## Run

```bash
python Lab4_NER_Eval/run_all.py --only classical
python Lab4_NER_Eval/run_all.py --only bert
python Lab4_NER_Eval/run_all.py --only bert --bert-model camembert-base
python Lab4_NER_Eval/run_all.py --only bert --bert-model Dr-BERT/DrBERT-7GB
```

`run_all.py` skips completed result files and can resume an interrupted grid. Use `--force` to repeat every run. Each classical run trains for 30 epochs; each BERT run trains for 5. Validation F1 chooses the checkpoint, and the selected checkpoint is evaluated on the test split. The resulting JSON files appear in `Lab4_NER_Eval/results`.

The classical grid comprises two datasets × two architectures (CNN, BiLSTM) × seven initializations (six TP1 embeddings and random), or 28 runs. Multilingual BERT adds two token classification runs. French CamemBERT and biomedical French DrBERT add four more, for 34 total. Static pretrained embeddings remain trainable during NER fitting. The Transformers use the first subtoken for each word label and ignore continuation subtokens.

On CPU, BERT training can be slow. A GPU runtime is recommended. The notebook shows missing results explicitly until the corresponding runs finish; it never treats an unrun configuration as a score.
