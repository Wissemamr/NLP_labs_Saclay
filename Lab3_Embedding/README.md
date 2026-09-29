# Lab 3: Word Embeddings (Word2Vec & FastText)

**Course:** M2 AI - NLP Today (Saclay)  
**Students:** Zakaria SOUALAH MOHAMMED & Wissam AMAR  

---

## 📌 Project Overview
This project trains, evaluates, and compares different word embedding models using **Gensim** and **FastText** on two French corpora:
1. **Medical corpus (specialized & small):** `QUAERO_FrenchMed` (~52k tokens, 3,021 sentences)
2. **Press corpus (general-domain & large):** `QUAERO_FrenchPress` (~1.25M tokens, 38,548 sentences)

We build **6 embedding models** in total across 3 approaches:
- **Word2Vec CBOW** (`sg=0`, Continuous Bag-of-Words)
- **Word2Vec Skip-gram** (`sg=1`)
- **FastText CBOW** (`sg=0`, subword character $n$-grams)

**Hyperparameters:**
- Dimension: `dim = 100`
- Minimum word count: `min_count = 1`
- Context window: `window = 5`
- Epochs: `epochs = 5`

---

## 📂 Repository Structure

```text
Lab3_Embedding/
├── data/
│   ├── QUAERO_FrenchMed/
│   │   └── QUAERO_FrenchMed_traindev.ospl
│   └── QUAERO_FrenchPress/
│       └── QUAERO_FrenchPress_traindev.ospl
├── models/                     # Saved full models (.model) and vectors (.vec)
│   ├── w2v_cbow_med.model / .vec
│   ├── w2v_sg_med.model / .vec
│   ├── fasttext_cbow_med.model / .vec
│   ├── w2v_cbow_press.model / .vec
│   ├── w2v_sg_press.model / .vec
│   └── fasttext_cbow_press.model / .vec
├── plots/                      # Generated 2D t-SNE figures
│   ├── tsne_comparison_all_models.png
│   ├── tsne_domain_shift_medical_vs_press.png
│   └── tsne_architecture_comparison_med.png
├── Lab3_Word_Embedding.ipynb  # Complete interactive notebook with all executed outputs
├── train_embeddings.py        # Python script to train all 6 models
├── train_embeddings.sh        # Bash script to run training (Python / FastText CLI)
├── evaluate_similarity.py     # Evaluation script (Scipy spatial cosine vs. Gensim)
├── plot_tsne.py               # 2D t-SNE projection and visualization script
└── README.md
```

---

## 🚀 How to Run

### 1. Training Embeddings
To train all 6 models and export vectors to `models/`:
```bash
python train_embeddings.py --corpus all --model all --dim 100 --min_count 1
```
Or run via Bash:
```bash
bash train_embeddings.sh
```

### 2. Evaluating Semantic Similarity
Computes top-10 nearest neighbors with both `scipy.spatial` cosine distance and Gensim's `most_similar` for the candidate words (`patient`, `traitement`, `maladie`, `solution`, `jaune`):
```bash
python evaluate_similarity.py
```

### 3. Generating t-SNE Visualizations
Generates 2D dimensionality reduction plots saved into `plots/`:
```bash
python plot_tsne.py
```

### 4. Running the Jupyter Notebook
Open `Lab3_Word_Embedding.ipynb` in VS Code, JupyterLab, or upload to Google Colab. The notebook already contains all executed cell outputs, comparison tables, and inline charts.

---

## 🔍 Key Findings & Comparative Insights

### 1. Architecture Comparison (Same Corpus):
- **Skip-Gram vs. CBOW**: On the small medical corpus, Skip-gram yields far more clinically meaningful associations (e.g. for `patient`: `adulte`, `femmes`, `sujets`, `enfants`, `population`), whereas CBOW averages context vectors and tends to highlight frequent grammatical stop words.
- **FastText**: By decomposing words into subword $n$-grams ($3 \le n \le 6$), FastText captures morphological variations (plurals, suffixes such as `-ement`, `-ite`, `-ique`). This makes it exceptionally strong for medical vocabularies with complex affixes.

### 2. Domain & Data Comparison (Same Model, Different Corpora):
- **Polysemy of `solution`**:
  - *Medical corpus:* Refers to a pharmaceutical liquid formulation (`perfusion`, `flacon`, `injectable`, `dissolution`, `dilution`).
  - *Press corpus:* Refers to a political or diplomatic compromise (`pacifique`, `légitimité`, `consensuelle`, `alternative`).
- **Cultural nuance of `jaune`**:
  - *Medical corpus:* Infrequent color/clinical descriptor.
  - *Press corpus:* Dominated by sports news (`maillot jaune`, `trophée`, `sprint`).

### 3. Implications for Downstream NER:
Clinical Named Entity Recognition benefits substantially from in-domain embeddings. **FastText Medical** and **Word2Vec Skip-Gram Medical** are best positioned for clinical entity extraction due to subword generalization and preservation of clinical semantic specificity.
