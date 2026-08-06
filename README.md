# SATD Extraction and Classification in Microservice Systems

This repository accompanies the paper **Self-Admitted Technical Debt in Microservice-Based Systems: Taxonomy, Characteristics, and LLM-Based Classification**. The paper is currently under submission.

It contains the data collection and experimental pipeline used to study **Self-Admitted Technical Debt (SATD)** in microservice-based software systems. It supports the extraction of source-code comments from GitHub repositories, keyword-based identification and manual labeling of SATD, and SATD classification using classical machine-learning algorithms, CodeBERT, and large language models (LLMs).

## Repository overview

The project has two main components:

1. **SATD extraction and manual labeling**: the scripts at the repository root clone the selected projects, analyze their commit history, extract source-code comments, and identify SATD candidates using a configurable keyword list. The candidates were then manually labeled.
2. **SATD classification**: the `classification/` directory contains experiments based on Random Forest, XGBoost, CodeBERT, and LLMs.

## Project structure

```text
.
├── main.py                         # Main repository-mining entry point
├── Extractor.py                    # Commit and changed-file analysis
├── identify_SATD_comments.py       # Keyword-based SATD candidate identification
├── comments_extract/               # Comment parsing and extraction helpers
├── conf/
│   └── conf.yml                    # Dataset, clone, output, and language settings
├── dataset/
│   ├── projects_finale.xlsx        # List of GitHub projects
│   └── keywords_list.txt           # SATD-identification keywords
├── outputs/                        # Extracted SATD/non-SATD candidates
├── utils/                          # Shared extraction utilities
├── requierments.txt                # Core extraction dependencies
└── classification/
    ├── data/
    │   └── SATD_Classification_WITH_CONTEXT.xlsx
    ├── preprocess.py               # Shared text preprocessing
    ├── run_expriment.py            # Experiment launcher
    ├── xgb/                        # XGBoost classifier and tuning
    ├── rf/                         # Random Forest classifier and tuning
    ├── codeBert/                   # CodeBERT and CodeBERT-LoRA experiments
    ├── LLMs/                       # Zero-shot and few-shot LLM experiments
    ├── models/                     # Serialized trained models/vectorizers
    └── results/                    # Metrics, predictions, ROC curves, and comparisons
```

## Component 1: SATD extraction and manual labeling

### 1. Configure the project list

The default configuration is stored in `conf/conf.yml`:

`dataset/projects_finale.xlsx` must contain a `project` column. Each value is expected to use the GitHub `owner/repository` format.

### 2. Extract comments from repositories

Run the following command from the repository root:

```bash
python main.py
```

`main.py` performs the following operations:

- reads the repositories listed in `dataset/projects_finale.xlsx`;
- clones repositories that are not already available locally;
- analyzes commits and changed files for the configured languages; and
- writes extracted comment records as JSON files under `dataset/comments/`.

### 3. Extract SATD 

```bash
python dentify_SATD_comments.py
```

## Component 2: SATD classification

The classification experiments compare several families of models:

- **Classical machine learning**: Random Forest and XGBoost.
- **Pretrained code model**: CodeBERT
- **Large language models**: zero-shot, static few-shot, and dynamically retrieved few-shot classification, with or without source-code context.

Experiments report accuracy, macro F1, weighted F1, Matthews correlation coefficient (MCC), confusion matrices, classification reports, predictions, and—where supported—one-vs-rest ROC/AUC results.

### Run LLM experiments

The LLM pipeline is implemented in `classification/LLMs/LLMs_eval.py`. Before running it:

1. Create `classification/LLMs/.env` locally and define `OPENROUTER_API_KEY`.
2. Select the models in `MODELS`.
3. Set `MODE` to `zero_shot`, `few_shot_static`, or `few_shot_dynamic`.
4. Set `INPUT_PATH` to `./data/SATD_Classification_WITH_CONTEXT.xlsx`.
5. Configure `use_context` and `type_context` (`code`, `summary`, or `content_file`) according to the available dataset columns.

Then run from `classification/`:

```bash
python LLMs/LLMs_eval.py
```

The LLM experiments use OpenRouter's OpenAI-compatible API. 

