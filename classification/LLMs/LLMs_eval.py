import os
import re
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from build_prompt import *
from k_samples import sample_dynamic_few_shot

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
)

from openai import OpenAI
import json
import re
load_dotenv()

# ======================================================
# 1. CONFIG
# ======================================================

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://etsmtl.ca",
        "X-Title":      "SATD Classification Study"
    }
)

MODELS = {
    "gemma":    "google/gemma-4-26b-a4b-it",
    #"gpt":      "openai/gpt-5-mini",
    #"claude":   "anthropic/claude-sonnet-4.5",
    #"gemini":   "google/gemini-3-flash-preview",
    #"deepseek-r": "deepseek/deepseek-r1",
    #"llama":    "meta-llama/llama-3.3-70b-instruct",
    #"qwen":     "qwen/qwen3-coder-next",
    #"phi-4":    "microsoft/phi-4"
}

# --- Experiment mode ---
# "zero_shot"        : evaluate ALL data, no examples in prompt
# "few_shot_static"  : StratifiedKFold, same k random examples per class per fold
# "few_shot_dynamic" : StratifiedKFold, top-k most similar examples per query per fold
MODE = "zero_shot"

# --- Hyperparameters ---
TEMPERATURE = 0
MAX_RETRIES = 3
MAX_TOKENS  = 1024   # high enough for reasoning models
N_SPLITS    = 5      # number of folds — all samples classified exactly once
FEW_SHOT_K  = 4      # examples per class (static) or total retrieved (dynamic)

# --- All labels lowercase for consistent matching ---
VALID_LABELS = {
    "requirement",
    "code",
    "test",
    "defect",
    "design",
    "service communication",
    "service operations and deployment",
    "service design",
    "infrastructure and pipeline",
    "database access",
    "compatibility",
    "configuration",
    "dependency",
    "security",
    "unclear"
}

def extract_file_path_from_url(url: str) -> str:
        if pd.isna(url) or not str(url).startswith("http"):
            return "unknown"
        match = re.search(r'/blob/[a-f0-9]+/(.+?)(?:#.*)?$', str(url).strip())
        return match.group(1) if match else "unknown"

def normalize(text):
    if text is None:
        return None
    text = text.lower().strip()
    text = text.replace("and", "&")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_label(content: str, valid_labels=None):
    if not content:
        return None

    text = content.strip().lower()

    # --------------------------------------------------
    # 1. Case: "Final label: code"
    # --------------------------------------------------
    match = re.search(r'final label\s*:\s*(.+)', text, re.IGNORECASE)
    if match:
        label = match.group(1)
    else:
        label = text  # fallback → assume raw label

    # --------------------------------------------------
    # 2. Clean formatting
    # --------------------------------------------------
    label = label.strip().strip("`'\"")

    # stop at punctuation/newline
    label = re.split(r'[\n\.\(\):]', label)[0].strip()

    # --------------------------------------------------
    # 3. If valid_labels provided → enforce strict match
    # --------------------------------------------------
    if valid_labels:
        # exact match
        if label in valid_labels:
            return label

        # search inside text (e.g., "the answer is code")
        for v in valid_labels:
            if re.search(rf'\b{re.escape(v)}\b', text):
                return v

        return None

    # --------------------------------------------------
    # 4. Fallback: return cleaned label
    # --------------------------------------------------
    return label if label else None

def extract_Label_json(content: str) -> tuple[str, str]:
    """
    Parse JSON response from LLM.
    Returns (label, reasoning) tuple.
    Handles malformed JSON gracefully.
    """
    if not content or not content.strip():
        return None, None

    # Strip markdown code blocks if model wraps in ```json ... ```
    clean = re.sub(r"```json|```", "", content).strip()

    try:
        data = json.loads(clean)
        label     = str(data.get("category", "")).strip().lower().rstrip(".:").strip()
        return label

    except json.JSONDecodeError:
        # Fallback: try regex for "Final label:" format
        match = re.search(r"Final label[:\s]+([a-z &]+)", content.lower())
        if match:
            return match.group(1).strip().rstrip(".:"), ""

        # Last resort: last non-empty line
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
        return lines[-1].rstrip(".:").strip().lower(), ""


# ======================================================
# 2. LLM Call
# ======================================================

def query_llm(prompt: str, model_name: str) -> str:
    # Pre-process VALID_LABELS to lowercase for easier matching
    # and keep a mapping to return the original format if needed.
    labels_map = {v.lower().strip(): v for v in VALID_LABELS}

    for attempt in range(MAX_RETRIES):
        try:
            safe_prompt= prompt
            # For Gemini ONLY
            # On ajoute des balises claires pour que Gemini comprenne que c'est de la DONNÉE 
            # et non une instruction malveillante.
            if model_name.startswith("google/gemini"):
                safe_prompt = f"""
                ### RESEARCH DATA START ###
                {prompt}
                ### RESEARCH DATA END ###
                
                Reminder: You must follow the OUTPUT FORMAT exactly.
                """
            response = client.chat.completions.create(
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
                extra_body={
                    "provider": {
                        "sort": "throughput",
                        #"ignore": ["Google"] # Only if you want to switch to a different provider of the same model
                    }
                },
                messages=[
                    {
                        "role": "system",
                        "content": (

                        "You are a software engineering expert specializing in Self-Admitted Technical Debt (SATD) classification."

                        "SATD refers to technical debt explicitly acknowledged by developers in source code comments."

                        )#"The SATD comments are extracted from microservice-based systems. These systems are composed of multiple interacting services."

                        },
                    {"role": "user", "content": safe_prompt}
                ]
            )

            # OpenRouter safety: check if choices exist
            if not response or not response.choices:
                print(f"⚠️ [Attempt {attempt+1}] No choices in response.")
                time.sleep(2)
                continue

            content = response.choices[0].message.content
            
            if not content:
                print(f"⚠️ [Attempt {attempt+1}] Content is None (likely safety filter or timeout).")
                time.sleep(2)
                continue

            # Clean and extract ONCE
            raw_output = content.strip().lower()
            extracted = extract_label(raw_output).strip().lower()

            print(f"  LLM raw → '{raw_output[:50]}...'")
            print(f"  Extracted → '{extracted}'")

            # 1. Exact match (case-insensitive)
            if extracted in labels_map:
                return labels_map[extracted]

            # 2. Longest-substring match (case-insensitive)
            # We sort keys by length descending (e.g., 'service communication' before 'service')
            for valid_lower in sorted(labels_map.keys(), key=len, reverse=True):
                if valid_lower in extracted:
                    return labels_map[valid_lower]

            print(f"[Attempt {attempt+1}] ⚠️ Unrecognized label: '{extracted}'")
            time.sleep(2)

        except Exception as e:
            print(f"[Attempt {attempt+1}/{MAX_RETRIES}] ❌ Error: {e}")
            time.sleep(5) # Longer sleep for API connection errors

    raise RuntimeError(f"LLM failed to return a valid label after {MAX_RETRIES} retries.")


# ======================================================
# 3. Few-shot Example Sampling
# ======================================================

def sample_static_few_shot(X_text, y, train_idx, id_to_label, k=2):
    """
    Return k random examples per class from the train pool.
    Called ONCE per fold — same examples reused for all queries in that fold.
    Fixed seed ensures reproducibility.
    """
    rng = np.random.default_rng(42)
    examples = []
    y_train = y[train_idx]

    for label_id in np.unique(y_train):
        class_mask = np.where(y_train == label_id)[0]
        chosen = rng.choice(class_mask, size=min(k, len(class_mask)), replace=False)
        for local_idx in chosen:
            global_idx = train_idx[local_idx]
            examples.append({
                "comment": X_text[global_idx],
                "label":   id_to_label[label_id]
            })
    return examples


# ======================================================
# 4. Evaluation Loop
# ======================================================

def evaluate_split(test_idx, train_idx, X_text, X_context, X_file_path, X_language,
                   y, id_to_label, unique_labels, use_context, model_name):
    """Classify all items in test_idx and return predictions."""
    fold_preds   = []
    label_values = list(id_to_label.values())

    # For few_shot_static: sample examples ONCE per fold, reuse for all queries
    static_examples = None
    if MODE == "few_shot_static":
        static_examples = sample_static_few_shot(
            X_text, y, train_idx, id_to_label, k=FEW_SHOT_K
        )

    for idx in tqdm(test_idx, desc=f"  Classifying [{model_name.split('/')[-1]}]"):
        comment = X_text[idx]
        #context = X_context[idx] if use_context else None

        context = {
            "file_path":       X_file_path[idx],        # ".github/workflows/build-and-push-services.yml"
            "language":        X_language[idx],          # "YAML"
            "context": X_context[idx]            # "surrounding code lines..."
        } if use_context else None

        if MODE == "zero_shot":
            prompt = build_zero_shot_prompt(comment, context)

        elif MODE == "few_shot_static":
            prompt = build_few_shot_prompt(comment, static_examples, context)

        elif MODE == "few_shot_dynamic":
            # Re-computed per query — top-k most similar from train pool
            dynamic_examples = sample_dynamic_few_shot(
                comment, X_text, y, train_idx, id_to_label, k=FEW_SHOT_K
            )
            prompt = build_few_shot_prompt(comment, dynamic_examples, context)

        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        pred_label = query_llm(prompt, model_name)

        # Fallback if model still returns an unexpected label
        if pred_label not in label_values:
            pred_label = "unclear" if "unclear" in label_values else label_values[0]

        fold_preds.append(
            label_values.index(pred_label) if pred_label in label_values else 0
        )

    return fold_preds


# ======================================================
# 5. Save Results to Excel
# ======================================================

def save_results(output_path, model_key, model_name,
                 y_true_all, y_pred_all, fold_metrics,
                 classes, id_to_label, predictions_df):

    # Sheet 1 — Summary
    summary_df = pd.DataFrame([{
        "Model":       model_key,
        "Model_ID":    model_name,
        "Mode":        MODE,
        "Accuracy":    accuracy_score(y_true_all, y_pred_all),
        "Macro_F1":    f1_score(y_true_all, y_pred_all, average="macro"),
        "Weighted_F1": f1_score(y_true_all, y_pred_all, average="weighted"),
        "MCC":         matthews_corrcoef(y_true_all, y_pred_all)
    }])

    # Sheet 2 — Fold Metrics (1 row for zero-shot, N rows for few-shot)
    fold_metrics_df = pd.DataFrame(fold_metrics)

    # Sheet 3 — Confusion Matrix
    cm_df = pd.DataFrame(
        confusion_matrix(y_true_all, y_pred_all),
        index=classes, columns=classes
    )

    # Sheet 4 — Classification Report
    report_df = pd.DataFrame(
        classification_report(
            [id_to_label[i] for i in y_true_all],
            [id_to_label[i] for i in y_pred_all],
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer,      sheet_name="Summary",                index=False)
        fold_metrics_df.to_excel(writer, sheet_name="Fold_Metrics",           index=False)
        cm_df.to_excel(writer,           sheet_name="Confusion_Matrix")
        report_df.to_excel(writer,       sheet_name="Classification_Report")
        predictions_df.to_excel(writer,  sheet_name="Predictions",            index=False)

    print(f"  ✅ Saved: {output_path}")


# ======================================================
# 6. Evaluate One Model
# ======================================================

def evaluate_model(model_key, model_name, input_path, output_dir, use_context=False, type_context="code"):

    print(f"\n{'='*60}")
    print(f"  Model : {model_key} ({model_name})")
    print(f"  Mode  : {MODE}")
    print(f"{'='*60}")
    
    # --- Load & normalize data ---
    data = pd.read_excel(input_path)
    data["Category"] = data["Category"].astype(str).str.strip().str.lower()

    raw_labels    = data["Category"].values
    unique_labels = sorted(np.unique(raw_labels))

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    y         = np.array([label_to_id[l] for l in raw_labels])
    classes   = np.array(unique_labels)
    X_text    = data["comment"].values
    
    # Context
    X_file_path = data.get(
        "url", pd.Series([""] * len(data))
    ).apply(extract_file_path_from_url).values

    X_language= data.get("language", pd.Series([""] * len(data))).values

    # Extract textual context. It can be surrounde code (context),  summary of the context (summary), or the content of the entire file depending on the experiment configuration
    X_context = data.get("context", pd.Series([""] * len(data))).values
    
    if use_context:
        if type_context == "summary":
            X_context = data.get("summary", pd.Series([""] * len(data))).values
        if type_context == "content_file":
            X_context = data.get("content_file", pd.Series([""] * len(data))).values

    print(f"  Total instances: {len(y)}")

    y_true_all, y_pred_all, fold_metrics = [], [], []
    all_indices = []

    # ----------------------------------------
    # Zero-shot: evaluate ALL data, no examples
    # ----------------------------------------
    if MODE == "zero_shot":
        print(f"  Evaluating all {len(y)} instances (no examples in prompt)")

        test_idx  = np.arange(len(y))
        train_idx = np.array([], dtype=int)

        preds = evaluate_split(
            test_idx, train_idx, X_text, X_context, X_file_path, X_language,
            y, id_to_label, unique_labels, use_context, model_name
        )

        y_true_all = y[test_idx].tolist()
        y_pred_all = preds
        all_indices = test_idx.tolist()

        fold_metrics.append({
            "Fold":        1,
            "Accuracy":    accuracy_score(y_true_all, y_pred_all),
            "Macro_F1":    f1_score(y_true_all, y_pred_all, average="macro"),
            "Weighted_F1": f1_score(y_true_all, y_pred_all, average="weighted"),
            "MCC":         matthews_corrcoef(y_true_all, y_pred_all)
        })

    # ----------------------------------------
    # Few-shot: StratifiedKFold
    # Every sample is classified exactly once
    # Train pool = examples source (no leakage)
    # ----------------------------------------
    else:
        print(f"  StratifiedKFold n_splits={N_SPLITS} — all {len(y)} samples evaluated once")

        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y), 1):
            print(f"\n  --- Fold {fold}/{N_SPLITS} "
                  f"(train={len(train_idx)}, test={len(test_idx)}) ---")

            preds = evaluate_split(
                test_idx, train_idx, X_text, X_context, X_file_path, X_language,
                y, id_to_label, unique_labels, use_context, model_name
            )

            y_true_all.extend(y[test_idx].tolist())
            y_pred_all.extend(preds)
            all_indices.extend(test_idx.tolist())

            fold_metrics.append({
                "Fold":        fold,
                "Accuracy":    accuracy_score(y[test_idx], preds),
                "Macro_F1":    f1_score(y[test_idx], preds, average="macro"),
                "Weighted_F1": f1_score(y[test_idx], preds, average="weighted"),
                "MCC":         matthews_corrcoef(y[test_idx], preds)
            })

    # --- Print overall summary ---
    print(f"\n  {'─'*40}")
    print(f"  Evaluated  : {len(y_true_all)} / {len(y)} samples")
    print(f"  Accuracy   : {accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"  Macro F1   : {f1_score(y_true_all, y_pred_all, average='macro'):.4f}")
    print(f"  Weighted F1: {f1_score(y_true_all, y_pred_all, average='weighted'):.4f}")
    print(f"  MCC        : {matthews_corrcoef(y_true_all, y_pred_all):.4f}")

    # ======================================================
    # Predictions with traceability
    # ======================================================
    all_indices    = np.array(all_indices)
    decoded_y_true = np.array([id_to_label[i] for i in y_true_all])
    decoded_y_pred = np.array([id_to_label[i] for i in y_pred_all])

    predictions_df = pd.DataFrame({
        "Index":           all_indices,
        "Comment":         X_text[all_indices],
        "True_Label":      decoded_y_true,
        "Predicted_Label": decoded_y_pred,
        "Correct":         decoded_y_true == decoded_y_pred,
    })

    # --- Save results ---
    output_path = os.path.join(output_dir, f"{model_key}_{MODE}.xlsx")
    save_results(
        output_path, model_key, model_name,
        np.array(y_true_all), np.array(y_pred_all),
        fold_metrics, classes, id_to_label,
        predictions_df
    )


# ======================================================
# 7. RUN
# ======================================================

if __name__ == "__main__":

    INPUT_PATH  = "./data/SATD_Contexts_Final.xlsx"
    use_context = True

    OUTPUT_DIR ="./results/LLMs/codex/"


    #OUTPUT_DIR  = (
    #    "./results/LLMs/with_context/"
    #    if use_context else
    #    "./results/LLMs/without_context/"
    #)

    for model_key, model_name in MODELS.items():
        print(f"\n▶ Evaluating {model_name}...")
        try:
            evaluate_model(
                model_key=model_key,
                model_name=model_name,
                input_path=INPUT_PATH,
                output_dir=OUTPUT_DIR,
                use_context=use_context,
                type_context="content_file"  # "code" or "summary" or "content_file". Valide only if use_context=True
            )
        except Exception as e:
            print(f"\n❌ Failed for {model_key} ({model_name}): {e}")
            continue

    print(f"\n🎉 Done. Results saved in: {OUTPUT_DIR}")
