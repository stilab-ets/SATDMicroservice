
import pandas as pd
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

def preprocess_satd(text: str) -> str:
    """
    Minimal preprocessing for SATD comments.
    Keeps code tokens, TODO/FIXME markers, and identifiers intact.
    """
    if text is None:
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing spaces
    text = text.strip()

    return text

def prepare(excel_path):
    df = pd.read_excel(excel_path)
    df["comment_text"] = df["comment_text"].apply(preprocess_text)
    df = df.dropna(subset=["comment_text", "category"])
    return df

def compute_metrics(y_true, y_pred, labels):
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "macro_precision": precision_score(y_true, y_pred, average="macro"),
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels)
    }



def aggregate_results(results, metric="macro_f1"):
    summary = {}
    for method, folds in results.items():
        values = [fold[metric] for fold in folds]
        summary[method] = {
            "mean": np.mean(values),
            "std": np.std(values)
        }
    return summary
