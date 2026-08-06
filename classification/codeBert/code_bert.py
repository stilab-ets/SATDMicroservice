import os
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from torch.optim import AdamW

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import Dataset, DataLoader


SEED = 42
batch_size = 8  # More gradient updates per epoch with smaller batch size, which can help convergence on small datasets.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs("./results/codeBert", exist_ok=True)


# ======================================================
# Dataset
# ======================================================
class CommentDataset(Dataset):
    def __init__(self, comments, labels, tokenizer, max_length=256):
        #  max_length : 256
        #   SATD comments can be long; truncating at 128 loses context
        self.comments   = comments
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.comments[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ======================================================
# Early stopping — val_loss based, saves best checkpoint
# ======================================================
class EarlyStopping:
    def __init__(self, patience=3):
        self.patience   = patience
        self.best_loss  = None
        self.counter    = 0
        self.best_state = None

    def step(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss  = val_loss
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter    = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ======================================================
# Main
# ======================================================
def main_code_bert():

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------
    df = pd.read_excel("./data/SATD_Classification_CLEANED_V3.xlsx")

    comments    = df["comment"].tolist()
    labels      = df["Category"].astype("category").cat.codes.values
    label_names = df["Category"].astype("category").cat.categories.tolist()

    print("Class distribution:")
    print(df["Category"].value_counts())

    num_classes = len(label_names)

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    # --------------------------------------------------
    # 5-fold: test set = ~116 samples → ~8/class → more stable metrics.
    # --------------------------------------------------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    y_true_all, y_pred_all, y_proba_all = [], [], []
    all_indices  = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(comments, labels), 1):
        print(f"\n{'='*50}")
        print(f"===== Fold {fold} =====")

        # --------------------------------------------------
        # For stoppin validaiton:
        # use train (80%) for gradient updates, 
        # use test  (20%) ONLY for early stopping val_loss check.
        # --------------------------------------------------

        train_dataset = CommentDataset(
            [comments[i] for i in train_idx], labels[train_idx], tokenizer
        )
        test_dataset = CommentDataset(
            [comments[i] for i in test_idx], labels[test_idx], tokenizer
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

        # --------------------------------------------------
        # Per-fold class weights
        # --------------------------------------------------
        fold_classes = np.unique(labels[train_idx])
        fold_weights = compute_class_weight(
            "balanced", classes=fold_classes, y=labels[train_idx]
        )
        weight_vector = np.ones(num_classes)
        for cls, w in zip(fold_classes, fold_weights):
            weight_vector[cls] = w
        weight_tensor = torch.tensor(weight_vector, dtype=torch.float).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

        # --------------------------------------------------
        # Model
        # --------------------------------------------------
        model = RobertaForSequenceClassification.from_pretrained(
            "microsoft/codebert-base",
            num_labels=num_classes,
        ).to(device)

        # Lower classifier head LR =5e-5
        optimizer = AdamW([
            {"params": model.roberta.parameters(),    "lr": 2e-5},
            {"params": model.classifier.parameters(), "lr": 5e-5},
        ])

        # Epochs= 20, patience= 4
        #   With more training samples per fold, model needs more epochs to converge
        epochs      = 20
        total_steps = len(train_loader) * epochs
        scheduler   = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        early_stopping = EarlyStopping(patience=4)

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_labels   = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = loss_fn(outputs.logits, batch_labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            # --------------------------------------------------
            # Validation on test fold (for early stopping only — no leakage)
            # --------------------------------------------------
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    outputs  = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                    val_loss += loss_fn(outputs.logits, batch["labels"].to(device)).item()

            val_loss       /= len(test_loader)
            avg_train_loss  = total_loss / len(train_loader)
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

            if early_stopping.step(val_loss, model):
                print(f"  → Early stopping at epoch {epoch+1}")
                break

        # Restore best checkpoint
        model.load_state_dict(
            {k: v.to(device) for k, v in early_stopping.best_state.items()}
        )

        # --------------------------------------------------
        # Fold predictions
        # --------------------------------------------------
        y_true_fold, y_pred_fold = [], []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                )
                proba = torch.softmax(outputs.logits, dim=1)
                preds = torch.argmax(proba, dim=1)

                y_true_fold.extend(batch["labels"].cpu().numpy())
                y_pred_fold.extend(preds.cpu().numpy())

                y_true_all.extend(batch["labels"].cpu().numpy())
                y_pred_all.extend(preds.cpu().numpy())
                y_proba_all.extend(proba.cpu().numpy())

        all_indices.extend(test_idx)

        fold_metrics.append({
            "Fold":        fold,
            "Accuracy":    accuracy_score(y_true_fold, y_pred_fold),
            "Macro_F1":    f1_score(y_true_fold, y_pred_fold, average="macro",     zero_division=0),
            "Weighted_F1": f1_score(y_true_fold, y_pred_fold, average="weighted",  zero_division=0),
            "MCC":         matthews_corrcoef(y_true_fold, y_pred_fold),
        })
        print(f"Fold {fold} → Acc={fold_metrics[-1]['Accuracy']:.3f}  "
              f"MacroF1={fold_metrics[-1]['Macro_F1']:.3f}  "
              f"MCC={fold_metrics[-1]['MCC']:.3f}")

        del model
        torch.cuda.empty_cache()

    # ======================================================
    # Aggregate OOF metrics
    # ======================================================
    y_true_all  = np.array(y_true_all)
    y_pred_all  = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all)
    all_indices = np.array(all_indices)

    summary_df = pd.DataFrame([{
        "Model":          "CodeBERT",
        "Accuracy":       accuracy_score(y_true_all, y_pred_all),
        "Macro_F1":       f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0),
        "Weighted_F1":    f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
        "MCC":            matthews_corrcoef(y_true_all, y_pred_all),
        "OvR_AUC_Macro":  roc_auc_score(
            label_binarize(y_true_all, classes=np.arange(num_classes)),
            y_proba_all,
            average="macro",
            multi_class="ovr",
        ),
    }])

    print("\n===== OVERALL RESULTS =====")
    print(summary_df.to_string(index=False))

    # ======================================================
    # Confusion matrix & classification report
    # ======================================================
    cm_df = pd.DataFrame(
        confusion_matrix(y_true_all, y_pred_all),
        index=label_names,
        columns=label_names,
    )

    decoded_y_true = np.array([label_names[i] for i in y_true_all])
    decoded_y_pred = np.array([label_names[i] for i in y_pred_all])

    report_df = pd.DataFrame(
        classification_report(
            decoded_y_true,
            decoded_y_pred,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()

    # ======================================================
    # Predictions with traceability
    # ======================================================
    predictions_df = pd.concat([
        pd.DataFrame({
            "Index":           all_indices,
            "Comment":         df.loc[all_indices, "comment"].values,
            "True_Label":      decoded_y_true,
            "Predicted_Label": decoded_y_pred,
        }),
        pd.DataFrame(
            y_proba_all,
            columns=[f"P({c})" for c in label_names],
        ),
    ], axis=1)

    # ======================================================
    # Per-class OvR AUC + ROC curves
    # ======================================================
    y_true_bin = label_binarize(y_true_all, classes=np.arange(num_classes))
    roc_rows, auc_rows = [], []

    for i, class_name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_all[:, i])
        class_auc   = auc(fpr, tpr)
        auc_rows.append({"Class": class_name, "OvR_AUC": class_auc})

        for fp, tp in zip(fpr, tpr):
            roc_rows.append({"Class": class_name, "FPR": fp, "TPR": tp})

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {class_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title(f"OvR ROC – {class_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"./results/codeBert/roc_{class_name}.png", dpi=300)
        plt.close()

    roc_all_df       = pd.DataFrame(roc_rows)
    per_class_auc_df = pd.DataFrame(auc_rows)

    # ======================================================
    # Save everything to Excel
    # ======================================================
    with pd.ExcelWriter(
        "./results/codeBert/codebert_full_results.xlsx",
        engine="xlsxwriter",
    ) as writer:
        summary_df.to_excel(writer,          sheet_name="Summary",               index=False)
        pd.DataFrame(fold_metrics).to_excel(writer, sheet_name="Fold_Metrics",   index=False)
        cm_df.to_excel(writer,               sheet_name="Confusion_Matrix")
        report_df.to_excel(writer,           sheet_name="Classification_Report")
        predictions_df.to_excel(writer,      sheet_name="Predictions",           index=False)
        per_class_auc_df.to_excel(writer,    sheet_name="OvR_AUC_Per_Class",     index=False)
        roc_all_df.to_excel(writer,          sheet_name="OvR_ROC_All",           index=False)

    print("\n✅ ALL CodeBERT results saved to ./results/codeBert/codebert_full_results.xlsx")


if __name__ == "__main__":
    main_code_bert()