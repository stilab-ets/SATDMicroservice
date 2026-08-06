import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc
)
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier
import preprocess


# ======================================================
# Helper: build a fresh vectorizer each time
# ======================================================
def build_vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95
        ))
    ])


# ======================================================
# Optuna objective — inner CV to find best params
# NOTE: uses SAME 7-fold structure but separate from
#       the final evaluation CV to avoid leakage
# ======================================================
def objective(trial, X_text, y, num_classes):
    params = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "max_depth":         trial.suggest_int("max_depth", 3, 10),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
        "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
        "objective":         "multi:softprob",
        "eval_metric":       "mlogloss",
        "num_class":         num_classes,
        "random_state":      42,
        "verbosity":         0,
    }

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    macro_f1_scores = []

    for train_idx, test_idx in skf.split(X_text, y):
        X_train_text = X_text[train_idx]
        X_test_text  = X_text[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Vectorizer fit only on training fold 
        vec = build_vectorizer()
        X_train = vec.fit_transform(X_train_text)
        X_test  = vec.transform(X_test_text)

        present_classes = np.unique(y_train)
        class_weights   = compute_class_weight("balanced", classes=present_classes, y=y_train)
        cw_dict         = dict(zip(present_classes, class_weights))
        sample_weights  = np.array([cw_dict[c] for c in y_train])

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        macro_f1_scores.append(f1_score(y_test, y_pred, average="macro", zero_division=0))

    return np.mean(macro_f1_scores)


# ======================================================
# MAIN
# ======================================================
def main(input_path, output_path=None):

    # ======================================================
    # 1. Load & preprocess data
    # ======================================================
    data = pd.read_excel(input_path)
    data["comment"] = data["comment"].apply(preprocess.preprocess_comment)

    # ======================================================
    # Label mapping — global, no LabelEncoder
    # ======================================================
    raw_labels    = data["Category"].values
    unique_labels = sorted(np.unique(raw_labels))
    print("Classes:", unique_labels)

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    y           = np.array([label_to_id[l] for l in raw_labels])
    classes     = np.array(unique_labels)
    num_classes = len(classes)
    X_text      = data["comment"].values

    # ======================================================
    # 2. Optuna hyperparameter search
    # ======================================================
    print("\n🔍 Running Optuna hyperparameter search (50 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_text, y, num_classes),
        n_trials=50,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_params.update({
        "objective":    "multi:softprob",
        "eval_metric":  "mlogloss",
        "num_class":    num_classes,
        "random_state": 42,
        "verbosity":    0,
    })

    print(f"\n✅ Best Optuna Macro F1 : {study.best_value:.4f}")
    print(f"✅ Best Params          : {best_params}")

    # ======================================================
    # 3. Final evaluation — Stratified K-Fold CV
    #    using best_params from Optuna
    # ======================================================
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    y_true_all, y_pred_all, y_proba_all = [], [], []
    all_indices = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y), 1):
        print(f"Fold {fold}")

        X_train_text = X_text[train_idx]
        X_test_text  = X_text[test_idx]
        y_train      = y[train_idx]
        y_test       = y[test_idx]

        # Vectorizer fit only on training fold — no leakage
        vectorizer = build_vectorizer()
        X_train = vectorizer.fit_transform(X_train_text)
        X_test  = vectorizer.transform(X_test_text)

        # Class-balanced sample weights (per fold)
        present_classes = np.unique(y_train)
        class_weights   = compute_class_weight("balanced", classes=present_classes, y=y_train)
        cw_dict         = dict(zip(present_classes, class_weights))
        sample_weights  = np.array([cw_dict[c] for c in y_train])

        # Use best params from Optuna
        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
        all_indices.extend(test_idx)

        fold_metrics.append({
            "Fold":        fold,
            "Accuracy":    accuracy_score(y_test, y_pred),
            "Macro_F1":    f1_score(y_test, y_pred, average="macro",     zero_division=0),
            "Weighted_F1": f1_score(y_test, y_pred, average="weighted",  zero_division=0),
            "MCC":         matthews_corrcoef(y_test, y_pred)
        })

    y_true_all  = np.array(y_true_all)
    y_pred_all  = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all)
    all_indices = np.array(all_indices)

    # ======================================================
    # 4. Aggregate metrics (OOF) — unchanged from original
    # ======================================================
    summary_df = pd.DataFrame([{
        "Model":        "XGBoost",
        "Accuracy":     accuracy_score(y_true_all, y_pred_all),
        "Macro_F1":     f1_score(y_true_all, y_pred_all, average="macro",    zero_division=0),
        "Weighted_F1":  f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0),
        "MCC":          matthews_corrcoef(y_true_all, y_pred_all),
        "OvR_AUC_Macro": roc_auc_score(
            label_binarize(y_true_all, classes=np.arange(num_classes)),
            y_proba_all,
            average="macro",
            multi_class="ovr"
        )
    }])

    # ======================================================
    # 5. Confusion matrix — unchanged
    # ======================================================
    cm_df = pd.DataFrame(
        confusion_matrix(y_true_all, y_pred_all),
        index=classes,
        columns=classes
    )

    # ======================================================
    # 6. Classification report — unchanged
    # ======================================================
    decoded_y_true = np.array([id_to_label[i] for i in y_true_all])
    decoded_y_pred = np.array([id_to_label[i] for i in y_pred_all])

    report_df = pd.DataFrame(
        classification_report(
            decoded_y_true,
            decoded_y_pred,
            output_dict=True,
            zero_division=0
        )
    ).transpose()

    # ======================================================
    # 7. Predictions with traceability — unchanged
    # ======================================================
    predictions_df = pd.concat([
        pd.DataFrame({
            "Index":           all_indices,
            "Comment":         data.loc[all_indices, "comment"].values,
            "True_Label":      decoded_y_true,
            "Predicted_Label": decoded_y_pred
        }),
        pd.DataFrame(y_proba_all, columns=[f"P({c})" for c in classes])
    ], axis=1)

    # ======================================================
    # 8. Per-class OvR AUC + ROC — unchanged
    # ======================================================
    y_true_bin = label_binarize(y_true_all, classes=np.arange(num_classes))
    roc_rows, auc_rows = [], []

    for i, class_name in enumerate(classes):
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
        plt.savefig(f"./results/xgb/roc_{class_name}.png", dpi=300)
        plt.close()

    roc_all_df       = pd.DataFrame(roc_rows)
    per_class_auc_df = pd.DataFrame(auc_rows)

    # Optuna summary sheet
    optuna_df = pd.DataFrame([{
        "Best_Macro_F1_Optuna": study.best_value,
        **study.best_params
    }])

    # ======================================================
    # 9. Save EVERYTHING in ONE Excel file
    # ======================================================
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer,       sheet_name="Summary",              index=False)
        optuna_df.to_excel(writer,        sheet_name="Optuna_Best_Params",   index=False)  # NEW
        pd.DataFrame(fold_metrics).to_excel(writer, sheet_name="Fold_Metrics", index=False)
        cm_df.to_excel(writer,            sheet_name="Confusion_Matrix")
        report_df.to_excel(writer,        sheet_name="Classification_Report")
        predictions_df.to_excel(writer,   sheet_name="Predictions",          index=False)
        per_class_auc_df.to_excel(writer, sheet_name="OvR_AUC_Per_Class",    index=False)
        roc_all_df.to_excel(writer,       sheet_name="OvR_ROC_All",          index=False)

    # ======================================================
    # 10. Refit FINAL model on full data with best params
    # ======================================================
    final_vectorizer = build_vectorizer()
    X_full = final_vectorizer.fit_transform(X_text)

    final_model = XGBClassifier(**best_params)
    final_model.fit(X_full, y)

    joblib.dump(final_vectorizer, "./models/tfidf_vectorizer.pkl")
    joblib.dump(final_model,      "./models/xgb_model.pkl")

    print(f"\n✅ ALL results saved to : {output_path}")
    print("✅ Final model/vectorizer saved to: ./models/")


if __name__ == "__main__":
    main(
        input_path="./data/SATD_Classification_CLEANED_V3.xlsx",
        output_path="./results/xgb/xgb_full_results.xlsx"
    )