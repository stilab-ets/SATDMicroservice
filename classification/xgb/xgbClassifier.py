import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

import joblib


def main(input_path, output_path=None):
    # ======================================================
    # 1. Load & preprocess data
    # ======================================================
    data = pd.read_excel(input_path)
    data["Category"] = data["Category"].astype(str)
    data["Category"] = preprocess.normalize_category(data["Category"])
    data["comment"] = data["comment"].apply(preprocess.preprocess_comment)

    # ======================================================
    # Label mapping (keep as-is)
    # ======================================================
    raw_labels = data["Category"].values
    unique_labels = sorted(np.unique(raw_labels))
    print(unique_labels)

    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    y = np.array([label_to_id[l] for l in raw_labels])
    classes = np.array(unique_labels)
    num_classes = len(classes)

    X_text = data["comment"].values

    # ======================================================
    # 2. Stratified K-Fold CV
    # ======================================================
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)

    y_true_all, y_pred_all, y_proba_all = [], [], []
    all_indices = []
    fold_metrics = []
   

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_text, y), 1):
        print(f"Fold {fold}")

        X_train_text = X_text[train_idx]
        X_test_text = X_text[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

         # --------------------------------------------------
        # Vectorizer created + fit ONLY on training fold
        # --------------------------------------------------
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
    

        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=0.95
        )

        vectorizer = FeatureUnion([
            ("word", word_vec),
            ("char", char_vec)
        ])

        X_train = vectorizer.fit_transform(X_train_text)
        # --------------------------------------------------
        # converts test text into feature vectors using the vocabulary and IDF learned from training data.
        #  If a word appears in test but was never seen in training, it is silently ignored
        X_test = vectorizer.transform(X_test_text) 

        # ==================================================
        # Class-balanced sample weights (per fold)
        # ==================================================
        present_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=present_classes,
            y=y_train
        )
        class_weight_dict = dict(zip(present_classes, class_weights))
        sample_weights = np.array([class_weight_dict[c] for c in y_train])

        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            eval_metric="mlogloss",
            num_class=num_classes,
            random_state=42
        )
        

        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)
        y_proba_all.extend(y_proba)
        all_indices.extend(test_idx)

        fold_metrics.append({
            "Fold": fold,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Macro_F1": f1_score(y_test, y_pred, average="macro"),
            "Weighted_F1": f1_score(y_test, y_pred, average="weighted"),
            "MCC": matthews_corrcoef(y_test, y_pred)
        })

    # ======================================================
    # 3. Aggregate metrics (OOF)
    # ======================================================
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all)
    all_indices = np.array(all_indices)

    summary_df = pd.DataFrame([{
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_true_all, y_pred_all),
        "Macro_F1": f1_score(y_true_all, y_pred_all, average="macro"),
        "Weighted_F1": f1_score(y_true_all, y_pred_all, average="weighted"),
        "MCC": matthews_corrcoef(y_true_all, y_pred_all),
        "OvR_AUC_Macro": roc_auc_score(
            label_binarize(y_true_all, classes=np.arange(num_classes)),
            y_proba_all,
            average="macro",
            multi_class="ovr"
        )
    }])

    # ======================================================
    # 4. Confusion matrix
    # ======================================================
    cm_df = pd.DataFrame(
        confusion_matrix(y_true_all, y_pred_all),
        index=classes,
        columns=classes
    )

    # ======================================================
    # 5. Classification report
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
    # 6. Predictions with traceability
    # ======================================================
    predictions_df = pd.concat([
        pd.DataFrame({
            "Index": all_indices,
            "Comment": data.loc[all_indices, "comment"].values,
            "True_Label": decoded_y_true,
            "Predicted_Label": decoded_y_pred
        }),
        pd.DataFrame(y_proba_all, columns=[f"P({c})" for c in classes])
    ], axis=1)

    # ======================================================
    # 7. Per-class OvR AUC + ROC data
    # ======================================================
    y_true_bin = label_binarize(y_true_all, classes=np.arange(num_classes))

    roc_rows, auc_rows = [], []

    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba_all[:, i])
        class_auc = auc(fpr, tpr)

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

    roc_all_df = pd.DataFrame(roc_rows)
    per_class_auc_df = pd.DataFrame(auc_rows)

    # ======================================================
    # 8. Save EVERYTHING in ONE Excel file (keep)
    # ======================================================
    with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(fold_metrics).to_excel(writer, sheet_name="Fold_Metrics", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion_Matrix")
        report_df.to_excel(writer, sheet_name="Classification_Report")
        predictions_df.to_excel(writer, sheet_name="Predictions", index=False)
        per_class_auc_df.to_excel(writer, sheet_name="OvR_AUC_Per_Class", index=False)
        roc_all_df.to_excel(writer, sheet_name="OvR_ROC_All", index=False)

    # ======================================================
    # 9. Refit FINAL model on full dataset for saving (clean)
    # ======================================================
    final_word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    final_char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_df=0.95
    )
    final_vectorizer = FeatureUnion([
        ("word", final_word_vec),
        ("char", final_char_vec)
    ])

    X_full = final_vectorizer.fit_transform(X_text)

    final_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=num_classes,
        random_state=42
    )
    final_model.fit(X_full, y)

    joblib.dump(final_vectorizer, "./models/tfidf_vectorizer.pkl")
    joblib.dump(final_model, "./models/xgb_model.pkl")

    print(f"\n✅ ALL results saved to: {output_path}")
    print("✅ Final model/vectorizer saved to: ./models/")


if __name__ == "__main__":
    main(
        input_path="./data/SATD_Classification_CLEANED_V3.xlsx",
        output_path="./results/xgb/xgb_full_results.xlsx"
    )
