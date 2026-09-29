import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display needed, just save plots to file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    recall_score,
)

import shap

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Step 3-4: Load the dataset and return it as a labeled DataFrame."""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target  # 0 = malignant, 1 = benign
    target_names = data.target_names  # ['malignant' 'benign']
    print(f"Dataset shape: {df.shape}")
    print(f"Classes: {dict(zip(target_names, np.bincount(df['target'])))}")
    return df, data.feature_names, target_names


def explore_data(df):
    """Step 4: Quick look at feature correlations, saved as a heatmap."""
    plt.figure(figsize=(14, 12))
    corr = df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True, cbar_kws={"shrink": 0.6})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()
    print("Saved correlation_heatmap.png")


def split_and_scale(df, feature_names):
    """Step 5: Train/test split + feature scaling."""
    X = df[feature_names]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train, X_test


def train_models(X_train, y_train):
    """Step 6: Train Logistic Regression, Random Forest, and SVM."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF kernel)": SVC(probability=True, random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")

    return models


def evaluate_models(models, X_test, y_test, target_names):
    """Step 7: Evaluate each model, with emphasis on recall for malignant cases.

    target_names[0] == 'malignant'. In this dataset malignant is encoded as 0,
    so we track recall for class 0 specifically -- missing a malignant case
    (a false negative) is the costly mistake to minimize.
    """
    results_lines = []
    best_name, best_model, best_malignant_recall = None, None, -1

    for name, model in models.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=target_names)
        malignant_recall = recall_score(y_test, y_pred, pos_label=0)  # 0 = malignant

        block = f"\n=== {name} ===\n{report}\nMalignant recall (sensitivity): {malignant_recall:.3f}\n"
        print(block)
        results_lines.append(block)

        if malignant_recall > best_malignant_recall:
            best_name, best_model, best_malignant_recall = name, model, malignant_recall

    with open(os.path.join(OUTPUT_DIR, "results.txt"), "w") as f:
        f.writelines(results_lines)
    print("Saved results.txt")

    # Confusion matrix for the best model (by malignant recall)
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")

    # ROC curve for the best model
    y_proba_best = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba_best)
    auc = roc_auc_score(y_test, y_proba_best)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"{best_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=150)
    plt.close()
    print("Saved roc_curve.png")

    print(f"\nBest model by malignant recall: {best_name} ({best_malignant_recall:.3f})")
    return best_name, best_model


def explain_model(best_name, best_model, X_train, X_test_scaled, feature_names):
    """Step 8: Use SHAP to explain which features drive predictions."""
    print(f"\nGenerating SHAP explanations for {best_name}...")

    # Random Forest / tree-based models use TreeExplainer (fast, exact).
    # Logistic Regression / SVM fall back to KernelExplainer (slower, sampled).
    if best_name == "Random Forest":
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test_scaled)
        # shap_values is a list [class0, class1] for binary classifiers
        values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        background = shap.sample(X_test_scaled, min(50, len(X_test_scaled)))
        explainer = shap.KernelExplainer(best_model.predict_proba, background)
        shap_values = explainer.shap_values(X_test_scaled[:50], nsamples=100)
        values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
        X_test_scaled = X_test_scaled[:50]

    plt.figure()
    shap.summary_plot(values_to_plot, X_test_scaled, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved shap_summary.png")


def main():
    print("=" * 60)
    print("Breast Cancer Early Detection — Model Training Pipeline")
    print("=" * 60)

    df, feature_names, target_names = load_data()
    explore_data(df)

    (X_train_scaled, X_test_scaled, y_train, y_test,
     scaler, X_train_raw, X_test_raw) = split_and_scale(df, feature_names)

    models = train_models(X_train_scaled, y_train)
    best_name, best_model = evaluate_models(models, X_test_scaled, y_test, target_names)
    explain_model(best_name, best_model, X_train_scaled, X_test_scaled, feature_names)

    print("\nDone. Check the 'outputs' folder for plots and results.txt")
    print("Reminder: this is a learning project, not a medical diagnostic tool.")


if __name__ == "__main__":
    main()