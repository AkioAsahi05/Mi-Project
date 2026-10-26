from pathlib import Path
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sms_sample.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = {"label", "message"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns: {sorted(required)}")

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()
    df = df[(df["message"] != "") & df["label"].isin(["spam", "ham"])]
    df = df.drop_duplicates(subset=["message"])
    if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
        raise ValueError("Need at least two examples for each label (spam and ham).")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.25,
        random_state=42, stratify=df["label"]
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                  ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "dataset_rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_spam": float(precision_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(y_test, y_pred, pos_label="spam", zero_division=0)),
        "note": "Illustrative small dataset; these scores are not evidence of production performance."
    }

    joblib.dump(model, MODEL_DIR / "spam_model.joblib")
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = classification_report(y_test, y_pred, labels=["ham", "spam"], zero_division=0)
    (REPORT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
    disp.plot()
    plt.title("SMS Spam Detection — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)
    print(f"\nSaved model: {MODEL_DIR / 'spam_model.joblib'}")
    print(f"Saved reports: {REPORT_DIR}")

if __name__ == "__main__":
    main()
