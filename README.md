# SMS Spam Detection — AI Capstone Project

A beginner-friendly end-to-end machine-learning project that classifies an SMS message as **Spam** or **Not Spam (Ham)**.

## Project overview

**Problem statement:** Mobile users receive unwanted promotional and fraudulent SMS messages. This project trains a text-classification model to flag likely spam messages and provides a small web interface for testing messages.

**AI technique:** TF-IDF text features + Logistic Regression.

**Data source:** A small illustrative dataset is included so the project can run immediately. For a stronger academic submission, replace/extend it with a reputable, appropriately licensed SMS Spam Collection dataset and cite its source in your report. The bundled sample is for demonstration only and is too small for reliable real-world performance claims.

## Project structure

```text
sms_spam_detection_capstone/
├── app.py                  # Streamlit web app
├── train.py                # Train, evaluate, save model
├── requirements.txt
├── README.md
├── reports/project_report.md       # Report template with project details
├── reports/presentation_outline.md         # Five-slide pitch content
├── data/
│   └── sms_sample.csv      # Illustrative sample data
├── models/                 # Created after training
├── reports/                # Created after training
├── src/
│   └── predict.py          # Prediction helper
└── tests/
    └── test_predict.py     # Basic tests
```

## Requirements

- Python 3.10 or newer
- pip

## Setup

```bash
python -m venv .venv
```

Activate the environment:

- Windows: `.venv\\Scripts\\activate`
- macOS/Linux: `source .venv/bin/activate`

Install packages:

```bash
pip install -r requirements.txt
```

## Train and evaluate

```bash
python train.py
```

This trains a TF-IDF + Logistic Regression pipeline, prints evaluation metrics, and creates:
- `models/spam_model.joblib`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

The split is stratified where possible. Metrics from the tiny included sample are only a software demonstration and must not be presented as evidence of production readiness.

## Run the web app

Train first, then run:

```bash
streamlit run app.py
```

Enter a message and select **Classify message**. The app shows the predicted class and, when supported by the model, a spam score.

## Test

```bash
python -m unittest discover -s tests
```

## Important limitations and responsible use

- The included data is small and illustrative; it is not representative of all languages, regions, or modern spam tactics.
- A spam score is a model estimate, not proof that a message is malicious.
- Do not automatically delete or block messages based only on this demo.
- Avoid entering private or sensitive messages into an untrusted deployment.
- Evaluate false positives and false negatives before any real-world use.
- For a formal submission, use a larger licensed dataset, document its source, and report measured results from that dataset.

## Suggested improvements

1. Use a larger, documented public SMS dataset.
2. Compare Multinomial Naive Bayes, Logistic Regression, and Linear SVM.
3. Tune the classification threshold based on the cost of false positives/negatives.
4. Add language support, drift monitoring, and a feedback mechanism.
5. Deploy to a suitable hosting service only after privacy and security review.
