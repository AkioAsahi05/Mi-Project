# Project Report: SMS Spam Detection Using Machine Learning

## 1. Abstract
This project demonstrates a text-classification workflow that estimates whether an SMS message is spam or ham (normal). It uses TF-IDF to represent text numerically and Logistic Regression to classify messages. A Streamlit interface allows a user to test a message.

## 2. Problem statement
Mobile users receive unwanted messages that may advertise products, offer fake prizes, or attempt to trick recipients into revealing information. The goal is to build a basic tool that flags messages resembling spam and helps users review them carefully.

## 3. Objectives
- Prepare and inspect a labeled text dataset.
- Clean missing, duplicate, and invalid records.
- Convert text into numeric features using TF-IDF.
- Train a Logistic Regression classifier.
- Evaluate predictions using accuracy, precision, recall, F1-score, and a confusion matrix.
- Provide a simple interface for predictions.
- Identify limitations and responsible-use considerations.

## 4. Dataset and data quality
The included `data/sms_sample.csv` is a small illustrative dataset with `label` and `message` columns. Labels are `spam` or `ham`. The training script removes missing rows, blank messages, invalid labels, and duplicate messages.

**Important:** This demonstration dataset is intentionally small. It is suitable for showing that the project runs, not for drawing meaningful conclusions about real-world spam detection. For a formal capstone, replace it with a larger public dataset whose terms permit use, cite the source, document its size and label distribution, and repeat the evaluation.

## 5. Methodology
1. Load the CSV file.
2. Clean missing values, blank messages, invalid labels, and duplicates.
3. Split data into training and testing subsets with stratification.
4. Use TF-IDF word and two-word features.
5. Train Logistic Regression.
6. Predict labels for the held-out test set.
7. Calculate evaluation metrics and generate a confusion matrix.
8. Save the model and metrics.
9. Load the saved model in a Streamlit interface.

## 6. Model and evaluation
The model is a scikit-learn Pipeline containing `TfidfVectorizer` and `LogisticRegression`.

Metrics:
- **Accuracy:** Fraction of all predictions that are correct.
- **Precision (spam):** Of messages predicted as spam, the fraction that are actually spam.
- **Recall (spam):** Of actual spam messages, the fraction detected.
- **F1-score (spam):** Harmonic mean of spam precision and recall.
- **Confusion matrix:** Counts correct and incorrect predictions by class.

Run `python train.py` to generate the actual metric values in `reports/metrics.json` and `reports/classification_report.txt`. Do not claim a target score before measuring it. Because the bundled dataset is tiny, scores may vary substantially with a different split and are not representative of deployment performance.

## 7. Interface and deployment
The Streamlit app accepts a message, runs the trained pipeline, and displays a predicted class and estimated spam score. For local use, run `streamlit run app.py`. A production deployment would require a larger validated dataset, secure hosting, privacy controls, monitoring, and a process for correcting errors.

## 8. Risks, ethics, and limitations
- **False positive:** A normal message is flagged as spam.
- **False negative:** A spam message is missed.
- The dataset may not represent all languages, regions, slang, or new scam techniques.
- The score is not a guarantee that a message is safe or malicious.
- Avoid storing user messages unnecessarily.
- Do not automatically delete or block messages based only on this demo.
- Evaluate error rates and communicate limitations clearly.

## 9. Future scope
- Train on a larger, documented dataset.
- Compare Logistic Regression with Naive Bayes and Linear SVM.
- Tune thresholds to balance false positives and false negatives.
- Add language support and user feedback.
- Monitor performance drift and retrain with reviewed data.
- Add API deployment and versioning.

## 10. Conclusion
This project demonstrates the core stages of an applied ML product: problem definition, data preparation, feature extraction, model training, evaluation, interface design, and responsible-use review. The prototype is a learning tool; further data and testing are required before real-world use.
