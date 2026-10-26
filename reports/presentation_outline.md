# Five-Slide Capstone Pitch: SMS Spam Detection

## Slide 1 — The Hook: Problem
- Unwanted and fraudulent SMS messages can waste time and expose users to scams.
- **Problem statement:** Users need a simple way to flag SMS messages that resemble spam.
- Goal: classify a message as likely spam or likely normal.
- AI technique: TF-IDF + Logistic Regression.

## Slide 2 — The Data
- Input: labeled SMS text with `spam` or `ham` labels.
- Data preparation: remove missing/blank rows, invalid labels, and duplicates.
- Feature engineering: TF-IDF represents words and two-word phrases numerically.
- Limitation: the included sample is small and illustrative; replace it with a larger documented dataset for a formal evaluation.

## Slide 3 — The Solution
- Train/test split with stratification.
- Pipeline: TF-IDF Vectorizer → Logistic Regression.
- Saved artifact: `models/spam_model.joblib`.
- Why this approach: fast, interpretable baseline suitable for a text-classification demonstration.

## Slide 4 — Impact and Demo
- Demo: enter a message in the Streamlit interface and view the predicted class and spam score.
- Report actual accuracy, spam precision, spam recall, F1-score, and confusion matrix after running training.
- Do not invent performance numbers.
- Real-world value: assist review, not automatically determine whether a message is malicious.

## Slide 5 — Future Scope
- Replace the sample with a larger, licensed public dataset.
- Compare multiple models and tune the decision threshold.
- Add language support, privacy safeguards, and feedback collection.
- Add model monitoring, versioning, and secure deployment.
- Main takeaway: reliable AI requires good data, measured evaluation, a usable interface, and responsible use.
