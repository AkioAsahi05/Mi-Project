from pathlib import Path
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "spam_model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run `python train.py` first.")
    return joblib.load(MODEL_PATH)

def predict_message(message: str, model=None):
    if not isinstance(message, str) or not message.strip():
        raise ValueError("Please provide a non-empty message.")
    if model is None:
        model = load_model()

    label = str(model.predict([message])[0])
    result = {"label": label}
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        if "spam" in classes:
            result["spam_probability"] = float(probabilities[classes.index("spam")])
    return result
