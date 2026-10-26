import streamlit as st
from src.predict import load_model, predict_message

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")
st.title("📩 SMS Spam Detector")
st.write("Enter an SMS message to estimate whether it resembles spam or a normal message.")
st.warning("Educational demo only. Do not rely on this model to make security decisions.")

@st.cache_resource
def get_model():
    return load_model()

try:
    model = get_model()
    message = st.text_area(
        "SMS message",
        placeholder="Example: Congratulations! You have won a prize. Claim now!",
        height=140
    )
    if st.button("Classify message", type="primary"):
        if not message.strip():
            st.error("Please enter a message first.")
        else:
            result = predict_message(message, model)
            if result["label"] == "spam":
                st.error("Prediction: Likely SPAM")
            else:
                st.success("Prediction: Likely NOT SPAM (HAM)")
            if "spam_probability" in result:
                st.metric("Estimated spam score", f"{result['spam_probability']:.1%}")
            st.caption("This is a model estimate, not proof. Check suspicious messages independently.")
except FileNotFoundError:
    st.error("Model file not found. In the project folder, run `python train.py` first.")
