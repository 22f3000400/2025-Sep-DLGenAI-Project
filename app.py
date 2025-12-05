import streamlit as st
import torch

from scripts.inference import load_all_models, predict, LABEL_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="DLGenAI Emotion Detection Demo", page_icon="😊")

st.title("DLGenAI Emotion Detection Demo")
st.write(
    "Enter any sentence and choose a model. "
    "The app will predict one or more emotions."
)


@st.cache_resource
def get_models():
    return load_all_models()


MODELS = get_models()
MODEL_NAMES = list(MODELS.keys())

text = st.text_area("Enter text:", height=120)
model_name = st.selectbox("Choose Model", MODEL_NAMES)
threshold = st.slider("Prediction Threshold", 0.10, 0.90, 0.50, 0.05)

clicked = st.button("Predict")

if clicked:
    if not text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        with st.spinner("Running inference..."):
            labels, probs, active = predict(text, model_name, MODELS, threshold)

        st.subheader("Predicted Emotions")
        if active:
            st.write(", ".join(active))
        else:
            st.write("No emotion crossed the chosen threshold.")

        st.subheader("Raw Probabilities")
        for lbl, p in zip(labels, probs):
            st.write(f"{lbl}: {p:.4f}")

        st.success("Prediction completed!")
