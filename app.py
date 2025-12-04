import streamlit as st
import torch
from scripts.inference import load_all_models, predict, LABEL_NAMES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="DLGenAI Emotion Classifier", page_icon="😊")

@st.cache_resource
def get_models():
    return load_all_models()

MODELS = get_models()
MODEL_NAMES = list(MODELS.keys())      # ["TextCNN"]

st.title("DLGenAI Emotion Detection Demo")
st.write("Enter any sentence and the model will predict emotions.")


user_text = st.text_area("Enter text:", height=120)

model_name = st.selectbox("Choose Model", MODEL_NAMES)

threshold = st.slider("Prediction Threshold", 0.1, 0.9, 0.5, 0.05)


if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        labels, probs, active = predict(
            text=user_text,
            model_name=model_name,
            models=MODELS,
            threshold=threshold,
        )

        st.subheader("Predicted Emotions")
        if len(active) == 0:
            st.write("No emotion above the threshold.")
        else:
            st.write(", ".join(active))

        st.subheader("Raw Probabilities")
        for lbl, pr in zip(labels, probs):
            st.write(f"**{lbl}**: {pr:.4f}")

        st.success("Prediction completed!")
