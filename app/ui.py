import streamlit as st
from app.inference import predict

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector (DistilBERT)")

text = st.text_area("Paste news text:", height=220)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please paste some text.")
    else:
        out = predict(text)
        st.subheader(f"Prediction: {out['label']}")
        st.write({"prob_fake": out["prob_fake"], "prob_real": out["prob_real"]})
