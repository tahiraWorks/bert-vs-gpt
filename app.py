import random
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BERT vs GPT-2", page_icon="🤖", layout="centered")

# Models are downloaded from the Hugging Face Hub
@st.cache_resource(max_entries=1)
def load_model(name):
    if name == "BERT":
        return pipeline("sentiment-analysis",
                        model="textattack/bert-base-uncased-SST-2")
    return pipeline("text-generation", model="gpt2")

@st.cache_data
def load_samples():
    try:
        return pd.read_csv("sample_reviews.csv")["review"].tolist()
    except Exception:
        return []

samples = load_samples()

st.title("BERT vs GPT-2")
st.caption("Encoder-only vs decoder-only Transformers, loaded from Hugging Face")

page = st.sidebar.radio(
    "Choose a model",
    ["BERT (encoder-only)", "GPT-2 (decoder-only)", "Comparison"],
)

# ---------------- BERT ----------------
if page == "BERT (encoder-only)":
    st.header("BERT: sentiment classification")
    st.write("BERT reads the whole text and predicts a label.")

    st.session_state.setdefault("bert_text", "The movie was excellent.")

    if samples and st.button("Load a random IMDB review"):
        st.session_state["bert_text"] = random.choice(samples)

    text = st.text_area("Input text", key="bert_text", height=160)

    if st.button("Classify", type="primary"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading BERT and classifying..."):
                clf = load_model("BERT")
                r = clf(text, truncation=True, max_length=512)[0]
            label = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}.get(r["label"], r["label"])
            if label == "POSITIVE":
                st.success(f"Output: {label}")
            else:
                st.error(f"Output: {label}")
            st.metric("Confidence", f"{r['score']:.2%}")
            st.progress(float(r["score"]))

# ---------------- GPT-2 ----------------
elif page == "GPT-2 (decoder-only)":
    st.header("GPT-2: text generation")
    st.write("GPT-2 uses the previous tokens to predict the next ones.")

    prompt = st.text_input("Start of a sentence", "Artificial intelligence will")
    c1, c2, c3 = st.columns(3)
    max_new = c1.slider("New tokens", 10, 100, 40)
    temp = c2.slider("Temperature", 0.1, 1.5, 0.8)
    top_k = c3.slider("Top-k", 1, 100, 50)

    if st.button("Generate", type="primary"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Loading GPT-2 and generating..."):
                gen = load_model("GPT2")
                out = gen(prompt, max_new_tokens=max_new, do_sample=True,
                          temperature=temp, top_k=top_k, pad_token_id=50256)
            st.subheader("Generated text")
            st.write(out[0]["generated_text"])
            st.info("Click Generate again: the output changes because GPT-2 samples the next token.")

# ---------------- Comparison ----------------
else:
    st.header("Comparison")
    st.markdown("""
| | BERT (encoder-only) | GPT-2 (decoder-only) |
|---|---|---|
| Input | Full sentence | Beginning of a sentence |
| Output | Label + confidence | New generated text |
| Attention | Bidirectional | Causal (left to right) |
| Pre-training | Masked word prediction | Next-token prediction |
| Best for | Classification, understanding | Text generation |
""")
