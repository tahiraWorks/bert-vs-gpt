import html
import random
import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="BERT vs GPT-2", page_icon="🧠", layout="wide")

# ---------------------------------------------------------------- config
BERT_ID = "TahiraAbbas/bert-multilingual-imdb-sentiment"
GPT_ID = "TahiraAbbas/gpt2-wikitext"
BERT_URL = f"https://huggingface.co/{BERT_ID}"
GPT_URL = f"https://huggingface.co/{GPT_ID}"

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
.block-container {padding-top: 1.2rem; max-width: 1150px;}

/* hero */
.hero {
    background: linear-gradient(120deg, #0f172a 0%, #0e7490 55%, #f59e0b 130%);
    padding: 2.2rem 2.2rem 1.6rem 2.2rem; border-radius: 22px; color: #ffffff;
    margin-bottom: 1rem; box-shadow: 0 12px 32px rgba(14,116,144,0.35);
}
.hero h1 {margin: 0; font-size: 2.5rem; color: #ffffff; letter-spacing: -0.5px;}
.hero p {margin: 0.4rem 0 0 0; font-size: 1.05rem; color: #e2e8f0;}
.hero .b {color: #67e8f9; font-weight: 700;}
.hero .g {color: #fcd34d; font-weight: 700;}

/* stat tiles */
.stats {display: flex; gap: 0.8rem; margin: 0 0 1.2rem 0; flex-wrap: wrap;}
.stat {
    flex: 1; min-width: 150px; border-radius: 14px; padding: 0.9rem 1rem;
    border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.07);
}
.stat .n {font-size: 1.5rem; font-weight: 800;}
.stat .l {font-size: 0.8rem; opacity: 0.75; text-transform: uppercase; letter-spacing: 0.06em;}

/* model cards */
.mcard {
    border-radius: 16px; padding: 1.1rem 1.3rem; height: 100%;
    border: 1px solid rgba(128,128,128,0.25); background: rgba(128,128,128,0.06);
}
.mcard.bert {border-top: 6px solid #06b6d4;}
.mcard.gpt {border-top: 6px solid #f59e0b;}
.mcard h3 {margin: 0 0 0.5rem 0;}
.tag {
    display: inline-block; padding: 0.15rem 0.65rem; margin: 0 0.3rem 0.4rem 0;
    border-radius: 999px; font-size: 0.78rem; font-weight: 600;
}
.tag.c {background: rgba(6,182,212,0.18); color: #06b6d4;}
.tag.o {background: rgba(245,158,11,0.2); color: #f59e0b;}

/* sidebar mini cards */
.side {
    border-radius: 12px; padding: 0.8rem 0.9rem; margin-bottom: 0.7rem;
    border: 1px solid rgba(128,128,128,0.3); background: rgba(128,128,128,0.08);
    font-size: 0.88rem; line-height: 1.5;
}
.side.bert {border-left: 5px solid #06b6d4;}
.side.gpt {border-left: 5px solid #f59e0b;}

/* results */
.result {
    border-radius: 16px; padding: 1.3rem; text-align: center; color: #ffffff;
    font-size: 1.8rem; font-weight: 800; margin-bottom: 0.8rem;
}
.pos {background: linear-gradient(135deg, #047857, #10b981);}
.neg {background: linear-gradient(135deg, #be123c, #f43f5e);}
.gen {
    border-left: 5px solid #f59e0b; border-radius: 12px; padding: 1rem 1.2rem;
    background: rgba(245,158,11,0.09); line-height: 1.65; margin-bottom: 0.7rem;
}
.gen .prompt {background: rgba(245,158,11,0.3); padding: 0 4px; border-radius: 4px; font-weight: 700;}
.small {font-size: 0.85rem; opacity: 0.75;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- models
@st.cache_resource(max_entries=1)
def load_model(name):
    if name == "BERT":
        return pipeline("sentiment-analysis", model=BERT_ID)
    return pipeline("text-generation", model=GPT_ID)

@st.cache_data
def load_samples():
    try:
        return pd.read_csv("sample_reviews.csv")["review"].tolist()
    except Exception:
        return []

samples = load_samples()
LABELS = {"LABEL_0": "NEGATIVE", "LABEL_1": "POSITIVE"}

st.session_state.setdefault("bert_text", "The movie was excellent.")
st.session_state.setdefault("gpt_prompt", "Artificial intelligence will")
st.session_state.setdefault("bert_history", [])
st.session_state.setdefault("gpt_history", [])

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Project overview")
    st.markdown(f"""
<div class="side bert">
<b>🔍 BERT</b> · multilingual encoder<br>
Base: <code>bert-base-multilingual-cased</code><br>
<b>Dataset:</b> IMDB movie reviews<br>
<a href="{BERT_URL}" target="_blank">Open on Hugging Face</a>
</div>
<div class="side gpt">
<b>✍️ GPT-2</b> · monolingual decoder<br>
Base: <code>gpt2</code> (English)<br>
<b>Dataset:</b> WikiText-2 (Wikipedia)<br>
<a href="{GPT_URL}" target="_blank">Open on Hugging Face</a>
</div>
""", unsafe_allow_html=True)
    st.caption("Two different datasets: IMDB for BERT, WikiText-2 for GPT-2.")
    st.caption("Models run on CPU, so the first request can take a while.")
    if st.button("Clear history", use_container_width=True):
        st.session_state["bert_history"] = []
        st.session_state["gpt_history"] = []
        st.rerun()

# ---------------------------------------------------------------- hero
st.markdown("""
<div class="hero">
  <h1>🧠 BERT vs GPT-2</h1>
  <p><span class="b">BERT</span> understands text. <span class="g">GPT-2</span> generates text.
  Two Transformer architectures, two datasets, one interface.</p>
</div>
<div class="stats">
  <div class="stat"><div class="n">2</div><div class="l">Models</div></div>
  <div class="stat"><div class="n">2</div><div class="l">Datasets</div></div>
  <div class="stat"><div class="n">104 / 1</div><div class="l">Languages (BERT / GPT-2)</div></div>
  <div class="stat"><div class="n">Encoder / Decoder</div><div class="l">Architectures</div></div>
</div>
""", unsafe_allow_html=True)

tab_bert, tab_gpt, tab_data, tab_cmp = st.tabs(
    ["🔍 BERT: Understand", "✍️ GPT-2: Generate", "📚 Models & Datasets", "⚖️ Comparison"])

# ================================================================ BERT
with tab_bert:
    st.subheader("Sentiment classification (multilingual BERT)")
    st.caption("Fine-tuned on English IMDB reviews. Try other languages to see cross-lingual transfer.")

    st.write("**Try an example:**")
    ex = st.columns(4)
    examples = [
        "The movie was excellent.",
        "La película fue excelente.",
        "Ce film était terrible.",
        "یہ فلم بہت اچھی تھی۔",
    ]
    for col, e in zip(ex, examples):
        if col.button(e, key="ex_" + e, use_container_width=True):
            st.session_state["bert_text"] = e
    if samples and st.button("🎲 Load a random IMDB review"):
        st.session_state["bert_text"] = random.choice(samples)

    text = st.text_area("Input text", key="bert_text", height=150)
    st.markdown(f"<span class='small'>{len(text.split())} words</span>", unsafe_allow_html=True)

    if st.button("Classify", type="primary", key="classify"):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Loading BERT and classifying..."):
                clf = load_model("BERT")
                res = clf(text, truncation=True, max_length=512, top_k=None)
                if res and isinstance(res[0], list):
                    res = res[0]
            scores = {LABELS.get(r["label"], r["label"]): r["score"] for r in res}
            best = max(scores, key=scores.get)
            css = "pos" if best == "POSITIVE" else "neg"
            icon = "😊" if best == "POSITIVE" else "😞"
            st.markdown(f"<div class='result {css}'>{icon} {best}</div>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            c1.metric("Confidence", f"{scores[best]:.1%}")
            c2.metric("Words analysed", min(len(text.split()), 512))
            for lab in ["POSITIVE", "NEGATIVE"]:
                if lab in scores:
                    st.write(f"{lab}: {scores[lab]:.1%}")
                    st.progress(float(scores[lab]))
            st.session_state["bert_history"].insert(0, {
                "Input": text[:80] + ("..." if len(text) > 80 else ""),
                "Output": best,
                "Confidence": f"{scores[best]:.1%}",
            })

    if st.session_state["bert_history"]:
        st.markdown("#### History")
        st.dataframe(pd.DataFrame(st.session_state["bert_history"]), use_container_width=True, hide_index=True)

# ================================================================ GPT-2
with tab_gpt:
    st.subheader("Text generation (monolingual GPT-2)")
    st.caption("Fine-tuned on WikiText-2. GPT-2 is monolingual: it works with English only.")

    st.write("**Try a prompt:**")
    pr = st.columns(4)
    prompts = ["Artificial intelligence will", "The history of",
               "Once upon a time", "In the future,"]
    for col, p in zip(pr, prompts):
        if col.button(p, key="pr_" + p, use_container_width=True):
            st.session_state["gpt_prompt"] = p

    prompt = st.text_input("Start of a sentence", key="gpt_prompt")

    c1, c2, c3, c4 = st.columns(4)
    max_new = c1.slider("New tokens", 10, 100, 40)
    temp = c2.slider("Temperature", 0.1, 1.5, 0.8)
    top_k = c3.slider("Top-k", 1, 100, 50)
    n_out = c4.slider("Variations", 1, 3, 2)
    st.caption("Higher temperature = more creative. Lower = more predictable.")

    if st.button("Generate", type="primary", key="generate"):
        if not prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            with st.spinner("Loading GPT-2 and generating..."):
                gen = load_model("GPT2")
                outs = gen(prompt, max_new_tokens=max_new, do_sample=True, temperature=temp,
                           top_k=top_k, num_return_sequences=n_out, pad_token_id=50256)
            st.markdown("#### Generated text")
            for i, o in enumerate(outs, 1):
                full = o["generated_text"]
                cont = full[len(prompt):] if full.startswith(prompt) else full
                st.markdown(
                    f"<div class='gen'><b>Version {i}</b><br>"
                    f"<span class='prompt'>{html.escape(prompt)}</span>{html.escape(cont)}</div>",
                    unsafe_allow_html=True,
                )
                st.session_state["gpt_history"].insert(0, {"Prompt": prompt, "Generated": full})
            st.info("Press Generate again: the output changes because GPT-2 samples the next token.")

    if st.session_state["gpt_history"]:
        st.markdown("#### History")
        st.dataframe(pd.DataFrame(st.session_state["gpt_history"][:10]), use_container_width=True, hide_index=True)

# ================================================================ Models & Datasets
with tab_data:
    st.subheader("Models and datasets used")
    a, b = st.columns(2)
    with a:
        st.markdown(f"""
<div class="mcard bert">
<h3>🔍 BERT</h3>
<span class="tag c">Encoder-only</span><span class="tag c">Multilingual</span><span class="tag c">104 languages</span>
<p><b>Base model:</b> bert-base-multilingual-cased<br>
<b>Fine-tuned model:</b> <a href="{BERT_URL}" target="_blank">{BERT_ID}</a><br>
<b>Task:</b> sentiment classification<br>
<b>Dataset:</b> IMDB movie reviews (50K reviews, positive / negative)<br>
<b>Why this dataset:</b> BERT needs labelled data to learn a classification task.</p>
</div>""", unsafe_allow_html=True)
    with b:
        st.markdown(f"""
<div class="mcard gpt">
<h3>✍️ GPT-2</h3>
<span class="tag o">Decoder-only</span><span class="tag o">Monolingual</span><span class="tag o">English</span>
<p><b>Base model:</b> gpt2<br>
<b>Fine-tuned model:</b> <a href="{GPT_URL}" target="_blank">{GPT_ID}</a><br>
<b>Task:</b> text generation<br>
<b>Dataset:</b> WikiText-2 (Wikipedia articles)<br>
<b>Why this dataset:</b> GPT-2 learns from plain text and needs no labels.</p>
</div>""", unsafe_allow_html=True)

    st.markdown("#### Summary")
    st.dataframe(pd.DataFrame({
        "Model": ["BERT", "GPT-2"],
        "Architecture": ["Encoder-only", "Decoder-only"],
        "Language": ["Multilingual (104)", "Monolingual (English)"],
        "Dataset": ["IMDB movie reviews", "WikiText-2"],
        "Task": ["Sentiment classification", "Text generation"],
        "Hosted on": ["Hugging Face", "Hugging Face"],
    }), use_container_width=True, hide_index=True)

# ================================================================ Comparison
with tab_cmp:
    st.subheader("Encoder-only vs decoder-only")
    a, b = st.columns(2)
    with a:
        st.markdown("""
<div class="mcard bert">
<h3>🔍 BERT (encoder-only)</h3>
<b>Goal:</b> understand text<br>
<b>Input:</b> full sentence<br>
<b>Output:</b> label + confidence<br>
<b>Attention:</b> bidirectional<br>
<b>Pre-training:</b> masked word prediction<br>
<b>Best for:</b> classification, NER, question answering
</div>""", unsafe_allow_html=True)
    with b:
        st.markdown("""
<div class="mcard gpt">
<h3>✍️ GPT-2 (decoder-only)</h3>
<b>Goal:</b> generate text<br>
<b>Input:</b> beginning of a sentence<br>
<b>Output:</b> new text<br>
<b>Attention:</b> causal (left to right)<br>
<b>Pre-training:</b> next-token prediction<br>
<b>Best for:</b> text generation, completion
</div>""", unsafe_allow_html=True)

    st.markdown("#### Side-by-side")
    st.dataframe(pd.DataFrame({
        "": ["Input", "Output", "Attention", "Pre-training", "Best for"],
        "BERT (encoder-only)": ["Full sentence", "Label + confidence", "Bidirectional",
                                "Masked word prediction", "Understanding / classification"],
        "GPT-2 (decoder-only)": ["Start of a sentence", "New generated text", "Causal (left to right)",
                                 "Next-token prediction", "Text generation"],
    }), use_container_width=True, hide_index=True)

    st.success("Encoder = understanding the whole input. Decoder = generating the output step by step.")
