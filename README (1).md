# BERT vs GPT-2

Streamlit app comparing an encoder-only model (BERT) and a decoder-only model (GPT-2).

- BERT model: https://huggingface.co/textattack/bert-base-uncased-SST-2
- GPT-2 model: https://huggingface.co/gpt2
- Dataset: IMDB 50K movie reviews (30 sample reviews in sample_reviews.csv)

Run locally:

    pip install -r requirements.txt
    streamlit run app.py
