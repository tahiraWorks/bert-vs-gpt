# BERT vs GPT-2: Multilingual Encoder vs Monolingual Decoder

Streamlit app comparing an encoder-only model (BERT) and a decoder-only model (GPT-2).

**Live app:** https://bert-vs-gpt-mc9m4glhmahelwisshjscb.streamlit.app/

## Models

| Model | Type | Language | Fine-tuned on | Task |
|---|---|---|---|---|
| BERT (`bert-base-multilingual-cased`) | Encoder-only | Multilingual (104 languages) | IMDB movie reviews | Sentiment classification |
| GPT-2 (`gpt2`) | Decoder-only | Monolingual (English) | WikiText-2 | Text generation |

- BERT model: https://huggingface.co/TahiraAbbas/bert-multilingual-imdb-sentiment
- GPT-2 model: https://huggingface.co/TahiraAbbas/gpt2-wikitext

## Datasets

Two different datasets are used, one for each model:

- **IMDB 50K Movie Reviews** (Kaggle CSV): used to fine-tune BERT. 30 sample reviews are in `sample_reviews.csv` and feed the random-review button in the app.
- **WikiText-2** (Hugging Face: `Salesforce/wikitext`, `wikitext-2-raw-v1`): used to fine-tune GPT-2.

## Files

- `app.py`: Streamlit interface
- `requirements.txt`: Python packages
- `sample_reviews.csv`: 30 IMDB reviews used as example inputs
- `Task1_Training_Colab.ipynb`: fine-tuning code for both models

## Run locally

    pip install -r requirements.txt
    streamlit run app.py
