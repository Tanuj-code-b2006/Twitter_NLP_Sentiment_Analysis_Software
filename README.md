# Twitter_NLP_Sentiment_Analysis_Software
This repo contains an end to end software for nlp sentiment analysis 

## Deploying the Streamlit app

The app entry point is `app.py`. Streamlit Community Cloud installs the runtime
packages listed in the root `requirements.txt`, including scikit-learn (imported
as `sklearn` in Python).

The generated `data/` files are not tracked by Git. If the preprocessed training
file is missing, the app rebuilds the training vocabulary from the tracked
`dataset.csv`; the light theme and readable text colors are configured in
`.streamlit/config.toml`. Text preprocessing uses whitespace tokenization, so
the app does not require NLTK's `punkt_tab`; missing NLTK stopwords are
downloaded on first use.
