"""Streamlit interface for the project's Twitter sentiment model."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.components.data_transformation import transform_text


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"
DATASET_PATH = PROJECT_DIR / "dataset.csv"
TRAIN_TEXT_PATH = PROJECT_DIR / "data" / "interim_data" / "train_preprocessed.csv"
TRAIN_LABEL_PATH = PROJECT_DIR / "data" / "raw_data" / "train.csv"

# Keep these settings in sync with feature_engineering.py. The project saves
# the classifier but not the fitted vectorizer, so the vocabulary is rebuilt.
VECTORIZER_SETTINGS = {
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_features": 18000,
}

SENTIMENT_STYLES = {
    "Positive": ("#16a878", "😊"),
    "Negative": ("#f05b67", "🌧️"),
    "Neutral": ("#6575f5", "☁️"),
    "Irrelevant": ("#9b7bf4", "✦"),
}


def load_training_split() -> pd.DataFrame:
    """Recreate the training split from the tracked source dataset."""
    source_data = pd.read_csv(DATASET_PATH)
    required_columns = {"ID", "Entity", "Sentiment", "Text"}
    if not required_columns.issubset(source_data.columns):
        raise ValueError(
            f"Expected columns {sorted(required_columns)} in {DATASET_PATH}."
        )

    source_data = source_data.drop(columns=["ID"])
    source_data["Combined_Text"] = (
        source_data["Entity"] + " " + source_data["Text"]
    )
    source_data = (
        source_data.drop(columns=["Entity", "Text"])
        .dropna(subset=["Combined_Text"])
        .drop_duplicates()
    )
    training_data, _ = train_test_split(
        source_data,
        test_size=0.4,
        random_state=2,
    )
    return training_data


@st.cache_resource(show_spinner="Loading the sentiment model…")
def load_artifacts() -> tuple[Any, CountVectorizer, dict[int, str]]:
    """Load the model and recreate the vectorizer using its training corpus."""
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)

    training_source: pd.DataFrame | None = None
    if TRAIN_TEXT_PATH.is_file():
        training_text = pd.read_csv(TRAIN_TEXT_PATH)
    else:
        training_source = load_training_split()
        training_text = training_source[["Combined_Text"]].copy()
        training_text["Combined_Text"] = training_text["Combined_Text"].apply(
            transform_text
        )

    if "Combined_Text" not in training_text.columns:
        raise ValueError(
            f"Expected a 'Combined_Text' column in {TRAIN_TEXT_PATH}."
        )

    vectorizer = CountVectorizer(**VECTORIZER_SETTINGS)
    vectorizer.fit(
        training_text["Combined_Text"].fillna("").astype(str)
    )

    if len(vectorizer.vocabulary_) != model.n_features_in_:
        raise ValueError(
            "The rebuilt text vocabulary does not match the saved model "
            f"({len(vectorizer.vocabulary_)} features vs "
            f"{model.n_features_in_} expected)."
        )

    if TRAIN_LABEL_PATH.is_file():
        raw_training_data = pd.read_csv(TRAIN_LABEL_PATH)
    else:
        if training_source is None:
            training_source = load_training_split()
        raw_training_data = training_source[["Sentiment"]]

    if "Sentiment" not in raw_training_data.columns:
        raise ValueError(
            f"Expected a 'Sentiment' column in {TRAIN_LABEL_PATH}."
        )

    label_encoder = LabelEncoder().fit(raw_training_data["Sentiment"])
    expected_class_ids = np.arange(len(label_encoder.classes_))
    if not np.array_equal(model.classes_, expected_class_ids):
        raise ValueError(
            "The model's class IDs do not match the labels in the training data."
        )

    class_names = {
        int(class_id): str(label)
        for class_id, label in enumerate(label_encoder.classes_)
    }
    return model, vectorizer, class_names


st.set_page_config(
    page_title="TweetTone · Sentiment Studio",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# A lightweight CSS-only visual layer keeps the app self-contained while
# adding a gradient backdrop, floating accents, and subtle entrance motion.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --ink: #17213b;
        --muted: #68748e;
        --line: rgba(118, 132, 170, 0.18);
        --glass: rgba(255, 255, 255, 0.78);
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(ellipse at 8% 2%, rgba(201, 218, 255, .55), transparent 34%),
            radial-gradient(ellipse at 94% 12%, rgba(232, 210, 255, .48), transparent 30%),
            linear-gradient(145deg, #f8faff 0%, #f4f6ff 48%, #fff8f5 100%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.6rem;
        border: 1px solid rgba(255, 255, 255, .8);
        border-radius: 30px;
        background: linear-gradient(120deg, rgba(255,255,255,.92), rgba(247,248,255,.72));
        box-shadow: 0 24px 70px rgba(67, 82, 139, .11);
        animation: rise-in .7s ease both;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 290px;
        height: 290px;
        right: -80px;
        top: -150px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(121, 145, 255, .28), rgba(211, 131, 255, .19));
        filter: blur(2px);
        animation: float 8s ease-in-out infinite;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .42rem .8rem;
        border: 1px solid rgba(111, 126, 234, .18);
        border-radius: 999px;
        background: rgba(255,255,255,.75);
        color: #626ed0;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .hero h1 {
        position: relative;
        z-index: 1;
        margin: 1rem 0 .45rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2.4rem, 6vw, 4.2rem);
        font-weight: 800;
        letter-spacing: -.065em;
        line-height: 1.05;
    }
    .gradient-word {
        color: #7564e8;
        background: linear-gradient(95deg, #6579ee, #a36be4 60%, #eb7fa9);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-copy {
        position: relative;
        z-index: 1;
        max-width: 600px;
        margin: 0;
        color: var(--muted);
        font-size: 1.06rem;
        line-height: 1.7;
    }
    .hero-orb {
        position: absolute;
        z-index: 1;
        right: 12%;
        bottom: 20%;
        display: grid;
        width: 76px;
        height: 76px;
        place-items: center;
        border: 1px solid rgba(255,255,255,.9);
        border-radius: 24px;
        background: linear-gradient(145deg, #8b9bff, #c689ee);
        box-shadow: 0 15px 32px rgba(115, 107, 221, .28);
        font-size: 2rem;
        transform: rotate(9deg);
        animation: float 5s ease-in-out infinite;
    }
    .section-label {
        margin: 1.7rem 0 .7rem;
        color: #596681;
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .info-card, .result-card {
        padding: 1.35rem 1.5rem;
        border: 1px solid var(--line);
        border-radius: 22px;
        background: var(--glass);
        box-shadow: 0 14px 40px rgba(67, 82, 139, .07);
        animation: rise-in .55s ease both;
    }
    .info-card h3 {
        margin: .15rem 0 .55rem;
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
        font-size: 1.16rem;
    }
    .info-card p {
        margin: 0;
        color: var(--muted);
        font-size: .94rem;
        line-height: 1.65;
    }
    .class-pills {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1rem;
    }
    .pill {
        padding: .42rem .7rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,.76);
        color: #596681;
        font-size: .82rem;
        font-weight: 600;
    }
    .result-card {
        margin: 1.2rem 0 .7rem;
        padding: 1.7rem;
        border-color: color-mix(in srgb, var(--sentiment) 23%, white);
        background: linear-gradient(120deg, color-mix(in srgb, var(--sentiment) 8%, white), rgba(255,255,255,.88));
    }
    .result-kicker {
        color: var(--muted);
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .11em;
        text-transform: uppercase;
    }
    .result-label {
        margin: .45rem 0 0;
        color: var(--sentiment);
        font-family: 'Manrope', sans-serif;
        font-size: 2rem;
        font-weight: 800;
    }
    .result-icon { margin-right: .4rem; }
    .footer {
        margin-top: 2.2rem;
        color: #8a93a9;
        font-size: .82rem;
        text-align: center;
    }
    div.stButton > button[kind="primaryFormSubmit"] {
        min-height: 3.1rem;
        border: 0;
        border-radius: 13px;
        background: linear-gradient(100deg, #687cf0, #9c6ee8 65%, #ca72c4);
        box-shadow: 0 9px 22px rgba(119, 105, 224, .24);
        color: white;
        font-weight: 700;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    div.stButton > button[kind="primaryFormSubmit"]:hover {
        box-shadow: 0 13px 28px rgba(119, 105, 224, .32);
        transform: translateY(-2px);
    }
    [data-testid="stTextArea"] textarea {
        border: 1px solid rgba(118, 132, 170, .22);
        border-radius: 15px;
        background: rgba(255,255,255,.82);
        line-height: 1.6;
    }
    @keyframes rise-in {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(9deg); }
        50% { transform: translateY(-11px) rotate(13deg); }
    }
    @media (max-width: 720px) {
        .hero { padding: 1.6rem 1.4rem; border-radius: 23px; }
        .hero-orb { right: 7%; bottom: 9%; width: 56px; height: 56px; font-size: 1.55rem; }
        .hero-copy { max-width: 78%; font-size: .97rem; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <span class="eyebrow">✦ &nbsp; Sentiment Studio</span>
        <h1>Read the <span class="gradient-word">room.</span></h1>
        <p class="hero-copy">
            A little context goes a long way. Paste a tweet and let the model
            reveal the feeling behind the words.
        </p>
        <div class="hero-orb" aria-hidden="true">💬</div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Your text, decoded</div>', unsafe_allow_html=True)
input_column, guide_column = st.columns([1.45, 1], gap="large")

with input_column:
    with st.form("sentiment_form"):
        user_text = st.text_area(
            "Paste a tweet or short message",
            placeholder="What's happening on your timeline?",
            height=190,
            max_chars=5000,
            help="The model removes links, punctuation, stopwords, and word endings before scoring.",
        )
        submitted = st.form_submit_button(
            "✨  Analyze sentiment",
            type="primary",
            use_container_width=True,
        )

with guide_column:
    st.markdown(
        """
        <div class="info-card">
            <h3>Four possible reads</h3>
            <p>
                The classifier was trained on Twitter text. It predicts one
                label for each message and shows how the model's class scores
                compare.
            </p>
            <div class="class-pills">
                <span class="pill">😊 Positive</span>
                <span class="pill">🌧️ Negative</span>
                <span class="pill">☁️ Neutral</span>
                <span class="pill">✦ Irrelevant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(
        """
        <div class="info-card">
            <h3>How to get the best read</h3>
            <p>
                Use a complete thought with the words that carry its tone.
                The model works best on short, English-language social posts.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    if not user_text.strip():
        st.warning("Add some text first, then run the sentiment analysis.")
    else:
        try:
            model, vectorizer, class_names = load_artifacts()
        except LookupError as error:
            st.error(
                "The NLTK language data used by the training pipeline is missing. "
                "Install the required tokenizer and stopword data, then restart "
                f"the app. Details: {error}"
            )
            st.stop()
        except (FileNotFoundError, ValueError) as error:
            st.error(f"The model could not be prepared: {error}")
            st.stop()

        try:
            cleaned_text = transform_text(user_text)
        except LookupError as error:
            st.error(
                "The NLTK language data used by the training pipeline is missing. "
                "Install the required tokenizer and stopword data, then restart "
                f"the app. Details: {error}"
            )
            st.stop()

        if not cleaned_text:
            st.warning(
                "No words remained after the model's text-cleaning steps. "
                "Try a message with a few descriptive words."
            )
        else:
            with st.spinner("Reading the tone…"):
                features = vectorizer.transform([cleaned_text])
                predicted_class = int(model.predict(features)[0])
                decision_scores = model.decision_function(features)[0]

            sentiment = class_names[predicted_class]
            color, icon = SENTIMENT_STYLES.get(sentiment, ("#6575f5", "✦"))
            st.markdown(
                f"""
                <div class="result-card" style="--sentiment: {color};">
                    <div class="result-kicker">The model reads this as</div>
                    <div class="result-label"><span class="result-icon">{icon}</span>{sentiment}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if features.nnz == 0:
                st.info(
                    "None of the cleaned words matched the model's learned vocabulary, "
                    "so this prediction is based on the model's default class margins."
                )

            score_rows = [
                {
                    "Sentiment": class_names[int(class_id)],
                    "Decision score": float(score),
                }
                for class_id, score in zip(model.classes_, decision_scores)
            ]
            score_rows.sort(key=lambda row: row["Decision score"], reverse=True)

            with st.expander("See how the model compared each sentiment"):
                st.caption(
                    "These are SVM decision scores, not probabilities or calibrated confidence."
                )
                st.bar_chart(
                    pd.DataFrame(score_rows),
                    x="Sentiment",
                    y="Decision score",
                    height=240,
                )

st.markdown(
    '<div class="footer">Built with the project\'s saved LinearSVC model · '
    'Predictions are model estimates, not a substitute for human context.</div>',
    unsafe_allow_html=True,
)
