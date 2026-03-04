import streamlit as st
import os
import cloudpickle
import pandas as pd
from emailanalyzer.utils import _normalize_and_mask_text
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH_RAW = "data/01_raw/spam_dataset.csv"
DATA_PATH_PREPROCESSED = "data/02_intermediate/preprocessed_emails.pq"
MODEL_PATH_SPAM = "data/06_models/regressor.pickl"
PLOT_PATH_EMAIL_SUMMARY = "data/08_reporting/plots/email_summary.png"


@st.cache_resource
def load_models():
    """Loads the trained models."""
    models = {}
    if os.path.exists(MODEL_PATH_SPAM):
        with open(MODEL_PATH_SPAM, "rb") as f:
            models["spam_detector"] = cloudpickle.load(f)
    return models


@st.cache_data
def load_data():
    """Loads available data layers."""
    data_dict = {}
    # Load Raw for trends
    if os.path.exists(DATA_PATH_RAW):
        try:
            data_dict["raw"] = pd.read_csv(DATA_PATH_RAW)
        except Exception:
            pass
    # Load Preprocessed for trends
    if os.path.exists(DATA_PATH_PREPROCESSED):
        try:
            data_dict["data"] = pd.read_parquet(DATA_PATH_PREPROCESSED)
        except Exception:
            pass

    return data_dict


def _create_template(data, models):
    df = data.get("data")
    raw = data.get("raw")
    model = models.get("spam_detector").get("model")
    total_customers = len(df)

    vectorizer = models.get("spam_detector").get("vectorizer")

    def clear_text():
        st.session_state.email_input = ""

    st.markdown("#### Test an email")
    st.text(
        "Example valid email ✅ : Can you bring your laptop tomorrow? I need to borrow it"
    )
    st.text(
        "Example invalid email ⚠️ : REMINDER FROM O2: To get 2.50 pounds free call credit and three months half price line rental, call free NOW on 0808 145 4742"
    )
    col1, col2 = st.columns([5, 1])
    with col1:
        email_test = st.text_input("Introduce an email to test", key="email_input")
    with col2:
        st.text("")  # Spacing to align button with input field
        st.text("")
        st.button("Clear", on_click=clear_text)

    if email_test:
        if vectorizer is not None:
            email_test = _normalize_and_mask_text(email_test)
            st.write("Email cleaned: ", email_test)
            email_features = vectorizer.transform([email_test])
            prediction = model.predict(email_features)
            probs = model.predict_proba(email_features)

            if prediction[0] == "spam":
                st.error(f"Prediction: Spam with {probs[0][1] * 100:.2f}% confidence")
            else:
                st.success(f"Prediction: Ham with {probs[0][0] * 100:.2f}% confidence")
        else:
            st.error(
                "Vectorizer not found in model package. Please re-run the Kedro pipeline."
            )

    st.markdown("#### Raw Data from CSV")
    with st.expander("View Raw Emails (First 7)"):
        st.dataframe(raw.head(7))

    st.markdown("#### Cleaned Data from Parquet")
    with st.expander("View Cleaned Emails (First 7)"):
        st.dataframe(df.head(7))

    st.markdown("#### Email Summary (Only test loaded images)")
    if os.path.exists(PLOT_PATH_EMAIL_SUMMARY):
        st.image(PLOT_PATH_EMAIL_SUMMARY, caption="Email Summary")


def main():
    with st.spinner("Synchronizing predictive cores..."):
        models = load_models()
        data = load_data()

    if not models or not data:
        st.error("⚠️ No data found. Please run the Kedro pipeline first.")
        st.code("kedro run")
        return

    st.title("Emails Spam Detector")

    _create_template(data, models)


if __name__ == "__main__":
    main()
