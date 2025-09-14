# src/dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from data_prep import load_and_clean

# ── Page config 
st.set_page_config(page_title="Retention Radar", layout="wide")
st.title("🚨 Retention Radar Dashboard")

# ── Data loading (cache) 
@st.cache_data
def load_data(n: int = 500) -> pd.DataFrame:
    df = load_and_clean("data/customers.csv")
    df = df.reset_index(drop=True).reset_index().rename(columns={"index": "customer_id"})
    return df.head(n)

# ── Model loading (cache) 
@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

# ── Predict all (cache; no in-place mutation) 
@st.cache_data
def predict_all(data: pd.DataFrame) -> pd.DataFrame:
    clf = load_model()
    df = data.copy()
    X = df.drop(columns=["customer_id", "churn"], errors="ignore")

    # If the model was trained on a fixed set of columns, align to them
    try:
        trained_cols = clf.feature_names_in_
        for c in trained_cols:
            if c not in X.columns:
                X[c] = 0
        X = X[trained_cols]
    except AttributeError:
        
        pass

    # Use probabilities (floats in [0,1]), not hard labels
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X)[:, 1]
    elif hasattr(clf, "decision_function"):
        # Fallback: squash decision scores to (0,1)
        scores = clf.decision_function(X)
        probs = 1 / (1 + np.exp(-scores))
    else:
        # Last resort: use predictions but cast to float (will be 0/1)
        probs = clf.predict(X).astype(float)

    df["churn_prob"] = probs.astype(float)

    df["recommendation"] = [
        "Offer 20% discount" if p > 0.8
        else "Offer 10% discount" if p > 0.5
        else "Send thank-you note"
        for p in df["churn_prob"]
    ]
    return df

# ── Load + predict ─────
raw_df = load_data()
df = predict_all(raw_df)

# ── Controls ──────
left, right = st.columns([2, 1])
with left:
    threshold = st.slider("Churn probability", 0.0, 1.0, 0.70, 0.05)
with right:
    mode = st.radio(
        "Filter mode",
        ["Around threshold (± 5%)", "≥ threshold"],  # default to band view
        index=0
    )

# ── Filtering ─────
if mode == "≥ threshold":
    filtered = df[df["churn_prob"] >= threshold].copy()
    # Keep highest first (makes sense for ≥ mode)
    filtered = filtered.sort_values("churn_prob", ascending=False)
else:
    band = 0.05
    lo, hi = max(0.0, threshold - band), min(1.0, threshold + band)
    within = df[(df["churn_prob"] >= lo) & (df["churn_prob"] <= hi)].copy()
    # Show rows closest to the slider value at the top
    filtered = within.assign(dist=(within["churn_prob"] - threshold).abs()) \
                     .sort_values("dist", ascending=True) \
                     .drop(columns="dist")

filtered = filtered.reset_index(drop=True)

# ── Display ────
st.markdown(
    f"**Showing {len(filtered)} of {len(df)} customers ({mode}; slider={threshold:.2f})**"
)


display_df = filtered[["customer_id", "churn", "churn_prob", "recommendation"]].copy()
display_df = display_df.assign(
    churn_prob_float = display_df["churn_prob"].round(4),
    churn_prob_pct   = (display_df["churn_prob"] * 100).round(2)
)[["customer_id", "churn", "churn_prob_float", "churn_prob_pct", "recommendation"]]
display_df = display_df.rename(columns={
    "churn_prob_float": "churn_prob",
    "churn_prob_pct": "churn_prob(%)"
})

st.dataframe(display_df, use_container_width=True, height=520)

# Debug panel 
with st.expander("Debug: probability distribution"):
    st.write(df["churn_prob"].describe())
    st.write("Unique (rounded to 3dp):",
             sorted(pd.Series(df["churn_prob"].round(3)).unique())[:20], "...")
