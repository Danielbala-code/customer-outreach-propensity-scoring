"""Streamlit interface for the customer outreach propensity prototype."""
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_FILE = ROOT / "models" / "outreach_propensity_model.joblib"

st.set_page_config(page_title="Outreach propensity scoring", page_icon="📈", layout="wide")
st.title("Customer Outreach Propensity Scoring")
st.caption("A public-data prototype: prioritise eligible customers for a re-engagement campaign. Not a live B2B CRM system.")

@st.cache_data
def load_metrics():
    return json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))

@st.cache_data
def load_queue():
    return pd.read_csv(OUTPUT_DIR / "scored_customers.csv")

metrics = load_metrics()
queue = load_queue()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Chronological test ROC-AUC", f"{metrics['roc_auc']:.3f}")
col2.metric("Top-decile precision", f"{metrics['precision_at_top_10_percent']:.1%}")
col3.metric("Lift vs base rate", f"{metrics['lift_at_top_10_percent']:.2f}×")
col4.metric("Test conversion base rate", f"{metrics['test_base_conversion_rate']:.1%}")

st.caption(
    f"Chronological holdout: {metrics['test_rows']:,} customer-snapshots; "
    f"top decile contains {metrics['top_10_percent_leads']:,} of them."
)
comparison = pd.DataFrame(
    {"Observed purchase rate": [metrics["test_base_conversion_rate"], metrics["precision_at_top_10_percent"]]},
    index=["All eligible customers", "Top 10% ranked customers"],
)
st.bar_chart(comparison, y="Observed purchase rate")
st.caption("This is historical ranking evidence, not proof of incremental campaign impact. A live campaign needs a randomised holdout.")

st.subheader("Current scored outreach queue")
priority = st.multiselect("Priority bands", ["High", "Medium", "Low"], default=["High"])
view = queue.loc[queue["priority_band"].isin(priority)].copy()
st.dataframe(view[["CustomerID", "Country", "propensity_score", "priority_band", "recommended_action", "score_reason"]], width="stretch", hide_index=True)
st.download_button("Download filtered queue", view.to_csv(index=False).encode("utf-8"), "scored_outreach_queue.csv", "text/csv")

st.subheader("Score a feature-ready CSV")
upload = st.file_uploader("Upload a CSV containing the documented feature columns", type="csv")
if upload is not None:
    incoming = pd.read_csv(upload)
    artefact = joblib.load(MODEL_FILE)
    required = artefact["feature_columns"]
    missing = [column for column in required if column not in incoming.columns]
    if missing:
        st.error(f"Cannot score: missing required columns: {', '.join(missing)}")
    else:
        scored = incoming.copy()
        scored["propensity_score"] = artefact["pipeline"].predict_proba(scored[required])[:, 1]
        scored = scored.sort_values("propensity_score", ascending=False)
        st.dataframe(scored, width="stretch", hide_index=True)

with st.expander("Model controls and limitations"):
    st.markdown("""
    - Target: positive purchase in the 28 days after the scoring date.
    - Features use only the preceding 90 days; future transactions are excluded.
    - This is a retail re-engagement proxy using public UCI Online Retail data, not proprietary B2B CRM data.
    - A production CRM deployment requires identity resolution, consent/suppression checks, monitoring, and a reviewed data contract.
    """)

