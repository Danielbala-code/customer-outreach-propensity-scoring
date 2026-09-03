"""Train a point-in-time customer outreach propensity model from public retail data.

This is a production-style prototype: the target is a purchase in the next 28 days,
and every feature is calculated from the 90 days available before a scoring date.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
RAW_PATH = Path(os.environ.get("RETAIL_DATA_PATH", ROOT.parent.parent / "data" / "uci_online_retail" / "Online Retail.xlsx"))
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

LOOKBACK_DAYS = 90
OUTCOME_DAYS = 28
TEST_SNAPSHOTS = 8


def read_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return valid positive sales and negative product returns with known customers."""
    raw = pd.read_excel(RAW_PATH, dtype={"InvoiceNo": "string", "StockCode": "string"})
    raw.columns = raw.columns.str.strip()
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"], errors="coerce")
    raw["StockCode"] = raw["StockCode"].str.strip()
    raw["CustomerID"] = raw["CustomerID"].astype("Int64")
    product = raw["StockCode"].str.fullmatch(r"\d{5}[A-Z]?", na=False)
    common = product & raw["InvoiceDate"].notna() & raw["CustomerID"].notna()

    sales = raw.loc[common & (raw["Quantity"] > 0) & (raw["UnitPrice"] > 0), [
        "InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice", "Country"
    ]].copy()
    sales["revenue"] = sales["Quantity"] * sales["UnitPrice"]

    returns = raw.loc[common & (raw["Quantity"] < 0) & (raw["UnitPrice"] >= 0), [
        "CustomerID", "InvoiceDate", "Quantity"
    ]].copy()
    returns["return_units"] = -returns["Quantity"]
    return sales, returns


def feature_rows(sales: pd.DataFrame, returns: pd.DataFrame, snapshot_date: pd.Timestamp, include_target: bool) -> pd.DataFrame:
    """Build one legal, point-in-time feature table at a scoring date."""
    history_start = snapshot_date - pd.Timedelta(days=LOOKBACK_DAYS)
    history = sales.loc[(sales["InvoiceDate"] >= history_start) & (sales["InvoiceDate"] < snapshot_date)].copy()
    if history.empty:
        return pd.DataFrame()

    history = history.sort_values("InvoiceDate")
    orders = history.groupby(["CustomerID", "InvoiceNo"], as_index=False).agg(
        order_revenue=("revenue", "sum"), order_units=("Quantity", "sum"), order_date=("InvoiceDate", "max")
    )
    features = orders.groupby("CustomerID", as_index=False).agg(
        purchases_90d=("InvoiceNo", "nunique"),
        spend_90d=("order_revenue", "sum"),
        units_90d=("order_units", "sum"),
        active_days_90d=("order_date", lambda x: x.dt.normalize().nunique()),
        avg_order_value=("order_revenue", "mean"),
        last_purchase=("order_date", "max"),
    )
    features["recency_days"] = (snapshot_date - features.pop("last_purchase")).dt.total_seconds() / 86400

    latest_country = history.sort_values("InvoiceDate").groupby("CustomerID", as_index=False).tail(1)[["CustomerID", "Country"]]
    features = features.merge(latest_country, on="CustomerID", how="left")

    prior_returns = returns.loc[(returns["InvoiceDate"] >= history_start) & (returns["InvoiceDate"] < snapshot_date)]
    return_units = prior_returns.groupby("CustomerID", as_index=False)["return_units"].sum()
    features = features.merge(return_units, on="CustomerID", how="left")
    features["return_units"] = features["return_units"].fillna(0)
    features["return_rate_units"] = features["return_units"] / (features["units_90d"] + features["return_units"])
    features["snapshot_date"] = snapshot_date

    if include_target:
        future_end = snapshot_date + pd.Timedelta(days=OUTCOME_DAYS)
        converted = sales.loc[(sales["InvoiceDate"] >= snapshot_date) & (sales["InvoiceDate"] < future_end), "CustomerID"].drop_duplicates()
        features["converted_next_28d"] = features["CustomerID"].isin(converted).astype(int)
    return features


def build_training_table(sales: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    first_date = sales["InvoiceDate"].min().normalize()
    final_date = sales["InvoiceDate"].max().normalize()
    snapshots = pd.date_range(first_date + pd.Timedelta(days=LOOKBACK_DAYS), final_date - pd.Timedelta(days=OUTCOME_DAYS), freq="7D")
    tables = [feature_rows(sales, returns, snapshot, include_target=True) for snapshot in snapshots]
    return pd.concat([table for table in tables if not table.empty], ignore_index=True)


def precision_at_fraction(y_true: pd.Series, scores: np.ndarray, fraction: float = 0.10) -> tuple[float, int]:
    k = max(1, math.ceil(len(y_true) * fraction))
    top = pd.DataFrame({"target": y_true.to_numpy(), "score": scores}).nlargest(k, "score")
    return float(top["target"].mean()), k


def reason(row: pd.Series) -> str:
    reasons = []
    if row["recency_days"] <= 14:
        reasons.append("purchased in the last 14 days")
    if row["purchases_90d"] >= 3:
        reasons.append("frequent 90-day buyer")
    if not reasons:
        reasons.append("eligible customer with recent purchase history")
    return "; ".join(reasons[:2])


def main() -> None:
    sales, returns = read_and_clean()
    training = build_training_table(sales, returns)
    training = training.sort_values("snapshot_date").reset_index(drop=True)

    snapshot_dates = sorted(training["snapshot_date"].unique())
    cutoff = pd.Timestamp(snapshot_dates[-TEST_SNAPSHOTS])
    train = training.loc[training["snapshot_date"] < cutoff].copy()
    test = training.loc[training["snapshot_date"] >= cutoff].copy()

    numeric_features = ["purchases_90d", "spend_90d", "units_90d", "active_days_90d", "avg_order_value", "recency_days", "return_units", "return_rate_units"]
    categorical_features = ["Country"]
    feature_columns = numeric_features + categorical_features

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric_features),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", LogisticRegression(max_iter=1500, solver="lbfgs")),
    ])
    model.fit(train[feature_columns], train["converted_next_28d"])

    test_scores = model.predict_proba(test[feature_columns])[:, 1]
    base_rate = float(test["converted_next_28d"].mean())
    precision_10, top_k = precision_at_fraction(test["converted_next_28d"], test_scores, 0.10)
    metrics = {
        "model": "Logistic regression with median imputation, scaling, and one-hot encoded country",
        "target": "Positive sale by an eligible customer in the 28 days after the scoring date",
        "lookback_days": LOOKBACK_DAYS,
        "test_cutoff": str(cutoff.date()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "test_base_conversion_rate": base_rate,
        "roc_auc": float(roc_auc_score(test["converted_next_28d"], test_scores)),
        "average_precision": float(average_precision_score(test["converted_next_28d"], test_scores)),
        "brier_score": float(brier_score_loss(test["converted_next_28d"], test_scores)),
        "precision_at_top_10_percent": precision_10,
        "top_10_percent_leads": top_k,
        "lift_at_top_10_percent": float(precision_10 / base_rate) if base_rate else None,
        "leakage_exclusions": ["Invoice activity after snapshot date", "Outcome-period purchase events", "Post-outcome delivery or return information"],
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Score a current customer queue at the end of the observed history. No future outcome is used here.
    current_snapshot = sales["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    scored = feature_rows(sales, returns, current_snapshot, include_target=False)
    scored["propensity_score"] = model.predict_proba(scored[feature_columns])[:, 1]
    scored = scored.sort_values("propensity_score", ascending=False).reset_index(drop=True)
    high_cut = scored["propensity_score"].quantile(0.90)
    medium_cut = scored["propensity_score"].quantile(0.70)
    scored["priority_band"] = np.select(
        [scored["propensity_score"] >= high_cut, scored["propensity_score"] >= medium_cut],
        ["High", "Medium"], default="Low"
    )
    scored["recommended_action"] = np.select(
        [scored["priority_band"].eq("High"), scored["priority_band"].eq("Medium")],
        ["Prioritise for outreach", "Add to normal outreach queue"], default="Do not prioritise this cycle"
    )
    scored["score_reason"] = scored.apply(reason, axis=1)
    scored.insert(0, "score_date", current_snapshot.date())
    scored.to_csv(OUTPUT_DIR / "scored_customers.csv", index=False)
    scored.head(200).to_csv(OUTPUT_DIR / "sample_scoring_input.csv", index=False)

    joblib.dump({"pipeline": model, "feature_columns": feature_columns, "numeric_features": numeric_features, "categorical_features": categorical_features}, MODEL_DIR / "outreach_propensity_model.joblib")
    training.to_csv(OUTPUT_DIR / "training_snapshots.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

