# app.py
"""
Stock Price Prediction — stonksFree data: yfinance (Yahoo Finance)
Models: RandomForest (default) and XGBoost (optional)
Features: MA, RSI, returns, volume; train/test, next-day & multi-day recursive forecast,
interactive candlestick + predictions, download predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor — Standalone Project")
st.markdown(
    "Upload a CSV with historical OHLCV (date, open, high, low, close, volume) "
    "or download data from Yahoo Finance. Choose model, train, and forecast next-day / multi-day prices."
)

# -----------------------
# Sidebar: Data Source
# -----------------------
st.sidebar.header("1) Data Source")
data_source = st.sidebar.radio("Source", ["Yahoo Finance (download)", "Upload CSV (local)"])
df = None

# -------- Yahoo Finance download --------
if data_source == "Yahoo Finance (download)":
    ticker = st.sidebar.text_input("Ticker (e.g. AAPL, TSLA, MSFT)", value="AAPL")
    period = st.sidebar.selectbox("History period", ["1y", "2y", "5y", "10y", "max"], index=2)
    interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
    if st.sidebar.button("Download data"):
        with st.spinner("Downloading from Yahoo Finance..."):
            raw = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
            if raw.empty:
                st.error("No data returned. Check ticker or period.")
                st.stop()

            raw.reset_index(inplace=True)

            # Flatten MultiIndex if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = ['_'.join([str(i) for i in col if i]) for col in raw.columns]

            # Lowercase and strip
            raw.columns = [c.strip().lower() for c in raw.columns]

            # Fix ticker suffix (e.g., 'open_aapl' -> 'open')
            ticker_suffix = f"_{ticker.lower()}"
            fixed_cols = {c: c.replace(ticker_suffix, '') for c in raw.columns if c.endswith(ticker_suffix)}
            raw.rename(columns=fixed_cols, inplace=True)

            # Use 'adj close' if 'close' is missing
            if 'close' not in raw.columns and 'adj close' in raw.columns:
                raw.rename(columns={'adj close': 'close'}, inplace=True)

            df = raw.copy()

# -------- CSV/XLSX upload --------
else:
    uploaded = st.sidebar.file_uploader("Upload CSV/XLSX (must contain date, open, high, low, close, volume)", type=["csv", "xlsx"])
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded)
            else:
                df = pd.read_csv(uploaded)

            # Flatten MultiIndex if any
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([str(i) for i in col if i]) for col in df.columns]

            # Lowercase and strip
            df.columns = [c.strip().lower() for c in df.columns]

            # Use 'adj close' if 'close' is missing
            if 'close' not in df.columns and 'adj close' in df.columns:
                df.rename(columns={'adj close': 'close'}, inplace=True)

        except Exception as e:
            st.error(f"Failed to read file: {e}")
            st.stop()

# -----------------------
# If df is loaded, continue
# -----------------------
if df is not None:
    st.write("Columns detected:", df.columns.tolist())

    # Ensure a date column exists
    date_cols = [c for c in df.columns if "date" in c]
    if not date_cols:
        st.error("No date column found in dataset. Ensure there's a 'date' column.")
        st.stop()
    date_col = date_cols[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Check required OHLCV columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.warning(f"Missing columns in dataset: {set(missing)}")

    # Prevent training if 'close' is missing
    if 'close' not in df.columns:
        st.error("Cannot train model without 'close' column.")
        st.stop()

    # Show head of data
    display_cols = [date_col] + [c for c in required_cols if c in df.columns]
    st.subheader("Raw data (head)")
    st.dataframe(df[display_cols].head(10))

    # -----------------------
    # Feature Engineering
    # -----------------------
    st.sidebar.header("2) Feature engineering")
    add_indicators = st.sidebar.checkbox("Auto feature-engineer indicators (MA, RSI, returns)", value=True)
    if add_indicators:
        df = df.copy()
        df["close"] = df["close"].astype(float)
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["ma20"] = df["close"].rolling(20).mean()
        df["price_change"] = df["close"].pct_change() * 100
        # RSI (14)
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / (loss.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))
        df.fillna(method="bfill", inplace=True)

    # -----------------------
    # Model & Training options
    # -----------------------
    st.sidebar.header("3) Model & training")
    model_choice = st.sidebar.selectbox(
        "Choose model",
        options=["RandomForest (fast)", "XGBoost (if installed)"],
        index=0
    )
    if model_choice.startswith("XGBoost") and not XGBOOST_AVAILABLE:
        st.sidebar.warning("XGBoost not installed — RandomForest will be used instead.")
        model_choice = "RandomForest (fast)"

    test_size = st.sidebar.slider("Test set size (%)", 5, 40, 20)
    n_estimators = st.sidebar.number_input("n_estimators", min_value=50, max_value=1000, value=300, step=50)
    forecast_days = st.sidebar.slider("Forecast days (multi-day recursive)", 1, 30, 7)

    # -----------------------
    # Prepare training data
    # -----------------------
    base_features = [f for f in ["close", "volume"] if f in df.columns]
    engineered_features = [f for f in ["ma5","ma10","ma20","price_change","rsi"] if f in df.columns]
    features = base_features + engineered_features

    df_model = df.copy()
    df_model["target"] = df_model["close"].shift(-1)
    df_model.dropna(inplace=True)
    X = df_model[features]
    y = df_model["target"]

    st.write(f"Using features: {', '.join(features)}")
    st.dataframe(df_model[[date_col] + features + ["target"]].tail(5))

    # -----------------------
    # Train-test split
    # -----------------------
    split_index = int(len(X)*(1-test_size/100))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    st.write(f"Training on {len(X_train)} rows — Testing on {len(X_test)} rows")

    # -----------------------
    # Train model
    # -----------------------
    if model_choice.startswith("XGBoost") and XGBOOST_AVAILABLE:
        model = XGBRegressor(n_estimators=n_estimators, verbosity=0, objective="reg:squarederror",
                             n_jobs=-1, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, random_state=42)

    with st.spinner("Training model..."):
        model.fit(X_train, y_train)

    # -----------------------
    # Evaluate
    # -----------------------
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))  # version-proof
    r2 = r2_score(y_test, y_pred_test)

    st.subheader("Model performance (on test set)")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.4f}")
    col2.metric("RMSE", f"{rmse:.4f}")
    col3.metric("R²", f"{r2:.4f}")

    # Feature importance
    try:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": features, "importance": importances}).sort_values("importance", ascending=False)
        st.dataframe(imp_df)
        fig_imp = px.bar(imp_df, x="feature", y="importance", title="Feature importances")
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Model does not provide feature importances.")

else:
    st.write("Waiting for data. Use the sidebar to download from Yahoo Finance or upload a CSV.")
