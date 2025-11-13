import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px

# ------------------------- Config -------------------------
st.set_page_config(page_title="SmartProcure Dashboard", layout="wide")
DATA_ENRICHED = "data/992800103_enriched.csv"   # تغییرپذیر
DATA_DAILY    = "data/daily.csv"                # اختیاری
DATA_MONTHLY  = "data/monthly.csv"              # اختیاری

# ------------------------- Helpers -------------------------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path)

def ensure_datetime(df):
    if "Date Applied" in df.columns and not np.issubdtype(df["Date Applied"].dtype, np.datetime64):
        df["Date Applied"] = pd.to_datetime(df["Date Applied"])
    return df

def build_daily_monthly(df):
    """محاسبه یکنواخت daily/monthly درصورت نبود CSVهای آماده."""
    df = ensure_datetime(df).sort_values("Date Applied").reset_index(drop=True)
    df["Inflow"]  = df["Transaction Qty"].apply(lambda x: x if x > 0 else 0.0)
    df["Outflow"] = df["Transaction Qty"].apply(lambda x: -x if x < 0 else 0.0)

    daily = (df.groupby(["Date Applied","Jalali_Date"], as_index=False)
               .agg({"Inflow":"sum","Outflow":"sum",
                     "On Hand Qty After Transaction":"last"})
               .rename(columns={"On Hand Qty After Transaction":"OnHand"}))
    # ضمیمهٔ جلالی
    if {"J_Year","J_Month"}.issubset(df.columns):
        jl = df.groupby("Date Applied", as_index=False)[["J_Year","J_Month"]].last()
        daily = daily.merge(jl, on="Date Applied", how="left")
    # ماهانه
    monthly = daily.groupby(["J_Year","J_Month"], as_index=False).agg({
        "Inflow":"sum","Outflow":"sum"
    })
    monthly["J_Month_Label"] = monthly["J_Year"].astype(str) + "-" + monthly["J_Month"].astype(str).str.zfill(2)
    return daily, monthly

def add_coverage_and_eri(daily, enriched_df, roll_window=30):
    eps = 1e-9
    daily = daily.sort_values("Date Applied").reset_index(drop=True)
    daily["AvgUsage_30d"] = daily["Outflow"].rolling(roll_window, min_periods=7).mean()
    daily["DaysOfCoverage"] = daily["OnHand"] / (daily["AvgUsage_30d"] + eps)

    # ERI اگر در enriched باشد
    if "Expiry_Risk_Index" in enriched_df.columns:
        eri_map = (enriched_df.groupby("Date Applied")["Expiry_Risk_Index"]
                   .last().reindex(daily["Date Applied"].values).values)
        daily["Expiry_Risk_Index"] = eri_map
    else:
        daily["Expiry_Risk_Index"] = np.nan
    return daily

def compute_outliers(daily):
    dz = daily["Outflow"].fillna(0.0)
    z = (dz - dz.mean()) / (dz.std(ddof=0) or 1.0)
    daily["Outlier_Z"] = z.abs() > 3
    Q1, Q3 = dz.quantile(0.25), dz.quantile(0.75)
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    daily["Outlier_IQR"] = (dz < lb) | (dz > ub)
    daily["Outlier"] = daily["Outlier_Z"] | daily["Outlier_IQR"]
    return daily, (lb, ub)

def compute_anomalies(daily, z_thresh=3.0, contam=0.05):
    # Rule-based
    mu, sigma = daily["Outflow"].mean(), daily["Outflow"].std(ddof=0) or 1.0
    daily["Usage_Z"] = (daily["Outflow"] - mu) / sigma
    rule_extreme = daily["Usage_Z"].abs() > z_thresh
    rule_low_cov_no_in = (daily["DaysOfCoverage"] < 6) & (daily["Inflow"] <= 0)  # هشداری/قابل‌تنظیم
    rule_high_eri_low_use = (daily["Expiry_Risk_Index"] > 0.5) & (daily["Outflow"] < daily["AvgUsage_30d"] * 0.3)
    daily["Anomaly_Rule"] = rule_extreme | rule_low_cov_no_in | rule_high_eri_low_use

    # IForest
    feats = daily[["Outflow","Inflow","OnHand","DaysOfCoverage","Expiry_Risk_Index"]].fillna(0.0)
    model = IsolationForest(n_estimators=300, contamination=contam, random_state=42)
    daily["Anomaly_IForest"] = (model.fit_predict(feats) == -1)
    daily["Anomaly_Combined"] = daily["Anomaly_Rule"] | daily["Anomaly_IForest"]
    return daily

# ------------------------- Sidebar -------------------------
st.sidebar.title("SmartProcure Controls")
data_path = st.sidebar.text_input("Enriched CSV path:", DATA_ENRICHED)
daily_path = st.sidebar.text_input("Daily CSV (optional):", DATA_DAILY)
monthly_path = st.sidebar.text_input("Monthly CSV (optional):", DATA_MONTHLY)

LEAD_TIME_DAYS = st.sidebar.number_input("Lead Time (days)", min_value=1, value=45, step=1)
ERI_WARN = st.sidebar.slider("Expiry Risk Index warning", 0.0, 1.0, 0.5, 0.05)
Z_THRESH = st.sidebar.slider("Z-score threshold (usage)", 2.0, 4.0, 3.0, 0.1)
IFOREST_CONTAM = st.sidebar.slider("IsolationForest contamination", 0.01, 0.15, 0.05, 0.01)
TOPK_LABELS = st.sidebar.slider("Max anomaly labels", 3, 30, 12, 1)

# ------------------------- Data Load -------------------------
enriched = load_csv(data_path)
enriched = ensure_datetime(enriched)

if daily_path and monthly_path:
    try:
        daily = load_csv(daily_path)
        monthly = load_csv(monthly_path)
        daily = ensure_datetime(daily)
    except Exception:
        daily, monthly = build_daily_monthly(enriched)
else:
    daily, monthly = build_daily_monthly(enriched)

daily = add_coverage_and_eri(daily, enriched)
daily, (lb_iqr, ub_iqr) = compute_outliers(daily)
daily = compute_anomalies(daily, z_thresh=Z_THRESH, contam=IFOREST_CONTAM)

# فیلتر بازهٔ تاریخ
min_d, max_d = daily["Date Applied"].min(), daily["Date Applied"].max()
st.sidebar.write(f"Date range: {min_d.date()} → {max_d.date()}")
date_from, date_to = st.sidebar.date_input("Filter range", [min_d.date(), max_d.date()])
mask = (daily["Date Applied"].dt.date >= date_from) & (daily["Date Applied"].dt.date <= date_to)
daily_f = daily.loc[mask].copy()

# ------------------------- Header -------------------------
st.title("SmartProcure – Procurement & Inventory Analytics (Jalali)")

# ==========================================================
# 1) Overview
# ==========================================================
st.header("1) Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Inflow", f"{daily_f['Inflow'].sum():,.0f}")
with col2:
    st.metric("Total Outflow", f"{daily_f['Outflow'].sum():,.0f}")
with col3:
    today_cov = daily_f["DaysOfCoverage"].iloc[-1] if len(daily_f) else np.nan
    st.metric("Days of Coverage (latest)", f"{today_cov:,.1f}" if pd.notna(today_cov) else "—")
with col4:
    eri_now = daily_f["Expiry_Risk_Index"].iloc[-1] if "Expiry_Risk_Index" in daily_f.columns and len(daily_f) else np.nan
    st.metric("Expiry Risk Index (latest)", f"{eri_now:.2f}" if pd.notna(eri_now) else "—")

# ==========================================================
# 2) Consumption & Replenishment (Daily/Monthly)
# ==========================================================
st.header("2) Consumption & Replenishment")

# Daily line
fig = go.Figure()
fig.add_trace(go.Scatter(x=daily_f["Date Applied"], y=daily_f["Outflow"],
                         mode="lines", name="Daily Outflow (Usage)"))
fig.add_trace(go.Scatter(x=daily_f["Date Applied"], y=daily_f["Inflow"],
                         mode="lines", name="Daily Inflow (Receipts)"))
fig.update_layout(height=380, xaxis_title="Date", yaxis_title="Quantity",
                  legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(fig, use_container_width=True)

# Monthly grouped bars
monthly["J_Month_Label"] = monthly["J_Month_Label"].astype(str)
figm = go.Figure()
figm.add_trace(go.Bar(x=monthly["J_Month_Label"], y=monthly["Outflow"], name="Outflow (Usage)"))
figm.add_trace(go.Bar(x=monthly["J_Month_Label"], y=monthly["Inflow"],  name="Inflow (Receipts)"))
figm.update_layout(barmode="group", height=380, xaxis_title="Jalali Month (YYYY-MM)", yaxis_title="Total Quantity",
                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
st.plotly_chart(figm, use_container_width=True)

# Net Consumption line
monthly_nc = monthly.copy()
monthly_nc["Net_Consumption"] = monthly_nc["Outflow"] - monthly_nc["Inflow"]
fign = go.Figure()
fign.add_trace(go.Scatter(x=monthly_nc["J_Month_Label"], y=monthly_nc["Net_Consumption"],
                          mode="lines+markers", name="Net Consumption"))
fign.update_layout(height=300, xaxis_title="Jalali Month", yaxis_title="Outflow - Inflow")
st.plotly_chart(fign, use_container_width=True)

# ==========================================================
# 3) Outliers
# ==========================================================
st.header("3) Outliers (Daily Usage)")
c1, c2 = st.columns(2)

with c1:
    # Boxplot (plotly)
    figb = go.Figure()
    figb.add_trace(go.Box(y=daily_f["Outflow"], name="Daily Outflow", boxpoints="outliers"))
    figb.update_layout(height=360, yaxis_title="Outflow Quantity")
    st.plotly_chart(figb, use_container_width=True)

with c2:
    # Histogram with IQR lines
    figh = go.Figure()
    figh.add_trace(go.Histogram(x=daily_f["Outflow"], nbinsx=30, name="Outflow"))
    figh.add_vline(x=lb_iqr, line_dash="dash", annotation_text="IQR Lower", line_color="red")
    figh.add_vline(x=ub_iqr, line_dash="dash", annotation_text="IQR Upper", line_color="red")
    figh.update_layout(height=360, xaxis_title="Outflow Quantity", yaxis_title="Frequency")
    st.plotly_chart(figh, use_container_width=True)

# Timeline with labels on outliers only
out_idx = daily_f.index[daily_f["Outlier"]].tolist()
topk = out_idx[:TOPK_LABELS]
figo = go.Figure()
figo.add_trace(go.Scatter(x=daily_f["Date Applied"], y=daily_f["Outflow"], mode="lines", name="Daily Usage"))
figo.add_trace(go.Scatter(x=daily_f.loc[out_idx, "Date Applied"], y=daily_f.loc[out_idx, "Outflow"],
                          mode="markers+text", name="Outliers",
                          text=daily_f.loc[out_idx, "Jalali_Date"],
                          textposition="top center",
                          textfont=dict(size=9)))
figo.update_layout(height=380, xaxis_title="Date", yaxis_title="Outflow Quantity")
st.plotly_chart(figo, use_container_width=True)

# ==========================================================
# 4) Lead Time Coverage
# ==========================================================
st.header("4) Lead Time Coverage")

figc = go.Figure()
figc.add_trace(go.Scatter(x=daily_f["Date Applied"], y=daily_f["DaysOfCoverage"],
                          mode="lines", name="Days of Coverage"))
figc.add_hline(y=LEAD_TIME_DAYS, line_dash="dash", annotation_text=f"Lead Time = {LEAD_TIME_DAYS}d")
# پرچم روزهای زیر LT
risk_mask = daily_f["DaysOfCoverage"] < LEAD_TIME_DAYS
figc.add_trace(go.Scatter(x=daily_f.loc[risk_mask, "Date Applied"],
                          y=daily_f.loc[risk_mask, "DaysOfCoverage"],
                          mode="markers", name="Below LT"))
figc.update_layout(height=360, xaxis_title="Date", yaxis_title="Days")
st.plotly_chart(figc, use_container_width=True)

# ==========================================================
# 5) Anomalies (Rules + Isolation Forest)
# ==========================================================
st.header("5) Anomalies (Rules + Isolation Forest)")

figa = go.Figure()
figa.add_trace(go.Scatter(x=daily_f["Date Applied"], y=daily_f["Outflow"], mode="lines", name="Daily Usage"))
anom_mask = daily_f["Anomaly_Combined"]
figa.add_trace(go.Scatter(x=daily_f.loc[anom_mask, "Date Applied"],
                          y=daily_f.loc[anom_mask, "Outflow"],
                          mode="markers+text",
                          name="Anomalies",
                          text=daily_f.loc[anom_mask, "Jalali_Date"],
                          textposition="top center",
                          textfont=dict(size=9)))
figa.update_layout(height=380, xaxis_title="Date", yaxis_title="Outflow Quantity")
st.plotly_chart(figa, use_container_width=True)

# Monthly anomaly counts
if {"J_Year","J_Month"}.issubset(daily_f.columns):
    m_anom = daily_f.groupby(["J_Year","J_Month"], as_index=False)["Anomaly_Combined"].sum()
    m_anom["J_Month_Label"] = m_anom["J_Year"].astype(str) + "-" + m_anom["J_Month"].astype(str).str.zfill(2)
    figmc = go.Figure()
    figmc.add_trace(go.Bar(x=m_anom["J_Month_Label"], y=m_anom["Anomaly_Combined"], name="Anomaly Count"))
    figmc.update_layout(height=320, xaxis_title="Jalali Month", yaxis_title="Count")
    st.plotly_chart(figmc, use_container_width=True)

# ==========================================================
# 6) Decisions (R & Q) – Placeholder for next phase
# ==========================================================
st.header("6) Decisions (Next Phase)")
st.info(
    "این بخش در فاز بعدی تکمیل می‌شود: محاسبه R و Q بر اساس LT، سطح خدمت هدف، نوسان تقاضا، و سناریوهای What-If. "
    "خروجی این بخش جدول سفارش پیشنهادی و اثر آن بر Coverage/ERI خواهد بود."
)

# Tables (download)
st.subheader("Data Snapshots")
st.download_button("Download filtered daily (CSV)", daily_f.to_csv(index=False).encode("utf-8"),
                   file_name="daily_filtered.csv")
st.download_button("Download monthly (CSV)", monthly.to_csv(index=False).encode("utf-8"),
                   file_name="monthly.csv")
