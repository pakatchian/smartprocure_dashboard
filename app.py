import streamlit as st
import pandas as pd
import numpy as np
import math

# ---------------------------
# 1. Page config
# ---------------------------
st.set_page_config(
    page_title="Inventory NSM & KPI Dashboard",
    page_icon="📦",
    layout="centered"
)

st.title("📦 Inventory NSM & KPI Dashboard")
st.caption("Dataset: 992800103.xlsx – NSM & KPIs computed directly from transaction log.")

# ---------------------------
# 2. Load data
# ---------------------------
FILE_PATH = "/content/992800103.xlsx"   # اگر اسم/مسیر فرق دارد این را عوض کن

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df['Date Applied'] = pd.to_datetime(df['Date Applied']).dt.date
    return df

try:
    df = load_data(FILE_PATH)
except Exception as e:
    st.error(f"Could not load file `{FILE_PATH}`. Error: {e}")
    st.stop()

# ---------------------------
# 3. Helper: end-of-day inventory per date
# ---------------------------
daily = (
    df.sort_values(['Date Applied'])
      .groupby('Date Applied', as_index=False)
      .tail(1)
)

daily_onhand = daily.set_index('Date Applied')['On Hand Qty After Transaction']
dates = daily_onhand.index
num_days = len(daily_onhand)

available_days = (daily_onhand > 0).sum()
stockout_days = (daily_onhand == 0).sum()

# ---------------------------
# 4. Compute NSM & KPIs
# ---------------------------

def fmt_rate(x, decimals=2):
    if pd.isna(x):
        return "NA"
    return f"{x*100:.{decimals}f}%"

def fmt_number(x, decimals=3):
    if pd.isna(x):
        return "NA"
    return f"{x:.{decimals}f}"

# --- NSM: Inventory Availability Rate (IAR) ---
IAR = available_days / num_days if num_days > 0 else np.nan

# --- KPI-O1: Stockout Days Rate ---
stockout_rate = stockout_days / num_days if num_days > 0 else np.nan

# --- KPI-O2: Average Lead Time Coverage at Issue ---
issues = df[df['Transaction Code'] == 'INTSHIP']
avg_ltc_issue = issues['Lead Time Coverage'].mean() if len(issues) > 0 else np.nan

# --- Receipts: q_i > 0 ---
receipts = df[df['Transaction Qty'] > 0].copy()
num_receipts = len(receipts)

# --- KPI-I1: Reorder Policy Adherence ---
if num_receipts > 0:
    receipts['OnHand_Before'] = (
        receipts['On Hand Qty After Transaction'] - receipts['Transaction Qty']
    )
    adherence_mask = receipts['OnHand_Before'] <= receipts['Order Point']
    reorder_adherence = adherence_mask.mean()
else:
    reorder_adherence = np.nan

# --- KPI-I2: Lot Size Adherence ---
if num_receipts > 0:
    valid_lot = receipts['Lot Size'] > 0
    if valid_lot.any():
        lot_dev = (
            receipts.loc[valid_lot, 'Transaction Qty']
            - receipts.loc[valid_lot, 'Lot Size']
        ).abs()
        lot_ratio = lot_dev / receipts.loc[valid_lot, 'Lot Size']
        lot_adherence = (lot_ratio <= 0.10).mean()
    else:
        lot_adherence = np.nan
else:
    lot_adherence = np.nan

# --- KPI-G1: Inventory Intensity (Avg On-hand vs Order Point) ---
avg_onhand = daily_onhand.mean() if num_days > 0 else np.nan
avg_order_point = df['Order Point'].replace(0, np.nan).mean()
inventory_intensity = (
    avg_onhand / avg_order_point
    if (avg_order_point is not None and not pd.isna(avg_order_point) and avg_order_point != 0)
    else np.nan
)

# --- KPI-G2: Order Frequency per Year ---
if num_receipts > 0:
    t_min = df['Date Applied'].min()
    t_max = df['Date Applied'].max()
    period_days = (t_max - t_min).days
    period_days = period_days if period_days > 0 else 1  # avoid zero division
    years = period_days / 365.25
    order_freq_per_year = num_receipts / years
else:
    order_freq_per_year = np.nan

# ---------------------------
# 5. UI helpers
# ---------------------------

st.markdown(
    """
    <style>
    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.3rem;
    }
    .metric-name {
        font-weight: 700;
        font-size: 14px;
    }
    .metric-desc {
        font-size: 12px;
        color: #555555;
    }
    .metric-value {
        font-weight: 700;
        font-size: 18px;
        text-align: right;
    }
    .divider {
        border-top: 1px solid #dddddd;
        margin-top: 0.8rem;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def metric_row(name, desc, value_str):
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"<div class='metric-name'>{name}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-desc'>{desc}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-value'>{value_str}</div>", unsafe_allow_html=True)

# ---------------------------
# 6. Layout like your sketch
# ---------------------------

# NSM section
st.markdown("<div class='section-title'>☆ NSM</div>", unsafe_allow_html=True)
metric_row(
    "Inventory Availability Rate (IAR)",
    "Share of days with end-of-day on-hand inventory > 0.",
    fmt_rate(IAR)
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Input / Behavior section
st.markdown("<div class='section-title'>⇩ Input / Behavior</div>", unsafe_allow_html=True)

metric_row(
    "Reorder Policy Adherence",
    "Receipts placed when pre-receipt on-hand was at or below Order Point.",
    fmt_rate(reorder_adherence)
)

metric_row(
    "Lot Size Adherence",
    "Receipts whose quantity is within 10% of the defined Lot Size.",
    fmt_rate(lot_adherence)
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Outcome section
st.markdown("<div class='section-title'>⬆ Outcome</div>", unsafe_allow_html=True)

metric_row(
    "Stockout Days Rate",
    "Share of days where end-of-day on-hand inventory equals zero.",
    fmt_rate(stockout_rate)
)

metric_row(
    "Avg Lead Time Coverage",
    "Average Lead Time Coverage after issue transactions (INTSHIP).",
    fmt_number(avg_ltc_issue)
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Guardrail section
st.markdown("<div class='section-title'>║ Guardrail</div>", unsafe_allow_html=True)

metric_row(
    "Inventory Intensity",
    "Average end-of-day on-hand divided by average Order Point.",
    fmt_number(inventory_intensity)
)

metric_row(
    "Order Frequency per Year",
    "Number of receipt events per year over the observed period.",
    fmt_number(order_freq_per_year)
)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.success("All NSM & KPI values are computed from the raw dataset on every run.")
