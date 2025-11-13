import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Inventory NSM Dashboard", layout="centered")

st.title("📦 Inventory NSM & KPI Dashboard")

# -----------------------------------
# LOAD CSV (no openpyxl required)
# -----------------------------------
FILE_PATH = "data/992800103.csv"   # فقط CSV آپلود کن

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    df['Date Applied'] = pd.to_datetime(df['Date Applied']).dt.date
    return df

try:
    df = load_data(FILE_PATH)
except Exception as e:
    st.error(f"Could not load file {FILE_PATH}. Error: {e}")
    st.stop()

# -----------------------------------
# DAILY INVENTORY
# -----------------------------------
daily = (
    df.sort_values(['Date Applied'])
      .groupby('Date Applied', as_index=False)
      .tail(1)
)

daily_onhand = daily.set_index('Date Applied')['On Hand Qty After Transaction']
num_days = len(daily_onhand)
available_days = (daily_onhand > 0).sum()
stockout_days = (daily_onhand == 0).sum()

# -----------------------------------
# METRICS
# -----------------------------------
def fmt_rate(x): return "NA" if pd.isna(x) else f"{x*100:.2f}%"
def fmt_num(x): return "NA" if pd.isna(x) else f"{x:.3f}"

IAR = available_days / num_days if num_days else np.nan
stockout_rate = stockout_days / num_days if num_days else np.nan

issues = df[df['Transaction Code'] == 'INTSHIP']
avg_ltc_issue = issues['Lead Time Coverage'].mean() if len(issues) else np.nan

receipts = df[df['Transaction Qty'] > 0].copy()
num_receipts = len(receipts)

if num_receipts:
    receipts['Before'] = receipts['On Hand Qty After Transaction'] - receipts['Transaction Qty']
    reorder_adherence = (receipts['Before'] <= receipts['Order Point']).mean()
else:
    reorder_adherence = np.nan

if num_receipts:
    valid = receipts['Lot Size'] > 0
    if valid.any():
        lot_dev = (receipts.loc[valid, 'Transaction Qty'] - receipts.loc[valid, 'Lot Size']).abs()
        lot_ratio = lot_dev / receipts.loc[valid, 'Lot Size']
        lot_adherence = (lot_ratio <= 0.10).mean()
    else:
        lot_adherence = np.nan
else:
    lot_adherence = np.nan

avg_onhand = daily_onhand.mean()
avg_op = df['Order Point'].replace(0, np.nan).mean()
inventory_intensity = avg_onhand / avg_op if avg_op else np.nan

if num_receipts:
    t_min = df['Date Applied'].min()
    t_max = df['Date Applied'].max()
    years = (t_max - t_min).days / 365.25
    order_freq_year = num_receipts / years if years > 0 else np.nan
else:
    order_freq_year = np.nan

# -----------------------------------
# UI STYLE
# -----------------------------------
st.markdown("""
<style>
.section-title { font-size:20px; font-weight:700; margin-top:1rem; }
.metric-name { font-size:14px; font-weight:700; }
.metric-value { font-size:18px; font-weight:700; text-align:right; }
.metric-desc { font-size:12px; color:#666; }
.divider { border-top:1px solid #ccc; margin-top:0.5rem; margin-bottom:0.5rem; }
</style>
""", unsafe_allow_html=True)

def block(name, desc, val):
    c1, c2 = st.columns([3,1])
    c1.markdown(f"<div class='metric-name'>{name}</div>", unsafe_allow_html=True)
    c1.markdown(f"<div class='metric-desc'>{desc}</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-value'>{val}</div>", unsafe_allow_html=True)

# -----------------------------------
# LAYOUT
# -----------------------------------

st.markdown("<div class='section-title'>☆ NSM</div>", unsafe_allow_html=True)
block("Inventory Availability Rate (IAR)", "Days with on-hand > 0", fmt_rate(IAR))

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>⇩ Input / Behavior</div>", unsafe_allow_html=True)
block("Reorder Policy Adherence", "Before-receipt ≤ Order Point", fmt_rate(reorder_adherence))
block("Lot Size Adherence", "Within 10% of Lot Size", fmt_rate(lot_adherence))

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>⬆ Outcome</div>", unsafe_allow_html=True)
block("Stockout Days Rate", "Days with on-hand = 0", fmt_rate(stockout_rate))
block("Avg Lead Time Coverage", "Average LTC after issues", fmt_num(avg_ltc_issue))

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

st.markdown("<div class='section-title'>║ Guardrail</div>", unsafe_allow_html=True)
block("Inventory Intensity", "Avg On-hand vs Avg Order Point", fmt_num(inventory_intensity))
block("Order Frequency per Year", "Yearly receipt events", fmt_num(order_freq_year))

st.success("Dashboard loaded successfully.")
