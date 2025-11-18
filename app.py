import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Inventory NSM & KPIs", layout="wide")

DATA_PATH = "data/992800103_withPartNumber.csv"  # مسیر را مطابق پروژه خودت تنظیم کن

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # محاسبه ستون‌های کمکی
    df["IsIssue"] = df["Transaction Qty"] < 0
    df["IssueVolume"] = np.where(df["IsIssue"], -df["Transaction Qty"], 0.0)
    return df

df = load_data(DATA_PATH)

# ---------------------------
# 1) فیلتر بازه زمانی (اختیاری)
# ---------------------------
years = sorted(df["Date Applied (Year)"].dropna().unique())
selected_years = st.sidebar.multiselect(
    "Select Shamsi Years",
    options=years,
    default=years
)

df_f = df[df["Date Applied (Year)"].isin(selected_years)].copy()

# اگر دیتای فیلتر شده خالی شد
if df_f.empty:
    st.warning("No data for selected years.")
    st.stop()

# ---------------------------
# 2) تعریف Stockout و محاسبه متریک‌ها
# ---------------------------
# فرض: Stockout یعنی تراکنشی که Issue بوده و موجودی بعد از تراکنش صفر یا منفی شده
df_f["IsStockoutEvent"] = (df_f["IsIssue"]) & (df_f["On Hand Qty After Transaction"] <= 0)

total_issue_txn = df_f["IsIssue"].sum()
stockout_events = df_f["IsStockoutEvent"].sum()

# جلوگیری از تقسیم بر صفر
if total_issue_txn > 0:
    stockout_rate = stockout_events / total_issue_txn
    sla = 1 - stockout_rate
else:
    stockout_rate = 0.0
    sla = 0.0

# Inventory Turnover ≈ مجموع Issue / متوسط موجودی
total_issue_volume = df_f.loc[df_f["IsIssue"], "IssueVolume"].sum()
avg_inventory = df_f["On Hand Qty After Transaction"].mean()

if avg_inventory and avg_inventory > 0:
    inventory_turnover = total_issue_volume / avg_inventory
else:
    inventory_turnover = 0.0

# ---------------------------
# 3) نمایش 3 کارت KPI
# ---------------------------
st.title("Inventory NSM & Key KPIs")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        label="NSM – Service Level Availability",
        value=f"{sla:.1%}",
        help="Percentage of issue transactions fulfilled without stockout."
    )

with c2:
    st.metric(
        label="Stockout Occurrence Rate",
        value=f"{stockout_rate:.2%}",
        help="Share of issue transactions that ended with stockout (On-hand ≤ 0)."
    )

with c3:
    st.metric(
        label="Inventory Turnover (Issue-based)",
        value=f"{inventory_turnover:.2f}",
        help="Total issue volume divided by average on-hand inventory."
    )

st.caption(
    "Definitions can be refined (e.g., stockout per day or per MR), "
    "but this version is consistent with typical inventory analytics practice."
)


# ==========================================================
# 6) Decisions (R & Q) – Placeholder for next phase
# ==========================================================
st.header("6) Decisions (Next Phase)")

# Try to run optimizer and display results
try:
    from optimizer import (
        Config, DataLoader, ChangePointDetector, FeatureEngineer,
        InventorySimulator, PolicyOptimizer, LeadTimeEstimator
    )
    import matplotlib.pyplot as plt
    
    config = Config()
    shelf_life_days = 90
    
    # Load and process data for optimization
    daily_opt, dates_opt = DataLoader.load_and_prepare(config.TRANSACTIONS_FILE)
    
    # Change point detection
    detector = ChangePointDetector(
        penalty=config.CHANGE_POINT_PENALTY,
        min_size=config.CHANGE_POINT_MIN_SIZE
    )
    detector.detect(daily_opt["usage_qty"])
    periods = detector.classify_periods(
        daily_opt["usage_qty"],
        dates_opt,
        config.HIGH_DEMAND_THRESHOLD,
        config.LOW_DEMAND_THRESHOLD
    )
    
    # Feature engineering
    daily_opt = FeatureEngineer.create_base_features(daily_opt, dates_opt)
    overall_mean = daily_opt["usage_qty"].mean()
    daily_opt = FeatureEngineer.create_period_features(daily_opt, periods, overall_mean)
    daily_opt = FeatureEngineer.create_proximity_features(daily_opt, periods, config)
    
    feature_columns = [
        "dayofweek", "month", "lag1", "rolling_mean_7", "rolling_mean_30",
        "period_type_high", "period_type_low", "period_mean_demand",
        "days_to_high_demand", "days_from_high_demand",
        "near_high_demand", "near_low_demand"
    ]
    X = daily_opt[feature_columns]
    y = daily_opt["inventory"]
    
    # Train model
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    
    model = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
    model.fit(X, y)
    pred_inventory = model.predict(X)
    pred_usage = np.maximum(0, np.concatenate(([0], -np.diff(pred_inventory))))
    
    # Lead time function
    lead_time_func = lambda current_date, idx: 2
    try:
        lead_est = LeadTimeEstimator(config)
        lead_est.train(config.LEAD_TIMES_FILE)
        lead_time_func = lambda current_date, idx: lead_est.predict(current_date)
    except:
        pass
    
    # Optimization
    simulator = InventorySimulator(config, shelf_life_days)
    optimizer = PolicyOptimizer(simulator, config)
    results_df = optimizer.grid_search(
        pred_usage,
        daily_opt["inventory"].iloc[0],
        dates_opt,
        lead_time_func=lead_time_func
    )
    best = results_df.iloc[0]
    
    # Simulate with best policy
    _, _, _, simulated_inv = simulator.simulate(
        pred_usage,
        daily_opt["inventory"].iloc[0],
        dates_opt,
        best.R,
        best.Q,
        lead_time_func=lead_time_func,
        track_inventory=True
    )
    
    # Create matplotlib figure and convert to Streamlit
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(dates_opt, daily_opt["inventory"].values, label="Actual Inventory", alpha=0.7, linewidth=2)
    ax.plot(dates_opt, simulated_inv, label="Simulated Optimal Policy", alpha=0.7, linewidth=2)
    ax.axhline(best.R, color="r", linestyle="--", alpha=0.5, label=f"Reorder Point R={best.R:,.0f}")
    
    for cp_idx in detector.change_points[:-1]:
        ax.axvline(dates_opt[cp_idx], color="orange", linestyle=":", alpha=0.5, linewidth=1)
    
    for period in periods:
        if period["period_type"] == "HIGH_DEMAND":
            ax.axvspan(period["start_date"], period["end_date"], alpha=0.15, color="red")
        elif period["period_type"] == "LOW_DEMAND":
            ax.axvspan(period["start_date"], period["end_date"], alpha=0.15, color="blue")
    
    ax.set_title("Optimal Policy vs Historical Inventory (with Change Point Detection)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Inventory Level")
    ax.set_xlabel("Date")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(fig)
    
    st.success(f"✓ Optimal Policy Found: **R={best.R:,.0f}, Q={best.Q:,.0f}**")
    st.metric("Waste", f"{best.waste:,.1f}")
    st.metric("Stockouts", f"{best.stockouts:,.1f}")
    st.metric("Service Level", f"{best.service_level:.1%}")
    
except Exception as e:
    st.info(
        "این بخش در فاز بعدی تکمیل می‌شود: محاسبه R و Q بر اساس LT، سطح خدمت هدف، نوسان تقاضا، و سناریوهای What-If. "
        "خروجی این بخش جدول سفارش پیشنهادی و اثر آن بر Coverage/ERI خواهد بود."
    )
    st.warning(f"Optimizer not available: {str(e)}")

# Tables (download)
st.subheader("Data Snapshots")
st.download_button("Download filtered daily (CSV)", daily_f.to_csv(index=False).encode("utf-8"),
                   file_name="daily_filtered.csv")
st.download_button("Download monthly (CSV)", monthly.to_csv(index=False).encode("utf-8"),
                   file_name="monthly.csv")
