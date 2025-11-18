import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------- Config --------------------
st.set_page_config(
    page_title="Inventory NSM & Time-based Analysis",
    layout="wide"
)

st.title("Inventory NSM & Time-based Analysis (Shamsi Calendar)")
st.caption("Data source: 992800103_withPartNumber.csv")

DATA_PATH_DEFAULT = "data/992800103_withPartNumber.csv"  # مسیر را مطابق پروژه خودت تنظیم کن


# -------------------- Load Data --------------------
@st.cache_data
def load_data(path_or_file):
    return pd.read_csv(path_or_file)


st.sidebar.header("Data Settings")

data_path = st.sidebar.text_input(
    "CSV file path",
    value=DATA_PATH_DEFAULT,
    help="Enter the path to 992800103_withPartNumber.csv"
)

uploaded_file = st.sidebar.file_uploader(
    "…or upload CSV file",
    type=["csv"],
    help="If you upload a file, it will override the path above."
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    df = load_data(data_path)

# -------------------- Required Columns Check --------------------
required_cols = [
    "Transaction Qty",
    "On Hand Qty After Transaction",
    "Date Applied (Year)",
    "Date Applied (Month)",
    "Date Applied (Day)",
    "Date Applied (Shamsi)",
    "Date Applied (Miladi)",
    "Quarter Number",
]
missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"Missing required columns in data: {missing}")
    st.stop()

# -------------------- Year Filter --------------------
years = sorted(df["Date Applied (Year)"].dropna().unique())
default_years = [y for y in years if 1402 <= y <= 1404] or years

selected_years = st.sidebar.multiselect(
    "Select Shamsi year(s)",
    options=years,
    default=default_years,
)

if not selected_years:
    st.warning("Please select at least one year from the sidebar.")
    st.stop()

df_f = df[df["Date Applied (Year)"].isin(selected_years)].copy()

if df_f.empty:
    st.warning("No data for selected years.")
    st.stop()

# -------------------- Common Transformations --------------------
# FlowType & FlowQty for Issue / Receive
df_f["FlowType"] = np.where(df_f["Transaction Qty"] < 0, "Issue", "Receive")
df_f["FlowQty"] = np.where(
    df_f["FlowType"] == "Issue",
    -df_f["Transaction Qty"],
    df_f["Transaction Qty"]
)

# Issue flags & IssueVolume
df_f["IsIssue"] = df_f["Transaction Qty"] < 0
df_f["IssueVolume"] = np.where(df_f["IsIssue"], -df_f["Transaction Qty"], 0.0)

# Stockout definition: Issue + موجودی بعد از تراکنش ≤ 0
df_f["IsStockoutEvent"] = (df_f["IsIssue"]) & (df_f["On Hand Qty After Transaction"] <= 0)

# Shamsi months in English (fixed order)
shamsi_months_en = [
    "Farvardin", "Ordibehesht", "Khordad",
    "Tir", "Mordad", "Shahrivar",
    "Mehr", "Aban", "Azar",
    "Dey", "Bahman", "Esfand"
]
month_map = {i + 1: name for i, name in enumerate(shamsi_months_en)}

df_f["MonthEN"] = df_f["Date Applied (Month)"].map(month_map)

# -------------------- KPI Computation --------------------
total_issue_txn = df_f["IsIssue"].sum()
stockout_events = df_f["IsStockoutEvent"].sum()

if total_issue_txn > 0:
    stockout_rate = stockout_events / total_issue_txn
    sla = 1 - stockout_rate
else:
    stockout_rate = 0.0
    sla = 0.0

total_issue_volume = df_f.loc[df_f["IsIssue"], "IssueVolume"].sum()
avg_inventory = df_f["On Hand Qty After Transaction"].mean()

if avg_inventory and avg_inventory > 0:
    inventory_turnover = total_issue_volume / avg_inventory
else:
    inventory_turnover = 0.0

# -------------------- KPI Cards --------------------
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
        value=f"{inventory_turnover:.2f}×",
        help="Total issue volume divided by average on-hand inventory."
    )

st.markdown("---")

# -------------------- Tabs --------------------
tab1, tab2 = st.tabs(["📊 Quarterly & Monthly Seasonality", "⚠️ Daily Outliers (IQR)"])

# ==================== TAB 1: Quarterly & Monthly Seasonality ====================
with tab1:
    st.subheader("Quarterly Seasonality of Issue & Receive")

    quarter_agg = (
        df_f
        .groupby(["Date Applied (Year)", "Quarter Number", "FlowType"], as_index=False)
        .agg({"FlowQty": "sum"})
        .rename(columns={"Date Applied (Year)": "Year"})
    )

    if quarter_agg.empty:
        st.info("No quarterly data available for the selected years.")
    else:
        fig_q = px.bar(
            quarter_agg,
            x="Quarter Number",
            y="FlowQty",
            color="FlowType",
            facet_col="Year",
            facet_col_wrap=3,
            barmode="group",
            title="Seasonality of Issue & Receive by Quarter",
            labels={
                "Quarter Number": "Quarter (Q1–Q4)",
                "FlowQty": "Total Quantity",
                "FlowType": "Flow Type",
                "Year": "Year",
            }
        )

        fig_q.update_layout(
            template="plotly_white",
            title_x=0.5,
            legend_title_text="Flow Type",
            font=dict(size=12),
            height=500
        )
        fig_q.update_yaxes(showgrid=True)

        st.plotly_chart(fig_q, use_container_width=True)

        with st.expander("Show aggregated quarterly data"):
            st.dataframe(quarter_agg, use_container_width=True)

    st.subheader("Monthly Issue Trend (Shamsi Months, English Labels)")

    # ---------------- Monthly aggregation ----------------
    monthly_issue = (
        df_f[df_f["IsIssue"]]
        .groupby(["Date Applied (Year)", "Date Applied (Month)"], as_index=False)
        .agg(MonthlyIssue=("IssueVolume", "sum"))
    )

    if monthly_issue.empty:
        st.info("No monthly issue data available.")
    else:
        monthly_issue["MonthEN"] = monthly_issue["Date Applied (Month)"].map(month_map)

        # Build label: e.g. "Tir 1403"
        monthly_issue["MonthYearLabel"] = (
            monthly_issue["MonthEN"] + " " +
            monthly_issue["Date Applied (Year)"].astype(int).astype(str)
        )

        # Sort by Year then Month index
        monthly_issue = monthly_issue.sort_values(
            ["Date Applied (Year)", "Date Applied (Month)"]
        ).reset_index(drop=True)

        # Make label categorical in correct order to avoid any date parsing
        ordered_labels = monthly_issue["MonthYearLabel"].tolist()
        monthly_issue["MonthYearLabel"] = pd.Categorical(
            monthly_issue["MonthYearLabel"],
            categories=ordered_labels,
            ordered=True
        )

        fig_m = px.line(
            monthly_issue,
            x="MonthYearLabel",
            y="MonthlyIssue",
            markers=True,
            color="Date Applied (Year)",
            title="Monthly Issue Volume (Shamsi Month-Year Labels)",
            labels={
                "MonthYearLabel": "Month-Year (Shamsi)",
                "MonthlyIssue": "Total Issue Volume",
                "Date Applied (Year)": "Year"
            }
        )

        fig_m.update_layout(
            template="plotly_white",
            title_x=0.5,
            font=dict(size=11),
            xaxis_tickangle=-45,
            height=450
        )

        st.plotly_chart(fig_m, use_container_width=True)

        with st.expander("Show monthly issue data"):
            st.dataframe(monthly_issue, use_container_width=True)

# ==================== TAB 2: Daily Outliers (IQR) ====================
with tab2:
    st.subheader("Daily Issue Outliers (IQR-based, Shamsi Calendar)")

    # DateMiladi only for internal sorting
    df_f["DateMiladi"] = pd.to_datetime(df_f["Date Applied (Miladi)"])

    # Build daily issue grouped by Shamsi date + components
    daily_issue = (
        df_f[df_f["IsIssue"]]
        .groupby(
            [
                "Date Applied (Shamsi)",
                "DateMiladi",
                "Date Applied (Year)",
                "Date Applied (Month)",
                "Date Applied (Day)"
            ],
            as_index=False
        )
        .agg(DailyIssue=("IssueVolume", "sum"))
    )

    if daily_issue.empty:
        st.info("No issue data available to calculate daily outliers.")
    else:
        daily_issue = daily_issue.sort_values("DateMiladi").reset_index(drop=True)

        # Map month to English name
        daily_issue["MonthEN"] = daily_issue["Date Applied (Month)"].map(month_map)

        # Pretty label: e.g. "Tir 02 1403"
        daily_issue["PrettyShamsi"] = (
            daily_issue["MonthEN"] + " " +
            daily_issue["Date Applied (Day)"].astype(int).astype(str).str.zfill(2) +
            " " +
            daily_issue["Date Applied (Year)"].astype(int).astype(str)
        )

        # Ensure axis is categorical in the correct time order
        ordered_daily_labels = daily_issue["PrettyShamsi"].tolist()
        daily_issue["PrettyShamsi"] = pd.Categorical(
            daily_issue["PrettyShamsi"],
            categories=ordered_daily_labels,
            ordered=True
        )

        # IQR
        Q1 = daily_issue["DailyIssue"].quantile(0.25)
        Q3 = daily_issue["DailyIssue"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        def classify_outlier(x):
            if x > upper_bound:
                return "High Outlier"
            elif x < lower_bound:
                return "Low Outlier"
            else:
                return "Normal"

        daily_issue["OutlierFlag"] = daily_issue["DailyIssue"].apply(classify_outlier)
        outliers = daily_issue[daily_issue["OutlierFlag"] != "Normal"]

        c_left, c_right = st.columns(2)
        with c_left:
            st.write(f"Q1: `{Q1:.2f}`  |  Q3: `{Q3:.2f}`  |  IQR: `{IQR:.2f}`")
        with c_right:
            st.write(f"Lower Bound: `{lower_bound:.2f}`  |  Upper Bound: `{upper_bound:.2f}`")

        # Horizontal Boxplot
        fig_box = px.box(
            daily_issue,
            x="DailyIssue",
            points="outliers",
            title="Daily Issue Distribution (Horizontal Boxplot, IQR Outliers)",
            labels={"DailyIssue": "Daily Issue Volume"}
        )
        fig_box.update_layout(
            template="plotly_white",
            title_x=0.5,
            font=dict(size=12),
            height=400
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Line chart with PrettyShamsi labels
        fig_line = px.line(
            daily_issue,
            x="PrettyShamsi",
            y="DailyIssue",
            markers=True,
            title="Daily Issue Trend with IQR Outliers (Shamsi, English Month Names)",
            labels={
                "PrettyShamsi": "Date (Shamsi)",
                "DailyIssue": "Daily Issue Volume"
            }
        )

        if not outliers.empty:
            fig_line.add_scatter(
                x=outliers["PrettyShamsi"],
                y=outliers["DailyIssue"],
                mode="markers",
                name="Outliers",
                marker=dict(
                    size=10,
                    color="red",
                    symbol="circle-open",
                    line=dict(width=2)
                )
            )

        fig_line.update_layout(
            template="plotly_white",
            title_x=0.5,
            font=dict(size=11),
            xaxis_tickangle=-60,
            height=450
        )

        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("Show daily issue data (with outlier flags)"):
            st.dataframe(daily_issue, use_container_width=True)



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
