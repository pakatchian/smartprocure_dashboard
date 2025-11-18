import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import DateOffset
from datetime import timedelta
import matplotlib.pyplot as plt
import ruptures as rpt

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    """System configuration parameters"""
    TRANSACTIONS_FILE = "data/data_transactions.csv-test"
    LEAD_TIMES_FILE = "data/data_lead_times.csv"
    TRAIN_TEST_SPLIT = 0.8
    RANDOM_SEED = 42

    CHANGE_POINT_PENALTY = 10
    CHANGE_POINT_MIN_SIZE = 7

    HIGH_DEMAND_THRESHOLD = 0.5
    LOW_DEMAND_THRESHOLD = 0.5

    DAYS_TO_HIGH_DEMAND_THRESHOLD = 14
    DAYS_FROM_HIGH_DEMAND_THRESHOLD = 7
    DAYS_TO_LOW_DEMAND_THRESHOLD = 14
    DAYS_FROM_LOW_DEMAND_THRESHOLD = 7

    CANDIDATE_R = [1000, 2000, 3000, 4000, 5000]
    CANDIDATE_Q = [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


# =============================================================================
# DATA LOADING
# =============================================================================
class DataLoader:
    """Handles data loading and initial preprocessing"""
    
    @staticmethod
    def load_and_prepare(file_path):
        """Load Excel data and prepare daily aggregates"""
        raw_df = pd.read_excel(file_path, 0)
        raw_df["date"] = pd.to_datetime(raw_df["date"])
        raw_df["date_only"] = raw_df["date"].dt.date
        
        usage_fn = lambda x: -x[x < 0].sum()
        
        daily = (
            raw_df.sort_values("date")
            .groupby("date_only")
            .agg(
                inventory=("inv", "last"),
                usage_qty=("trans_qty", usage_fn)
            )
        )
        
        full_dates = pd.date_range(daily.index.min(), daily.index.max())
        daily = daily.reindex(full_dates)
        daily["inventory"] = daily["inventory"].ffill()
        daily["usage_qty"] = daily["usage_qty"].fillna(0)
        
        return daily, full_dates


# =============================================================================
# CHANGE POINT DETECTION
# =============================================================================
class ChangePointDetector:
    """Automatically detects regime changes in demand patterns"""
    
    def __init__(self, penalty=10, min_size=7):
        self.penalty = penalty
        self.min_size = min_size
        self.change_points = None
        self.periods = None
    
    def detect(self, usage_series):
        """Detect change points using PELT algorithm"""
        signal = usage_series.fillna(0).values.reshape(-1, 1)
        model = rpt.Pelt(model="rbf", min_size=self.min_size, jump=1).fit(signal)
        self.change_points = model.predict(pen=self.penalty)
        return self.change_points
    
    def classify_periods(self, usage_series, dates, high_threshold=0.5, low_threshold=0.5):
        """Classify periods between change points"""
        overall_mean = usage_series.mean()
        overall_std = usage_series.std()
        
        periods = []
        for i in range(len(self.change_points)):
            start_idx = 0 if i == 0 else self.change_points[i-1]
            end_idx = self.change_points[i]
            
            period_data = usage_series.iloc[start_idx:end_idx]
            period_mean = period_data.mean()
            period_std = period_data.std()
            period_dates = dates[start_idx:end_idx]
            
            if period_mean > overall_mean + high_threshold * overall_std:
                period_type = "HIGH_DEMAND"
                marker = "🔴"
            elif period_mean < overall_mean - low_threshold * overall_std:
                period_type = "LOW_DEMAND"
                marker = "🔵"
            else:
                period_type = "NORMAL"
                marker = "⚪"
            
            periods.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_date": period_dates[0] if len(period_dates) > 0 else None,
                "end_date": period_dates[-1] if len(period_dates) > 0 else None,
                "mean_demand": period_mean,
                "std_demand": period_std,
                "period_type": period_type,
                "marker": marker
            })
        
        self.periods = periods
        return periods
    
    def print_summary(self):
        """Display detected periods summary"""
        print("=" * 60)
        print(f"✓ Detected {len(self.change_points)-1} change points (regime shifts)")
        print("=" * 60)
        print("\nDETECTED SPECIAL PERIODS:")
        print("=" * 60)
        
        for idx, period in enumerate(self.periods):
            print(f"{period['marker']} Period {idx+1}: {period['start_date']} to {period['end_date']}")
            print(f"   Type: {period['period_type']}, Avg Demand: {period['mean_demand']:.1f} units/day\n")


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
class FeatureEngineer:
    """Creates features for machine learning models"""
    
    @staticmethod
    def create_base_features(daily_df, dates):
        """Create basic temporal and lag features"""
        daily_df["dayofweek"] = dates.dayofweek
        daily_df["month"] = dates.month
        daily_df["lag1"] = daily_df["inventory"].shift(1).fillna(daily_df["inventory"].iloc[0])
        daily_df["rolling_mean_7"] = daily_df["inventory"].shift(1).rolling(7, min_periods=1).mean()
        daily_df["rolling_mean_30"] = daily_df["inventory"].shift(1).rolling(30, min_periods=1).mean()
        return daily_df
    
    @staticmethod
    def create_period_features(daily_df, periods, overall_mean):
        """Add change point derived features"""
        daily_df["period_type_high"] = 0
        daily_df["period_type_low"] = 0
        daily_df["period_mean_demand"] = overall_mean
        
        for period in periods:
            mask = (daily_df.index >= period["start_date"]) & (daily_df.index <= period["end_date"])
            daily_df.loc[mask, "period_mean_demand"] = period["mean_demand"]
            
            if period["period_type"] == "HIGH_DEMAND":
                daily_df.loc[mask, "period_type_high"] = 1
            elif period["period_type"] == "LOW_DEMAND":
                daily_df.loc[mask, "period_type_low"] = 1
        
        return daily_df
    
    @staticmethod
    def calculate_temporal_distance(daily_df, periods, period_type, direction="next"):
        """Calculate days to/from special periods"""
        special_dates = []
        for period in periods:
            if period["period_type"] == period_type:
                date_key = "start_date" if direction == "next" else "end_date"
                special_dates.append(period[date_key])
        
        if not special_dates:
            return pd.Series([999] * len(daily_df), index=daily_df.index)
        
        special_dates = pd.DatetimeIndex(special_dates)
        days_list = []
        
        for current_date in daily_df.index:
            if direction == "next":
                relevant_dates = special_dates[special_dates > current_date]
                days = (relevant_dates[0] - current_date).days if len(relevant_dates) > 0 else 999
            else:
                relevant_dates = special_dates[special_dates <= current_date]
                days = (current_date - relevant_dates[-1]).days if len(relevant_dates) > 0 else 999
            
            days_list.append(days)
        
        return pd.Series(days_list, index=daily_df.index)
    
    @staticmethod
    def create_proximity_features(daily_df, periods, config):
        """Create features indicating proximity to special periods"""
        daily_df["days_to_high_demand"] = FeatureEngineer.calculate_temporal_distance(
            daily_df, periods, "HIGH_DEMAND", "next"
        )
        daily_df["days_from_high_demand"] = FeatureEngineer.calculate_temporal_distance(
            daily_df, periods, "HIGH_DEMAND", "last"
        )
        daily_df["days_to_low_demand"] = FeatureEngineer.calculate_temporal_distance(
            daily_df, periods, "LOW_DEMAND", "next"
        )
        daily_df["days_from_low_demand"] = FeatureEngineer.calculate_temporal_distance(
            daily_df, periods, "LOW_DEMAND", "last"
        )
        
        daily_df["near_high_demand"] = (
            (daily_df["days_to_high_demand"] <= config.DAYS_TO_HIGH_DEMAND_THRESHOLD) | 
            (daily_df["days_from_high_demand"] <= config.DAYS_FROM_HIGH_DEMAND_THRESHOLD)
        ).astype(int)
        
        daily_df["near_low_demand"] = (
            (daily_df["days_to_low_demand"] <= config.DAYS_TO_LOW_DEMAND_THRESHOLD) | 
            (daily_df["days_from_low_demand"] <= config.DAYS_FROM_LOW_DEMAND_THRESHOLD)
        ).astype(int)
        
        return daily_df


# =============================================================================
# INVENTORY SIMULATION
# =============================================================================
class InventorySimulator:
    """Simulates FEFO inventory management"""
    
    def __init__(self, config, shelf_life_days):
        self.config = config
        self.shelf_life_days = shelf_life_days
    
    def simulate(self, pred_usage, initial_inventory, dates, R, Q, 
                 lead_time_func=None, track_inventory=False):
        """Simulate inventory with (R,Q) policy using FEFO"""
        np.random.seed(self.config.RANDOM_SEED)
        
        batches = [{
            "qty": initial_inventory,
            "expiry": dates[0] + DateOffset(years=100)
        }]
        pending_orders = []
        waste = 0.0
        stockouts = 0.0
        daily_inv = [] if track_inventory else None
        
        for i, current_date in enumerate(dates):
            arrivals = [o for o in pending_orders if o["arrival_date"] <= current_date]
            for order in arrivals:
                batches.append({
                    "qty": order["qty"],
                    "expiry": current_date + timedelta(days=self.shelf_life_days)
                })
                pending_orders.remove(order)
            
            expired = [b for b in batches if b["expiry"] <= current_date]
            for batch in expired:
                waste += batch["qty"]
                batches.remove(batch)
            
            demand = pred_usage[i] if i < len(pred_usage) else 0
            remaining_demand = demand
            batches.sort(key=lambda b: b["expiry"])
            
            for batch in batches:
                if remaining_demand <= 0:
                    break
                used = min(batch["qty"], remaining_demand)
                batch["qty"] -= used
                remaining_demand -= used
            
            if remaining_demand > 0:
                stockouts += remaining_demand
            
            batches = [b for b in batches if b["qty"] > 1e-6]
            
            current_inv = sum(b["qty"] for b in batches)
            if track_inventory:
                daily_inv.append(current_inv)
            
            if current_inv <= R and not pending_orders:
                lead_time = lead_time_func(current_date, i) if lead_time_func else 2
                pending_orders.append({
                    "arrival_date": current_date + timedelta(days=lead_time),
                    "qty": Q
                })
        
        total_demand = pred_usage.sum()
        service_level = 1 - (stockouts / total_demand) if total_demand > 0 else 1
        
        if track_inventory:
            return waste, stockouts, service_level, daily_inv
        return waste, stockouts, service_level


# =============================================================================
# OPTIMIZATION ENGINE
# =============================================================================
class PolicyOptimizer:
    """Optimizes (R, Q) policy parameters"""

    def __init__(self, simulator, config):
        self.simulator = simulator
        self.config = config

    def grid_search(self, pred_usage, initial_inventory, dates, lead_time_func=None):
        """Perform grid search over R and Q candidates"""
        print("\nRunning simulations with change-point-aware predictions...")

        results = []
        for R in self.config.CANDIDATE_R:
            for Q in self.config.CANDIDATE_Q:
                waste, stockouts, service = self.simulator.simulate(
                    pred_usage, initial_inventory, dates, R, Q, lead_time_func=lead_time_func
                )
                results.append({
                    "R": R,
                    "Q": Q,
                    "waste": waste,
                    "stockouts": stockouts,
                    "service_level": service
                })
                print(f"R={R}, Q={Q} → waste={waste:.1f}, stockouts={stockouts:.1f}, service={service:.2%}")

        results_df = pd.DataFrame(results)
        results_df["stockout_flag"] = results_df["stockouts"] > 0
        results_df = results_df.sort_values(["stockout_flag", "stockouts", "waste", "R"])

        return results_df
    
    @staticmethod
    def print_results(results_df):
        """Display optimization results"""
        print("\n=== TOP 10 POLICIES ===")
        print(results_df.head(10).to_string(index=False))
        
        best = results_df.iloc[0]
        print("\n" + "=" * 60)
        print("=== OPTIMAL POLICY ===")
        print(f"R={best.R:.0f}, Q={best.Q:.0f}, Waste={best.waste:.1f}, "
              f"Stockouts={best.stockouts:.1f}, Service={best.service_level:.2%}")
        print("=" * 60)
        
        return best


# =============================================================================
# VISUALIZATION
# =============================================================================
class Visualizer:
    """Creates visualizations for analysis results"""
    
    @staticmethod
    def plot_optimization_results(dates, actual_inventory, simulated_inventory,
                                   best_R, change_points, periods, model_name):
        """Create comprehensive optimization visualization"""
        fig, ax = plt.subplots(figsize=(15, 5))
        
        ax.plot(dates, actual_inventory, label="Actual Inventory", 
                alpha=0.7, linewidth=2)
        ax.plot(dates, simulated_inventory, label="Simulated Optimal Policy", 
                alpha=0.7, linewidth=2)
        ax.axhline(best_R, color="r", linestyle="--", alpha=0.5, 
                   label=f"Reorder Point R={best_R:,.0f}")
        
        for cp_idx in change_points[:-1]:
            ax.axvline(dates[cp_idx], color="orange", linestyle=":", 
                       alpha=0.5, linewidth=1)
        
        for period in periods:
            if period["period_type"] == "HIGH_DEMAND":
                ax.axvspan(period["start_date"], period["end_date"], 
                          alpha=0.15, color="red")
            elif period["period_type"] == "LOW_DEMAND":
                ax.axvspan(period["start_date"], period["end_date"], 
                          alpha=0.15, color="blue")
        
        ax.set_title(f"Optimal Policy vs Historical Inventory ({model_name} with Change Point Detection)",
                    fontsize=12, fontweight='bold')
        ax.set_ylabel("Inventory Level")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# LEAD TIME ESTIMATION
# =============================================================================
class LeadTimeEstimator:
    """Train a simple ML model to predict lead time (days) from ORDER_DATE -> DATE_ENTERED."""

    def __init__(self, config):
        self.config = config
        self.model = None

    def _make_features(self, dates):
        d = pd.to_datetime(pd.Series(dates))
        return pd.DataFrame({
            "dayofweek": d.dt.dayofweek,
            "month": d.dt.month,
            "day": d.dt.day,
            "day_of_year": d.dt.dayofyear
        })

    def train(self, file_path):
        df = pd.read_excel(file_path)
        df["ORDER_DATE"] = pd.to_datetime(df["ORDER_DATE"])
        df["DATE_ENTERED"] = pd.to_datetime(df["DATE_ENTERED"])
        df["lead_days"] = (df["DATE_ENTERED"] - df["ORDER_DATE"]).dt.days.clip(lower=0)

        X = self._make_features(df["ORDER_DATE"])
        y = df["lead_days"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.config.RANDOM_SEED
        )

        model = RandomForestRegressor(n_estimators=200, random_state=self.config.RANDOM_SEED, n_jobs=-1)
        model.fit(X_train, y_train)
        self.model = model

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print("\n=== Lead Time Model Evaluation ===")
        print(f"Samples: {len(y)}  MAE: {mae:.2f} days  RMSE: {rmse:.2f} days  R²: {r2:.3f}")
        print("=" * 60)

    def predict(self, order_date):
        if self.model is None:
            raise RuntimeError("Lead time model not trained.")
        X = self._make_features([order_date])
        pred = int(np.round(self.model.predict(X)[0]))
        return pred


# =============================================================================
# MAIN EXECUTION
# =============================================================================
def run_direct():
    """Run full pipeline automatically with 90 days shelf life and Linear Regression"""
    config = Config()
    shelf_life_days = 90
    model_name = "Linear Regression (direct)"

    print("=" * 60)
    print("DIRECT OPTIMIZATION: 90-day shelf life + Linear Regression")
    print("=" * 60)

    # 1) Load data
    print("\n[1/7] Loading data...")
    daily, dates = DataLoader.load_and_prepare(config.TRANSACTIONS_FILE)

    # 2) Change point detection
    print("\n[2/7] Detecting change points...")
    detector = ChangePointDetector(
        penalty=config.CHANGE_POINT_PENALTY,
        min_size=config.CHANGE_POINT_MIN_SIZE
    )
    detector.detect(daily["usage_qty"])
    periods = detector.classify_periods(
        daily["usage_qty"],
        dates,
        config.HIGH_DEMAND_THRESHOLD,
        config.LOW_DEMAND_THRESHOLD
    )
    detector.print_summary()

    # 3) Feature engineering
    print("\n[3/7] Creating features...")
    daily = FeatureEngineer.create_base_features(daily, dates)
    overall_mean = daily["usage_qty"].mean()
    daily = FeatureEngineer.create_period_features(daily, periods, overall_mean)
    daily = FeatureEngineer.create_proximity_features(daily, periods, config)

    feature_columns = [
        "dayofweek", "month", "lag1", "rolling_mean_7", "rolling_mean_30",
        "period_type_high", "period_type_low", "period_mean_demand",
        "days_to_high_demand", "days_from_high_demand",
        "near_high_demand", "near_low_demand"
    ]
    X = daily[feature_columns]
    y = daily["inventory"]

    # 4) Train Linear Regression
    print("\n[4/7] Training Linear Regression on full dataset...")
    model = make_pipeline(SimpleImputer(strategy="mean"), LinearRegression())
    model.fit(X, y)
    pred_inventory = model.predict(X)
    daily["pred_inventory"] = pred_inventory

    # 5) Compute predicted usage
    print("\n[5/7] Computing predicted usage...")
    pred_inventory = np.asarray(pred_inventory)
    pred_usage = np.maximum(0, np.concatenate(([0], -np.diff(pred_inventory))))

    # 6) Lead time function
    print("\n[6/7] Training lead time estimator...")
    lead_time_func = None
    try:
        lead_est = LeadTimeEstimator(config)
        lead_est.train(config.LEAD_TIMES_FILE)
        lead_time_func = lambda current_date, idx: lead_est.predict(current_date)
    except Exception as e:
        print(f"⚠️ Lead time training failed: {e}")
        print("   Using fallback constant lead time of 2 days.")
        lead_time_func = lambda current_date, idx: 2

    # 7) Optimization
    print("\n[7/7] Running policy optimization...")
    simulator = InventorySimulator(config, shelf_life_days)
    optimizer = PolicyOptimizer(simulator, config)

    results_df = optimizer.grid_search(
        pred_usage,
        daily["inventory"].iloc[0],
        dates,
        lead_time_func=lead_time_func
    )

    best = PolicyOptimizer.print_results(results_df)

    # Simulate with best policy
    _, _, _, simulated_inv = simulator.simulate(
        pred_usage,
        daily["inventory"].iloc[0],
        dates,
        best.R,
        best.Q,
        lead_time_func=lead_time_func,
        track_inventory=True
    )

    # Plot final results
    print("\nGenerating visualization...")
    Visualizer.plot_optimization_results(
        dates,
        daily["inventory"].values,
        simulated_inv,
        best.R,
        detector.change_points,
        periods,
        model_name
    )

    print("\n✓ direct.py run complete — shelf_life_days=90, Linear Regression used.")


if __name__ == "__main__":
    run_direct()
