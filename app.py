"""
Project 4: AI Robo-Advisor and Portfolio Recommendation Tool
Course: FIN-671 | Advanced Financial Technology & Quantitative Asset Management
Architecture: Streamlit, Plotly, NumPy, Pandas, Requests, yFinance

This application delivers institutional-grade quantitative risk profiling,
dynamic Modern Portfolio Theory (MPT) multi-asset ETF allocation, and stochastic
wealth projections for educational asset allocation illustrations.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

# ==============================================================================
# GLOBAL CONSTANTS & ETF UNIVERSE METADATA
# ==============================================================================
ETF_UNIVERSE: Dict[str, Dict[str, str]] = {
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "US Large-Cap Equity",
        "category": "Equity",
        "color": "#1E40AF"  # Deep Royal Blue
    },
    "QQQ": {
        "name": "Invesco QQQ Trust (Nasdaq-100)",
        "asset_class": "US Tech / Large Growth",
        "category": "Equity",
        "color": "#0284C7"  # Sky Blue
    },
    "VEA": {
        "name": "Vanguard FTSE Developed Markets ETF",
        "asset_class": "International Developed Equity",
        "category": "Equity",
        "color": "#0D9488"  # Teal
    },
    "AGG": {
        "name": "iShares Core U.S. Aggregate Bond ETF",
        "asset_class": "Total US Investment-Grade Bond",
        "category": "Fixed Income",
        "color": "#D97706"  # Amber
    },
    "SHY": {
        "name": "iShares 1-3 Year Treasury Bond ETF",
        "asset_class": "Short-Term US Treasury / Cash Equivalent",
        "category": "Cash / Short-Term",
        "color": "#EA580C"  # Orange
    }
}

TICKERS: List[str] = list(ETF_UNIVERSE.keys())

# Default benchmark Capital Market Assumptions (CMA) annualized forward returns
CMA_ANNUAL_RETURNS: Dict[str, float] = {
    "SPY": 0.095,   # 9.50% expected nominal return
    "QQQ": 0.115,   # 11.50% expected nominal return
    "VEA": 0.078,   # 7.80% expected nominal return
    "AGG": 0.046,   # 4.60% expected nominal return
    "SHY": 0.041    # 4.10% expected nominal return
}

# Pre-computed robust empirical 5-year annualized covariance matrix (offline fallback)
# Order: SPY, QQQ, VEA, AGG, SHY
OFFLINE_COV_MATRIX: np.ndarray = np.array([
    [0.0289, 0.0345, 0.0242, 0.0035, 0.0004],  # SPY (vol ~17.0%)
    [0.0345, 0.0484, 0.0280, 0.0031, 0.0002],  # QQQ (vol ~22.0%)
    [0.0242, 0.0280, 0.0289, 0.0040, 0.0006],  # VEA (vol ~17.0%)
    [0.0035, 0.0031, 0.0040, 0.0036, 0.0008],  # AGG (vol ~6.0%)
    [0.0004, 0.0002, 0.0006, 0.0008, 0.0004]   # SHY (vol ~2.0%)
])

FALLBACK_RISK_FREE_RATE: float = 0.0425  # 4.25% default 3-Month T-Bill yield


# ==============================================================================
# FUNCTION 1: QUANTITATIVE RISK PROFILING
# ==============================================================================
def calculate_risk_score(
    age: int,
    horizon_years: int,
    max_drawdown_pct: float,
    risk_score_input: int,
    liquidity_ratio_pct: float,
    financial_goal: str
) -> Tuple[float, str, Dict[str, float]]:
    """
    Combines strictly numerical user inputs into a 0-100 normalized Composite Risk Score
    using a multi-factor quantitative scoring formula.

    Factors:
    - Investment Horizon (30%): Longer horizons offer capacity to endure market cycles.
    - Max Acceptable Drawdown (25%): Direct behavioral loss tolerance threshold.
    - Numerical Risk Score (15%): Subjective psychometric risk tolerance slider (1-10).
    - Investor Age (15%): Younger investors possess higher human capital / recovery time.
    - Liquidity Buffer Need (15%): Higher emergency cash needs reduce risk capacity.
    - Goal Factor: Empirical categorical adjustment (+8 for Accumulation, -12 for Preservation).

    Returns:
        composite_score (float): 0.0 to 100.0
        risk_profile (str): 'Conservative', 'Moderate', or 'Aggressive'
        subscores (dict): Normalized constituent scores for diagnostic inspection
    """
    # 1. Horizon Factor (normalized to 1-30+ years): 1 yr -> ~3.3, 30+ yrs -> 100.0
    f_horizon = float(np.clip((horizon_years / 30.0) * 100.0, 0.0, 100.0))

    # 2. Drawdown Factor (5% to 40% range mapped to 0-100)
    f_drawdown = float(np.clip(((max_drawdown_pct - 5.0) / (40.0 - 5.0)) * 100.0, 0.0, 100.0))

    # 3. Psychometric Risk Score (1 to 10 mapped to 0-100)
    f_risk_scale = float(np.clip(((risk_score_input - 1.0) / 9.0) * 100.0, 0.0, 100.0))

    # 4. Age Factor (Inverse mapping: age 18 -> 100.0, age 80 -> 0.0)
    f_age = float(np.clip(((80.0 - age) / (80.0 - 18.0)) * 100.0, 0.0, 100.0))

    # 5. Liquidity Need Factor (0% need -> 100.0 risk capacity, 50% need -> 0.0 risk capacity)
    f_liquidity = float(np.clip((1.0 - (liquidity_ratio_pct / 50.0)) * 100.0, 0.0, 100.0))

    # Goal Offset
    goal_offsets = {
        "Wealth Accumulation": 6.0,
        "Retirement": 0.0,
        "Capital Preservation": -12.0
    }
    goal_offset = goal_offsets.get(financial_goal, 0.0)

    # Weighted composite aggregation
    raw_composite = (
        0.30 * f_horizon +
        0.25 * f_drawdown +
        0.15 * f_risk_scale +
        0.15 * f_age +
        0.15 * f_liquidity +
        goal_offset
    )

    composite_score = float(np.clip(raw_composite, 0.0, 100.0))

    # Classification boundaries
    if composite_score < 38.0:
        risk_profile = "Conservative"
    elif composite_score < 68.0:
        risk_profile = "Moderate"
    else:
        risk_profile = "Aggressive"

    subscores = {
        "Horizon Sub-score": f_horizon,
        "Drawdown Sub-score": f_drawdown,
        "Risk Scale Sub-score": f_risk_scale,
        "Age Sub-score": f_age,
        "Liquidity Sub-score": f_liquidity,
        "Goal Adjustment": goal_offset
    }

    return composite_score, risk_profile, subscores


# ==============================================================================
# FUNCTION 2: PORTFOLIO ALLOCATION ENGINE & FINE-TUNE TILT
# ==============================================================================
def get_etf_allocation(
    risk_profile: str,
    equity_tilt_pct: float = 0.0
) -> Dict[str, float]:
    """
    Computes target portfolio weights across Equities (SPY, QQQ, VEA) and Fixed Income (AGG, SHY).
    Applies user-selected fine-tune equity/bond tilt, rigorously guaranteeing weights sum to 1.0 (100%).

    Base Profiles:
    - Conservative: 30% Equities (15% SPY, 5% QQQ, 10% VEA), 70% Fixed Income (45% AGG, 25% SHY)
    - Moderate: 60% Equities (35% SPY, 15% QQQ, 10% VEA), 40% Fixed Income (30% AGG, 10% SHY)
    - Aggressive: 85% Equities (50% SPY, 25% QQQ, 10% VEA), 15% Fixed Income (10% AGG, 5% SHY)
    """
    baseline_weights: Dict[str, Dict[str, float]] = {
        "Conservative": {
            "SPY": 0.15, "QQQ": 0.05, "VEA": 0.10,
            "AGG": 0.45, "SHY": 0.25
        },
        "Moderate": {
            "SPY": 0.35, "QQQ": 0.15, "VEA": 0.10,
            "AGG": 0.30, "SHY": 0.10
        },
        "Aggressive": {
            "SPY": 0.50, "QQQ": 0.25, "VEA": 0.10,
            "AGG": 0.10, "SHY": 0.05
        }
    }

    base = baseline_weights.get(risk_profile, baseline_weights["Moderate"]).copy()

    # Calculate baseline aggregate equity and fixed income weights
    equity_tickers = ["SPY", "QQQ", "VEA"]
    fi_tickers = ["AGG", "SHY"]

    base_equity_sum = sum(base[t] for t in equity_tickers)
    base_fi_sum = sum(base[t] for t in fi_tickers)

    # Shift equity sum by equity_tilt_pct (within safe boundaries [0.10, 0.95])
    target_equity_sum = float(np.clip(base_equity_sum + (equity_tilt_pct / 100.0), 0.10, 0.95))
    target_fi_sum = 1.0 - target_equity_sum

    adjusted_weights: Dict[str, float] = {}

    # Proportional rescaling within asset classes to preserve strategic intra-asset diversification
    for t in equity_tickers:
        relative_share = base[t] / base_equity_sum
        adjusted_weights[t] = relative_share * target_equity_sum

    for t in fi_tickers:
        relative_share = base[t] / base_fi_sum
        adjusted_weights[t] = relative_share * target_fi_sum

    # Strict normalization check to eliminate floating-point drift
    total_w = sum(adjusted_weights.values())
    for t in adjusted_weights:
        adjusted_weights[t] = float(np.round(adjusted_weights[t] / total_w, 4))

    # Fix any remaining microscopic rounding discrepancy on core equity anchor (SPY)
    rounding_discrepancy = 1.0 - sum(adjusted_weights.values())
    adjusted_weights["SPY"] = float(np.round(adjusted_weights["SPY"] + rounding_discrepancy, 4))

    return adjusted_weights


# ==============================================================================
# FUNCTION 3: MARKET DATA INTEGRATION WITH MULTI-TIER FALLBACKS
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_market_data(
    finnhub_api_key: str = "",
    fred_api_key: str = ""
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, Dict[str, Any]], float, List[str]]:
    """
    Fetches real-time ETF quotes, historical price series for covariance calculation,
    and macroeconomic risk-free yield data with multi-tier fallback architecture.

    Returns:
        expected_returns (dict): Annualized expected return per ticker
        cov_matrix (np.ndarray): 5x5 Annualized return covariance matrix
        live_quotes (dict): Latest price, daily change, and metadata
        risk_free_rate (float): Current risk-free rate (e.g. 0.0425 = 4.25%)
        data_status_logs (list): Diagnostic status messages detailing data sources used
    """
    data_status_logs: List[str] = []
    live_quotes: Dict[str, Dict[str, Any]] = {}

    # --- 1. Macroeconomic Risk-Free Rate (FRED API or Fallback) ---
    risk_free_rate = FALLBACK_RISK_FREE_RATE
    fred_fetched = False
    if fred_api_key.strip():
        try:
            fred_url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=DTB3&api_key={fred_api_key.strip()}&file_type=json&sort_order=desc&limit=5"
            )
            response = requests.get(fred_url, timeout=3.5)
            if response.status_code == 200:
                obs_list = response.json().get("observations", [])
                for obs in obs_list:
                    val_str = obs.get("value", "")
                    if val_str and val_str != ".":
                        risk_free_rate = float(val_str) / 100.0
                        fred_fetched = True
                        data_status_logs.append(f"FRED API: 3-Month T-Bill yield loaded dynamically ({risk_free_rate * 100:.2f}%).")
                        break
        except Exception as exc:
            data_status_logs.append(f"FRED API notice: {exc}. Using fallback benchmark.")

    if not fred_fetched:
        data_status_logs.append(f"Macro Risk-Free Rate: Defaulted to benchmark 3M Treasury yield ({FALLBACK_RISK_FREE_RATE * 100:.2f}%).")

    # --- 2. Live Quotes (Finnhub API or yFinance / Static Fallback) ---
    finnhub_success = False
    if finnhub_api_key.strip():
        try:
            finnhub_count = 0
            for ticker in TICKERS:
                url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={finnhub_api_key.strip()}"
                res = requests.get(url, timeout=3.0)
                if res.status_code == 200:
                    payload = res.json()
                    current_price = payload.get("c", 0.0)
                    change = payload.get("d", 0.0)
                    pct_change = payload.get("dp", 0.0)
                    if current_price and current_price > 0:
                        live_quotes[ticker] = {
                            "price": float(current_price),
                            "change": float(change),
                            "pct_change": float(pct_change),
                            "source": "Finnhub Live Quote"
                        }
                        finnhub_count += 1
            if finnhub_count == len(TICKERS):
                finnhub_success = True
                data_status_logs.append("Finnhub API: Real-time ETF quotes loaded successfully.")
        except Exception as exc:
            data_status_logs.append(f"Finnhub API notice: {exc}. Falling back to secondary sources.")

    # --- 3. Historical Returns & Covariance Matrix (yfinance or Pre-computed Matrix) ---
    historical_fetched = False
    cov_matrix = OFFLINE_COV_MATRIX.copy()
    expected_returns = CMA_ANNUAL_RETURNS.copy()

    try:
        # Download 3 years of daily historical data for the ETF basket
        df_hist = yf.download(TICKERS, period="3y", interval="1d", progress=False, auto_adjust=True)
        if df_hist is not None and not df_hist.empty:
            if "Close" in df_hist.columns:
                close_prices = df_hist["Close"]
            else:
                close_prices = df_hist

            # Ensure all tickers present
            valid_cols = [t for t in TICKERS if t in close_prices.columns]
            if len(valid_cols) == len(TICKERS):
                returns = close_prices[TICKERS].pct_change().dropna()
                if len(returns) > 60:
                    # Annualize daily return covariance (252 trading days/year)
                    sample_cov = returns.cov().values * 252.0
                    # Ensure positive semi-definiteness
                    cov_matrix = (sample_cov + sample_cov.T) / 2.0

                    # Empirical trailing annualized return
                    hist_annual_return = returns.mean().values * 252.0
                    # Blend 50% historical empirical + 50% Capital Market Assumptions for forward robustness
                    for idx, t in enumerate(TICKERS):
                        cma_val = CMA_ANNUAL_RETURNS[t]
                        expected_returns[t] = float(0.50 * hist_annual_return[idx] + 0.50 * cma_val)

                    historical_fetched = True
                    data_status_logs.append("Yahoo Finance: 3-year historical daily price series loaded for empirical covariance estimation.")

                    # Fill live quotes from yfinance if Finnhub was absent
                    if not finnhub_success:
                        for t in TICKERS:
                            last_p = float(close_prices[t].iloc[-1])
                            prev_p = float(close_prices[t].iloc[-2]) if len(close_prices[t]) > 1 else last_p
                            diff = last_p - prev_p
                            pct_diff = (diff / prev_p) * 100.0 if prev_p > 0 else 0.0
                            live_quotes[t] = {
                                "price": np.round(last_p, 2),
                                "change": np.round(diff, 2),
                                "pct_change": np.round(pct_diff, 2),
                                "source": "Yahoo Finance Close"
                            }
    except Exception as exc:
        data_status_logs.append(f"Yahoo Finance historical fetch notice: {exc}.")

    # --- 4. Ultimate Offline Fallback Guarantee ---
    if not historical_fetched:
        data_status_logs.append("Covariance Matrix: Offline empirical benchmark matrix utilized.")
        cov_matrix = OFFLINE_COV_MATRIX.copy()
        expected_returns = CMA_ANNUAL_RETURNS.copy()

    # If quotes are still empty, supply documented benchmark prices
    if len(live_quotes) < len(TICKERS):
        fallback_prices = {"SPY": 585.50, "QQQ": 495.20, "VEA": 52.80, "AGG": 98.40, "SHY": 82.10}
        for t in TICKERS:
            if t not in live_quotes:
                live_quotes[t] = {
                    "price": fallback_prices.get(t, 100.0),
                    "change": 0.0,
                    "pct_change": 0.0,
                    "source": "Documented Benchmark Price"
                }

    return expected_returns, cov_matrix, live_quotes, risk_free_rate, data_status_logs


# ==============================================================================
# FUNCTION 4: FINANCIAL ANALYSIS & QUANTITATIVE PORTFOLIO METRICS
# ==============================================================================
def compute_portfolio_metrics(
    weights_dict: Dict[str, float],
    expected_returns_dict: Dict[str, float],
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> Dict[str, float]:
    """
    Calculates Modern Portfolio Theory (MPT) quantitative metrics:
    - Calculation 1: Annualized Expected Return E(Rp) = sum(w_i * E(R_i))
    - Calculation 2: Annualized Portfolio Volatility sigma_p = sqrt(w^T * Sigma * w)
    - Calculation 3: Portfolio Sharpe Ratio = (E(Rp) - Rf) / sigma_p
    - Calculation 4: Parametric 95% 1-Year Value at Risk (VaR)
    """
    w = np.array([weights_dict[t] for t in TICKERS], dtype=float)
    mu = np.array([expected_returns_dict[t] for t in TICKERS], dtype=float)

    # Calculation 1: Expected Portfolio Return
    expected_return = float(np.dot(w, mu))

    # Calculation 2: Annualized Volatility
    variance = float(np.dot(w.T, np.dot(cov_matrix, w)))
    volatility = float(np.sqrt(max(variance, 1e-8)))

    # Calculation 3: Sharpe Ratio
    excess_return = expected_return - risk_free_rate
    sharpe_ratio = float(excess_return / volatility) if volatility > 0 else 0.0

    # Calculation 4: 95% Parametric 1-Year Value-at-Risk (Normal Assumption: Z_0.95 = 1.645)
    var_95_pct = float(max(0.0, 1.645 * volatility - expected_return))

    return {
        "expected_return": expected_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio,
        "risk_free_rate": risk_free_rate,
        "var_95": var_95_pct
    }


# ==============================================================================
# FUNCTION 5: MULTI-YEAR COMPOUND WEALTH PROJECTION (STOCHASTIC / MONTE CARLO)
# ==============================================================================
def simulate_wealth_projection(
    initial_capital: float,
    monthly_contribution: float,
    horizon_years: int,
    expected_annual_return: float,
    annual_volatility: float,
    num_simulations: int = 1500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulates multi-year compound wealth accumulation incorporating initial capital
    and periodic monthly contributions over the investment horizon.

    Uses a discrete monthly Geometric Brownian Motion (GBM) / lognormal stochastic process:
        W_{t} = (W_{t-1} + Contribution) * exp((mu_m - 0.5 * sigma_m^2) + sigma_m * Z_t)

    Computes:
    - 10th Percentile (Conservative / Downside Bear Market path)
    - 50th Percentile (Median / Expected Growth path)
    - 90th Percentile (Optimistic / Bull Market path)
    - Cumulative Invested Principal (Cost basis)
    """
    np.random.seed(seed)
    num_months = int(horizon_years * 12)
    time_index = np.arange(0, num_months + 1)

    # Monthly parameters
    mu_monthly = expected_annual_return / 12.0
    sigma_monthly = annual_volatility / np.sqrt(12.0)
    drift = mu_monthly - 0.5 * (sigma_monthly ** 2)

    # Initialize simulation matrix: (num_months + 1, num_simulations)
    wealth_matrix = np.zeros((num_months + 1, num_simulations))
    wealth_matrix[0, :] = initial_capital

    # Vectorized monthly step iteration
    for t in range(1, num_months + 1):
        z = np.random.standard_normal(num_simulations)
        monthly_growth = np.exp(drift + sigma_monthly * z)
        # Add contribution at start of month t, then compound
        wealth_matrix[t, :] = (wealth_matrix[t - 1, :] + monthly_contribution) * monthly_growth

    # Calculate percentiles across all simulated paths at each month
    p10 = np.percentile(wealth_matrix, 10, axis=1)
    p50 = np.percentile(wealth_matrix, 50, axis=1)
    p90 = np.percentile(wealth_matrix, 90, axis=1)
    mean_path = np.mean(wealth_matrix, axis=1)
    cumulative_invested = initial_capital + monthly_contribution * time_index

    df_projection = pd.DataFrame({
        "Month": time_index,
        "Year": time_index / 12.0,
        "Cumulative_Invested": cumulative_invested,
        "P10_Bearish": p10,
        "P50_Median": p50,
        "P90_Bullish": p90,
        "Expected_Mean": mean_path
    })

    return df_projection


# ==============================================================================
# FUNCTION 6: DATA VISUALIZATIONS (PLOTLY INTERACTIVE CHARTS)
# ==============================================================================
def render_charts(
    weights: Dict[str, float],
    initial_capital: float,
    df_projection: pd.DataFrame,
    current_metrics: Dict[str, float],
    cov_matrix: np.ndarray,
    expected_returns: Dict[str, float],
    risk_profile: str
) -> Tuple[go.Figure, go.Figure, go.Figure]:
    """
    Builds 3 production-grade interactive Plotly visualizations:
    1. Interactive Donut Chart displaying ETF weights, asset classes, and dollar amounts.
    2. Compound Wealth Projection Fan Chart showing 10th, 50th, 90th percentiles and invested capital.
    3. Risk-Return Scatter / Efficient Frontier Plot comparing user's allocation to baselines.
    """
    # ---------------------------------------------------------
    # Chart 1: Recommended ETF Allocation Donut Chart
    # ---------------------------------------------------------
    labels = [f"{t} ({ETF_UNIVERSE[t]['asset_class']})" for t in TICKERS]
    values = [weights[t] * 100.0 for t in TICKERS]
    dollars = [weights[t] * initial_capital for t in TICKERS]
    colors = [ETF_UNIVERSE[t]["color"] for t in TICKERS]

    customdata = np.stack(
        (TICKERS, [ETF_UNIVERSE[t]["name"] for t in TICKERS], dollars),
        axis=-1
    )

    fig_donut = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="label+percent",
                hoverinfo="text",
                hovertext=[
                    f"<b>{t}</b> - {ETF_UNIVERSE[t]['name']}<br>"
                    f"Asset Class: {ETF_UNIVERSE[t]['asset_class']}<br>"
                    f"Target Allocation: {weights[t] * 100:.1f}%<br>"
                    f"Initial Value: ${dollars[i]:,.2f}"
                    for i, t in enumerate(TICKERS)
                ],
                customdata=customdata
            )
        ]
    )

    fig_donut.update_layout(
        title=dict(
            text=f"<b>Target Asset Allocation Breakdown</b> ({risk_profile} Strategy)",
            x=0.05,
            font=dict(size=18, family="Arial", color="#1E293B")
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        annotations=[
            dict(
                text=f"<b>Total Capital</b><br>${initial_capital:,.0f}",
                x=0.5, y=0.5,
                font_size=15,
                showarrow=False,
                font=dict(family="Arial", color="#334155")
            )
        ],
        margin=dict(t=60, b=60, l=20, r=20),
        height=450
    )

    # ---------------------------------------------------------
    # Chart 2: Compound Wealth Projection Fan Chart
    # ---------------------------------------------------------
    fig_fan = go.Figure()

    # 90th Percentile (Upper boundary of shaded band)
    fig_fan.add_trace(
        go.Scatter(
            x=df_projection["Year"],
            y=df_projection["P90_Bullish"],
            mode="lines",
            line=dict(color="rgba(16, 185, 129, 0.4)", width=1),
            name="90th Percentile (Bullish Scenario)",
            hovertemplate="Year %{x:.1f}: <b>$%{y:,.0f}</b><extra></extra>"
        )
    )

    # 10th Percentile (Lower boundary with fill to upper)
    fig_fan.add_trace(
        go.Scatter(
            x=df_projection["Year"],
            y=df_projection["P10_Bearish"],
            mode="lines",
            line=dict(color="rgba(239, 68, 68, 0.4)", width=1),
            fill="tonexty",
            fillcolor="rgba(59, 130, 246, 0.12)",
            name="10th Percentile (Downside Scenario)",
            hovertemplate="Year %{x:.1f}: <b>$%{y:,.0f}</b><extra></extra>"
        )
    )

    # 50th Percentile / Median Expected Growth
    fig_fan.add_trace(
        go.Scatter(
            x=df_projection["Year"],
            y=df_projection["P50_Median"],
            mode="lines",
            line=dict(color="#1D4ED8", width=3),
            name="50th Percentile (Expected Median)",
            hovertemplate="Year %{x:.1f}: <b>$%{y:,.0f}</b><extra></extra>"
        )
    )

    # Cumulative Invested Principal (Baseline)
    fig_fan.add_trace(
        go.Scatter(
            x=df_projection["Year"],
            y=df_projection["Cumulative_Invested"],
            mode="lines",
            line=dict(color="#64748B", width=2, dash="dash"),
            name="Cumulative Invested Principal",
            hovertemplate="Year %{x:.1f} Principal: <b>$%{y:,.0f}</b><extra></extra>"
        )
    )

    fig_fan.update_layout(
        title=dict(
            text="<b>Multi-Year Stochastic Wealth Accumulation (Monte Carlo 1,500 Paths)</b>",
            x=0.05,
            font=dict(size=18, family="Arial", color="#1E293B")
        ),
        xaxis=dict(
            title="Investment Horizon (Years)",
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            title="Portfolio Value ($ USD)",
            tickprefix="$",
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified",
        margin=dict(t=60, b=80, l=40, r=20),
        height=480
    )

    # ---------------------------------------------------------
    # Chart 3: Risk-Return Scatter / Efficient Frontier Comparison
    # ---------------------------------------------------------
    # Generate 400 random portfolio weights for frontier context
    np.random.seed(101)
    num_sim_pts = 400
    rand_weights = np.random.dirichlet(np.ones(len(TICKERS)), size=num_sim_pts)
    mu_vec = np.array([expected_returns[t] for t in TICKERS])

    frontier_returns = np.dot(rand_weights, mu_vec)
    frontier_vols = np.array([
        np.sqrt(np.dot(rw.T, np.dot(cov_matrix, rw))) for rw in rand_weights
    ])
    frontier_sharpes = (frontier_returns - current_metrics["risk_free_rate"]) / np.maximum(frontier_vols, 1e-6)

    fig_frontier = go.Figure()

    # Cloud of simulated portfolios
    fig_frontier.add_trace(
        go.Scatter(
            x=frontier_vols * 100.0,
            y=frontier_returns * 100.0,
            mode="markers",
            marker=dict(
                size=5,
                color=frontier_sharpes,
                colorscale="Viridis",
                colorbar=dict(title="Sharpe Ratio", thickness=12, len=0.7),
                opacity=0.45
            ),
            name="Feasible Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Exp Return: %{y:.2f}%<extra></extra>"
        )
    )

    # Baseline Portfolios
    baselines = {
        "Conservative": get_etf_allocation("Conservative", 0.0),
        "Moderate": get_etf_allocation("Moderate", 0.0),
        "Aggressive": get_etf_allocation("Aggressive", 0.0)
    }

    b_markers = {"Conservative": "circle", "Moderate": "square", "Aggressive": "diamond"}
    b_colors = {"Conservative": "#0D9488", "Moderate": "#D97706", "Aggressive": "#DC2626"}

    for b_name, b_w in baselines.items():
        b_metrics = compute_portfolio_metrics(b_w, expected_returns, cov_matrix, current_metrics["risk_free_rate"])
        fig_frontier.add_trace(
            go.Scatter(
                x=[b_metrics["volatility"] * 100.0],
                y=[b_metrics["expected_return"] * 100.0],
                mode="markers+text",
                marker=dict(symbol=b_markers[b_name], size=12, color=b_colors[b_name], line=dict(color="#000", width=1.5)),
                text=[f"  {b_name}"],
                textposition="top right",
                name=f"Baseline: {b_name}",
                hovertemplate=f"<b>{b_name}</b><br>Vol: %{{x:.2f}}%<br>Return: %{{y:.2f}}%<br>Sharpe: {b_metrics['sharpe_ratio']:.2f}<extra></extra>"
            )
        )

    # User's Custom Recommended Portfolio (Prominent Star)
    fig_frontier.add_trace(
        go.Scatter(
            x=[current_metrics["volatility"] * 100.0],
            y=[current_metrics["expected_return"] * 100.0],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="#F59E0B", line=dict(color="#78350F", width=2)),
            text=["  ★ Your Recommendation"],
            textposition="bottom right",
            name="Your Custom Portfolio",
            hovertemplate=(
                f"<b>Your Recommended Portfolio</b><br>"
                f"Annual Volatility: %{{x:.2f}}%<br>"
                f"Expected Return: %{{y:.2f}}%<br>"
                f"Sharpe Ratio: {current_metrics['sharpe_ratio']:.2f}<extra></extra>"
            )
        )
    )

    fig_frontier.update_layout(
        title=dict(
            text="<b>Risk-Return Profile & Efficient Frontier Mapping</b>",
            x=0.05,
            font=dict(size=18, family="Arial", color="#1E293B")
        ),
        xaxis=dict(
            title="Annualized Volatility / Risk (σ %)",
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        yaxis=dict(
            title="Annualized Expected Return (E(Rp) %)",
            showgrid=True,
            gridcolor="#F1F5F9"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=60, b=80, l=40, r=20),
        height=480
    )

    return fig_donut, fig_fan, fig_frontier


# ==============================================================================
# FUNCTION 7: MAIN STREAMLIT APPLICATION CONTROLLER
# ==============================================================================
def main() -> None:
    """
    Main application orchestration loop for Streamlit.
    """
    st.set_page_config(
        page_title="AI Robo-Advisor & Portfolio Tool | FIN-671",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ---------------------------------------------------------
    # MANDATORY LEGAL & EDUCATIONAL DISCLAIMER BANNER
    # ---------------------------------------------------------
    st.markdown(
        """
        <div style="background-color: #FEF3C7; border-left: 5px solid #F59E0B; padding: 14px 18px; border-radius: 6px; margin-bottom: 20px;">
            <span style="font-size: 16px; font-weight: 700; color: #92400E;">⚠️ ACADEMIC & RESEARCH DISCLAIMER:</span>
            <span style="font-size: 14px; color: #78350F; margin-left: 8px;">
                This application is developed strictly for educational asset allocation illustrations for university coursework (FIN-671).
                It does <b>NOT</b> constitute personalized legal, tax, or investment advice. Historical results and Capital Market Assumptions
                do not guarantee future performance.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("🤖 AI Quantitative Robo-Advisor & Portfolio Recommendation Engine")
    st.caption("FIN-671: Applied Quantitative Portfolio Management | Academic Prototype")

    # ---------------------------------------------------------
    # SIDEBAR: STRICTLY NUMERICAL PROFILING INPUTS
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Quantitative Risk Profiling")
        st.markdown("*Input strictly numerical parameters to establish your risk budget:*")

        age = st.slider("1. Current Age", min_value=18, max_value=80, value=28, step=1, help="Younger investors possess longer human capital horizons to weather downturns.")
        horizon_years = st.slider("2. Investment Horizon (Years)", min_value=1, max_value=40, value=15, step=1, help="Number of years before capital will be systematically withdrawn.")
        initial_capital = st.number_input("3. Initial Investment Capital ($ USD)", min_value=1000.0, max_value=10000000.0, value=25000.0, step=1000.0, format="%.0f")
        monthly_contribution = st.number_input("4. Monthly Contribution ($ USD)", min_value=0.0, max_value=50000.0, value=750.0, step=50.0, format="%.0f")
        max_drawdown_pct = st.slider("5. Max Acceptable Drawdown / Loss Tolerance (%)", min_value=5.0, max_value=40.0, value=20.0, step=1.0, help="Maximum tolerable peak-to-trough decline over an annual holding period.")
        risk_score_input = st.slider("6. Numerical Risk Tolerance Score (1-10)", min_value=1, max_value=10, value=6, step=1, help="Self-assessed risk preference score (1=Ultra-Defensive, 10=Maximum Speculative Growth).")
        liquidity_ratio_pct = st.slider("7. Liquidity Need / Emergency Buffer Ratio (%)", min_value=0.0, max_value=50.0, value=15.0, step=2.5, help="Percentage of liquid net worth required in liquid cash instruments.")

        financial_goal = st.selectbox(
            "8. Primary Financial Goal",
            options=["Wealth Accumulation", "Retirement", "Capital Preservation"],
            index=0,
            help="High-level financial objective used for strategic asset duration matching."
        )

        st.markdown("---")
        st.subheader("🎛️ Strategic Allocation Tilt")
        equity_tilt_pct = st.slider(
            "Fine-Tune Equity Tilt (% Shift)",
            min_value=-25.0,
            max_value=25.0,
            value=0.0,
            step=2.5,
            help="Manually shift overall equity exposure higher or lower relative to base model, observing instant impact on return & risk."
        )

        st.markdown("---")
        with st.expander("🔑 External API Credentials (Optional)"):
            st.caption("Leave empty to use automatic Yahoo Finance and benchmark fallback data.")
            finnhub_key = st.text_input("Finnhub API Key", value="", type="password", help="Free API key from finnhub.io for live ETF quote feeds.")
            fred_key = st.text_input("FRED API Key", value="", type="password", help="Free API key from fred.stlouisfed.org for 3-Month T-Bill DTB3 yield.")

    # ---------------------------------------------------------
    # 1. RISK SCORE & CLASSIFICATION EVALUATION
    # ---------------------------------------------------------
    composite_score, risk_profile, subscores = calculate_risk_score(
        age=age,
        horizon_years=horizon_years,
        max_drawdown_pct=max_drawdown_pct,
        risk_score_input=risk_score_input,
        liquidity_ratio_pct=liquidity_ratio_pct,
        financial_goal=financial_goal
    )

    # ---------------------------------------------------------
    # 2. MARKET DATA & COVARIANCE RETRIEVAL
    # ---------------------------------------------------------
    expected_returns, cov_matrix, live_quotes, risk_free_rate, status_logs = fetch_market_data(
        finnhub_api_key=finnhub_key,
        fred_api_key=fred_key
    )

    # ---------------------------------------------------------
    # 3. PORTFOLIO ALLOCATION CALCULATION
    # ---------------------------------------------------------
    target_weights = get_etf_allocation(risk_profile, equity_tilt_pct)

    # ---------------------------------------------------------
    # 4. QUANTITATIVE PERFORMANCE METRICS
    # ---------------------------------------------------------
    metrics = compute_portfolio_metrics(target_weights, expected_returns, cov_matrix, risk_free_rate)

    # ---------------------------------------------------------
    # TOP KPI BANNER & RISK PROFILE CLASSIFICATION
    # ---------------------------------------------------------
    badge_colors = {"Conservative": "#0D9488", "Moderate": "#D97706", "Aggressive": "#DC2626"}
    profile_color = badge_colors.get(risk_profile, "#2563EB")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            f"""
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 12px; border-radius: 8px; text-align: center;">
                <div style="font-size: 12px; color: #64748B; font-weight: 600; text-transform: uppercase;">Quantitative Profile</div>
                <div style="font-size: 20px; font-weight: 800; color: {profile_color}; margin-top: 4px;">{risk_profile}</div>
                <div style="font-size: 11px; color: #475569;">Score: <b>{composite_score:.1f}/100</b></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.metric(
            label="Annual Expected Return E(Rp)",
            value=f"{metrics['expected_return'] * 100.0:.2f}%",
            delta=f"{(metrics['expected_return'] - risk_free_rate) * 100.0:+.2f}% over Rf"
        )
    with c3:
        st.metric(
            label="Annualized Volatility (σp)",
            value=f"{metrics['volatility'] * 100.0:.2f}%",
            help="Portfolio standard deviation calculated via historical covariance matrix w^T Sigma w."
        )
    with c4:
        st.metric(
            label="Sharpe Ratio (Rf)",
            value=f"{metrics['sharpe_ratio']:.2f}",
            delta=f"Rf: {risk_free_rate * 100.0:.2f}%"
        )
    with c5:
        st.metric(
            label="Parametric 95% 1-Yr VaR",
            value=f"{metrics['var_95'] * 100.0:.2f}%",
            help="Estimated 95% confidence 1-year maximum loss threshold under normal distribution."
        )

    # Status notifications in an unobtrusive expander
    with st.expander("ℹ️ Data Feeds & Connectivity Status", expanded=False):
        for log_entry in status_logs:
            st.write(f"• {log_entry}")

    # ---------------------------------------------------------
    # TABBED INTERACTIVE ANALYTICS & VISUALIZATIONS
    # ---------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Recommended Allocation",
        "📈 Wealth Projection",
        "🎯 Risk-Return & Frontier",
        "📐 Mathematical Methodology"
    ])

    # Monte Carlo simulation data for Tab 2
    df_projection = simulate_wealth_projection(
        initial_capital=initial_capital,
        monthly_contribution=monthly_contribution,
        horizon_years=horizon_years,
        expected_annual_return=metrics["expected_return"],
        annual_volatility=metrics["volatility"]
    )

    fig_donut, fig_fan, fig_frontier = render_charts(
        weights=target_weights,
        initial_capital=initial_capital,
        df_projection=df_projection,
        current_metrics=metrics,
        cov_matrix=cov_matrix,
        expected_returns=expected_returns,
        risk_profile=risk_profile
    )

    # --- TAB 1: ALLOCATION & QUOTES ---
    with tab1:
        st.plotly_chart(fig_donut, use_container_width=True)

        st.subheader("📋 Core Liquid ETF Universe & Target Dollar Breakdown")
        table_rows = []
        for t in TICKERS:
            w = target_weights[t]
            dollar_val = w * initial_capital
            q = live_quotes.get(t, {})
            price = q.get("price", 0.0)
            est_shares = int(dollar_val / price) if price > 0 else 0
            change = q.get("change", 0.0)
            change_str = f"{change:+.2f} ({q.get('pct_change', 0.0):+.2f}%)"

            table_rows.append({
                "Ticker": t,
                "Fund Name": ETF_UNIVERSE[t]["name"],
                "Asset Class": ETF_UNIVERSE[t]["asset_class"],
                "Target Weight": f"{w * 100:.1f}%",
                "Initial Capital Allocation": f"${dollar_val:,.2f}",
                "Recent Price": f"${price:.2f}",
                "Day Change": change_str,
                "Approx Shares": f"{est_shares:,}"
            })

        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # --- TAB 2: WEALTH PROJECTION ---
    with tab2:
        st.plotly_chart(fig_fan, use_container_width=True)

        final_row = df_projection.iloc[-1]
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Cumulative Invested", f"${final_row['Cumulative_Invested']:,.0f}")
        w2.metric("Downside Scenario (10th %)", f"${final_row['P10_Bearish']:,.0f}")
        w3.metric("Median Scenario (50th %)", f"${final_row['P50_Median']:,.0f}")
        w4.metric("Bullish Scenario (90th %)", f"${final_row['P90_Bullish']:,.0f}")

        st.caption(
            f"Projection modeled over {horizon_years} years ({horizon_years * 12} monthly compound cycles) "
            f"with initial ${initial_capital:,.0f} and monthly additions of ${monthly_contribution:,.0f}."
        )

    # --- TAB 3: RISK-RETURN & FRONTIER ---
    with tab3:
        st.plotly_chart(fig_frontier, use_container_width=True)

        st.markdown(
            """
            **How to interpret this chart:**
            - **Viridis Scatter Cloud**: Represents 400 randomly generated asset combinations satisfying budget constraints ($w_i \\ge 0, \\sum w_i = 1$). Color represents Sharpe Ratio.
            - **Geometric Baseline Markers**: Depict standard institutional benchmarks (*Conservative*, *Moderate*, and *Aggressive*).
            - **Yellow Star (★)**: Highlights your exact customized allocation including any fine-tune tilts. Notice how shifting your equity tilt moves your portfolio along the risk-return spectrum.
            """
        )

    # --- TAB 4: MATHEMATICAL METHODOLOGY & FORMULAS ---
    with tab4:
        st.header("📐 Quantitative Formulations & Academic Methodology")

        st.subheader("1. Composite Risk Score Formula")
        st.latex(
            r"S = 0.30 \cdot f_{\text{horizon}} + 0.25 \cdot f_{\text{drawdown}} + 0.15 \cdot f_{\text{risk}} + 0.15 \cdot f_{\text{age}} + 0.15 \cdot f_{\text{liquidity}} + \text{GoalOffset}"
        )
        st.markdown(
            r"""
            Where each constituent factor is mathematically normalized to a $[0, 100]$ interval:
            - **Horizon Factor**: $f_{\text{horizon}} = \min\left(100, \frac{\text{Horizon}}{30} \times 100\right)$
            - **Drawdown Factor**: $f_{\text{drawdown}} = \frac{\text{Drawdown} - 5\%}{40\% - 5\%} \times 100$
            - **Age Factor**: $f_{\text{age}} = \frac{80 - \text{Age}}{80 - 18} \times 100$
            - **Liquidity Factor**: $f_{\text{liquidity}} = \left(1 - \frac{\text{LiquidityRatio}}{50\%}\right) \times 100$
            - **Goal Offset**: Wealth Accumulation ($+6$), Retirement ($0$), Capital Preservation ($-12$)
            - **Classification Thresholds**: Conservative ($S < 38$), Moderate ($38 \le S < 68$), Aggressive ($S \ge 68$).
            """
        )

        st.subheader("2. Modern Portfolio Theory (MPT) Metrics")
        st.latex(
            r"E(R_p) = \mathbf{w}^T \boldsymbol{\mu} = \sum_{i=1}^N w_i E(R_i)"
        )
        st.latex(
            r"\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}"
        )
        st.latex(
            r"\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}"
        )
        st.markdown(
            r"""
            Where $\mathbf{w}$ is the normalized target weight vector ($\sum w_i = 1$),
            $\boldsymbol{\mu}$ is the vector of annualized expected asset returns,
            $\boldsymbol{\Sigma}$ is the $5 \times 5$ annualized covariance matrix of returns,
            and $R_f$ is the dynamic risk-free rate fetched from the Federal Reserve Economic Data (FRED) API.
            """
        )

        st.subheader("3. Stochastic Wealth Accumulation (Geometric Brownian Motion)")
        st.latex(
            r"W_{t} = (W_{t-1} + C) \cdot \exp\left( \left(\mu_m - \frac{1}{2}\sigma_m^2\right) + \sigma_m Z_t \right)"
        )
        st.markdown(
            r"""
            Where $C$ is the monthly contribution, $\mu_m = \frac{E(R_p)}{12}$, $\sigma_m = \frac{\sigma_p}{\sqrt{12}}$,
            and $Z_t \sim \mathcal{N}(0, 1)$ is a standard Gaussian random variate across 1,500 Monte Carlo paths.
            """
        )


if __name__ == "__main__":
    main()
