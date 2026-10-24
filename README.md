# AI Robo-Advisor and Portfolio Recommendation Engine
**Course:** FIN-671 | Advanced Financial Technology & Quantitative Asset Management  
**Project:** Assignment 4 — Quantitative Robo-Advisor & Portfolio Illustration Tool  
**Technologies:** Python 3.10+, Streamlit, Plotly, NumPy, Pandas, Requests, yFinance  

---

> [!WARNING]
> ### ⚠️ Mandatory Legal & Educational Disclaimer
> **FOR EDUCATIONAL ILLUSTRATIONS ONLY — NOT PERSONALIZED INVESTMENT ADVICE.**  
> This software is an academic engineering prototype developed exclusively for the university curriculum of **FIN-671**. All calculations, asset allocation models, risk profiles, and stochastic wealth projections are provided strictly for illustrative, pedagogical, and simulation purposes.  
> - It does **not** constitute legal, tax, accounting, or personalized financial advice.
> - No fiduciary relationship is established by running or interacting with this software.
> - Past performance, historical ETF returns, and Capital Market Assumptions (CMA) are **not** guarantees of future performance. Real-world investing involves the risk of loss, including the possible loss of principal.

---

## 📌 Executive Summary & Academic Objective

Modern robo-advisory platforms (e.g., Wealthfront, Betterment) operationalize quantitative portfolio management by marrying **behavioral risk elicitation** with **Harry Markowitz's Modern Portfolio Theory (MPT)**. Traditional retail questionnaires often rely on ambiguous psychological prompts; in contrast, this tool implements a **strictly numerical, multi-factor risk profiling engine** that maps investor constraints (age, investment horizon, maximum drawdown threshold, liquidity buffer, and self-assessed risk score) directly into an asset allocation framework.

The engine dynamically recommends and balances a core basket of five highly liquid exchange-traded funds (ETFs) spanning domestic equities, growth tech, international equities, intermediate bonds, and short-term Treasuries.

```
       [ Quantitative User Inputs ]
   (Age, Horizon, Drawdown %, Buffer %)
                    │
                    ▼
   [ Multi-Factor Risk Scoring Engine ]
      Composite Score S ∈ [0, 100]
                    │
   ┌────────────────┼────────────────┐
   ▼                ▼                ▼
Conservative     Moderate        Aggressive
 (0 ≤ S < 38)   (38 ≤ S < 68)   (68 ≤ S ≤ 100)
   │                │                │
   └────────────────┼────────────────┘
                    │
                    ▼
     [ Core ETF Allocation Engine ]
  (SPY, QQQ, VEA, AGG, SHY + Fine-Tune Tilt)
                    │
   ┌────────────────┴────────────────┐
   ▼                                 ▼
[ Modern Portfolio Theory ]   [ Stochastic Wealth Simulation ]
 • Expected Return E(Rp)       • Monte Carlo (1,500 paths)
 • Covariance Matrix Σ         • 10th, 50th, 90th percentiles
 • Portfolio Volatility σp     • Monthly contributions & compounding
 • Dynamic Sharpe Ratio (Rf)   • Cost basis tracking
```

---

## 🧮 Quantitative Methodology & Financial Formulations

### 1. Multi-Factor Risk Scoring Model
To eliminate ambiguity, user inputs are strictly numerical. Each variable is mapped into a normalized sub-score $f \in [0, 100]$:

1. **Investment Horizon Factor ($f_{\text{horizon}}$)**: Longer time horizons grant superior capacity to recover from drawdown events:
   $$f_{\text{horizon}} = \min\left(100, \frac{\text{Horizon}}{30} \times 100\right)$$

2. **Loss Tolerance / Drawdown Factor ($f_{\text{drawdown}}$)**: Quantifies the investor's maximum acceptable annual peak-to-trough decline (calibrated across $[5\%, 40\%]$):
   $$f_{\text{drawdown}} = \left(\frac{\text{Drawdown} - 5\%}{40\% - 5\%}\right) \times 100$$

3. **Psychometric Risk Score Factor ($f_{\text{risk}}$)**: Direct numerical preference slider ($1 \text{ to } 10$):
   $$f_{\text{risk}} = \left(\frac{\text{Score} - 1}{9}\right) \times 100$$

4. **Age Factor ($f_{\text{age}}$)**: Proxy for human capital duration and remaining career earnings (calibrated for ages $18 \text{ to } 80$):
   $$f_{\text{age}} = \left(\frac{80 - \text{Age}}{80 - 18}\right) \times 100$$

5. **Liquidity Need Factor ($f_{\text{liquidity}}$)**: Reflects emergency reserve constraints. High liquidity needs reduce risk-bearing capacity (calibrated across $[0\%, 50\%]$):
   $$f_{\text{liquidity}} = \left(1.0 - \frac{\text{Liquidity Need}}{50\%}\right) \times 100$$

6. **Primary Financial Goal Offset ($\Delta_G$)**:
   $$\Delta_G = \begin{cases} +6.0 & \text{Wealth Accumulation} \\ 0.0 & \text{Retirement} \\ -12.0 & \text{Capital Preservation} \end{cases}$$

#### Composite Score Formulation:
$$S = 0.30 \cdot f_{\text{horizon}} + 0.25 \cdot f_{\text{drawdown}} + 0.15 \cdot f_{\text{risk}} + 0.15 \cdot f_{\text{age}} + 0.15 \cdot f_{\text{liquidity}} + \Delta_G$$
$$\text{Composite Risk Score} = \text{clip}(S, 0.0, 100.0)$$

#### Classification Boundaries:
- **Conservative** ($0 \le S < 38$): Focus on capital preservation, drawdown defense, and income generation.
- **Moderate** ($38 \le S < 68$): Balanced strategy balancing long-term purchasing power expansion with dampening volatility.
- **Aggressive** ($68 \le S \le 100$): Growth-oriented profile prioritizing equity appreciation over full market cycles.

---

### 2. Core Liquid ETF Universe & Baseline Allocations

The platform utilizes five ultra-liquid, low-cost institutional building blocks:

| Ticker | ETF Full Name | Asset Class | Category | Expense Ratio | Role in Portfolio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SPY** | SPDR S&P 500 ETF Trust | US Large-Cap Equity | Core Equity | 0.09% | Core economic engine & corporate earnings driver |
| **QQQ** | Invesco QQQ Trust (Nasdaq-100) | US Tech / Large Growth | Growth Equity | 0.20% | High-beta technological innovation tilt |
| **VEA** | Vanguard FTSE Developed Markets | Int'l Developed Equity | Foreign Equity | 0.06% | Geographic and sovereign currency diversification |
| **AGG** | iShares Core U.S. Aggregate Bond | Total US Investment Grade | Fixed Income | 0.03% | Income yield and deflationary equity hedge |
| **SHY** | iShares 1-3 Year Treasury Bond | Short-Term US Treasury | Cash Equivalent | 0.15% | Liquidity buffer and capital protection anchor |

#### Baseline Strategic Asset Allocations:
- **Conservative**: 30% Equities (`SPY`: 15%, `QQQ`: 5%, `VEA`: 10%) | 70% Fixed Income (`AGG`: 45%, `SHY`: 25%)
- **Moderate**: 60% Equities (`SPY`: 35%, `QQQ`: 15%, `VEA`: 10%) | 40% Fixed Income (`AGG`: 30%, `SHY`: 10%)
- **Aggressive**: 85% Equities (`SPY`: 50%, `QQQ`: 25%, `VEA`: 10%) | 15% Fixed Income (`AGG`: 10%, `SHY`: 5%)

#### Fine-Tune Tilt Mechanism:
Users can adjust the **Fine-Tune Tilt** slider ($\Delta_{\text{tilt}} \in [-25\%, +25\%]$). The engine shifts total equity exposure while proportionately rebalancing constituent weights, guaranteeing that portfolio weights strictly sum to 100.00% ($\sum w_i = 1.0$) with zero negative weights ($w_i \ge 0$).

---

### 3. Financial Analysis Operations & Mathematical Metrics

#### Calculation 1: Annualized Expected Portfolio Return ($E(R_p)$)
Given target asset weights $\mathbf{w} = [w_1, w_2, \dots, w_N]^T$ and annualized expected asset return vector $\boldsymbol{\mu}$:
$$E(R_p) = \mathbf{w}^T \boldsymbol{\mu} = \sum_{i=1}^{N} w_i E(R_i)$$
Expected returns incorporate a blended average of trailing empirical performance and institutional Capital Market Assumptions (CMA):
$$\boldsymbol{\mu}_{\text{blended}} = 0.50 \cdot \boldsymbol{\mu}_{\text{hist}} + 0.50 \cdot \boldsymbol{\mu}_{\text{CMA}}$$

#### Calculation 2: Annualized Portfolio Volatility ($\sigma_p$)
Portfolio variance accounts for the full covariance matrix of asset returns $\boldsymbol{\Sigma}$:
$$\sigma_p^2 = \mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w} = \sum_{i=1}^N \sum_{j=1}^N w_i w_j \sigma_{ij}$$
$$\sigma_p = \sqrt{\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}}$$
Where $\boldsymbol{\Sigma} = 252 \cdot \text{Cov}(\mathbf{R}_{\text{daily}})$ is the annualized covariance matrix derived from daily historical price data.

#### Calculation 3: Dynamic Portfolio Sharpe Ratio ($\text{SR}_p$)
Measures excess return earned per unit of total risk, dynamically utilizing the live or benchmark 3-Month Treasury yield ($R_f$):
$$\text{Sharpe Ratio} = \frac{E(R_p) - R_f}{\sigma_p}$$

#### Calculation 4: Parametric 1-Year Value-at-Risk ($\text{VaR}_{95\%}$)
Quantifies the minimum loss threshold expected in the worst 5% of annual holding periods under normal market distributions ($Z_{0.95} = 1.645$):
$$\text{VaR}_{95\%} = \max\left(0.0, 1.645 \cdot \sigma_p - E(R_p)\right)$$

#### Calculation 5: Multi-Year Stochastic Wealth Accumulation (Monte Carlo)
To realistically model portfolio growth under volatility drag with regular monthly savings, the engine executes a discrete monthly lognormal Geometric Brownian Motion (GBM) simulation over $M = 1,500$ paths:
$$\mu_m = \frac{E(R_p)}{12}, \quad \sigma_m = \frac{\sigma_p}{\sqrt{12}}$$
$$W_t^{(k)} = \left(W_{t-1}^{(k)} + C\right) \cdot \exp\left(\left(\mu_m - \frac{1}{2}\sigma_m^2\right) + \sigma_m Z_t^{(k)}\right)$$
Where $C$ is the monthly contribution and $Z_t^{(k)} \sim \mathcal{N}(0, 1)$.  
At each month $t \in [0, 12 \cdot T]$, the distribution across all 1,500 runs is evaluated to obtain:
- **10th Percentile ($P_{10}$)**: Downside bear market trajectory.
- **50th Percentile ($P_{50}$)**: Expected median wealth trajectory.
- **90th Percentile ($P_{90}$)**: Optimistic bull market trajectory.
- **Cumulative Invested Capital ($P_t = W_0 + C \cdot t$)**: Cost basis benchmark.

---

## 🌐 Data Integration & Multi-Tier Resilience Architecture

The application implements an institutional, fail-safe data pipeline designed to execute smoothly under any network or API constraint:

```
                  [ fetch_market_data() ]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [ Macro Yield (Rf) ]              [ ETF Market Quotes ]
            │                                 │
     Is FRED Key Set?               Is Finnhub Key Set?
      ├── YES ──► FRED API (DTB3)     ├── YES ──► Finnhub Quote API
      └── NO  ──► 4.25% Benchmark     └── NO  ──► Yahoo Finance (yfinance)
                                                       │
                                              Did yfinance Succeed?
                                               ├── YES ──► Live Close & Covariance
                                               └── NO  ──► Empirical Covariance Matrix
```

1. **FRED API (Federal Reserve Bank of St. Louis)**:
   - Queries series `DTB3` (3-Month Treasury Bill Secondary Market Rate).
   - *Fallback:* Automatically uses default benchmark $R_f = 4.25\%$ if the key is empty or network times out.
2. **Finnhub API**:
   - Queries endpoint `/api/v1/quote?symbol={ticker}` for real-time bid, ask, and daily percentage change.
   - *Fallback:* If omitted or rate-limited, smoothly cascades to Yahoo Finance.
3. **Yahoo Finance (`yfinance`)**:
   - Downloads 3 years of daily adjusted close prices for all five tickers to estimate the dynamic covariance matrix $\boldsymbol{\Sigma}$.
   - Cached locally via `@st.cache_data(ttl=3600)`.
4. **Offline Robust Empirical Matrix**:
   - If completely disconnected from the internet, the app activates a pre-computed empirical 5-year annualized covariance matrix and Capital Market Assumptions. The tool **never crashes** due to network issues.

---

## 🖼️ Application Interface & Interactive Visualizations

### Visualizations Included:
1. **Interactive Donut Chart (Plotly)**: Visualizes asset allocation breakdown between Equities and Fixed Income, showing individual ETF percentage weights and initial dollar allocations ($w_i \times W_0$).
2. **Compound Wealth Fan Chart (Plotly)**: Shows stochastic wealth trajectories across $10\text{th}$, $50\text{th}$, and $90\text{th}$ percentiles with a shaded confidence corridor, plotted against cumulative invested capital.
3. **Risk-Return Scatter / Efficient Frontier (Plotly)**: Plots a cloud of 400 simulated portfolios in $(\sigma, E(R))$ space colored by Sharpe ratio, marking institutional baselines alongside a prominent gold star denoting the user's custom portfolio.

### Application Screenshots:

#### 1. Main Dashboard & Target Asset Allocation
![Main Dashboard Overview](screenshots/dashboard_overview.png)
*Placeholder: Place screenshot showing numerical sidebar, KPI cards, and asset allocation donut chart.*

#### 2. Risk-Return Mapping & Efficient Frontier
![Efficient Frontier and Stochastic Wealth Fan Chart](screenshots/efficient_frontier.png)
*Placeholder: Place screenshot showing the efficient frontier scatter plot and Monte Carlo wealth fan chart.*

---

## 🚀 Step-by-Step Installation & Reproducibility Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14.
- Git and a command terminal (Bash, Zsh, or Windows PowerShell).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/FIN-671-Prototype.git
cd FIN-671-Prototype
```

### 2. Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Or use `py -m venv venv` if using the Python Launcher).*

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
*(Or `python -m streamlit run app.py`).*

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🧪 Automated Testing & Verification

A dedicated verification suite is provided in `scratch/verify_calculations.py` to validate mathematical invariants:
```bash
python scratch/verify_calculations.py
```
Tests include:
- Risk scoring boundary conditions (age 18 vs 80, drawdown 5% vs 40%, goals).
- Strict weight normalization ($\sum w_i = 1.0000$ and $w_i \ge 0.0$ across all tilts).
- Covariance matrix positive semi-definiteness and MPT formulas.
- Stochastic Monte Carlo monotonicity ($P_{10} \le P_{50} \le P_{90}$).
- Zero-crash API fallback resilience.

---

## 🤖 AI-Assisted Development: Google Antigravity Reflection

This project was architected and developed in collaboration with **Google Antigravity**, DeepMind's agentic coding assistant:
1. **Mathematical Formalization**: Antigravity aided in formulating the multi-factor risk score equation ($S \in [0, 100]$) and translating the discrete monthly Geometric Brownian Motion equation into vectorized NumPy operations, enabling 1,500 Monte Carlo paths to run in $<15\text{ ms}$.
2. **Defensive Software Engineering**: Designed the multi-tier API fallback hierarchy (Finnhub $\rightarrow$ yfinance $\rightarrow$ Empirical Covariance Matrix) to ensure uninterrupted operation during university grading and offline demonstrations.
3. **Interactive Visualizations**: Rapidly generated clean, responsive Plotly charts with custom color schemes, hover tooltips, and unified hover modes.

---

## ⚠️ Known Limitations & Future Research Roadmap

1. **Static vs. Dynamic Covariance**: The current prototype employs a constant annualized sample covariance matrix. Future iterations could incorporate **GARCH(1,1)** or **DCC-GARCH** models to capture time-varying volatility clustering.
2. **Fat-Tailed Return Distributions**: The stochastic wealth simulation utilizes lognormal returns. In real-world market crashes, asset returns exhibit excess kurtosis and fat tails; integrating **Student's t-copula** or **Historical Block Bootstrap** would enhance downside tail risk assessment.
3. **Factor Investing & Black-Litterman**: Expanding beyond Markowitz mean-variance optimization to include the **Black-Litterman model** would allow blending subjective macroeconomic views with market equilibrium weights.
4. **Tax-Loss Harvesting & Cash-Flow Drag**: Modeling tax drag (capital gains taxes, qualified dividends) and transaction rebalancing costs for taxable accounts.

---

**Author / Student:** FIN-671 Quantitative Finance Candidate  
**Academic Institution:** Graduate School of Business / Quantitative Finance  
**License:** MIT Open Academic License
