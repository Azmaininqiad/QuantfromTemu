import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
import pymc as pm
import arviz as az
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="QuantTerminal | Stock Probability Engine",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# GLOBAL STYLE — ELITE TRADING TERMINAL THEME
# ==========================================
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-0: #06080d;
        --bg-1: #0b0f17;
        --bg-2: #10151f;
        --line: rgba(255,255,255,0.07);
        --line-strong: rgba(255,255,255,0.14);
        --text-dim: rgba(230,235,245,0.55);
        --text-mid: rgba(230,235,245,0.78);
        --accent-gold: #d4af37;
        --accent-cyan: #2dd4bf;
        --green: #16c784;
        --red: #ef4560;
        --amber: #f0b90b;
        --mono: 'JetBrains Mono', monospace;
        --sans: 'Inter', sans-serif;
    }

    html, body, [class*="css"] { font-family: var(--sans); }

    .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(212,175,55,0.05) 0%, transparent 45%),
            radial-gradient(circle at 85% 15%, rgba(45,212,191,0.05) 0%, transparent 45%),
            var(--bg-0);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] { background: transparent; }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: var(--bg-1);
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] * { font-family: var(--sans); }

    /* ---------- Terminal top bar ---------- */
    .term-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.55rem 1.1rem;
        border: 1px solid var(--line);
        border-radius: 10px;
        background: var(--bg-1);
        margin-bottom: 1.1rem;
        font-family: var(--mono);
        font-size: 0.78rem;
        color: var(--text-dim);
        letter-spacing: 0.04em;
    }
    .term-bar .dot {
        display:inline-block; width:7px; height:7px; border-radius:50%;
        background: var(--green); margin-right:6px;
        box-shadow: 0 0 8px var(--green);
    }

    /* ---------- Hero / ticker header ---------- */
    .app-hero {
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(212,175,55,0.10) 0%, rgba(45,212,191,0.07) 100%);
        border: 1px solid var(--line-strong);
        margin-bottom: 1.3rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }
    .app-hero .tick-symbol {
        font-family: var(--mono);
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: #f4f1ea;
        line-height: 1;
    }
    .app-hero .tick-sub {
        font-family: var(--sans);
        font-size: 0.82rem;
        color: var(--text-dim);
        margin-top: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .app-hero .price-block { text-align: right; font-family: var(--mono); }
    .app-hero .price-main { font-size: 2rem; font-weight: 700; color: #f4f1ea; }
    .app-hero .price-chg { font-size: 1rem; font-weight: 600; margin-top: 0.15rem; }
    .up { color: var(--green); }
    .down { color: var(--red); }

    /* ---------- Section headers ---------- */
    .sec-head {
        display:flex; align-items:center; gap:0.6rem;
        margin: 2.1rem 0 0.85rem 0;
    }
    .sec-head .bar {
        width: 4px; height: 20px; border-radius: 2px;
        background: linear-gradient(180deg, var(--accent-gold), var(--accent-cyan));
    }
    .sec-head h3 {
        font-size: 1.05rem; font-weight: 700; margin: 0;
        color: #f0ede4; letter-spacing: 0.01em;
    }
    .sec-head .tag {
        font-family: var(--mono); font-size: 0.68rem; color: var(--text-dim);
        border: 1px solid var(--line-strong); border-radius: 6px;
        padding: 0.12rem 0.5rem; text-transform: uppercase; letter-spacing: 0.06em;
    }
    .sec-desc { font-size: 0.86rem; color: var(--text-dim); margin: -0.4rem 0 0.9rem 0; max-width: 900px; }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: var(--bg-2);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        height: 100%;
        transition: border-color 0.15s ease;
    }
    .metric-card:hover { border-color: var(--line-strong); }
    .metric-card .label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.07em;
        color: var(--text-dim); margin-bottom: 0.45rem; font-weight: 600;
    }
    .metric-card .value {
        font-family: var(--mono); font-size: 1.55rem; font-weight: 700; color: #f4f1ea;
    }
    .metric-card .sub { font-size: 0.74rem; color: var(--text-dim); margin-top: 0.3rem; }

    /* ---------- Recommendation / signal boxes ---------- */
    .rec-box {
        border-radius: 12px; padding: 0.95rem 1.3rem; font-size: 0.94rem;
        border: 1px solid var(--line); margin: 0.6rem 0 1rem 0;
        font-family: var(--sans);
    }
    .rec-invest  { background: rgba(22,199,132,0.08);  border-color: rgba(22,199,132,0.35); }
    .rec-caution { background: rgba(240,185,11,0.08);  border-color: rgba(240,185,11,0.35); }
    .rec-avoid   { background: rgba(239,69,96,0.08);   border-color: rgba(239,69,96,0.35); }

    /* ---------- Data table styling ---------- */
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; overflow: hidden; }

    .ticker-chip button { border-radius: 8px !important; font-family: var(--mono) !important; }

    div[data-testid="stMetricValue"] { font-weight: 700; font-family: var(--mono); }

    hr { border-color: var(--line) !important; }

    /* ---------- Sidebar widget spacing (prevents label/tooltip overlap) ---------- */
    section[data-testid="stSidebar"] div[data-testid="stTextInput"],
    section[data-testid="stSidebar"] div[data-testid="stSlider"],
    section[data-testid="stSidebar"] div[data-testid="stExpander"] {
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] label { position: relative; z-index: 1; }

    /* ---------- Growth leaderboard ---------- */
    .lb-badge {
        display:inline-block; font-family: var(--mono); font-size: 0.7rem;
        color: var(--bg-0); background: var(--accent-gold);
        border-radius: 6px; padding: 0.1rem 0.45rem; font-weight: 700; margin-right: 0.4rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def section_header(title: str, tag: str, desc: str = ""):
    st.markdown(
        f"""
        <div class="sec-head"><div class="bar"></div><h3>{title}</h3>
            <span class="tag">{tag}</span>
        </div>
        {f'<div class="sec-desc">{desc}</div>' if desc else ''}
        """,
        unsafe_allow_html=True,
    )


PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, l=10, r=10, b=10),
    font=dict(family="Inter, sans-serif", color="rgba(230,235,245,0.85)"),
)

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "AAPL"

# ==========================================
# TOP TERMINAL BAR
# ==========================================
st.markdown(
    f"""
    <div class="term-bar">
        <div><span class="dot"></span>LIVE SESSION &nbsp;|&nbsp; DATA FEED: YAHOO FINANCE</div>
        <div>QUANT-PROB ENGINE v2.0</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# TOP GROWTH SCREENER — TOP 10 PICKS
# ==========================================
GROWTH_UNIVERSE = [
    "NVDA", "MSFT", "AAPL", "GOOGL", "AMZN", "META", "TSLA", "AVGO", "AMD", "CRM",
    "ADBE", "NFLX", "ORCL", "SHOP", "PANW", "CRWD", "SNOW", "PLTR", "UBER", "ABNB",
    "NOW", "INTU", "ISRG", "LLY", "COST", "V", "MA", "ASML", "TSM", "SMCI",
]


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_growth_leaderboard(tickers: list) -> pd.DataFrame:
    rows = []
    for t in tickers:
        try:
            info = yf.Ticker(t).get_info()
        except Exception:
            continue
        rev_g = info.get("revenueGrowth")
        earn_g = info.get("earningsGrowth")
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        name = info.get("shortName", t)
        sector = info.get("sector", "—")
        scores = [x for x in [rev_g, earn_g] if isinstance(x, (int, float))]
        if not scores:
            continue
        growth_score = float(np.mean(scores))
        rows.append({
            "Ticker": t, "Company": name, "Sector": sector,
            "Revenue Growth (YoY)": rev_g, "Earnings Growth (YoY)": earn_g,
            "Growth Score": growth_score, "Price": price,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("Growth Score", ascending=False).head(10).reset_index(drop=True)


section_header(
    "Top 10 Growth Picks", "SCREENER · FUNDAMENTALS",
    "Ranked by trailing revenue &amp; earnings growth (YoY), pulled live from Yahoo Finance across a curated "
    "large-cap / growth-sector universe. This is a fixed-list screen, not a full-market scan, and it is "
    "not investment advice — treat it as a starting point for your own research."
)

with st.spinner("Screening growth universe..."):
    try:
        growth_board = load_growth_leaderboard(GROWTH_UNIVERSE)
    except Exception:
        growth_board = pd.DataFrame()

if growth_board.empty:
    st.info("Growth screener data is temporarily unavailable — try again shortly.")
else:
    display_board = growth_board.copy()
    display_board.index = display_board.index + 1
    display_board["Revenue Growth (YoY)"] = display_board["Revenue Growth (YoY)"].map(
        lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—")
    display_board["Earnings Growth (YoY)"] = display_board["Earnings Growth (YoY)"].map(
        lambda x: f"{x * 100:.1f}%" if isinstance(x, (int, float)) else "—")
    display_board["Growth Score"] = display_board["Growth Score"].map(lambda x: f"{x * 100:.1f}%")
    display_board["Price"] = display_board["Price"].map(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else "—")

    st.dataframe(
        display_board[["Ticker", "Company", "Sector", "Revenue Growth (YoY)",
                        "Earnings Growth (YoY)", "Growth Score", "Price"]],
        use_container_width=True,
    )

    st.caption("Load any of these straight into the analysis engine below:")
    lb_cols = st.columns(5)
    for idx, row in growth_board.iterrows():
        with lb_cols[idx % 5]:
            if st.button(row["Ticker"], key=f"lb_pick_{row['Ticker']}", use_container_width=True):
                st.session_state.ticker_input = row["Ticker"]
                st.rerun()

st.divider()

# ==========================================
# SIDEBAR — INPUTS
# ==========================================
st.sidebar.markdown("### ⚙️ Terminal Settings")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value=st.session_state.ticker_input,
    help="Enter a valid ticker symbol, e.g. AAPL, MSFT, TSLA.",
    key="main_ticker_input",
).strip().upper()

st.sidebar.caption("Quick picks")
chip_cols = st.sidebar.columns(4)
for chip_col, chip_ticker in zip(chip_cols, ["AAPL", "MSFT", "NVDA", "TSLA"]):
    with chip_col:
        if st.button(chip_ticker, key=f"chip_{chip_ticker}", use_container_width=True):
            st.session_state.ticker_input = chip_ticker
            st.rerun()

st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

years = st.sidebar.slider("Years of Historical Data", min_value=1, max_value=5, value=2, key="years_slider")

st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

benchmark_ticker = st.sidebar.text_input(
    "Benchmark (for Beta / Joint analysis)",
    value="SPY",
    key="benchmark_ticker_input",
    help="Ticker used as the market benchmark for the beta/correlation section.",
).strip().upper()

st.sidebar.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

with st.sidebar.expander("ℹ️ Model glossary"):
    st.markdown(
        """
- **MLE** — single "best guess" Normal fit to historical returns.
- **Bayesian** — full posterior distribution over the true mean return (MCMC via PyMC).
- **Frequentist CI / t-test** — classical confidence interval & significance test.
- **Chernoff / tail bound** — probability of a large single-day loss.
- **Poisson process** — models the arrival rate of extreme "jump" days.
- **Markov chain** — bull / bear / sideways regime transition probabilities.
- **GBM Monte Carlo** — Brownian-motion-based forward price simulation.
- **Joint RV / Beta** — linear estimation of stock returns given benchmark returns.
- This is an educational statistics demo, **not investment advice**.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Data source: Yahoo Finance via `yfinance`")

# ==========================================
# DATA LOADING
# ==========================================
@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_data(symbol: str, yrs: int) -> pd.DataFrame:
    data = yf.download(symbol, period=f"{yrs}y", progress=False, auto_adjust=True)

    if data is None or data.empty:
        raise ValueError(f"No price data returned for '{symbol}'. Check the ticker symbol.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if "Close" not in data.columns:
        raise ValueError(f"No 'Close' price column found for '{symbol}'.")

    data["Returns"] = data["Close"].pct_change()
    data = data.dropna(subset=["Returns"])

    if len(data) < 30:
        raise ValueError(
            f"Only {len(data)} usable data points found for '{symbol}'. "
            "Try a longer time range or a different ticker."
        )

    return data


@st.cache_data(ttl=60 * 60, show_spinner=False)
def run_bayesian_model(returns: np.ndarray):
    with pm.Model():
        bayes_mu = pm.Normal("mean_return", mu=0.0, sigma=0.01)
        bayes_sigma = pm.HalfNormal("volatility", sigma=0.05)
        pm.Normal("y", mu=bayes_mu, sigma=bayes_sigma, observed=returns)
        trace = pm.sample(
            draws=500,
            tune=500,
            chains=2,
            cores=1,
            progressbar=False,
            return_inferencedata=True,
            random_seed=42,
        )

    summary = az.summary(trace, hdi_prob=0.94)
    hdi_low = summary.loc["mean_return", "hdi_3%"]
    hdi_high = summary.loc["mean_return", "hdi_97%"]
    bayes_mu_mean = summary.loc["mean_return", "mean"]
    posterior_samples = trace.posterior["mean_return"].values.flatten()

    return bayes_mu_mean, hdi_low, hdi_high, posterior_samples


@st.cache_data(ttl=60 * 60, show_spinner=False)
def run_gbm_simulation(last_price: float, mu_daily: float, sigma_daily: float,
                        n_days: int = 60, n_sims: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_days, n_sims))
    daily_log_returns = (mu_daily - 0.5 * sigma_daily ** 2) + sigma_daily * z
    price_paths = last_price * np.exp(np.cumsum(daily_log_returns, axis=0))
    return price_paths


# ==========================================
# MAIN LOGIC
# ==========================================
if not ticker:
    st.info("Enter a stock ticker in the sidebar to get started.")
    st.stop()

try:
    with st.spinner(f"Downloading price history for {ticker}..."):
        stock_data = load_data(ticker, years)

    returns = stock_data["Returns"].to_numpy()
    n = len(returns)

    last_close = float(stock_data["Close"].iloc[-1])
    prev_close = float(stock_data["Close"].iloc[-2])
    day_chg = (last_close / prev_close - 1) * 100
    chg_class = "up" if day_chg >= 0 else "down"
    chg_arrow = "▲" if day_chg >= 0 else "▼"

    # ---- MLE ----
    mle_mu, mle_sigma = stats.norm.fit(returns)

    # ---- Bayesian ----
    with st.spinner("Running Bayesian MCMC sampling..."):
        bayes_mu_mean, hdi_low, hdi_high, posterior_mu_samples = run_bayesian_model(returns)

    # ==========================================
    # HERO / TICKER HEADER
    # ==========================================
    st.markdown(
        f"""
        <div class="app-hero">
            <div>
                <div class="tick-symbol">{ticker}</div>
                <div class="tick-sub">Quantitative Probability &amp; Prediction Engine</div>
            </div>
            <div class="price-block">
                <div class="price-main">${last_close:,.2f}</div>
                <div class="price-chg {chg_class}">{chg_arrow} {abs(day_chg):.2f}% &nbsp;(last session)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==========================================
    # 1. MLE vs BAYESIAN ESTIMATES
    # ==========================================
    section_header(
        "Model Estimates", "MLE · BAYESIAN",
        "Comparing a classic point estimate against a full posterior belief distribution over the true average daily return."
    )

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">MLE Best Guess (Daily)</div>
                    <div class="value">{mle_mu * 100:.4f}%</div>
                    <div class="sub">Single point estimate, no uncertainty</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Bayesian Expected Return (Daily)</div>
                    <div class="value">{bayes_mu_mean * 100:.4f}%</div>
                    <div class="sub">Posterior mean</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">94% Credible Interval</div>
                    <div class="value">{hdi_low * 100:.3f}% → {hdi_high * 100:.3f}%</div>
                    <div class="sub">Range of plausible true average returns</div>
                </div>""",
            unsafe_allow_html=True,
        )

    st.write("")

    if hdi_low > 0:
        st.markdown(
            '<div class="rec-box rec-invest">🟢 <b>Strong signal:</b> the entire 94% '
            "credible interval sits above zero — historical daily returns have been "
            "consistently positive over this window.</div>",
            unsafe_allow_html=True,
        )
    elif hdi_high < 0:
        st.markdown(
            '<div class="rec-box rec-avoid">🔴 <b>Weak signal:</b> the entire 94% '
            "credible interval sits below zero — historical daily returns have been "
            "consistently negative over this window.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="rec-box rec-caution">🟡 <b>Inconclusive:</b> the 94% credible '
            f"interval spans zero ({hdi_low * 100:.2f}% to {hdi_high * 100:.2f}%). "
            "The historical trend is not statistically distinguishable from noise.</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        "⚠️ This tool describes **historical** statistical patterns only. It is not "
        "financial advice and does not predict future performance."
    )

    # ==========================================
    # 2. FREQUENTIST HYPOTHESIS TEST / CI
    # ==========================================
    section_header(
        "Classical Hypothesis Test", "SAMPLE MEAN · CI · t-TEST",
        "H₀: the true mean daily return is zero. A one-sample t-test and 94% confidence interval, "
        "shown alongside the Bayesian credible interval for direct comparison."
    )

    se = returns.std(ddof=1) / np.sqrt(n)
    t_stat = mle_mu / se
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    ci_low, ci_high = stats.t.interval(0.94, df=n - 1, loc=mle_mu, scale=se)
    reject_null = p_value < 0.05

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">t-statistic</div>
                    <div class="value">{t_stat:.3f}</div>
                    <div class="sub">df = {n - 1}</div>
                </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">p-value</div>
                    <div class="value">{p_value:.4f}</div>
                    <div class="sub">{"Reject H₀ at α=0.05" if reject_null else "Fail to reject H₀ at α=0.05"}</div>
                </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">94% Frequentist CI</div>
                    <div class="value">{ci_low * 100:.3f}% → {ci_high * 100:.3f}%</div>
                    <div class="sub">vs. Bayesian: {hdi_low * 100:.3f}% → {hdi_high * 100:.3f}%</div>
                </div>""", unsafe_allow_html=True)

    fig_ci = go.Figure()
    fig_ci.add_trace(go.Scatter(
        x=[ci_low * 100, ci_high * 100], y=["Frequentist t-CI", "Frequentist t-CI"],
        mode="lines+markers", line=dict(color="#f0b90b", width=6), marker=dict(size=10),
        name="Frequentist 94% CI",
    ))
    fig_ci.add_trace(go.Scatter(
        x=[hdi_low * 100, hdi_high * 100], y=["Bayesian HDI", "Bayesian HDI"],
        mode="lines+markers", line=dict(color="#2dd4bf", width=6), marker=dict(size=10),
        name="Bayesian 94% HDI",
    ))
    fig_ci.add_vline(x=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
    fig_ci.update_layout(
        title="Frequentist CI vs. Bayesian Credible Interval", xaxis_title="Daily Return %",
        showlegend=False, height=260, **PLOT_LAYOUT,
    )
    st.plotly_chart(fig_ci, use_container_width=True)

    # ==========================================
    # 3. RAW RETURNS TIME SERIES
    # ==========================================
    section_header("Raw Daily Returns", "TIME SERIES", "The underlying return sequence driving every model below.")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=stock_data.index, y=returns * 100, name="Daily Returns",
        line=dict(color="#6C5CE7", width=1.4),
    ))
    fig1.update_layout(title=f"Raw Daily Returns Over Time — {ticker}", yaxis_title="Daily Return %", **PLOT_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    # ==========================================
    # 4. MLE FIT
    # ==========================================
    section_header("Maximum Likelihood Fit", "DISCRETE/CONT. RV · PDF",
                   "A single best-fit Normal density overlaid on the empirical return distribution.")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=returns * 100, histnorm="probability density", name="Actual Data",
        marker_color="#00CEC9", opacity=0.55,
    ))
    x_range = np.linspace(returns.min() * 100, returns.max() * 100, 200)
    p_mle = stats.norm.pdf(x_range / 100, mle_mu, mle_sigma) / 100
    fig2.add_trace(go.Scatter(x=x_range, y=p_mle, name="MLE Fitted Bell Curve", line=dict(color="#fdcb6e", width=3)))
    fig2.update_layout(title="Maximum Likelihood Estimation (MLE) Fit", xaxis_title="Daily Return %",
                        yaxis_title="Density", **PLOT_LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================
    # 5. BAYESIAN POSTERIOR
    # ==========================================
    section_header("Bayesian Posterior Belief", "MCMC · POSTERIOR",
                   "Full distribution of plausible values for the true mean daily return, sampled via PyMC.")
    fig3 = go.Figure()
    fig3.add_trace(go.Histogram(
        x=posterior_mu_samples * 100, histnorm="probability density", name="Belief Spectrum",
        marker_color="#a29bfe", opacity=0.65,
    ))
    fig3.add_vrect(x0=hdi_low * 100, x1=hdi_high * 100, fillcolor="#fdcb6e", opacity=0.2, line_width=0,
                   annotation_text="94% Credible Zone", annotation_position="top left")
    fig3.update_layout(title="Bayesian Posterior Belief Distribution (Uncertainty)",
                        xaxis_title="True Average Daily Return %", yaxis_title="Belief Density", **PLOT_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    # ==========================================
    # 6. TAIL RISK / CHERNOFF BOUND
    # ==========================================
    section_header(
        "Tail Risk — Large-Loss Probability", "CHERNOFF BOUND · VaR",
        "Empirical vs. Normal-model probability of a single-day loss beyond a threshold — a practical use of tail-bound reasoning."
    )
    loss_threshold_pct = st.slider("Loss threshold (daily return %)", -10.0, -0.5, -3.0, 0.5, key="loss_thresh")
    loss_threshold = loss_threshold_pct / 100
    empirical_p = float((returns < loss_threshold).mean())
    z = (loss_threshold - mle_mu) / mle_sigma
    normal_tail_p = float(stats.norm.cdf(z))

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Empirical P(return &lt; {loss_threshold_pct:.1f}%)</div>
                    <div class="value">{empirical_p * 100:.2f}%</div>
                    <div class="sub">Observed frequency in {n} trading days</div>
                </div>""", unsafe_allow_html=True)
    with t2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Normal-Model Estimate</div>
                    <div class="value">{normal_tail_p * 100:.2f}%</div>
                    <div class="sub">Gaussian tail approximation (MLE fit)</div>
                </div>""", unsafe_allow_html=True)

    # ==========================================
    # 7. POISSON PROCESS — JUMP / EVENT ARRIVALS
    # ==========================================
    section_header(
        "Jump / Event Arrival Process", "POISSON PROCESS",
        "Extreme moves (|return| beyond 3 standard deviations) treated as rare events arriving according to a Poisson process."
    )
    jump_mask = np.abs(returns - mle_mu) > 3 * mle_sigma
    n_jumps = int(jump_mask.sum())
    lam_per_day = n_jumps / n  # events per trading day
    horizon = 21  # ~1 trading month
    p_at_least_one = 1 - np.exp(-lam_per_day * horizon)

    j1, j2, j3 = st.columns(3)
    with j1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Observed Jump Days</div>
                    <div class="value">{n_jumps}</div>
                    <div class="sub">out of {n} trading days (&gt;3σ moves)</div>
                </div>""", unsafe_allow_html=True)
    with j2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Estimated Rate (λ)</div>
                    <div class="value">{lam_per_day:.4f}/day</div>
                    <div class="sub">≈ {lam_per_day * 252:.2f} jumps / year</div>
                </div>""", unsafe_allow_html=True)
    with j3:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">P(≥1 jump in next {horizon} days)</div>
                    <div class="value">{p_at_least_one * 100:.2f}%</div>
                    <div class="sub">Poisson process approximation</div>
                </div>""", unsafe_allow_html=True)

    # ==========================================
    # 8. MARKOV CHAIN — REGIME TRANSITIONS
    # ==========================================
    section_header(
        "Regime Transition Model", "MARKOV CHAIN",
        "Each day classified as Bull, Bear, or Sideways; transition probabilities estimated from the historical sequence."
    )

    def classify_regime(r, thresh=0.005):
        if r > thresh:
            return "Bull"
        elif r < -thresh:
            return "Bear"
        return "Sideways"

    regime_states = [classify_regime(r) for r in returns]
    labels = ["Bull", "Bear", "Sideways"]
    trans_counts = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for a, b in zip(regime_states[:-1], regime_states[1:]):
        trans_counts.loc[a, b] += 1
    row_sums = trans_counts.sum(axis=1).replace(0, np.nan)
    trans_matrix = trans_counts.div(row_sums, axis=0).fillna(0)

    current_state = regime_states[-1]
    next_probs = trans_matrix.loc[current_state]

    rc1, rc2 = st.columns([1.3, 1])
    with rc1:
        st.markdown("**Transition probability matrix** — P(next regime | current regime)")
        st.dataframe(trans_matrix.style.format("{:.1%}"), use_container_width=True)
    with rc2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Current Regime</div>
                    <div class="value">{current_state}</div>
                    <div class="sub">Bull {next_probs['Bull']:.1%} · Bear {next_probs['Bear']:.1%} · Sideways {next_probs['Sideways']:.1%}</div>
                </div>""", unsafe_allow_html=True)

    # ==========================================
    # 9. GBM MONTE CARLO FORWARD SIMULATION
    # ==========================================
    section_header(
        "Forward Price Simulation", "BROWNIAN MOTION · MONTE CARLO",
        "Geometric Brownian Motion paths simulated forward from today's price, using the fitted drift and volatility."
    )
    sim_days = st.slider("Forecast horizon (trading days)", 10, 120, 60, 5, key="gbm_days")
    price_paths = run_gbm_simulation(last_close, mle_mu, mle_sigma, n_days=sim_days, n_sims=2000)

    fig_gbm = go.Figure()
    sample_idx = np.random.default_rng(1).choice(price_paths.shape[1], size=min(60, price_paths.shape[1]), replace=False)
    for i in sample_idx:
        fig_gbm.add_trace(go.Scatter(y=price_paths[:, i], line=dict(width=0.6, color="#6C5CE7"), opacity=0.25, showlegend=False))
    fig_gbm.add_trace(go.Scatter(y=np.median(price_paths, axis=1), line=dict(color="#fdcb6e", width=3), name="Median path"))
    fig_gbm.add_trace(go.Scatter(y=np.percentile(price_paths, 95, axis=1), line=dict(color="#2dd4bf", width=1.5, dash="dot"), name="95th pct"))
    fig_gbm.add_trace(go.Scatter(y=np.percentile(price_paths, 5, axis=1), line=dict(color="#ef4560", width=1.5, dash="dot"), name="5th pct"))
    fig_gbm.update_layout(title=f"{sim_days}-Day GBM Monte Carlo Simulation ({ticker})",
                           xaxis_title="Trading Days Ahead", yaxis_title="Simulated Price ($)", **PLOT_LAYOUT)
    st.plotly_chart(fig_gbm, use_container_width=True)

    final_prices = price_paths[-1]
    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">Median Forecast Price</div>
                    <div class="value">${np.median(final_prices):,.2f}</div>
                    <div class="sub">in {sim_days} trading days</div>
                </div>""", unsafe_allow_html=True)
    with g2:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">90% Simulation Interval</div>
                    <div class="value">${np.percentile(final_prices, 5):,.2f} → ${np.percentile(final_prices, 95):,.2f}</div>
                    <div class="sub">5th–95th percentile of simulated outcomes</div>
                </div>""", unsafe_allow_html=True)
    with g3:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="label">P(price above current)</div>
                    <div class="value">{(final_prices > last_close).mean() * 100:.1f}%</div>
                    <div class="sub">Share of simulated paths ending higher</div>
                </div>""", unsafe_allow_html=True)

    # ==========================================
    # 10. JOINT RANDOM VARIABLES — MARKET BETA
    # ==========================================
    section_header(
        "Joint Distribution vs. Benchmark", "PAIRS OF RV · LINEAR ESTIMATION",
        f"Correlation and linear (MAP/ML) estimation of {ticker} returns given {benchmark_ticker or 'benchmark'} returns — the classic market-beta regression."
    )
    try:
        with st.spinner(f"Downloading benchmark history for {benchmark_ticker}..."):
            bench_data = load_data(benchmark_ticker, years)
        aligned = pd.concat([stock_data["Returns"], bench_data["Returns"]], axis=1, join="inner").dropna()
        aligned.columns = [ticker, benchmark_ticker]

        beta, alpha = np.polyfit(aligned[benchmark_ticker], aligned[ticker], 1)
        corr = aligned.corr().iloc[0, 1]

        b1, b2, b3 = st.columns(3)
        with b1:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="label">Beta (β) vs {benchmark_ticker}</div>
                        <div class="value">{beta:.2f}</div>
                        <div class="sub">Market sensitivity (linear ML estimate)</div>
                    </div>""", unsafe_allow_html=True)
        with b2:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="label">Alpha (α), daily</div>
                        <div class="value">{alpha * 100:.4f}%</div>
                        <div class="sub">Intercept — benchmark-independent return</div>
                    </div>""", unsafe_allow_html=True)
        with b3:
            st.markdown(
                f"""<div class="metric-card">
                        <div class="label">Correlation (ρ)</div>
                        <div class="value">{corr:.3f}</div>
                        <div class="sub">Linear dependence with {benchmark_ticker}</div>
                    </div>""", unsafe_allow_html=True)

        fig_joint = go.Figure()
        fig_joint.add_trace(go.Scatter(
            x=aligned[benchmark_ticker] * 100, y=aligned[ticker] * 100, mode="markers",
            marker=dict(color="#2dd4bf", size=5, opacity=0.5), name="Daily returns",
        ))
        x_line = np.linspace(aligned[benchmark_ticker].min(), aligned[benchmark_ticker].max(), 50)
        y_line = alpha + beta * x_line
        fig_joint.add_trace(go.Scatter(x=x_line * 100, y=y_line * 100, mode="lines",
                                        line=dict(color="#fdcb6e", width=3), name="Linear fit (β, α)"))
        fig_joint.update_layout(title=f"{ticker} vs {benchmark_ticker} — Joint Return Distribution",
                                 xaxis_title=f"{benchmark_ticker} Daily Return %",
                                 yaxis_title=f"{ticker} Daily Return %", **PLOT_LAYOUT)
        st.plotly_chart(fig_joint, use_container_width=True)
    except ValueError as bench_err:
        st.warning(f"Benchmark analysis unavailable: {bench_err}")

    # ==========================================
    # 11. RAW DATA
    # ==========================================
    section_header("Raw Historical Data", "DATASET", "Full price and return history used across every model above.")
    st.dataframe(
        stock_data[["Close", "Returns"]].sort_index(ascending=False),
        use_container_width=True,
        height=420,
    )
    csv = stock_data.to_csv().encode("utf-8")
    st.download_button(
        "⬇️ Download full dataset as CSV",
        data=csv,
        file_name=f"{ticker}_history.csv",
        mime="text/csv",
    )

except ValueError as ve:
    st.error(f"⚠️ {ve}")
except Exception as e:
    st.error(f"Something went wrong while analyzing '{ticker}'. Detail: {e}")