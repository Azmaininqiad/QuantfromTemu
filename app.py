import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.stats as stats
import pymc as pm
import arviz as az
import plotly.graph_objects as go

# ==========================================
# PAGE CONFIG & GLOBAL STYLE
# ==========================================
st.set_page_config(
    page_title="Stock Probability Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    :root {
        --accent: #6C5CE7;
        --accent-2: #00CEC9;
        --card-bg: rgba(255, 255, 255, 0.04);
        --card-border: rgba(255, 255, 255, 0.08);
    }

    /* Hide default streamlit chrome we don't need */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .app-hero {
        padding: 1.75rem 2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(108,92,231,0.25) 0%, rgba(0,206,201,0.15) 100%);
        border: 1px solid var(--card-border);
        margin-bottom: 1.5rem;
    }
    .app-hero h1 {
        font-size: 2.1rem;
        margin: 0 0 0.35rem 0;
        background: linear-gradient(90deg, #a29bfe, #74ebd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .app-hero p {
        margin: 0;
        opacity: 0.85;
        font-size: 1.02rem;
    }

    .metric-card {
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        height: 100%;
    }
    .metric-card .label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        opacity: 0.65;
        margin-bottom: 0.35rem;
    }
    .metric-card .value {
        font-size: 1.9rem;
        font-weight: 750;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        opacity: 0.6;
        margin-top: 0.2rem;
    }

    .rec-box {
        border-radius: 14px;
        padding: 1rem 1.3rem;
        font-size: 1.02rem;
        border: 1px solid var(--card-border);
        margin: 0.75rem 0 1.25rem 0;
    }
    .rec-invest { background: rgba(0, 200, 120, 0.12); border-color: rgba(0,200,120,0.35); }
    .rec-caution { background: rgba(255, 190, 30, 0.12); border-color: rgba(255,190,30,0.35); }
    .rec-avoid { background: rgba(255, 70, 70, 0.12); border-color: rgba(255,70,70,0.35); }

    .ticker-chip button {
        border-radius: 999px !important;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--card-border);
    }

    div[data-testid="stMetricValue"] { font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================
# HERO HEADER
# ==========================================
st.markdown(
    """
    <div class="app-hero">
        <h1>📈 Stock Quantitative Probability Dashboard</h1>
        <p>Compare a classic Maximum Likelihood Estimate against a full Bayesian posterior
        to see not just <i>what</i> a stock's average daily return might be, but <i>how confident</i>
        we should be about it.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# SIDEBAR — INPUTS
# ==========================================
st.sidebar.header("⚙️ Settings")

if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "AAPL"

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value=st.session_state.ticker_input,
    help="Enter a valid ticker symbol, e.g. AAPL, MSFT, TSLA.",
).strip().upper()

st.sidebar.caption("Quick picks")
chip_cols = st.sidebar.columns(4)
for chip_col, chip_ticker in zip(chip_cols, ["AAPL", "MSFT", "NVDA", "TSLA"]):
    with chip_col:
        if st.button(chip_ticker, key=f"chip_{chip_ticker}", use_container_width=True):
            st.session_state.ticker_input = chip_ticker
            st.rerun()

years = st.sidebar.slider("Years of Historical Data", min_value=1, max_value=5, value=2)

with st.sidebar.expander("ℹ️ How this works"):
    st.markdown(
        """
- **MLE** fits a single "best guess" Normal distribution to historical daily returns.
- **Bayesian** treats the true average return as uncertain and produces a full
  probability distribution (posterior) plus a 94% credible interval, using MCMC
  sampling via PyMC.
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

    # yfinance can return MultiIndex columns (Price, Ticker) even for a single symbol.
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

    # ---- MLE ----
    mle_mu, mle_sigma = stats.norm.fit(returns)

    # ---- Bayesian ----
    with st.spinner("Running Bayesian MCMC sampling..."):
        bayes_mu_mean, hdi_low, hdi_high, posterior_mu_samples = run_bayesian_model(returns)

    # ==========================================
    # METRIC CARDS
    # ==========================================
    st.subheader(f"📊 Model Estimates for {ticker}")

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

    # ---- Recommendation ----
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
    # CHARTS (in tabs)
    # ==========================================
    st.subheader("📈 Graphical Analysis")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Raw Returns", "MLE Fit", "Bayesian Posterior", "Raw Data"]
    )

    plot_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, l=10, r=10, b=10),
    )

    with tab1:
        fig1 = go.Figure()
        fig1.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=returns * 100,
                name="Daily Returns",
                line=dict(color="#6C5CE7", width=1.5),
            )
        )
        fig1.update_layout(
            title=f"Raw Daily Returns Over Time — {ticker}",
            yaxis_title="Daily Return %",
            **plot_layout,
        )
        st.plotly_chart(fig1, use_container_width=True)

    with tab2:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Histogram(
                x=returns * 100,
                histnorm="probability density",
                name="Actual Data",
                marker_color="#00CEC9",
                opacity=0.55,
            )
        )
        x_range = np.linspace(returns.min() * 100, returns.max() * 100, 200)
        p_mle = stats.norm.pdf(x_range / 100, mle_mu, mle_sigma) / 100
        fig2.add_trace(
            go.Scatter(
                x=x_range,
                y=p_mle,
                name="MLE Fitted Bell Curve",
                line=dict(color="#fdcb6e", width=3),
            )
        )
        fig2.update_layout(
            title="Maximum Likelihood Estimation (MLE) Fit",
            xaxis_title="Daily Return %",
            yaxis_title="Density",
            **plot_layout,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig3 = go.Figure()
        fig3.add_trace(
            go.Histogram(
                x=posterior_mu_samples * 100,
                histnorm="probability density",
                name="Belief Spectrum",
                marker_color="#a29bfe",
                opacity=0.65,
            )
        )
        fig3.add_vrect(
            x0=hdi_low * 100,
            x1=hdi_high * 100,
            fillcolor="#fdcb6e",
            opacity=0.2,
            line_width=0,
            annotation_text="94% Credible Zone",
            annotation_position="top left",
        )
        fig3.update_layout(
            title="Bayesian Posterior Belief Distribution (Uncertainty)",
            xaxis_title="True Average Daily Return %",
            yaxis_title="Belief Density",
            **plot_layout,
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
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