## QuantfromTemu
# 📈 Stock Quantitative Probability Dashboard

A Bayesian and classical statistical analysis dashboard for stock return modeling.

This project provides an interactive **Streamlit web application** that compares:

- **Maximum Likelihood Estimation (MLE)** — a traditional point estimate approach
- **Bayesian Inference** — a probabilistic approach that estimates uncertainty using MCMC sampling

The dashboard fetches historical stock prices from **Yahoo Finance**, analyzes daily returns, estimates expected returns, and visualizes uncertainty using statistical distributions.

---

## 🚀 Features

### 📊 Statistical Modeling

The application compares two approaches:

### 1. Maximum Likelihood Estimation (MLE)

MLE estimates the parameters of a Normal distribution:

\[
R \sim N(\mu, \sigma)
\]

where:

- `μ` = average daily return
- `σ` = daily volatility

The model provides a single best estimate of historical average return.

---

### 2. Bayesian Inference

The Bayesian model treats the true average return as uncertain.

A probabilistic model is built using:

- Prior distributions
- Likelihood function
- Posterior distribution

The model uses:

- **PyMC**
- **Hamiltonian Monte Carlo (MCMC)** sampling
- **ArviZ** posterior analysis

Output includes:

- Posterior mean return
- 94% Highest Density Interval (HDI)
- Full uncertainty distribution


---

## ✨ Application Highlights

### 📌 Interactive Stock Analysis

Users can:

- Enter any valid stock ticker
- Select historical analysis period
- Compare statistical estimates
- Download processed stock data


Supported examples:

```
AAPL
MSFT
NVDA
TSLA
GOOGL
AMZN
```

---

## 📈 Dashboard Components

### 1. Model Estimate Cards

Displays:

| Metric | Description |
|-|-|
| MLE Daily Return | Classical best estimate |
| Bayesian Expected Return | Posterior mean |
| 94% Credible Interval | Range of probable true return |


---

### 2. Statistical Interpretation

The dashboard automatically evaluates Bayesian uncertainty:


🟢 **Positive Signal**

The entire credible interval is above zero.


🟡 **Inconclusive**

The credible interval crosses zero.


🔴 **Negative Signal**

The entire credible interval is below zero.


> Note: This is a statistical interpretation of historical data and is not financial advice.

---

## 📉 Visualization

The application provides four interactive analysis tabs:

---

### 1. Raw Returns

Shows daily percentage returns over time.

Features:

- Time-series visualization
- Return fluctuations
- Market volatility behavior


---

### 2. MLE Distribution Fit

Displays:

- Histogram of actual returns
- Fitted Normal distribution curve
- Estimated mean and volatility


---

### 3. Bayesian Posterior Distribution

Visualizes:

- Posterior belief of true average return
- 94% credible interval
- Uncertainty around expected return


---

### 4. Raw Historical Data

Provides:

- Closing prices
- Daily returns
- CSV download functionality


---

# 🏗️ Project Architecture


```
Stock Probability Dashboard
│
├── app.py
│
├── requirements.txt
│
└── README.md
```


---

# 🛠️ Installation

## 1. Clone Repository


```bash
git clone https://github.com/yourusername/stock-probability-dashboard.git

cd stock-probability-dashboard
```


---

## 2. Create Virtual Environment


### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```


### Windows

```bash
python -m venv venv

venv\Scripts\activate
```


---

## 3. Install Dependencies


```bash
pip install -r requirements.txt
```


---

# ▶️ Running the Application


Start Streamlit:

```bash
streamlit run app.py
```


The application will open automatically:

```
http://localhost:8501
```


---

# 📦 Dependencies


Main libraries:

| Library | Purpose |
|-|-|
| Streamlit | Web dashboard framework |
| yfinance | Stock market data retrieval |
| Pandas | Data processing |
| NumPy | Numerical computation |
| SciPy | Statistical fitting |
| PyMC | Bayesian modeling |
| ArviZ | Bayesian diagnostics |
| Plotly | Interactive visualization |


Full dependencies:

```
streamlit>=1.40.0
yfinance>=0.2.50
pandas>=2.2.0,<3.0.0
numpy>=1.26.0,<2.0.0
scipy>=1.12.0
pymc==5.16.2
arviz==0.18.0
matplotlib>=3.7.0,<3.11.0
plotly>=6.0.0
```

---

# 🧠 Mathematical Background


## Daily Return Calculation

The application calculates daily returns:

\[
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
\]


where:

- \(P_t\) = closing price at day t


---

## MLE Estimation


The likelihood function assumes:

\[
R_i \sim N(\mu,\sigma)
\]


MLE finds:

\[
\hat{\mu},\hat{\sigma}
\]


that maximize likelihood.


---

## Bayesian Model


Prior:


\[
\mu \sim N(0,0.01)
\]


\[
\sigma \sim HalfNormal(0.05)
\]


Likelihood:


\[
R_i \sim N(\mu,\sigma)
\]


Posterior:


\[
P(\mu,\sigma|R)
\]


is sampled using MCMC.


---

# 📁 Data Source

Historical market data is retrieved from:

**Yahoo Finance**

through:

```
yfinance
```


The application automatically downloads:

- Historical closing prices
- Daily returns


---

# ⚡ Performance Notes

The Bayesian model performs MCMC sampling:

Configuration:

```
Samples:
500 draws

Tuning:
500 steps

Chains:
2

CPU cores:
1
```


The first analysis may take longer because the Bayesian sampler needs initialization.

---

# 🔒 Disclaimer

This project is for:

- Educational purposes
- Statistical modeling demonstrations
- Learning Bayesian inference


It does **not** provide:

- Financial advice
- Trading recommendations
- Future stock predictions


Past performance does not guarantee future results.

---

# 🔮 Future Improvements

Possible extensions:

- [ ] Add Monte Carlo price forecasting
- [ ] Add GARCH volatility modeling
- [ ] Add portfolio optimization
- [ ] Add multiple stock comparison
- [ ] Add Bayesian regression models
- [ ] Add real-time market updates
- [ ] Add technical indicators
- [ ] Add model diagnostics dashboard
- [ ] Deploy using Streamlit Cloud


---

# 🤝 Contributing

Contributions are welcome.

Steps:

1. Fork the repository
2. Create a new branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Add new feature"
```

4. Push changes

```bash
git push origin feature/new-feature
```

5. Open a Pull Request


---

# 👨‍💻 Author

**Your Name**

GitHub:
```
https://github.com/yourusername
```


---

# ⭐ Acknowledgements

Built using:

- Streamlit
- PyMC
- ArviZ
- SciPy
- Yahoo Finance API
- Plotly

