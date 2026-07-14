# 📊 QuantFromTemu — Bayesian Stock Probability Intelligence Dashboard

An interactive quantitative finance dashboard that combines **classical statistical estimation** with **Bayesian probabilistic modeling** to analyze historical stock returns.

QuantFromTemu provides an uncertainty-aware approach to stock analysis by comparing:

- **Maximum Likelihood Estimation (MLE)** — traditional point-based statistical estimation
- **Bayesian Inference** — probability-based modeling with uncertainty estimation using MCMC sampling

Instead of only predicting a single expected return, QuantFromTemu analyzes **how confident we should be about that estimate**.

---

# 🌐 Live Demo

Try the deployed application:

🔗 **Website:**  
https://quantfromtemu.streamlit.app/

---

# 🚀 Features

## 📈 Quantitative Stock Analysis

Analyze any publicly traded stock using historical market data.

Supported examples:

```
AAPL
MSFT
NVDA
TSLA
```
You can give any valid name.The dashboard automatically retrieves historical price data from Yahoo Finance and performs statistical analysis on daily returns.

---

# 🧠 Statistical Modeling

QuantFromTemu compares two powerful statistical approaches.

---

## 1. Maximum Likelihood Estimation (MLE)

MLE assumes that daily stock returns follow a normal distribution:

\[
R \sim N(\mu,\sigma)
\]


where:

- \( \mu \) = average daily return
- \( \sigma \) = daily volatility


The model estimates:

- Expected daily return
- Historical volatility


MLE provides a **single best estimate** based on observed historical data.

---

## 2. Bayesian Probability Modeling

Unlike MLE, Bayesian inference considers uncertainty in the parameters.

The model defines prior beliefs:

\[
\mu \sim N(0,0.01)
\]


\[
\sigma \sim HalfNormal(0.05)
\]


The likelihood function:

\[
R_i \sim N(\mu,\sigma)
\]


The posterior distribution:

\[
P(\mu,\sigma | R)
\]


is estimated using:

- PyMC
- Markov Chain Monte Carlo (MCMC)
- ArviZ


The Bayesian model provides:

- Posterior expected return
- Probability distribution of possible returns
- 94% credible interval


---

# 📊 Dashboard Overview


## 1. Model Comparison Cards

The dashboard displays:


| Metric | Description |
|---|---|
| MLE Best Guess | Classical estimated daily return |
| Bayesian Expected Return | Posterior mean return |
| 94% Credible Interval | Range of probable true returns |


---

# 🔍 Statistical Signal Interpretation


QuantFromTemu evaluates the Bayesian credible interval:


## 🟢 Positive Signal

When:

\[
HDI_{low} > 0
\]

The entire credible interval is above zero.

Interpretation:

> Historical returns show a statistically positive trend during the selected period.


---

## 🟡 Inconclusive Signal

When:

\[
HDI_{low}<0<HDI_{high}
\]


The uncertainty range crosses zero.

Interpretation:

> Historical performance is not statistically distinguishable from random variation.


---

## 🔴 Negative Signal

When:

\[
HDI_{high}<0
\]


The credible interval is below zero.

Interpretation:

> Historical returns show a statistically negative trend.


---

# 📈 Visualization


The dashboard provides four interactive analysis sections.


---

## 1. Raw Daily Returns

Displays:

- Daily percentage return movement
- Market fluctuations
- Volatility behavior over time


---

## 2. MLE Distribution Fit

Shows:

- Historical return histogram
- Normal distribution fitting
- Estimated probability density


---

## 3. Bayesian Posterior Distribution

Visualizes:

- Posterior belief distribution
- Uncertainty around expected return
- 94% credible interval


---

## 4. Historical Data Explorer

Allows users to:

- View closing prices
- Inspect calculated returns
- Download processed CSV data


---

# 🏗️ Project Architecture


```
QuantFromTemu
│
├── app.py                 # Streamlit dashboard application
│
├── requirements.txt       # Python dependencies
│
└── README.md              # Documentation
```


---

# 🛠️ Installation


## Clone Repository


```bash
git clone https://github.com/yourusername/QuantFromTemu.git

cd QuantFromTemu
```


---

## Create Virtual Environment


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

## Install Dependencies


```bash
pip install -r requirements.txt
```


---

# ▶️ Running Locally


Launch the Streamlit application:


```bash
streamlit run app.py
```


The application will open at:


```
http://localhost:8501
```


---

# 📦 Technology Stack


## Frontend

| Technology | Purpose |
|-|-|
| Streamlit | Interactive web dashboard |
| Plotly | Interactive financial visualization |


## Data Processing

| Technology | Purpose |
|-|-|
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| SciPy | Statistical analysis |


## Machine Learning / Statistics

| Technology | Purpose |
|-|-|
| PyMC | Bayesian probabilistic modeling |
| ArviZ | Bayesian inference analysis |


## Data Source

| Source | Purpose |
|-|-|
| Yahoo Finance | Historical stock market data |
| yfinance | Python API interface |


---

# 📚 Mathematical Foundation


## Daily Return Calculation


The application calculates:


\[
R_t=\frac{P_t-P_{t-1}}{P_{t-1}}
\]


where:

- \(P_t\) = closing price at time t


---

# ⚙️ Bayesian Model Configuration


MCMC Sampling:


```
Samples:
500 draws

Tuning:
500 iterations

Chains:
2

CPU cores:
1
```


The sampler generates posterior samples representing possible values of the true average return.

---

# 📁 Data Pipeline


```
Yahoo Finance
       |
       ↓
Historical Prices
       |
       ↓
Daily Return Calculation
       |
       ↓
       ├───────────────┐
       ↓               ↓
     MLE          Bayesian Model
       ↓               ↓
Point Estimate   Posterior Distribution
       |
       ↓
Interactive Dashboard
```


---

# ⚡ Performance Notes


- Historical data is cached for faster repeated analysis.
- Bayesian inference requires MCMC sampling, which may take longer during first execution.
- Longer historical periods provide more observations for statistical estimation.


---

# 🔒 Disclaimer


QuantFromTemu is an educational quantitative finance project.

It is intended for:

✅ Statistical learning  
✅ Bayesian modeling demonstrations  
✅ Financial data exploration  


It does **not provide**:

❌ Financial advice  
❌ Trading signals  
❌ Guaranteed predictions  


Past market performance does not guarantee future results.

---

# 🔮 Future Improvements


Planned enhancements:


- [ ] Multi-stock portfolio analysis
- [ ] Monte Carlo price simulation
- [ ] GARCH volatility modeling
- [ ] Technical indicator integration
- [ ] Risk metrics (Sharpe ratio, VaR, CVaR)
- [ ] Portfolio optimization
- [ ] Real-time market monitoring
- [ ] Bayesian regression forecasting
- [ ] Model comparison dashboard
- [ ] Cloud deployment improvements


---

# 🤝 Contributing


Contributions are welcome.


Steps:


### 1. Fork the repository


### 2. Create a feature branch


```bash
git checkout -b feature/new-feature
```


### 3. Commit changes


```bash
git commit -m "Add new feature"
```


### 4. Push changes


```bash
git push origin feature/new-feature
```


### 5. Create a Pull Request


---

# 👨‍💻 Author


**Azmain Adeeb**


GitHub:

```
https://github.com/Azmaininqiad
```


LinkedIn:

```
https://linkedin.com/in/azmain-adeeb-5b134230a
```


---

# ⭐ Acknowledgements


Built with:


- Streamlit
- PyMC
- ArviZ
- SciPy
- Plotly
- Yahoo Finance API


Special thanks to the open-source statistical computing community.

---

# 📌 Project Links


🌐 Live Application:

https://quantfromtemu.streamlit.app/



