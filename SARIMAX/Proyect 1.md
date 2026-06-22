# SARIMAX Predictive Model for Time Series Analysis

A production-grade predictive analytics system for monthly time series data. Built around a **SARIMAX model with square-root variance stabilization** and **Moving Block Bootstrap (MBB) confidence intervals**, the pipeline covers the full Data Science lifecycle — from data wrangling to diagnostic visualization.

---

## Overview

This project demonstrates how to build a robust end-to-end forecasting pipeline for any domain where you need to predict the evolution of a monthly event count (incidents, sales, orders, usage metrics, etc.) while accounting for infrastructure or contextual variables.

The model was originally developed to forecast IT support ticket volumes as a function of infrastructure growth (RAM, CPU, active machines), and has been generalized for reuse in any similar context.

---

## Pipeline Architecture

```
Data Source 
        │
        ▼
┌─────────────────────────┐
│  Module 0               │  RAM/CPU temporal expansion
│  Data Integration       │  (project-level → monthly aggregation)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 1               │  Priority: real RAM → assigned RAM → OLS proxy
│  Infrastructure Proxy   │  Falls back gracefully when data is missing
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 2               │  Moving Block Bootstrap (MBB, block_size=6)
│  Confidence Intervals   │  Preserves temporal autocorrelation of residuals
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 3               │  SARIMAX(1,1,1)(1,1,0)[12]
│  SARIMAX Forecasting    │  √-transform · top-3 exogenous by correlation
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  Module 5               │  8 diagnostic and forecast charts (G1–G8)
│  Visualization (G1–G8)  │
└─────────────────────────┘
```

---

## Model Design Decisions

### Why square-root transformation?
Count time series (incidents, tickets, events) exhibit **heteroscedasticity**: variance grows with the mean. Applying `y → √y` stabilizes the variance before fitting SARIMAX, making the model's assumptions more valid and reducing forecast error.

### Why MBB instead of classical bootstrap?
Time series residuals are **autocorrelated**. Classical bootstrap (independent random sampling) destroys this structure. Moving Block Bootstrap resamples contiguous blocks of residuals (block_size=6 months), preserving the dependence structure. This produces more honest confidence intervals, especially at long forecast horizons.

### Why SARIMAX(1,1,1)(1,1,0)[12]?
- **d=1, D=1**: one regular + one seasonal differencing to remove trend and multiplicative annual seasonality.
- **p=1, q=1**: minimal AR and MA terms — parsimonious model that avoids overfitting on ~8 years of monthly data.
- **Seasonal period=12**: monthly data with annual seasonality.
- The exogenous variables (RAM, CPU, active clients) capture structural growth that ARIMA terms alone cannot explain.

### Exogenous variable selection
From all available candidate variables, the model automatically selects the **top 3 by absolute Pearson correlation** with the target series, subject to a minimum coverage requirement (≥12 non-zero months). This avoids multicollinearity while keeping the model interpretable.

---

## Output Charts

| Chart | Description |
|-------|-------------|
| **G1** | Historical series + model fit + forecast + 95% CI |
| **G2** | Historical growth slopes: incidents vs. RAM |
| **G3** | STL decomposition: trend · seasonality · residual |
| **G4** | Correlation: incidents vs. active clients + load ratio |
| **G5** | Min-Max normalized comparison: incidents vs. infrastructure |
| **G6** | Infrastructure inventory: machines · CPU · RAM (individual panels) |
| **G7** | Infrastructure inventory normalized (single axis) |
| **G8** | Full model diagnostics: residuals, Q-Q, ACF/PACF, metrics card |

The **G8 metrics card** evaluates: R², MAE, RMSE, MAPE, AIC, Ljung-Box, Jarque-Bera, and ADF with traffic-light indicators (✅ / ⚠️) against standard thresholds.

---

## Adapting to Your Data Source

The script is designed to be **data-source agnostic**. Locate the `SECCIÓN DE DATOS` block at the top of the file and replace the synthetic data example with your own loader:

```python
# Option A — CSV
df_maestro = pd.read_csv("your_data.csv", parse_dates=["date_column"])
df_maestro = df_maestro.set_index("date_column").sort_index()

# Option B — SQL (any engine via SQLAlchemy)
import sqlalchemy
engine = sqlalchemy.create_engine("postgresql+psycopg2://user:pass@host/db")
df_maestro = pd.read_sql("SELECT * FROM monthly_events", engine,
                          index_col="period", parse_dates=["period"])

# Option C — World Bank API
import wbgapi as wb
df_raw = wb.data.DataFrame("NY.GDP.PCAP.CD", time=range(2000, 2024))
```

The **minimum required column** is `INCIDENCIAS` (your monthly event count). All infrastructure columns (`RAM_TOTAL_MB`, `CPU_TOTAL_MB`, `MAQUINAS_REALES`, `CLIENTES_ACTIVOS`) are optional — the model degrades gracefully to univariate mode if they are absent.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
scipy>=1.10
scikit-learn>=1.3
statsmodels>=0.14
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
sarimax-predictive-model/
│
├── sarimax_predictive_model.py   # Main pipeline
├── requirements.txt
├── README.md
└── outputs/                      # Generated charts (G1–G8)
```

---

## Skills Demonstrated

`Python` · `Pandas` · `Statsmodels` · `SARIMAX` · `Time Series Analysis` · `Bootstrap Methods` · `Signal Decomposition (STL)` · `Matplotlib` · `Statistical Diagnostics` · `Data Wrangling` · `OLS Regression`
