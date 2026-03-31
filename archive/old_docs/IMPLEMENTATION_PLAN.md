# Implementation Plan — What's Left to Build

> Estado al 2026-03-25. Fase 0 y Fase 1 completadas.
> Este documento lista exactamente que falta por implementar, donde va cada cosa,
> y que datos/inputs necesita cada modulo.

---

## Status actual

```
✅ FASE 0 — Infraestructura base (COMPLETA)
   src/data/fetcher.py         → yfinance + cache parquet
   src/analysis/portfolio.py   → retornos, Sharpe, drawdown, correlaciones
   src/analysis/risk.py        → VaR/CVaR (historical, parametric, Monte Carlo)
   src/analysis/optimization.py → Max Sharpe, Min Var, Risk Parity, frontera eficiente
   src/utils/config.py         → carga YAML
   src/utils/reporting.py      → guarda runs a disco
   run_analysis.py             → CLI entry point

✅ FASE 1 — Data Layer (COMPLETA)
   src/data/fmp_client.py      → FMP /stable/ API (profile, financials, ratios,
                                  earnings, estimates, news, sp500 list)
   src/data/database.py        → SQLite 10 tablas, 288 fundamentals, 96 ratios,
                                  143 earnings, 120 news, 12 profiles
   src/data/universe.py        → universe manager (watchlist activo)
   config/universe.yaml        → 3 universos + factor proxies
   scripts/test_phase1.py      → integration test validado

   DB POBLADA:
   fundamentals: 288 | ratios: 96 | earnings: 143 | profiles: 12 | news: 120
```

---

## FASE 2 — Senales + Validacion Simultanea

### 2A. Senales Tecnicas

**Archivo**: `src/signals/technical.py`
**Input**: precios OHLCV de yfinance (ya tenemos via fetcher.py)
**No necesita FMP** — solo precios.

```python
# Funciones a implementar:

def momentum_12_1(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorno 12m menos retorno 1m, cross-sectional z-score → [-1, +1]"""

def momentum_1m(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorno ultimo mes, z-score → [-1, +1]"""

def rsi_14(prices: pd.DataFrame) -> pd.DataFrame:
    """RSI 14 dias, normalizado a [-1, +1] (0.5 → 0, 0.7+ → positivo, 0.3- → negativo)"""

def macd_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """(MACD - Signal) / precio, z-score → [-1, +1]"""

def bollinger_position(prices: pd.DataFrame) -> pd.DataFrame:
    """Posicion dentro de Bollinger Bands (20d, 2std), normalizado a [-1, +1]"""

def volume_ratio(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    """Volumen actual / media 20d, z-score → [-1, +1]"""
```

Cada funcion retorna `pd.DataFrame` con shape `(dates, tickers)` y valores en `[-1, +1]`.

### 2B. Senales Fundamentales

**Archivo**: `src/signals/fundamental.py`
**Input**: datos de SQLite (fundamentals, ratios, earnings)
**Necesita FMP** — usa datos ya cacheados en DB.

```python
# Funciones a implementar:

def pe_relative(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """P/E vs mediana sectorial, invertido y z-scored → [-1, +1]"""

def ev_ebitda_relative(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """EV/EBITDA relativo al sector → [-1, +1]"""

def fcf_yield(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """Free Cash Flow / Market Cap → [-1, +1]"""

def roe(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """Return on Equity → [-1, +1]"""

def gross_margin_delta(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """Cambio en margenes brutos YoY → [-1, +1]"""

def earnings_surprise(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """(epsActual - epsEstimated) / |epsEstimated| → [-1, +1]"""

def revenue_growth(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """Crecimiento ingresos YoY → [-1, +1]"""

def debt_equity_inv(db: MarketDB, tickers: list, date: str) -> dict[str, float]:
    """Inverso de deuda/equity (menor deuda = mejor) → [-1, +1]"""
```

### 2C. Validacion de Senales

**Archivo**: `src/validation/signal_tester.py`
**Input**: scores de senales + retornos futuros realizados

```python
class SignalTester:
    """Evalua una senal segun el Research Protocol."""

    def compute_ic(self, signal: pd.DataFrame, returns: pd.DataFrame,
                   horizon: int = 21) -> dict:
        """IC = rank_corr(senal_t, retorno_{t+h}), promediado cross-sectional."""

    def decile_analysis(self, signal: pd.DataFrame, returns: pd.DataFrame,
                        horizon: int = 21, n_groups: int = 5) -> pd.DataFrame:
        """Retorno promedio por quintil. Busca: monotonia y spread Q5-Q1."""

    def factor_attribution(self, signal: pd.DataFrame, returns: pd.DataFrame,
                           factor_returns: pd.DataFrame) -> dict:
        """IC residual despues de controlar market, size, value, momentum, quality."""

    def stability_by_regime(self, signal: pd.DataFrame, returns: pd.DataFrame,
                            benchmark: pd.Series) -> dict:
        """IC en bull vs bear, high vol vs low vol."""

    def compute_turnover(self, signal: pd.DataFrame, top_pct: float = 0.2) -> float:
        """Turnover mensual del top quintil."""

    def full_evaluation(self, signal_name: str, signal: pd.DataFrame,
                        returns: pd.DataFrame, **kwargs) -> dict:
        """Corre todo el protocolo y retorna el evaluation report."""
```

**Archivo**: `src/validation/baselines.py`
**Input**: precios del universo + factor proxies

```python
def buy_and_hold_benchmark(prices: pd.DataFrame, benchmark: str = "SPY") -> pd.Series:
    """Retorno de buy & hold SPY."""

def momentum_simple(prices: pd.DataFrame, lookback: int = 252,
                    skip: int = 21) -> pd.DataFrame:
    """Top quintil por retorno 12-1m. El baseline mas duro."""

def value_simple(db: MarketDB, tickers: list) -> pd.DataFrame:
    """Top quintil por FCF yield o P/E inverso."""

def quality_simple(db: MarketDB, tickers: list) -> pd.DataFrame:
    """Top quintil por ROE + bajo leverage."""
```

### 2D. Notebook de Validacion

**Archivo**: `notebooks/02_signal_validation.ipynb`
- Corre cada senal sobre el universo
- Muestra IC, deciles, factor attribution
- Compara contra baselines
- Veredicto: PASS / FAIL / MARGINAL por senal

---

## FASE 3 — Ranking Cross-Sectional

**Archivo**: `src/screener/scorer.py`

```python
class CompositeScorer:
    """Combina senales validadas en score compuesto."""

    def __init__(self, signals: dict[str, pd.DataFrame],
                 weights: dict[str, float] | None = None):
        """weights=None → pesos proporcionales a IC."""

    def compute_composite(self, date: str, horizon: str = "21d") -> pd.DataFrame:
        """Score compuesto por ticker para una fecha."""

    def compute_rolling(self, dates: list[str]) -> pd.DataFrame:
        """Rolling composite para backtest."""
```

**Archivo**: `src/screener/ranker.py`

```python
class UniverseRanker:
    """Ranking del universo con score desglosado."""

    def rank(self, composite: pd.DataFrame) -> pd.DataFrame:
        """Ranking ordenado con breakdown por senal."""

    def compare_vs_baselines(self, ranking: pd.DataFrame,
                              baselines: dict) -> pd.DataFrame:
        """Top quintil del ranking vs baselines."""
```

**Test central**: top quintil vs bottom quintil a 21d y 63d.

---

## FASE 4 — Portfolio Baseline

Integra modulos existentes (`src/analysis/`) con ranking de Fase 3.

**Archivo**: `src/screener/portfolio_baseline.py`

```python
def backtest_top_quintile(
    ranking_history: pd.DataFrame,
    prices: pd.DataFrame,
    rebalance_freq: int = 21,       # dias
    max_weight: float = 0.10,       # 10% max por nombre
    max_sector: float = 0.30,       # 30% max por sector
    cost_per_trade: float = 0.0012, # 12 bps
) -> dict:
    """Backtest del portafolio equal-weight top quintil con costos."""
```

**Output**: Sharpe neto, max drawdown, alpha vs SPY, turnover.

---

## FASE 5 — LLM Intelligence

**Archivo**: `src/signals/sentiment.py`

```python
class SentimentScorer:
    """Pipeline: news text → Claude API → structured score."""

    def score_article(self, ticker: str, title: str, text: str) -> dict:
        """Score individual article. Returns JSON structured output."""

    def score_batch(self, articles: list[dict]) -> list[dict]:
        """Score multiple articles efficiently."""

    def web_search_sentiment(self, ticker: str) -> dict:
        """Claude web search: '[ticker] analyst outlook Q2 2026' → score."""
```

**Requiere**: `anthropic` SDK en requirements + ANTHROPIC_API_KEY en .env

---

## FASE 6 — Risk Modeling

**Archivo**: `src/risk/garch.py`

```python
class GARCHModel:
    """GJR-GARCH(1,1) fit + forecast via arch library."""

    def fit(self, returns: pd.Series) -> dict:
        """Fit GARCH a serie de retornos."""

    def forecast_volatility(self, horizon: int = 21) -> pd.Series:
        """Forecast de sigma(t) a horizon dias."""
```

**Archivo**: `src/risk/montecarlo.py`

```python
def simulate_paths(
    mu: float, sigma_forecast: np.ndarray,
    S0: float, horizon: int = 21, n_paths: int = 10000,
) -> np.ndarray:
    """Euler-Maruyama vectorizado. Returns (n_paths, horizon) array."""

def path_metrics(paths: np.ndarray, S0: float) -> dict:
    """P(gain), VaR, CVaR, expected return, fan chart data."""
```

**Archivo**: `src/risk/regime.py`

```python
def classify_regime(benchmark_returns: pd.Series) -> pd.Series:
    """Bull/Bear + High/Low vol classification by date."""
```

---

## FASE 7 — Dynamic E[r] (condicional a Fases 2-4)

**Archivo**: `src/models/state_space.py`

```python
class DynamicReturnModel:
    """State-space con covariables: beta_t' * x_t via Kalman filter."""

    def fit(self, returns: pd.Series, signals: pd.DataFrame,
            sigma: pd.Series) -> dict:
        """Fit Kalman con senales como covariables."""

    def estimate_expected_return(self, current_signals: dict) -> tuple[float, float]:
        """Returns (E[r], uncertainty)."""
```

**Solo se implementa si Fases 2-4 demuestran senal predictiva neta de costos.**

---

## FASE 8 — Portfolio Optimization

**Archivo**: `src/portfolio/black_litterman.py`

```python
class BlackLitterman:
    """BL model: prior CAPM + views cuantitativos + views humanos."""

    def compute_posterior(self, market_weights: np.ndarray,
                         sigma: np.ndarray,
                         views: dict, confidence: dict) -> np.ndarray:
        """Returns posterior expected returns."""
```

**Archivo**: `src/portfolio/kelly.py`

```python
def fractional_kelly(mu: float, sigma: float, rf: float = 0.045,
                     fraction: float = 0.5) -> float:
    """Position size as fraction of portfolio. f*/2 by default."""
```

---

## Dependencias pendientes por fase

| Fase | Requiere instalar | Requiere API key |
|------|------------------|-----------------|
| 2 | pandas-ta | — |
| 3 | — | — |
| 4 | — | — |
| 5 | anthropic | ANTHROPIC_API_KEY |
| 6 | (arch ya instalado) | — |
| 7 | filterpy | — |
| 8 | — | — |

---

## Orden de implementacion sugerido para proxima sesion

```
1. src/signals/technical.py          ← no necesita FMP, solo precios
2. src/validation/signal_tester.py   ← IC + deciles
3. notebooks/02_signal_validation    ← visualizar resultados
4. src/signals/fundamental.py        ← usa DB existente
5. src/validation/baselines.py       ← momentum, value, quality simples
6. src/screener/scorer.py            ← composite score
7. src/screener/ranker.py            ← ranking + comparacion
```

Esto completa Fases 2-3 y es donde sabremos si el sistema tiene senal real o no.
