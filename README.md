PRODUCCIÓN (correr regularmente):
──────────────────────────────────────────────────────────────

  python run_daily.py
    → Ranking diario + alertas LLM + PDF report
    → Correr cada noche después del cierre

  python run_daily.py --calibrate
    → Recalibrar pesos por GICS y clusters
    → Correr cada mes (primer domingo del mes)

  python run_daily.py --weekly
    → Refresh fundamentales de FMP
    → Correr cada domingo

  python run_daily.py --refresh-alerts
    → Forzar regenerar alertas (ignora cache del día)


  python run_portfolio.py --mode discover
    → Optimizar portafolio B-L con top del ranking
    → Correr cuando quieras invertir/rebalancear

  python run_portfolio.py --mode watchlist
    → Optimizar pesos solo para tu watchlist actual

  python run_portfolio.py --tickers NVDA,MU,CIEN,ASTS
    → Optimizar pesos para tickers específicos

  python run_portfolio.py --target-vol 40
    → Optimizar a nivel de volatilidad específico


BACKTESTING (correr cuando quieras validar):
──────────────────────────────────────────────────────────────

  python backtests/01_walkforward_models.py
    → Compara: Equal vs GICS vs Clusters vs Momentum vs SPY
    → El que validó que GICS calibrado da +30.8%/año Sharpe 1.49

  python backtests/02_bl_optimization.py
    → Compara B-L con diferentes max weights (5%, 10%, 15%)
    → El que mostró +61%/año con B-L 10%


UTILIDADES (one-off):
──────────────────────────────────────────────────────────────

  python scripts/batch_sentiment.py
    → Scoring LLM de todo el universo SP500
    → Correr semanalmente o cuando quieras actualizar sentimiento
