PRODUCCIÓN (correr regularmente):
──────────────────────────────────────────────────────────────

  python3 run_daily.py
    → Ranking diario + alertas LLM + PDF report
    → Correr cada noche después del cierre

  python3 run_daily.py --calibrate
    → Recalibrar pesos por GICS y clusters
    → Correr cada mes (primer domingo del mes)

  python3 run_daily.py --weekly
    → Refresh fundamentales de FMP
    → Correr cada domingo

  python3 run_daily.py --refresh-alerts
    → Forzar regenerar alertas (ignora cache del día)


  python3 run_portfolio.py --mode discover
    → Optimizar portafolio B-L con top del ranking
    → Correr cuando quieras invertir/rebalancear

  python3 run_portfolio.py --mode watchlist
    → Optimizar pesos solo para tu watchlist actual

  python3 run_portfolio.py --tickers NVDA,MU,CIEN,ASTS
    → Optimizar pesos para tickers específicos

  python3 run_portfolio.py --target-vol 40
    → Optimizar a nivel de volatilidad específico


BACKTESTING (correr cuando quieras validar):
──────────────────────────────────────────────────────────────

  python3 backtests/01_walkforward_models.py
    → Compara: Equal vs GICS vs Clusters vs Momentum vs SPY
    → El que validó que GICS calibrado da +30.8%/año Sharpe 1.49

  python3 backtests/02_bl_optimization.py
    → Compara B-L con diferentes max weights (5%, 10%, 15%)
    → El que mostró +61%/año con B-L 10%


UTILIDADES (one-off):
──────────────────────────────────────────────────────────────

  python3 scripts/batch_sentiment.py
    → Scoring LLM de todo el universo SP500
    → Correr semanalmente o cuando quieras actualizar sentimiento

PORTAFOLIO ACTUAL (IBKR):
──────────────────────────────────────────────────────────────

python run_tracker.py --ibkr          # lee posiciones + compara con ranking
python run_tracker.py -- history       # ver cómo ha ido el P&L
