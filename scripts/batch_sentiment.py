"""Batch sentiment scoring with incremental saves and progress tracking."""
import sys, os, time, json, sqlite3
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-dff8bfb2b4124f1ab82c381cabf4179d")

from src.data.database import MarketDB
from src.signals.sentiment import SentimentScorer, get_deepseek_provider

db = MarketDB("data/db/market.db")
scorer = SentimentScorer(get_deepseek_provider())
# Faster timeout
scorer.client.timeout = 15.0

# Get tickers with news, skip already scored today
conn = sqlite3.connect("data/db/market.db")
all_tickers = [r[0] for r in conn.execute(
    "SELECT DISTINCT ticker FROM news ORDER BY ticker"
).fetchall()]
already = set(r[0] for r in conn.execute(
    "SELECT DISTINCT ticker FROM signal_scores WHERE signal_name='llm_sentiment' AND signal_date=date('now')"
).fetchall())
conn.close()

tickers = [t for t in all_tickers if t not in already]
print(f"To score: {len(tickers)} (skipping {len(already)} already done today)")
print(f"Estimated: ~{len(tickers) * 2 / 60:.0f} minutes\n")

results = []
errors = 0
t0 = time.time()

for i, ticker in enumerate(tickers):
    try:
        r = scorer.score_ticker(ticker, db, max_articles=5)
        results.append(r)
        # Save incrementally every ticker
        db.upsert_signal(
            ticker=ticker,
            signal_date=time.strftime("%Y-%m-%d"),
            signal_name="llm_sentiment",
            signal_value=r.score,
            score=r.score,
            horizon="medium",
        )
    except Exception as e:
        errors += 1
        if errors <= 10:
            print(f"  ERR {ticker}: {str(e)[:60]}")
        continue

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(tickers) - i - 1) / rate
        print(f"  [{i+1:3d}/{len(tickers)}] {rate*60:.0f}/min ETA {eta:.0f}s | {ticker}={r.score:+.2f}")

    time.sleep(0.2)

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s ({len(results)} scored, {errors} errors)")

# Summary
if results:
    scores = np.array([r.score for r in results])
    print(f"\n{'=' * 60}")
    print(f"SENTIMENT SUMMARY — {len(results)} tickers")
    print(f"{'=' * 60}")
    print(f"  Mean:   {scores.mean():+.3f}")
    print(f"  Std:    {scores.std():.3f}")
    print(f"  Median: {np.median(scores):+.3f}")

    for label, cond in [
        ("strong bull  (>=+0.5)", scores >= 0.5),
        ("moderate bull (+0.2)", (scores >= 0.2) & (scores < 0.5)),
        ("slight bull  (0,+0.2)", (scores > 0) & (scores < 0.2)),
        ("neutral      (0)", scores == 0),
        ("slight bear  (-0.2,0)", (scores < 0) & (scores > -0.2)),
        ("moderate bear(-0.5)", (scores <= -0.2) & (scores > -0.5)),
        ("strong bear  (<=-0.5)", scores <= -0.5),
    ]:
        n = cond.sum()
        bar = "#" * (n // 2)
        print(f"  {label:25s} {n:>3} ({n/len(results)*100:4.1f}%) {bar}")

    results.sort(key=lambda x: -x.score)
    print(f"\n  TOP 15:")
    for r in results[:15]:
        print(f"    {r.ticker:6s} {r.score:+.2f}  {r.summary[:65]}")
    print(f"\n  BOTTOM 15:")
    for r in results[-15:]:
        print(f"    {r.ticker:6s} {r.score:+.2f}  {r.summary[:65]}")
