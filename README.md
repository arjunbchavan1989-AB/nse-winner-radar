# NSE Winner Radar

A 10-year study of **NSE big winners** — every episode where a liquid stock rose
**25%+ within 60 trading sessions** — plus a daily screener built to find the
next ones, and a complete swing-trading framework distilled from the findings.

Live: https://arjunbchavan1989-AB.github.io/nse-winner-radar/

## Tabs

**Screener** — all Nifty 500 stocks scored 0-100 against the winner profile:
Minervini Stage-2 trend template, 3-month relative strength vs Nifty, and setup
tags (BASE BREAK, VCP COIL, HT FLAG, HANDLE, PULLBACK, VOL SURGE, 52W HIGH).
Click a row for the 6-month candlestick chart with EMA 21/50 overlays.

**Study** — the findings from ~1,500+ winner episodes:
- How big (median gain, max extension) and how fast (median sessions)
- Where winners start: 52w-high proximity, EMA state, RS, base depth/length,
  breakout volume, sector and year distribution
- Whether winners keep running: median forward returns at 5/10/20/40/60
  sessions after the +25% day
- Notable-winner table (top 100 by gain)

**Framework** — the trading system the study supports:
1. Scanning (liquidity gate, trend template, RS, market regime)
2. Watchlist rules (pivots and stops written in advance, 15-25 names)
3. Setup catalog (VCP, cup & handle, high-tight flag, Qullamaggie continuation,
   EP/pullback, breakout-retest)
4. Trading style for India: swing 3-15 sessions as the primary engine,
   positional 2-6 months as the leader overlay; daily chart for entries,
   weekly for context
5. Timeframes & risk (0.5-1% risk per trade, circuit-gap sizing, exits)

## Method & caveats

- Universe: current Nifty 500 constituents (auto-refreshed from NSE) —
  survivorship bias applies; delisted losers are excluded
- Prices: Yahoo Finance daily OHLCV, auto-adjusted, 10 years
- Episode rule: non-overlapping, start at a 10-session local low, threshold
  +25% within ≤60 sessions
- Updated every trading day at 19:00 IST by GitHub Actions

## Local run

```bash
pip install -r requirements.txt
python pipeline.py            # writes data/screener.json, study.json, history.json
python3 -m http.server
```

## Disclaimer

Research and education only — not investment advice.
