#!/usr/bin/env python3
"""
NSE Winner Radar — 10-year big-winner study + daily screener pipeline.

STUDY (universe: current Nifty 500 — survivorship bias disclosed):
  Finds every episode where a stock rose >=25% within <=60 trading sessions
  over the past 10 years, then measures:
    - how the move started (base depth/length, 52w-high proximity, EMA state,
      RS vs Nifty, Nifty regime, sector, year)
    - what happened after (continuation returns at +5/10/20/40/60 sessions,
      max extension beyond the +25% day)

SCREENER (daily, "future winners"): Stage-2 trend template + RS leadership +
setup tags (base breakout, VCP coil, high-tight flag, handle, pullback,
volume surge, 52w-high proximity) + composite winner score.

Output: data/study.json (episodes + aggregates), data/screener.json,
        data/history.json (drawer charts)
"""

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent
NSE_CSV = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
MIRROR_CSV = "https://raw.githubusercontent.com/varunsrik/nifty_dashboard/main/data/nifty_500_constituents.csv"

STUDY_YEARS = 10
MIN_GAIN_PCT = 25.0          # episode threshold
MAX_WINDOW = 60              # max sessions for the move
WARMUP = 260                 # need 52w history before an episode can start
MIN_TURNOVER = 30_000_000    # ₹3 cr 30-day avg turnover (screener default)
RS_LOOKBACK = 63
CHUNK = 60


def log(msg):
    print(msg, flush=True)


# --- universe -----------------------------------------------------------------
def fetch_constituents():
    base = {}
    with open(ROOT / "constituents.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Symbol"):
                base[row["Symbol"]] = {
                    "industry": row.get("Industry", "").strip(),
                    "sector": row.get("Sector", "").strip(),
                }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for name, url in [("nse", NSE_CSV), ("mirror", MIRROR_CSV)]:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            rows = list(csv.reader(r.text.splitlines()))
            if len(rows) < 400:
                continue
            hdr = rows[0]
            sym_i = hdr.index("Symbol")
            ind_i = hdr.index("Industry") if "Industry" in hdr else None
            fresh = {}
            for row in rows[1:]:
                if len(row) > sym_i and row[sym_i]:
                    fresh[row[sym_i]] = {
                        "industry": row[ind_i] if ind_i is not None and len(row) > ind_i else "",
                        "sector": "",
                    }
            if len(fresh) > 400:
                for s, v in base.items():
                    if s in fresh and v["sector"]:
                        fresh[s]["sector"] = v["sector"]
                log(f"constituents: {len(fresh)} symbols ({name})")
                base = fresh
                break
        except Exception as e:
            log(f"constituents refresh via {name} failed: {e}")
    return base


# --- prices -------------------------------------------------------------------
def fetch_prices(symbols, period="1y", index=False):
    out = {}
    syms = symbols if index else [s + ".NS" for s in symbols]
    for i in range(0, len(syms), CHUNK):
        sub = syms[i:i + CHUNK]
        try:
            data = yf.download(sub, period=period, interval="1d",
                               auto_adjust=True, group_by="ticker",
                               progress=False, threads=True)
        except Exception as e:
            log(f"chunk {i // CHUNK} failed: {e}")
            continue
        for s in sub:
            raw = s if index else s[:-3]
            try:
                d = data[s] if isinstance(data.columns, pd.MultiIndex) else data
                if len(d) and d["Close"].notna().any():
                    out[raw] = d
            except Exception:
                continue
        time.sleep(1)
    return out


# --- study: episode detection ---------------------------------------------------
def find_episodes(close, min_gain=MIN_GAIN_PCT, max_window=MAX_WINDOW):
    """Non-overlapping episodes: start at a local low, +25% within 60 sessions."""
    c = close.values
    n = len(c)
    if n < WARMUP + 70:
        return []
    # forward max of close over (i, i+max_window]
    s = pd.Series(c)
    fwd_max = s.shift(-1).rolling(max_window, min_periods=1).max().shift(-(max_window - 1))
    fwd_max = fwd_max.reindex(s.index).values
    eps = []
    i = WARMUP
    while i < n - 2:
        fm = fwd_max[i]
        if np.isnan(fm) or fm < c[i] * (1 + min_gain / 100):
            i += 5
            continue
        # refine start: lowest close in [i-10, i]
        s0 = i
        for k in range(max(0, i - 10), i + 1):
            if c[k] < c[s0]:
                s0 = k
        target = c[s0] * (1 + min_gain / 100)
        j = None
        for k in range(s0 + 1, min(s0 + max_window, n - 1) + 1):
            if c[k] >= target:
                j = k
                break
        if j is None:
            i += 5
            continue
        eps.append((s0, j))
        i = j + 20  # skip past the move
    return eps


def median(vals):
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    return round(float(np.median(vals)), 2) if vals else None


def build_study(prices, meta, nifty):
    nc = nifty["Close"].dropna()
    n_ema200 = nc.ewm(span=200, adjust=False).mean()

    episodes = []
    for sym, df in prices.items():
        d = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(d) < WARMUP + 70:
            continue
        closes = d["Close"]
        vols = d["Volume"]
        eps = find_episodes(closes)
        if not eps:
            continue
        ema50 = closes.ewm(span=50, adjust=False).mean().values
        ema200 = closes.ewm(span=200, adjust=False).mean().values
        vol50 = vols.rolling(50).mean().values
        dates = d.index
        hi52 = closes.rolling(250, min_periods=120).max().values
        for (s0, j) in eps:
            # base metrics: 120 sessions before start
            lo = max(0, s0 - 120)
            win = closes.iloc[lo:s0 + 1]
            base_high = float(win.max())
            base_depth = (float(closes.iloc[s0]) / base_high - 1) * 100 if base_high else 0
            # base length: days within 12% of the 120d high before launch
            near_hi = int(((win >= base_high * 0.88).sum()))
            # volume ratio on threshold day
            vr = float(vols.iloc[j]) / vol50[j] if vol50[j] and not np.isnan(vol50[j]) else None
            # 52w-high proximity at start
            h = hi52[s0]
            prox = (float(closes.iloc[s0]) / h - 1) * 100 if h and not np.isnan(h) else None
            # RS vs Nifty at start (63 sessions)
            try:
                sret = float(closes.iloc[s0]) / float(closes.iloc[s0 - RS_LOOKBACK]) - 1
                nd = nc.index.get_loc(dates[s0]) if dates[s0] in nc.index else None
                nret = float(nc.iloc[nd]) / float(nc.iloc[nd - RS_LOOKBACK]) - 1 if nd else 0
                rs = (sret - nret) * 100
            except Exception:
                rs = None
            # continuation returns from threshold day
            def fwd(m):
                k = j + m
                return round((float(closes.iloc[k]) / float(closes.iloc[j]) - 1) * 100, 1) if k < len(closes) else None
            # max extension within 60 sessions after start
            ext_win = closes.iloc[s0:min(s0 + MAX_WINDOW + 30, len(closes))]
            ext = (float(ext_win.max()) / float(closes.iloc[s0]) - 1) * 100
            # Nifty regime at start
            try:
                nd = nc.index.get_loc(dates[s0]) if dates[s0] in nc.index else None
                n_bull = bool(nc.iloc[nd] > n_ema200.iloc[nd]) if nd else None
            except Exception:
                n_bull = None
            episodes.append({
                "symbol": sym,
                "sector": meta.get(sym, {}).get("sector", ""),
                "start": str(dates[s0].date()),
                "end": str(dates[j].date()),
                "year": int(str(dates[s0].date())[:4]),
                "gainPct": round((float(closes.iloc[j]) / float(closes.iloc[s0]) - 1) * 100, 1),
                "days": j - s0,
                "maxExtPct": round(ext, 1),
                "baseDepthPct": round(base_depth, 1),
                "baseLenDays": near_hi,
                "volRatioAtBreak": round(vr, 2) if vr else None,
                "from52wHAtStart": round(prox, 1) if prox is not None else None,
                "aboveEma50": bool(closes.iloc[s0] > ema50[s0]),
                "aboveEma200": bool(closes.iloc[s0] > ema200[s0]),
                "rsAtStart": round(rs, 1) if rs is not None else None,
                "niftyBull": n_bull,
                "fwd5": fwd(5), "fwd10": fwd(10), "fwd20": fwd(20),
                "fwd40": fwd(40), "fwd60": fwd(60),
            })
    episodes.sort(key=lambda e: e["gainPct"], reverse=True)

    def pct_true(key):
        vals = [e[key] for e in episodes if e[key] is not None]
        return round(100 * sum(1 for v in vals if v) / len(vals), 1) if vals else None

    summary = {
        "episodes": len(episodes),
        "stocks": len({e["symbol"] for e in episodes}),
        "medianGain": median([e["gainPct"] for e in episodes]),
        "medianDays": median([e["days"] for e in episodes]),
        "medianExt": median([e["maxExtPct"] for e in episodes]),
        "medianBaseDepth": median([e["baseDepthPct"] for e in episodes]),
        "medianBaseLen": median([e["baseLenDays"] for e in episodes]),
        "medianVolRatio": median([e["volRatioAtBreak"] for e in episodes if e["volRatioAtBreak"]]),
        "pctAboveEma50": pct_true("aboveEma50"),
        "pctAboveEma200": pct_true("aboveEma200"),
        "pctNiftyBull": pct_true("niftyBull"),
        "fwd5": median([e["fwd5"] for e in episodes]),
        "fwd10": median([e["fwd10"] for e in episodes]),
        "fwd20": median([e["fwd20"] for e in episodes]),
        "fwd40": median([e["fwd40"] for e in episodes]),
        "fwd60": median([e["fwd60"] for e in episodes]),
        "pctWithin5OfHigh": round(100 * sum(1 for e in episodes if e["from52wHAtStart"] is not None and e["from52wHAtStart"] >= -5) / max(1, sum(1 for e in episodes if e["from52wHAtStart"] is not None)), 1),
        "pctWithin15OfHigh": round(100 * sum(1 for e in episodes if e["from52wHAtStart"] is not None and e["from52wHAtStart"] >= -15) / max(1, sum(1 for e in episodes if e["from52wHAtStart"] is not None)), 1),
        "pctRsPositive": round(100 * sum(1 for e in episodes if e["rsAtStart"] is not None and e["rsAtStart"] > 0) / max(1, sum(1 for e in episodes if e["rsAtStart"] is not None)), 1),
    }
    return episodes, summary


# --- screener -------------------------------------------------------------------
def compute_screener(sym, df, meta, nifty):
    d = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if len(d) < 210:
        return None
    closes, vols = d["Close"], d["Volume"]
    last = float(closes.iloc[-1])
    prev = float(closes.iloc[-2])

    ema21 = float(closes.ewm(span=21, adjust=False).mean().iloc[-1])
    ema50 = float(closes.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(closes.ewm(span=200, adjust=False).mean().iloc[-1])
    ema200_21d = float(closes.ewm(span=200, adjust=False).mean().iloc[-22])

    avg_vol50 = float(vols.iloc[-50:].mean())
    avg_turn30 = float((closes * vols).iloc[-30:].mean())
    vol_ratio = float(vols.iloc[-1]) / max(avg_vol50, 1)
    day_chg = (last / prev - 1) * 100
    hi52 = float(closes.iloc[-250:].max())
    pct_hi = (last / hi52 - 1) * 100

    # RS vs Nifty (3m)
    nc = nifty["Close"].dropna()
    sret = last / float(closes.iloc[-RS_LOOKBACK - 1]) - 1
    nret = float(nc.iloc[-1]) / float(nc.iloc[-RS_LOOKBACK - 1]) - 1
    rs3m = (sret - nret) * 100

    runup20 = (last / float(closes.iloc[-21]) - 1) * 100
    runup60 = (last / float(closes.iloc[-61]) - 1) * 100

    # recent 60d high and drawdown from it
    hi60 = float(closes.iloc[-60:].max())
    dd60 = (last / hi60 - 1) * 100
    # days since 60d high
    ihi = int(closes.iloc[-60:].values.argmax())

    # tightness: 5d range and ATR contraction
    rng5 = (float(closes.iloc[-5:].max()) / float(closes.iloc[-5:].min()) - 1) * 100
    tr = (d["High"] - d["Low"]).iloc[-20:]
    atr5 = float(tr.iloc[-5:].mean()) / last * 100
    atr20 = float(tr.mean()) / last * 100

    # volume dry-up: 5d avg vol vs 50d
    vol5 = float(vols.iloc[-5:].mean()) / max(avg_vol50, 1)

    # pullback proximity: touched 21 EMA in last 3 sessions
    near21 = any(float(closes.iloc[-3:].min()) <= ema21 * 1.02 for _ in [0])

    stage2 = last > ema50 > ema200 and ema200 > ema200_21d and pct_hi >= -25
    rs_leader = rs3m >= 15
    base_breakout = last >= hi60 * 0.995 and vol_ratio >= 1.8
    vcp_coil = (rng5 <= 5 and atr5 <= 0.85 * atr20 and vol5 <= 0.8
                and last >= hi60 * 0.93 and stage2)
    htf = runup60 >= 60 and dd60 >= -15 and ihi >= 10
    handle = (-12 <= dd60 <= -4 and vol5 <= 0.9 and pct_hi >= -20 and stage2)
    pullback = (stage2 and near21 and vol_ratio < 1 and runup60 >= 20)
    vol_surge = vol_ratio >= 2
    new_hi = pct_hi >= -2

    tags = []
    if stage2: tags.append("STAGE 2")
    if rs_leader: tags.append("RS LEADER")
    if base_breakout: tags.append("BASE BREAK")
    if vcp_coil: tags.append("VCP COIL")
    if htf: tags.append("HT FLAG")
    if handle: tags.append("HANDLE")
    if pullback: tags.append("PULLBACK")
    if vol_surge: tags.append("VOL SURGE")
    if new_hi: tags.append("52W HIGH")

    score = 0
    score += 20 if stage2 else 0
    score += 15 if rs_leader else 10 if rs3m >= 8 else 0
    score += 15 if base_breakout else 0
    score += 15 if vcp_coil else 0
    score += 15 if htf else 0
    score += 10 if handle else 0
    score += 10 if pullback else 0
    score += 10 if vol_surge else 0
    score += 10 if new_hi else 0
    score += 10 if runup20 >= 15 else 5 if runup20 >= 8 else 0
    score += 5 if avg_turn30 >= 100_000_000 else 0
    score = min(score, 100)

    return {
        "symbol": sym,
        "company": "",
        "sector": meta.get(sym, {}).get("sector", ""),
        "industry": meta.get(sym, {}).get("industry", ""),
        "close": round(last, 2),
        "dayChangePct": round(day_chg, 2),
        "ema21": round(ema21, 2), "ema50": round(ema50, 2), "ema200": round(ema200, 2),
        "avgTurnover30": int(avg_turn30),
        "volRatio": round(vol_ratio, 2),
        "rs3m": round(rs3m, 1),
        "runup20": round(runup20, 1), "runup60": round(runup60, 1),
        "pctFrom52wHigh": round(pct_hi, 1),
        "ddFrom60dHigh": round(dd60, 1),
        "rng5": round(rng5, 1),
        "vol5Ratio": round(vol5, 2),
        "stage2": bool(stage2),
        "rsLeader": bool(rs_leader),
        "tags": tags,
        "winnerScore": score,
        "candidate": bool((stage2 or (rs_leader and tags)) and avg_turn30 >= MIN_TURNOVER),
    }


def market_regime(nifty, stocks):
    c = nifty["Close"].dropna()
    last = float(c.iloc[-1])
    e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
    e200 = float(c.ewm(span=200, adjust=False).mean().iloc[-1])
    above = [s for s in stocks if s["close"] > s["ema50"]]
    part = round(100 * len(above) / max(1, len(stocks)), 1)
    if last > e50 > e200:
        label = f"RISK-ON — {part}% of the universe above its 50 EMA; hunt breakouts"
    elif last > e200:
        label = f"NEUTRAL — {part}% above the 50 EMA; trade only the strongest RS leaders"
    else:
        label = f"RISK-OFF — only {part}% above the 50 EMA; build the watchlist, don't force trades"
    return {
        "niftyClose": round(last, 1),
        "niftyDayPct": round((last / float(c.iloc[-2]) - 1) * 100, 2),
        "above50": last > e50, "above200": last > e200,
        "pctFrom52wHigh": round((last / float(c.iloc[-250:].max()) - 1) * 100, 1),
        "breadth": part,
        "regime": label,
    }


HISTORY_BARS = 140


def build_history(prices):
    hist = {}
    for sym, df in prices.items():
        if sym == "^NSEI":
            continue
        try:
            d = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        except KeyError:
            continue
        if len(d) < 30:
            continue
        d = d.iloc[-HISTORY_BARS:]
        hist[sym] = {
            "t": [int(ts.timestamp() // 86400) for ts in d.index],
            "o": [round(float(x), 2) for x in d["Open"]],
            "h": [round(float(x), 2) for x in d["High"]],
            "l": [round(float(x), 2) for x in d["Low"]],
            "c": [round(float(x), 2) for x in d["Close"]],
            "v": [int(x) for x in d["Volume"]],
        }
    return hist


def main():
    constituents = fetch_constituents()
    symbols = list(constituents.keys())
    log(f"fetching 10y prices for {len(symbols)} symbols + NIFTY…")

    prices = fetch_prices(symbols, period=f"{STUDY_YEARS}y")
    nifty = fetch_prices(["^NSEI"], period=f"{STUDY_YEARS}y", index=True).get("^NSEI")
    if nifty is None:
        log("FATAL: NIFTY fetch failed")
        sys.exit(1)
    log(f"prices: {len(prices)} symbols, {len(nifty)} nifty rows")

    log("building study…")
    episodes, summary = build_study(prices, constituents, nifty)
    log(f"study: {summary['episodes']} episodes across {summary['stocks']} stocks")
    log(f"median: +{summary['medianGain']}% in {summary['medianDays']} sessions; "
        f"fwd20 {summary['fwd20']}%, fwd60 {summary['fwd60']}%")

    log("building screener…")
    stocks = []
    for sym in symbols:
        df = prices.get(sym)
        if df is None:
            continue
        rec = compute_screener(sym, df, constituents, nifty)
        if rec:
            stocks.append(rec)
    log(f"screener: {len(stocks)} stocks, "
        f"{len([s for s in stocks if s['candidate']])} candidates")

    regime = market_regime(nifty, stocks)
    log(f"regime: {regime['regime']}")

    doc = {
        "asOf": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "market": regime,
        "universeSize": len(symbols),
        "stocks": sorted(stocks, key=lambda s: -s["winnerScore"]),
        "stats": {
            "computed": len(stocks),
            "candidates": len([s for s in stocks if s["candidate"]]),
            "stage2": len([s for s in stocks if s["stage2"]]),
            "rsLeaders": len([s for s in stocks if s["rsLeader"]]),
            "breakouts": len([s for s in stocks if "BASE BREAK" in s["tags"]]),
            "vcp": len([s for s in stocks if "VCP COIL" in s["tags"]]),
            "htf": len([s for s in stocks if "HT FLAG" in s["tags"]]),
        },
    }
    out = ROOT / "data" / "screener.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")

    study = {
        "asOf": doc["asOf"],
        "params": {"minGainPct": MIN_GAIN_PCT, "maxWindowSessions": MAX_WINDOW,
                   "years": STUDY_YEARS, "universe": "current Nifty 500 (survivorship applies)"},
        "summary": summary,
        "episodes": episodes[:1500],  # cap for payload
    }
    sout = ROOT / "data" / "study.json"
    with open(sout, "w", encoding="utf-8") as f:
        json.dump(study, f, ensure_ascii=False, separators=(",", ":"))
    log(f"wrote {sout} ({sout.stat().st_size / 1024:.0f} KB)")

    hist_out = ROOT / "data" / "history.json"
    with open(hist_out, "w", encoding="utf-8") as f:
        json.dump(build_history(prices), f, separators=(",", ":"))
    log(f"wrote {hist_out} ({hist_out.stat().st_size / 1024:.0f} KB)")

    if len(stocks) < 300 or summary["episodes"] < 300:
        log(f"QUALITY GATE FAIL: stocks {len(stocks)}, episodes {summary['episodes']}")
        sys.exit(2)
    log("quality gate passed")


if __name__ == "__main__":
    main()
