"""
Agent 1: Technical Scanner — batch compute 13 technical indicators.
Pure Python computation, no LLM, no scoring. Scoring is the responsibility of A5 Fusion.
"""
import json
import time
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


def load_ohlcv(conn, code, days=120):
    rows = conn.execute(
        "SELECT trade_date, open, high, low, close, volume, amount FROM daily_quotes "
        "WHERE ts_code=? ORDER BY trade_date ASC",
        (code,),
    ).fetchall()
    if len(rows) < 30:
        return None
    df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low", "close", "volume", "amount"])
    for col in ["close", "high", "low", "volume"]:
        df[col] = df[col].astype(float)
    return df.tail(days)


def compute_indicators(df):
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    volumes = df["volume"].values
    n = len(closes)

    # --- MACD ---
    ema12 = pd.Series(closes).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(closes).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd_hist = 2 * (dif - dea)
    macd_signal = "bullish" if dif[-1] > dea[-1] and macd_hist[-1] > 0 else "bearish" if dif[-1] < dea[-1] else "neutral"

    # --- RSI-14 (Wilder's smoothing: alpha = 1/14) ---
    delta = np.diff(closes, prepend=closes[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values[-1]
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values[-1]
    rsi = 100 - 100 / (1 + avg_gain / avg_loss) if avg_loss > 0 else 100

    # --- Bollinger Bands ---
    ma20 = np.mean(closes[-20:])
    std20 = np.std(closes[-20:])
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_position = (closes[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    bb_bandwidth = (bb_upper - bb_lower) / ma20 if ma20 > 0 else 0

    # --- MA alignment ---
    ma5 = np.mean(closes[-5:])
    ma10 = np.mean(closes[-10:])
    ma60 = np.mean(closes[-60:]) if n >= 60 else ma10
    ma_alignment = "bullish" if ma5 > ma10 > ma60 else "bearish" if ma5 < ma10 < ma60 else "mixed"

    # --- OBV trend ---
    obv = np.zeros(n)
    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]
    obv_trend = "rising" if obv[-1] > np.mean(obv[-10:]) else "falling"

    # --- Volume ratio ---
    vol_ratio = 1.0
    if len(volumes) >= 25 and np.mean(volumes[-25:-5]) > 0:
        vol_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-25:-5])

    # --- KDJ (full recursive computation over 9-period rolling RSV) ---
    kdj_k, kdj_d = 50.0, 50.0
    for i in range(max(0, n - 60), n):
        window_lows = lows[max(0, i-8):i+1]
        window_highs = highs[max(0, i-8):i+1]
        low_n = np.min(window_lows)
        high_n = np.max(window_highs)
        rsv = (closes[i] - low_n) / (high_n - low_n) * 100 if high_n != low_n else 50.0
        kdj_k = 2/3 * kdj_k + 1/3 * rsv
        kdj_d = 2/3 * kdj_d + 1/3 * kdj_k
    kdj_j = 3 * kdj_k - 2 * kdj_d
    k = kdj_k
    d_val = kdj_d
    j = kdj_j

    # --- Candlestick patterns ---
    body = abs(closes[-1] - df["open"].values[-1])
    lower_shadow = min(closes[-1], df["open"].values[-1]) - lows[-1]
    upper_shadow = highs[-1] - max(closes[-1], df["open"].values[-1])
    is_hammer = bool(lower_shadow > body * 2 and upper_shadow < body * 0.3)
    # Engulfing
    prev_body = abs(df["close"].values[-2] - df["open"].values[-2])
    is_bullish_engulfing = bool(closes[-2] < df["open"].values[-2]
                                and closes[-1] > df["open"].values[-1]
                                and body > prev_body * 1.5)
    is_bearish_engulfing = bool(closes[-2] > df["open"].values[-2]
                                and closes[-1] < df["open"].values[-1]
                                and body > prev_body * 1.5)
    # Doji
    is_doji = bool(body < (highs[-1] - lows[-1]) * 0.1) if (highs[-1] - lows[-1]) > 0 else False

    return {
        "macd": {"dif": round(float(dif[-1]), 4), "dea": round(float(dea[-1]), 4),
                 "hist": round(float(macd_hist[-1]), 4), "signal": macd_signal},
        "rsi_14": round(float(rsi), 1),
        "bollinger": {"upper": round(float(bb_upper), 2), "mid": round(float(ma20), 2),
                       "lower": round(float(bb_lower), 2), "position": round(float(bb_position), 3),
                       "bandwidth": round(float(bb_bandwidth), 4)},
        "ma_alignment": ma_alignment,
        "ma5": round(float(ma5), 2),
        "obv_trend": obv_trend,
        "volume_ratio": round(float(vol_ratio), 2),
        "kdj": {"k": round(float(k), 1), "d": round(float(d_val), 1), "j": round(float(j), 1)},
        "hammer": is_hammer,
        "bullish_engulfing": is_bullish_engulfing,
        "bearish_engulfing": is_bearish_engulfing,
        "doji": is_doji,
    }




def run(tiers=None, trade_date=None):
    if tiers is None:
        tiers = ["HOLDING", "FAVORED", "NEUTRAL"]
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    start = time.time()
    conn = get_connection()
    logger.info(f"Agent 1 starting: tiers={tiers} date={trade_date}")

    try:
        placeholders = ",".join("?" * len(tiers))
        codes = [r["ts_code"] for r in conn.execute(
            f"SELECT ts_code FROM tier_assignments WHERE tier IN ({placeholders})", tiers
        ).fetchall()]
        logger.info(f"Target: {len(codes)} stocks")

        processed = 0
        for i, code in enumerate(codes):
            df = load_ohlcv(conn, code)
            if df is None:
                continue
            indicators = compute_indicators(df)
            conn.execute(
                "INSERT OR REPLACE INTO indicators (ts_code, calc_date, indicators_json) VALUES (?,?,?)",
                (code, trade_date, json.dumps(indicators)),
            )
            processed += 1
            if (i + 1) % 200 == 0:
                conn.commit()
                logger.info(f"  {i+1}/{len(codes)} scored")

        conn.commit()
        elapsed = time.time() - start
        logger.info(f"Agent 1 complete: {processed} stocks in {elapsed:.1f}s")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, stocks_processed, duration_s, summary) "
            "VALUES (1, ?, 'SUCCESS', ?, ?, ?)",
            (trade_date, processed, elapsed, f"{processed} indicators computed"),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 1 failed: {e}")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (1, ?, 'FAILED', ?, ?)",
            (trade_date, time.time() - start, str(e)[:200]),
        )
        conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", type=str, nargs="+", default=["HOLDING", "FAVORED", "NEUTRAL"])
    args = p.parse_args()
    run(tiers=args.tiers)
