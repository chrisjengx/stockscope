"""
Focus List Manager — unified factor engine (merged A5 + FL).

Two strategies, each with dual-path selection:
  long_term: A2 fundamental ranking (120) + FL trend_quality (80) → union
  hot_picks: 早起上涨 early_momentum (80) + 持续上涨 d5>0+d60>0 (40) → union

All factor computation is pure math — no LLM in the scoring path.
A2's LLM analysis (fundamental_reports) is consumed as structured data.
Missing A2 data → auto-switch to independent weight formulas (no binary reject).
"""
import json
import logging
import math
from datetime import datetime, timedelta
from collections import defaultdict

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

TOP_PCT = 0.12
MIN_TOTAL_SCORE = 25


def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════
# Factor computation (from former A5)
# ═══════════════════════════════════════════════

def calc_momentum(conn, target_codes):
    """Multi-timeframe momentum: 3/5/10/20/60 trading-day lookback."""
    result = {}
    for code in target_codes:
        rows = conn.execute(
            "SELECT trade_date, close, volume FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT 125",
            (code,),
        ).fetchall()
        if len(rows) < 22:
            continue
        dates = [r["trade_date"] for r in rows]
        closes = [r["close"] for r in rows]
        volumes = [r["volume"] for r in rows]
        def _pct(days):
            if len(closes) > days and closes[days] > 0:
                return round((closes[0] - closes[days]) / closes[days] * 100, 1)
            return 0.0
        if len(volumes) >= 25:
            vol_5d = sum(volumes[:5]) / 5
            vol_20d = sum(volumes[5:25]) / 20
            vol_ratio = round(vol_5d / vol_20d, 2) if vol_20d > 0 else 1.0
        else:
            vol_ratio = 1.0
        raw = {"d3": _pct(3), "d5": _pct(5), "d10": _pct(10),
               "d20": _pct(20), "d60": _pct(60) if len(closes) > 60 else _pct(20),
               "vol_ratio": vol_ratio}
        accel = round(raw["d3"] * 5/3 - raw["d5"], 1)
        has_gap = False
        for i in range(min(10, len(dates) - 1)):
            try:
                d1 = datetime.strptime(dates[i], "%Y-%m-%d")
                d2 = datetime.strptime(dates[i + 1], "%Y-%m-%d")
                if (d1 - d2).days > 5:
                    has_gap = True; break
            except (ValueError, TypeError): pass
        result[code] = {"raw": raw, "acceleration": accel, "has_gap": has_gap, "latest_date": dates[0] if dates else ""}
    return result


def classify_trend_type(momentum_data):
    types = {}
    for code, m in momentum_data.items():
        raw = m["raw"]
        if raw["d60"] > 0 and raw["d20"] > 0: types[code] = "long_term"
        elif raw["d60"] > 0: types[code] = "short_term"
        else: types[code] = "declining"
    return types


def composite_momentum_score(m_raw, trend_type):
    if trend_type == "long_term":
        weights = {"d3": 0.17, "d5": 0.17, "d10": 0.09, "d20": 0.25, "d60": 0.32}
    else:
        weights = {"d3": 0.35, "d5": 0.30, "d10": 0.10, "d20": 0.15, "d60": 0.10}
    score = sum(m_raw[k] * weights[k] for k in weights)
    return round(50 + math.tanh(score / 10) * 40, 1)


def _compute_tech_score(ind, strategy="long_term"):
    components, weights = [], []
    macd = ind.get("macd", {})
    hist = macd.get("histogram", 0)
    if isinstance(hist, (int, float)):
        components.append(50 + math.tanh(hist * 10) * 40); weights.append(1.0)
    rsi = ind.get("rsi_14", 50)
    if isinstance(rsi, (int, float)):
        components.append(rsi)  # use RSI directly — strong RSI = strong trend
        weights.append(1.0)
    bb = ind.get("bollinger", {})
    bb_pos = bb.get("position", 0.5)
    if isinstance(bb_pos, (int, float)):
        # Asymmetric: upper band = strength (reward), lower band = weakness (penalize)
        if bb_pos >= 0.5:
            score_bb = 50 + (bb_pos - 0.5) * 100   # 0.50→50, 0.75→75, 1.00→100
        else:
            score_bb = bb_pos * 100                  # 0.00→0, 0.25→25, 0.50→50
        components.append(score_bb); weights.append(0.5)
    ma = ind.get("ma_alignment", "mixed")
    components.append({"bullish": 75, "mixed": 50, "bearish": 25}.get(ma, 50)); weights.append(1.0)
    obv_trend = ind.get("obv_trend", "flat")
    components.append({"rising": 65, "flat": 50, "falling": 35}.get(obv_trend, 50)); weights.append(0.5)
    vr = ind.get("volume_ratio", 1.0)
    if isinstance(vr, (int, float)):
        components.append(50 + math.tanh((vr - 1.0) * 2) * 25); weights.append(0.3)
    if not components: return 50.0
    return max(0.0, min(100.0, sum(c * w for c, w in zip(components, weights)) / sum(weights)))


# ═══════════════════════════════════════════════
# Gate functions
# ═══════════════════════════════════════════════

def _gate_hot_picks(a2, vol_quality, d3, d5, d60):
    """Hot-picks gates: fundamental hard-fail + volume floor + trend + short direction."""
    # Gate 1: fundamental hard-fail
    if a2 is not None:
        fh = a2.get("financial_health", {}) if isinstance(a2.get("financial_health"), dict) else {}
        eq = a2.get("earnings_quality", {}) if isinstance(a2.get("earnings_quality"), dict) else {}
        fh_rating = fh.get("rating", "?") if isinstance(fh, dict) else "?"
        eq_rating = eq.get("rating", "?") if isinstance(eq, dict) else "?"
        rfs = a2.get("red_flags", [])
        if fh_rating == "POOR" and eq_rating == "LOW":
            return False, "基本面崩溃(财务差+盈利差)"
        high_rfs = [rf for rf in rfs if isinstance(rf, dict) and rf.get("severity") == "HIGH"]
        if len(rfs) >= 5 and len(high_rfs) >= 3:
            return False, f"多项高风险红旗({len(high_rfs)}HIGH/{len(rfs)}total)"
    # Gate 2: volume floor
    if vol_quality < 30:
        return False, f"无量上涨(量价质量{vol_quality:.0f}<30)"
    # Gate 3: trend direction
    if (d5 or 0) < 0 and (d60 or 0) < 0:
        return False, f"下跌趋势(d5={d5:.0f}%<0 d60={d60:.0f}%<0)"
    # Gate 4: short-term must be rising
    if (d3 or 0) <= 0:
        return False, f"短期未涨(d3={d3:.0f}%≤0)"
    return True, "PASS"


def _gate_long_term(a2, d20, d60):
    """Long-term gates: fundamental floor + trend + valuation."""
    # Gate 1: fundamental floor
    if a2 is not None:
        fs = a2.get("fundamental_score")
        if fs is not None and fs < 30:
            return False, f"基本面过弱(评分{fs:.0f}<30)"
    # Gate 2: trend direction
    if (d20 or 0) < 0 and (d60 or 0) < 0:
        return False, f"下跌趋势(d20={d20:.0f}%<0 d60={d60:.0f}%<0)"
    # Gate 3: valuation ceiling
    if a2 is not None:
        fs = a2.get("fundamental_score")
        if fs is not None:
            val = a2.get("valuation", {}) if isinstance(a2.get("valuation"), dict) else {}
            val_rating = val.get("rating", "?") if isinstance(val, dict) else "?"
            if val_rating == "OVERPRICED" and fs < 50:
                return False, "高估+基本面一般(双杀风险)"
    return True, "PASS"


# ═══════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════

def _fundamental_fallback(conn, code):
    """Pure-computation fundamental score for stocks without A2 report."""
    row = conn.execute("""
        SELECT roe, gross_margin, debt_ratio, revenue_yoy, fcf_ratio, pe_percentile
        FROM financials WHERE ts_code=? ORDER BY report_date DESC LIMIT 1
    """, (code,)).fetchone()
    if not row: return 40.0, 0.2
    score = 40.0
    if row["roe"]: score += min(20, max(-20, row["roe"] * 1.0))
    if row["gross_margin"]: score += min(15, max(-10, (row["gross_margin"] - 15) * 0.5))
    if row["revenue_yoy"]: score += min(10, max(-10, row["revenue_yoy"] * 0.3))
    if row["debt_ratio"] and row["debt_ratio"] < 60: score += 10
    elif row["debt_ratio"] and row["debt_ratio"] > 80: score -= 10
    if row["fcf_ratio"] and row["fcf_ratio"] > 0: score += 10
    if row["pe_percentile"] and row["pe_percentile"] < 40: score += 5
    return max(10.0, min(85.0, score)), 0.3


def _load_vol_price(conn, codes):
    """Load 5-day volume + price + MA data."""
    if not codes: return {}, {}, {}
    ph = ",".join("?" * len(codes))
    vol_price = defaultdict(list)
    all_closes = defaultdict(list)
    avg_vol = {}
    for r in conn.execute(f"SELECT ts_code, amount, close FROM daily_quotes WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date DESC", codes).fetchall():
        if len(vol_price[r["ts_code"]]) < 5: vol_price[r["ts_code"]].append((r["amount"], r["close"]))
    for r in conn.execute(f"SELECT ts_code, AVG(amount) as avg_amt FROM daily_quotes WHERE ts_code IN ({ph}) AND trade_date >= date('now','-25 days') GROUP BY ts_code", codes).fetchall():
        avg_vol[r["ts_code"]] = r["avg_amt"] or 0
    for r in conn.execute(f"SELECT ts_code, close FROM daily_quotes WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date DESC", codes).fetchall():
        if len(all_closes[r["ts_code"]]) < 65: all_closes[r["ts_code"]].append(r["close"])
    ma_data = {}
    for code, closes in all_closes.items():
        closes.reverse(); n = len(closes)
        up_days = sum(1 for i in range(max(1, n-10), n) if closes[i] > closes[i-1]) if n >= 11 else 5
        up_ratio = up_days / min(10, max(1, n-1))
        ma_data[code] = {
            "ma5": sum(closes[-5:]) / 5 if n >= 5 else None,
            "ma10": sum(closes[-10:]) / 10 if n >= 10 else None,
            "ma20": sum(closes[-20:]) / 20 if n >= 20 else None,
            "ma60": sum(closes[-60:]) / 60 if n >= 60 else None,
            "up_day_ratio": round(up_ratio, 2),
        }
    result = {}
    for code, entries in vol_price.items():
        entries.reverse(); padded = []
        for i, (amt, close) in enumerate(entries):
            padded.append((amt, close, entries[i-1][1] if i > 0 else None))
        result[code] = padded
    return result, avg_vol, ma_data


# ═══════════════════════════════════════════════
# Sub-scoring functions
# ═══════════════════════════════════════════════

def _direction_structure(d3, d5, d20, d60):
    """Score multi-timeframe trend direction (0-100).

    16 patterns ranked by predictive power for continued upward movement.
    Longer timeframes carry more weight — d60 matters more than d3.
    """
    s = (1 if d3 > 0 else 0, 1 if d5 > 0 else 0, 1 if d20 > 0 else 0, 1 if d60 > 0 else 0)
    # ── Tier 1: all timeframes aligned up ──
    if s == (1, 1, 1, 1): return 90    # 全周期共振上涨 — 最强信号
    # ── Tier 2: strong uptrend with one minor concern ──
    if s == (0, 1, 1, 1): return 70    # 上升中小回调(d3微跌,中长期全涨) — 经典买点
    if s == (1, 1, 0, 1): return 85    # 短中周涨,20日有回调 — 趋势持续中
    if s == (1, 0, 1, 1): return 80    # 回调结束(d3翻正确认反弹,d5仍负因回调深,中长期涨)
    if s == (1, 1, 1, 0): return 75    # 短中周全涨,长期未翻正 — 早期反转
    # ── Tier 3: positive but with notable concerns ──
    if s == (1, 1, 0, 0): return 60    # 短期(3日+5日)上涨确立,中长期待确认
    if s == (1, 0, 0, 1): return 55    # 长期上升中,3日反弹(5日+20日中间有回调)
    if s == (0, 1, 1, 0): return 55    # 中期反弹确立(5日+20日涨),长期未确认
    # ── Tier 4: weak positive or mixed ──
    if s == (1, 0, 0, 0): return 45    # 仅3日上涨 — 微弱反弹
    if s == (0, 1, 0, 1): return 45    # 长期上升+5日反弹,3日+20日信号分歧
    if s == (0, 0, 1, 1): return 40    # 中长期涨但短期(3日+5日)走弱 — 较深回调中
    # ── Tier 5: contradictory or very weak ──
    if s == (1, 0, 1, 0): return 35    # 信号矛盾(3日+20日涨 vs 5日+60日跌)
    # ── Tier 6: bearish — only one timeframe positive ──
    if s == (0, 0, 0, 1): return 25    # 仅60日涨 — 长期上升但近期持续走弱
    if s == (0, 0, 1, 0): return 20    # 仅20日涨 — 孤立弱信号
    if s == (0, 1, 0, 0): return 18    # 仅5日涨 — 极弱
    # ── Tier 7: all down ──
    if s == (0, 0, 0, 0): return 10    # 全周期下跌 — 最强下跌信号
    return 30  # unreachable — all 16 patterns covered above


def _acceleration_state(accel, is_early_reversal, d20=0, d60=0, d3=0):
    # Early reversal + positive accel = strongest signal (fresh breakout)
    if accel > 5 and is_early_reversal: return 95
    if accel > 5: return 85
    if accel > 1: return 80
    if accel >= -1: return 65
    # In confirmed uptrend, d3>0 means short-term momentum is still positive.
    # Deceleration (d3<d5) in this context is healthy consolidation, not exhaustion.
    # Math: d5越大→d3越难跑赢→负accel是数学必然，不是趋势衰竭。
    in_uptrend = (d20 or 0) > 0 and (d60 or 0) > 0
    d3_still_rising = (d3 or 0) > 0
    if in_uptrend and d3_still_rising:
        if accel >= -3: return 60
        if accel >= -5: return 55
        return 45  # d3>0, uptrend intact — even severe decel isn't a reversal
    if in_uptrend:
        if accel >= -3: return 50
        if accel >= -5: return 42
        return 30  # genuine dip within uptrend
    # No confirmed trend — deceleration is more concerning (dead cat risk)
    if accel >= -5: return 35
    return 15


def _ma_alignment(ma5, ma10, ma20, ma60):
    if ma5 is None or ma10 is None or ma20 is None or ma60 is None: return 50
    if ma5 > ma10 > ma20 > ma60: return 85
    if ma5 > ma10 and ma20 > ma60: return 70
    if ma5 > ma10 and ma60 > ma20 and ma20 > ma5: return 65
    if ma5 > ma10 and ma20 < ma60: return 55
    if ma5 < ma10 and ma20 > ma60: return 45
    if ma5 < ma10 and ma20 < ma60 and ma20 > ma5: return 30
    return 20


def _trend_quality(r, is_early_reversal, ma=None):
    d3 = r.get("momentum_d3") or 0; d5 = r.get("momentum_d5") or 0
    d20 = r.get("momentum_d20") or 0; d60 = r.get("momentum_d60") or 0
    accel = r.get("momentum_accel") or 0; ma = ma or {}
    direction = _direction_structure(d3, d5, d20, d60)
    acceleration = _acceleration_state(accel, is_early_reversal, d20, d60, d3)
    ma_score = _ma_alignment(ma.get("ma5"), ma.get("ma10"), ma.get("ma20"), ma.get("ma60"))
    consistency = _clamp(ma.get("up_day_ratio", 0.5) * 100, 0, 100)
    # In confirmed uptrend, direction dominates (the trend IS the signal).
    # Acceleration matters more when the trend is still forming (early reversal).
    in_uptrend = d20 > 0 and d60 > 0
    if in_uptrend:
        return direction * 0.45 + acceleration * 0.20 + ma_score * 0.15 + consistency * 0.20
    return direction * 0.30 + acceleration * 0.35 + ma_score * 0.15 + consistency * 0.20


def _buy_sell_ratio(vol_price):
    up_vols, dn_vols = [], []
    for amt, close, prev_close in vol_price:
        if prev_close is None or prev_close == 0: continue
        if close > prev_close: up_vols.append(amt)
        elif close < prev_close: dn_vols.append(amt)
    if not up_vols and not dn_vols: return 50
    if not dn_vols: return 90
    if not up_vols: return 15
    up_avg = sum(up_vols) / len(up_vols); dn_avg = sum(dn_vols) / len(dn_vols)
    ratio = up_avg / dn_avg if dn_avg > 0 else 1.5
    if ratio > 1.5: return 90
    if ratio >= 1.2: return 75
    if ratio >= 0.8: return 50
    if ratio >= 0.6: return 30
    return 15


def _volume_pattern(vol_price, avg_vol):
    if len(vol_price) < 5: return 50
    amts = [v[0] for v in vol_price]; closes = [v[1] for v in vol_price]
    avg20 = avg_vol if avg_vol and avg_vol > 0 else ((sum(amts) / len(amts)) if amts and sum(amts) > 0 else 1)
    vol_ratio = [a / avg20 for a in amts]; v3 = vol_ratio[-3:]
    p_chg_5d = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0
    p_chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 and closes[-2] > 0 else 0
    if len(closes) >= 3 and closes[-2] < closes[-3] and v3[0] < 1.0 and v3[1] < 1.0 and v3[2] > 1.0 and closes[-1] > closes[-2]: return 90
    if p_chg_1d > 9 and max(v3) < 1.2 and vol_ratio[-1] < 1.5: return 85
    if p_chg_5d > 0 and vol_ratio[0] > 1.0 and vol_ratio[-1] < vol_ratio[0] and p_chg_1d < 0: return 85
    if p_chg_5d > 0 and max(vol_ratio) < 1.0 and p_chg_5d < 8: return 80
    if max(vol_ratio) > 1.5 and vol_ratio[-1] < 1.0 and abs(p_chg_5d) < 5: return 80
    if p_chg_5d > 0 and vol_ratio[-1] >= 0.8: return 85
    if abs(p_chg_5d) < 3 and max(vol_ratio) < 1.3 and min(vol_ratio) > 0.7: return 50
    if max(vol_ratio) > 1.3 and abs(p_chg_5d) < 2: return 20
    if p_chg_5d < -3 and vol_ratio[0] > 1.2 and max(vol_ratio[-2:]) < 0.8: return 25
    if p_chg_1d < -5 and vol_ratio[-1] > 1.5: return 15
    if p_chg_5d < 0 and max(vol_ratio) < 1.0: return 30
    return 50


def _price_efficiency(vol_price, avg_vol):
    if not vol_price or not avg_vol or avg_vol <= 0: return 50
    amts = [v[0] for v in vol_price]; closes = [v[1] for v in vol_price]
    if len(closes) < 2: return 50
    daily_ret = [abs(closes[i] / closes[i-1] - 1) * 100 for i in range(1, len(closes))]
    avg_ret = sum(daily_ret) / len(daily_ret)
    avg_vol_ratio = (sum(amts) / len(amts)) / avg_vol
    if avg_vol_ratio <= 0: return 50
    efficiency = avg_ret / avg_vol_ratio
    if efficiency > 1.5: return 85
    if efficiency >= 0.7: return 60
    return 25


def _volume_quality(vol_price, avg_vol):
    if not vol_price: return 50
    return _buy_sell_ratio(vol_price) * 0.45 + _volume_pattern(vol_price, avg_vol) * 0.35 + _price_efficiency(vol_price, avg_vol) * 0.20


def _short_momentum(r):
    d3 = r.get("momentum_d3") or 0; d5 = r.get("momentum_d5") or 0; d20 = r.get("momentum_d20") or 0
    return _clamp(50 + math.tanh((d3 * 0.40 + d5 * 0.35 + d20 * 0.25) / 15) * 50)


def _direction_health(d3, d5, d20, d60):
    """0-100: Trend health. Sustained uptrend = strong positive. Only truly parabolic or dead-cat penalized."""
    if d60 > 0 and d3 > 0:
        base = 75 if d5 <= 0 else 70  # d5 still negative = pullback recovery, higher value
    elif d60 > 0 and d5 > 0: base = 60
    elif d60 > 0: base = 45
    elif d3 > 0 and d5 > 0: base = 60
    elif d3 > 0: base = 30
    else: base = 10
    # Sustained bonus (d3/d5 already in base score, not double-counted)
    if d20 > 0 and d5 > 0: sustained = 15
    elif d20 > 0: sustained = 10
    else: sustained = 0
    # Declining penalty
    declining = 30 if (d20 < 0 and d60 < 0) else (10 if (d20 < 0 and d3 <= 0) else 0)
    return _clamp(base + sustained - declining)


def _early_momentum_score(r, vol_price, avg_vol, is_early_reversal):
    """0-100: Early-stage momentum for hot_picks. Rewards acceleration + volume + trend health."""
    accel = r.get("momentum_accel") or 0; d3 = r.get("momentum_d3") or 0
    d5 = r.get("momentum_d5") or 0; d20 = r.get("momentum_d20") or 0
    d60 = r.get("momentum_d60") or 0
    accel_q = _acceleration_state(accel, is_early_reversal, d20, d60, d3)
    vol_q = _volume_quality(vol_price, avg_vol)
    dir_q = _direction_health(d3, d5, d20, d60)
    return accel_q * 0.40 + vol_q * 0.35 + dir_q * 0.25


# ═══════════════════════════════════════════════
# Main rebuild
# ═══════════════════════════════════════════════

def rebuild(strategy="long_term"):
    """Rebuild focus list with gate + dual-path selection."""
    conn = get_connection()
    try:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        # Step 1: A0 pool
        tiers = ("'HOLDING'", "'FAVORED'", "'NEUTRAL'")
        rows = conn.execute(f"SELECT s.ts_code, s.name, t.tier FROM stocks s JOIN tier_assignments t ON s.ts_code=t.ts_code WHERE t.tier IN ({','.join(tiers)})").fetchall()
        if not rows: logger.warning("No stocks in A0 pool"); return []
        target_codes = [r["ts_code"] for r in rows]; name_map = {r["ts_code"]: r["name"] for r in rows}
        logger.info(f"FL [{strategy}]: {len(target_codes)} stocks in A0 pool")
        # Step 2: A1 indicators
        ph_all = ",".join("?" * len(target_codes))
        tech_scores = {}
        for r in conn.execute(f"SELECT ts_code, indicators_json FROM indicators WHERE ts_code IN ({ph_all}) AND calc_date = (SELECT MAX(calc_date) FROM indicators)", target_codes).fetchall():
            try: tech_scores[r["ts_code"]] = _compute_tech_score(json.loads(r["indicators_json"]), strategy)
            except: tech_scores[r["ts_code"]] = 50.0
        # Step 3: A2 reports
        a2_cache = {}
        for r in conn.execute(f"SELECT ts_code, report_json FROM fundamental_reports WHERE ts_code IN ({ph_all}) AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)", target_codes).fetchall():
            try: a2_cache[r["ts_code"]] = json.loads(r["report_json"])
            except: pass
        # Step 4: momentum
        momentum_data = calc_momentum(conn, target_codes)
        trend_types = classify_trend_type(momentum_data)
        # Step 5: volume/MA
        vol_price, avg_vol, ma_data = _load_vol_price(conn, target_codes)
        # Step 6: per-stock scoring + gates
        rejected = []; passing = []; skipped = []
        for code in target_codes:
            a2 = a2_cache.get(code); vp = vol_price.get(code, []); av = avg_vol.get(code, 0)
            m = momentum_data.get(code, {}); m_raw = m.get("raw", {}) if m else {}
            tt = trend_types.get(code, "declining"); md = ma_data.get(code, {})
            if not vp or (av <= 0 and all(a == 0 for a, _, _ in vp)): skipped.append(code); continue
            # Compute scores
            vol_q = _volume_quality(vp, av)
            short_mom = _short_momentum({"momentum_d3": m_raw.get("d3", 0), "momentum_d5": m_raw.get("d5", 0), "momentum_d20": m_raw.get("d20", 0)})
            md_fields = {"momentum_d3": m_raw.get("d3", 0), "momentum_d5": m_raw.get("d5", 0), "momentum_d20": m_raw.get("d20", 0), "momentum_d60": m_raw.get("d60", 0), "momentum_accel": m.get("acceleration", 0)}
            # Early reversal signal: acceleration is positive, short-term (d3) is rising,
            # but at least one longer timeframe (d5/d20/d60) has not yet turned positive.
            # This indicates a fresh reversal rather than a sustained trend — higher reward potential.
            is_early_reversal = (
                (m.get("acceleration", 0) or 0) > 0
                and (m_raw.get("d3", 0) or 0) > 0
                and (
                    (m_raw.get("d5", 0) or 0) <= 0
                    or (m_raw.get("d20", 0) or 0) <= 0
                    or (m_raw.get("d60", 0) or 0) <= 0
                )
            )
            trend_q = _trend_quality(md_fields, is_early_reversal, md)
            early_mom = _early_momentum_score(md_fields, vp, av, is_early_reversal)
            # Fundamental score
            fund_from_fallback = False
            if a2 is not None:
                fund_score = a2.get("fundamental_score"); fund_conf = a2.get("confidence", 0.3)
                if fund_score is None: fund_score, fund_conf = _fundamental_fallback(conn, code); fund_from_fallback = True
            else: fund_score, fund_conf = _fundamental_fallback(conn, code); fund_from_fallback = True
            # Gates
            if strategy == "hot_picks":
                passed, gate_reason = _gate_hot_picks(a2, vol_q, m_raw.get("d3", 0), m_raw.get("d5", 0), m_raw.get("d60", 0))
            else:
                passed, gate_reason = _gate_long_term(a2, m_raw.get("d20", 0), m_raw.get("d60", 0))
            if not passed: rejected.append({"ts_code": code, "reason": gate_reason}); continue
            has_a2 = a2 is not None
            passing.append({"ts_code": code, "name": name_map.get(code, "?"), "has_a2": has_a2,
                "short_momentum": short_mom, "early_momentum": early_mom, "vol_quality": vol_q,
                "trend_quality": trend_q, "fund_score": fund_score, "fund_conf": fund_conf,
                "fund_from_fallback": fund_from_fallback, "trend_type": tt,
                "tech_score": tech_scores.get(code, 50), "momentum_raw": m_raw,
                "momentum_accel": m.get("acceleration"), "ma_data": md, "source_path": ""})

        # Step 7: Dual-path selection
        if strategy == "hot_picks":
            # Path 1: 早起上涨 (60) — early_momentum
            early_ranking = sorted(passing, key=lambda s: -s["early_momentum"])
            early_picks = {s["ts_code"]: s for s in early_ranking[:60]}
            for s in early_picks.values(): s["source_path"] = "早起上涨"
            # Path 2: 持续上涨 (60) — d5>0 AND d60>0
            sustained_pool = [s for s in passing if (s["momentum_raw"].get("d5", 0) or 0) > 0 and (s["momentum_raw"].get("d60", 0) or 0) > 0]
            sustained_ranking = sorted(sustained_pool, key=lambda s: -(s["short_momentum"] * 0.5 + s["vol_quality"] * 0.5))
            sustained_picks = {s["ts_code"]: s for s in sustained_ranking[:60]}
            for s in sustained_picks.values(): s["source_path"] = "持续上涨" if s["ts_code"] not in early_picks else "双路径"
            # Union
            fl_picks = early_picks.copy()
            for code, s in sustained_picks.items():
                if code not in fl_picks: fl_picks[code] = s
            if len(fl_picks) < 120:
                for s in sustained_ranking:
                    if len(fl_picks) >= 120: break
                    if s["ts_code"] not in fl_picks: s["source_path"] = "持续上涨"; fl_picks[s["ts_code"]] = s
            # A2 path
            a2_eligible = [s for s in passing if s["has_a2"] and not s["fund_from_fallback"] and s["short_momentum"] >= 60]
            a2_ranking = sorted(a2_eligible, key=lambda s: -s["fund_score"])
            a2_picks = {s["ts_code"]: s for s in a2_ranking[:80]}
            for s in a2_picks.values(): s["source_path"] = "A2基本面排名" if s["ts_code"] not in fl_picks else "双路径"
        else:
            # long_term: A2 (120) + FL trend_quality (80)
            a2_eligible = [s for s in passing if s["has_a2"] and not s["fund_from_fallback"]]
            a2_ranking = sorted(a2_eligible, key=lambda s: -s["fund_score"])
            a2_picks = {s["ts_code"]: s for s in a2_ranking[:120]}
            for s in a2_picks.values(): s["source_path"] = "A2基本面排名"
            fl_ranking = sorted(passing, key=lambda s: -s["trend_quality"])
            fl_picks = {}
            for s in fl_ranking:
                if len(fl_picks) >= 80: break
                # FL quality floor: ≥2 HIGH-severity red flags = reliable signal
                # of genuine fundamental deterioration. Single HIGH flag could be
                # a lagging indicator on a turn-around story; two+ means real trouble.
                a2_rpt = a2_cache.get(s["ts_code"])
                if a2_rpt is not None:
                    rfs = a2_rpt.get("red_flags", [])
                    if isinstance(rfs, list):
                        high_n = sum(1 for rf in rfs if isinstance(rf, dict) and rf.get("severity") == "HIGH")
                        if high_n >= 2:
                            continue
                fl_picks[s["ts_code"]] = s
            for s in fl_picks.values(): s["source_path"] = "FL技术排名" if s["ts_code"] not in a2_picks else "双路径"

        # Union
        selected_codes = set(fl_picks.keys()) | set(a2_picks.keys())
        selected = [s for s in passing if s["ts_code"] in selected_codes]

        # Log
        gate_reject_n = len(rejected)
        from collections import Counter
        src_counts = Counter(s.get("source_path", "?") for s in selected)
        dual_n = sum(1 for s in selected if s.get("source_path") == "双路径")
        logger.info(f"FL [{strategy}]: {len(passing)} passed gates, {gate_reject_n} rejected → final={len(selected)} ({dict(src_counts)})")
        if gate_reject_n > 0:
            reason_counts = Counter(r["reason"] for r in rejected)
            for reason, count in reason_counts.most_common(5):
                logger.info(f"  Gate reject: {reason} ({count}只)")

        # Step 8: Persist
        conn.execute("DELETE FROM composite_scores WHERE strategy=? AND calc_date=?", (strategy, trade_date))
        conn.execute("DELETE FROM focus_list WHERE strategy=?", (strategy,))
        for i, s in enumerate(selected):
            ts_code = s["ts_code"]
            # Multi-factor total_score: combines primary driver with supporting factors.
            # All component scores are already computed and in the 0-100 range.
            if strategy == "long_term":
                # 2-4 week hold: fundamental quality + trend timing + technical confirm.
                # Base weights: fund 50% / trend 30% / tech 20%.
                # When fundamental data is thin (low LLM confidence), shift weight
                # to observable factors: trend and price tell their own story.
                conf = s.get("fund_conf", 0.5)
                fw = 0.50 * max(0.1, conf)   # fund weight proportional to confidence
                released = 0.50 - fw
                tw = 0.30 + released * 0.6   # trend gets 60% of released weight
                pw = 0.20 + released * 0.4   # tech gets 40% of released weight
                total_score_raw = (
                    s["fund_score"] * fw +
                    s["trend_quality"] * tw +
                    s["tech_score"] * pw
                )
            else:
                # 3-5 day hold: momentum burst (50%) + volume confirmation (25%) + direction (15%) + structure (10%)
                total_score_raw = (
                    s["early_momentum"] * 0.50 +
                    s["vol_quality"] * 0.25 +
                    s["short_momentum"] * 0.15 +
                    s["trend_quality"] * 0.10
                )
            total_score = round(max(0.0, min(100.0, total_score_raw)), 1)
            conn.execute("""INSERT OR REPLACE INTO composite_scores (ts_code, calc_date, strategy, tech_score, fundamental_score, macro_fit, momentum, total_score, rank, tier_action, trend_type, momentum_d3, momentum_d5, momentum_d20, momentum_d60, momentum_accel) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts_code, trade_date, strategy, round(s["tech_score"], 1), round(s["fund_score"], 1), 0,
                 composite_momentum_score(s["momentum_raw"], s["trend_type"]) if s["momentum_raw"] else 50,
                 round(total_score, 1), i + 1, "PROMOTE" if i < len(selected) * 0.2 else "STAY",
                 s["trend_type"], s["momentum_raw"].get("d3"), s["momentum_raw"].get("d5"),
                 s["momentum_raw"].get("d20"), s["momentum_raw"].get("d60"), s["momentum_accel"]))
            tier = "价值优选" if strategy == "long_term" else "动能热点"
            conn.execute("INSERT INTO focus_list (ts_code, name, total_score, rank, list_date, position, tier, strategy, value_score, momentum_score) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (ts_code, s["name"], round(total_score, 1), i + 1, trade_date[:10], i + 1, tier, strategy,
                 round(s.get("fund_score", 0), 1), round(s.get("early_momentum", 0), 1)))
        conn.commit()
        return selected
    finally:
        conn.close()


def get_focus_codes(conn, strategy=None):
    if strategy:
        rows = conn.execute("SELECT ts_code, name, total_score, position, value_score, momentum_score FROM focus_list WHERE strategy=? ORDER BY position", (strategy,)).fetchall()
    else:
        rows = conn.execute("SELECT ts_code, name, industry, total_score, position, value_score, momentum_score FROM focus_list ORDER BY position").fetchall()
    return [dict(r) for r in rows]
