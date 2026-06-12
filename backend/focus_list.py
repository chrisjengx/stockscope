"""
Focus List Manager — dual-strategy independent scoring.

Two continuous scores per stock:
  value_score    → long_term FL (fundamentals + trend sustainability)
  momentum_score → hot_picks FL  (short-term momentum + volume + sector flow)

Missing A2 data → auto-switch to independent weight formulas (no binary reject).
"""
import json
import logging
import math
from datetime import datetime
from collections import defaultdict

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

TOP_PCT = 0.12     # default, overridden per strategy in rebuild()
MIN_TOTAL_SCORE = 25  # quality floor for focus list inclusion

# ── helpers ──

def _clamp(v, lo=0, hi=100):
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════
# Sub-scoring functions
# ═══════════════════════════════════════════════

def _direction_structure(d3, d5, d20, d60):
    """Score 0-100 based on which timeframes are trending up."""
    s = (1 if d3 > 0 else 0, 1 if d5 > 0 else 0,
         1 if d20 > 0 else 0, 1 if d60 > 0 else 0)

    if s == (1, 1, 1, 1):   return 95
    if s == (1, 1, 0, 1):   return 75   # short+long up, mid perturbed
    if s == (1, 1, 1, 0):   return 70   # broad recovery, long about to confirm
    if s == (1, 1, 0, 0):   return 60   # short rebound, watch
    if s == (1, 0, 0, 0):   return 45   # earliest reversal signal
    if s == (0, 0, 1, 1):   return 40   # long intact, short pullback
    if s == (1, 0, 1, 1):   return 55   # d3 just flipped, long still strong
    if s == (0, 0, 0, 1):   return 25   # long-term tail
    if s == (0, 0, 0, 0):   return 10
    return 30


def _acceleration_state(accel, accel_prev3):
    """Score 0-100 based on acceleration value and recent trend.
    accel_prev3: True if all of the prior 3 days' accel values were ≤ 0.
    """
    if accel > 5 and accel_prev3:   return 95   # just flipped positive — early reversal
    if accel > 5:                   return 55   # sustained acceleration, may be late
    if accel > 1:                   return 80   # gentle acceleration
    if accel >= -1:                 return 60   # stable
    if accel >= -5:                 return 35   # gentle deceleration
    return 15   # strong deceleration


def _ma_alignment(ma5, ma10, ma20, ma60):
    """Score 0-100 based on MA ordering."""
    if ma5 is None or ma10 is None or ma20 is None or ma60 is None:
        return 50
    if ma5 > ma10 > ma20 > ma60:                        return 85
    if ma5 > ma10 and ma20 > ma60:                      return 70
    if ma5 > ma10 and ma60 > ma20 and ma20 > ma5:       return 65
    if ma5 > ma10 and ma20 < ma60:                      return 55
    if ma5 < ma10 and ma20 > ma60:                      return 45
    if ma5 < ma10 and ma20 < ma60 and ma20 > ma5:       return 30
    return 20


def _trend_quality(r, accel_prev3, ma=None):
    """0-100: How sustainable is this stock's uptrend?"""
    d3  = r.get("momentum_d3") or 0
    d5  = r.get("momentum_d5") or 0
    d20 = r.get("momentum_d20") or 0
    d60 = r.get("momentum_d60") or 0
    accel = r.get("momentum_accel") or 0
    ma = ma or {}

    direction   = _direction_structure(d3, d5, d20, d60)
    acceleration = _acceleration_state(accel, accel_prev3)
    ma_score    = _ma_alignment(ma.get("ma5"), ma.get("ma10"), ma.get("ma20"), ma.get("ma60"))
    consistency = _clamp(ma.get("up_day_ratio", 0.5) * 100, 0, 100)

    return direction * 0.30 + acceleration * 0.35 + ma_score * 0.15 + consistency * 0.20


def _buy_sell_ratio(vol_price):
    """0-100: Avg volume on up-days / avg volume on down-days."""
    up_vols, dn_vols = [], []
    for amt, close, prev_close in vol_price:
        if prev_close is None or prev_close == 0:
            continue
        if close > prev_close:
            up_vols.append(amt)
        elif close < prev_close:
            dn_vols.append(amt)
    if not up_vols and not dn_vols:
        return 50
    if not dn_vols:
        return 90
    if not up_vols:
        return 15
    up_avg = sum(up_vols) / len(up_vols)
    dn_avg = sum(dn_vols) / len(dn_vols)
    ratio = up_avg / dn_avg if dn_avg > 0 else 1.5
    if ratio > 1.5:       return 90
    if ratio >= 1.2:      return 75
    if ratio >= 0.8:      return 50
    if ratio >= 0.6:      return 30
    return 15


def _volume_pattern(vol_price, avg_vol):
    """0-100: Best-matching 5-day price-volume pattern."""
    if len(vol_price) < 5:
        return 50

    amts     = [v[0] for v in vol_price]
    closes   = [v[1] for v in vol_price]
    avg20 = 1  # safe default
    if avg_vol and avg_vol > 0:
        avg20 = avg_vol
    elif amts:
        vsum = sum(amts)
        avg20 = (vsum / len(amts)) if vsum > 0 else 1

    vol_ratio = [a / avg20 for a in amts]  # 1.0 = at average

    # Recent 3-day volume trends
    v3 = vol_ratio[-3:]
    p_chg_5d = (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0
    p_chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 and closes[-2] > 0 else 0

    # 缩量回调后放量反弹 (90)
    if len(closes) >= 3:
        mid_lo = closes[-3]
        if closes[-2] < mid_lo and v3[0] < 1.0 and v3[1] < 1.0 and v3[2] > 1.0 and closes[-1] > closes[-2]:
            return 90

    # 缩量涨停/一字板 (85)
    if p_chg_1d > 9 and max(v3) < 1.2 and vol_ratio[-1] < 1.5:
        return 85

    # 持续放量上涨 + 缩量回调 (85)
    if p_chg_5d > 0 and vol_ratio[0] > 1.0 and vol_ratio[-1] < vol_ratio[0] and p_chg_1d < 0:
        return 85

    # 缩量缓慢爬升 (80)
    if p_chg_5d > 0 and max(vol_ratio) < 1.0 and p_chg_5d < 8:
        return 80

    # 放量突破 + 缩量整固 (80)
    if max(vol_ratio) > 1.5 and vol_ratio[-1] < 1.0 and abs(p_chg_5d) < 5:
        return 80

    # 放量上涨 + 量平价升 (70)
    if p_chg_5d > 0 and vol_ratio[-1] >= 0.8:
        return 70

    # 横盘 + 量平 (50)
    if abs(p_chg_5d) < 3 and max(vol_ratio) < 1.3 and min(vol_ratio) > 0.7:
        return 50

    # 放量滞涨 (20)
    if max(vol_ratio) > 1.3 and abs(p_chg_5d) < 2:
        return 20

    # 放量下跌 + 缩量反弹 (25)
    if p_chg_5d < -3 and vol_ratio[0] > 1.2 and max(vol_ratio[-2:]) < 0.8:
        return 25

    # 放量长阴 (15)
    if p_chg_1d < -5 and vol_ratio[-1] > 1.5:
        return 15

    # 缩量阴跌 (30)
    if p_chg_5d < 0 and max(vol_ratio) < 1.0:
        return 30

    return 50


def _price_efficiency(vol_price, avg_vol):
    """0-100: How much price movement per unit of volume?"""
    if not vol_price or not avg_vol or avg_vol <= 0:
        return 50
    amts   = [v[0] for v in vol_price]
    closes = [v[1] for v in vol_price]
    if len(closes) < 2:
        return 50
    daily_ret = [abs(closes[i] / closes[i-1] - 1) * 100 for i in range(1, len(closes))]
    avg_ret = sum(daily_ret) / len(daily_ret)
    avg_vol_ratio = (sum(amts) / len(amts)) / avg_vol
    if avg_vol_ratio <= 0:
        return 50
    efficiency = avg_ret / avg_vol_ratio
    if efficiency > 1.5:    return 85
    if efficiency >= 0.7:   return 60
    return 25


def _volume_quality(vol_price, avg_vol):
    """0-100: Price-volume relationship quality."""
    if not vol_price:
        return 50
    a = _buy_sell_ratio(vol_price)
    b = _volume_pattern(vol_price, avg_vol)
    c = _price_efficiency(vol_price, avg_vol)
    return a * 0.45 + b * 0.35 + c * 0.20


def _short_momentum(r):
    """0-100: Short-term momentum from d3/d5/d20."""
    d3  = r.get("momentum_d3") or 0
    d5  = r.get("momentum_d5") or 0
    d20 = r.get("momentum_d20") or 0
    raw = d3 * 0.40 + d5 * 0.35 + d20 * 0.25
    return _clamp(50 + math.tanh(raw / 15) * 50)


def _fundamental_score_v(a2):
    """0-100: Fundamental quality (for value_score)."""
    if not a2:
        return 25
    fh  = a2.get("financial_health", {}).get("rating", "?") if isinstance(a2.get("financial_health"), dict) else "?"
    eq  = a2.get("earnings_quality", {}).get("rating", "?") if isinstance(a2.get("earnings_quality"), dict) else "?"
    gq  = a2.get("growth_quality", {}).get("rating", "?") if isinstance(a2.get("growth_quality"), dict) else "?"
    val = a2.get("valuation", {}).get("rating", "?") if isinstance(a2.get("valuation"), dict) else "?"
    rfs = len(a2.get("red_flags", []))

    base = {"GOOD": 85, "MEDIUM": 70, "UNKNOWN": 40}.get(fh, 15)
    if eq == "HIGH":    base += 10
    elif eq == "LOW":   base -= 10
    if gq == "HIGH":    base += 5
    if val == "GOOD" or (isinstance(val, str) and ("FAIR" in val.upper() or "UNDER" in val.upper())):
        base += 5
    elif val == "OVERPRICED":
        base -= 15
    base -= min(40, rfs * 8)
    return _clamp(base)


def _valuation_score_v(a2):
    """0-100: Standalone valuation rating."""
    if not a2:
        return 50
    val = a2.get("valuation", {}).get("rating", "?") if isinstance(a2.get("valuation"), dict) else "?"
    v = str(val).upper()
    if "UNDER" in v:    return 95
    if "FAIR" in v:     return 80
    if "GOOD" in v:     return 70
    if "OVER" in v:     return 30
    return 50


def _fundamental_floor_v(a2):
    """0-100: Minimal quality check (for hot_picks). High unless truly bad."""
    if not a2:
        return 80
    fh = a2.get("financial_health", {}).get("rating", "?") if isinstance(a2.get("financial_health"), dict) else "?"
    eq = a2.get("earnings_quality", {}).get("rating", "?") if isinstance(a2.get("earnings_quality"), dict) else "?"
    rfs = len(a2.get("red_flags", []))
    if fh == "POOR" and eq == "LOW":
        return 25
    if fh == "POOR":
        return 50
    if rfs >= 3:
        return 35
    return 80


# ═══════════════════════════════════════════════
# Top-level scoring functions
# ═══════════════════════════════════════════════

def compute_value_score(r, a2, vol_price, avg_vol, accel_prev3, ma=None):
    """0-100: Value-investing conviction (long_term FL)."""
    has_a2 = a2 is not None
    momentum    = r.get("momentum") or 50
    trend       = _trend_quality(r, accel_prev3, ma)
    vol         = _volume_quality(vol_price, avg_vol)

    if has_a2:
        fund = _fundamental_score_v(a2)
        valv = _valuation_score_v(a2)
        return fund * 0.40 + trend * 0.35 + momentum * 0.15 + valv * 0.10
    else:
        return trend * 0.50 + vol * 0.25 + momentum * 0.25


def compute_momentum_score(r, a2, vol_price, avg_vol):
    """0-100: Short-term momentum conviction (hot_picks FL)."""
    has_a2 = a2 is not None
    short   = _short_momentum(r)
    vol     = _volume_quality(vol_price, avg_vol)

    if has_a2:
        floor = _fundamental_floor_v(a2)
        return short * 0.45 + vol * 0.35 + floor * 0.20
    else:
        return short * 0.50 + vol * 0.50


# ═══════════════════════════════════════════════
# Tier label
# ═══════════════════════════════════════════════

def _value_tier(score):
    if score >= 70: return "价值优选"
    if score >= 50: return "价值关注"
    return "一般关注"


def _momentum_tier(score):
    if score >= 70: return "动能热点"
    if score >= 50: return "动能追踪"
    return "一般关注"


# ═══════════════════════════════════════════════
# Legacy classifier (preserved for reference; not used in FL selection)
# ═══════════════════════════════════════════════
def classify_stock(r, a2_report=None, momentum_d3=None, momentum_d5=None, momentum_d20=None,
                   p80=None, p60=None, p50=None, p40=None, p30=None):
    """Return list of matching categories (multi-label). Strategy-specific views, not mutually exclusive.

    A stock can be both 稳健型 (for long_term) AND 动量型 (for hot_picks) simultaneously.
    """
    has_a2 = r.get("has_a2") == 1 and a2_report is not None
    momentum = r.get("momentum") or 50
    total_score = r.get("total_score") or 40
    fund_score = r.get("fundamental_score") or 40

    # Percentile-based thresholds, fallback to hardcoded defaults
    mp80 = p80 if p80 is not None else 85
    mp60 = p60 if p60 is not None else 65
    mp50 = p50 if p50 is not None else 55
    mp40 = p40 if p40 is not None else 50
    mp30 = p30 if p30 is not None else 40

    # ── Parse A2 quality ──
    eq = "?"; gq = "?"; fh = "?"; val = "?"; rfs = 0
    if a2_report:
        eq = (a2_report.get("earnings_quality") or {}).get("rating", "?") if isinstance(a2_report.get("earnings_quality"), dict) else "?"
        gq = (a2_report.get("growth_quality") or {}).get("rating", "?") if isinstance(a2_report.get("growth_quality"), dict) else "?"
        fh = (a2_report.get("financial_health") or {}).get("rating", "?") if isinstance(a2_report.get("financial_health"), dict) else "?"
        val = (a2_report.get("valuation") or {}).get("rating", "?") if isinstance(a2_report.get("valuation"), dict) else "?"
        rfs = len(a2_report.get("red_flags", []))

    d3 = momentum_d3 or 0; d5 = momentum_d5 or 0; d20 = momentum_d20 or 0

    # ── Hard gate: too many red flags → 观望型 only ──
    if rfs >= 5:
        return ["观望型"]
    # eq=LOW+fh=POOR no longer hard-gated — A7 LLM has discretion to include with rationale

    tags = set()

    # ── Long-term categories ──
    # 价值型: good health, reasonable valuation, moderate flags, momentum ≥ p40
    if fh == "GOOD" and val not in ("OVERPRICED", "?") and rfs <= 2 and momentum >= mp40:
        tags.add("价值型")

    # 成长型: high growth OR high earnings quality, reasonable valuation, momentum ≥ p50
    if (gq == "HIGH" or eq == "HIGH") and val != "OVERPRICED" and rfs <= 1 and momentum >= mp50:
        tags.add("成长型")

    # 稳健型: not poor health, known fh, momentum ≥ p30, rfs ≤ 2
    if fh != "POOR" and fh != "?" and momentum >= mp30 and rfs <= 2:
        tags.add("稳健型")

    # ── Hot-picks categories ──
    # 动量型: strong momentum, quality floor
    if momentum >= mp80 and eq != "LOW":
        tags.add("动量型")

    # 突破型: moderate momentum, quality floor
    if momentum >= mp60 and eq != "LOW" and rfs <= 3:
        tags.add("突破型")

    # 短期交易型: multi-timeframe momentum positive, fundamentals not terrible
    if d3 > 0 and d5 > 0 and d20 > 0:
        if fund_score >= 30 or not has_a2:
            tags.add("短期交易型")

    # ── Fallbacks if nothing matched ──
    if not tags:
        if has_a2 and fh != "POOR" and fh != "?":
            tags.add("稳健型")
        elif has_a2 and momentum >= mp60 and eq != "LOW":
            tags.add("动量型")
        elif not has_a2 and d3 > 0 and d5 > 0:
            tags.add("短期交易型")
        else:
            tags.add("观望型")

    return sorted(tags)


def _load_vol_price(conn, codes):
    """Load 5-day volume + price + MA data per code for FL scoring."""
    if not codes:
        return {}, {}, {}
    ph = ",".join("?" * len(codes))
    vol_price = defaultdict(list)
    all_closes = defaultdict(list)
    avg_vol = {}
    # Last 5 trading days per code (for volume analysis)
    for r in conn.execute(
        f"SELECT ts_code, amount, close FROM daily_quotes "
        f"WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date DESC",
        codes
    ).fetchall():
        if len(vol_price[r["ts_code"]]) < 5:
            vol_price[r["ts_code"]].append((r["amount"], r["close"]))
    # 20-day average volume
    for r in conn.execute(
        f"SELECT ts_code, AVG(amount) as avg_amt FROM daily_quotes "
        f"WHERE ts_code IN ({ph}) AND trade_date >= date('now','-25 days') "
        f"GROUP BY ts_code", codes
    ).fetchall():
        avg_vol[r["ts_code"]] = r["avg_amt"] or 0
    # MA data: close history for MA5/MA10/MA20/MA60 (up to 65 days)
    for r in conn.execute(
        f"SELECT ts_code, close FROM daily_quotes "
        f"WHERE ts_code IN ({ph}) ORDER BY ts_code, trade_date DESC",
        codes
    ).fetchall():
        if len(all_closes[r["ts_code"]]) < 65:
            all_closes[r["ts_code"]].append(r["close"])
    ma_data = {}
    for code, closes in all_closes.items():
        closes.reverse()  # oldest first
        n = len(closes)
        up_days = sum(1 for i in range(max(1, n-10), n) if closes[i] > closes[i-1]) if n >= 11 else 5
        up_ratio = up_days / min(10, max(1, n-1))
        ma_data[code] = {
            "ma5":  sum(closes[-5:]) / 5 if n >= 5 else None,
            "ma10": sum(closes[-10:]) / 10 if n >= 10 else None,
            "ma20": sum(closes[-20:]) / 20 if n >= 20 else None,
            "ma60": sum(closes[-60:]) / 60 if n >= 60 else None,
            "up_day_ratio": round(up_ratio, 2),
        }
    # Reverse to chronological order, build prev_close
    result = {}
    for code, entries in vol_price.items():
        entries.reverse()  # oldest first
        padded = []
        for i, (amt, close) in enumerate(entries):
            prev = entries[i-1][1] if i > 0 else None
            padded.append((amt, close, prev))
        result[code] = padded
    return result, avg_vol, ma_data


def rebuild(strategy="long_term"):
    """Rebuild focus list with dual independent scoring.

    long_term → top N by value_score
    hot_picks  → top N by momentum_score
    """
    conn = get_connection()
    try:
        tiers = ("'HOLDING'", "'FAVORED'", "'NEUTRAL'")

        rows = conn.execute(f"""
            SELECT s.ts_code, s.name,
                   cs.total_score, cs.rank, t.tier, t.confidence,
                   cs.momentum, cs.fundamental_score, cs.tech_score,
                   cs.trend_type, cs.momentum_d3, cs.momentum_d5, cs.momentum_d20,
                   cs.momentum_d60, cs.momentum_accel,
                   CASE WHEN fr.ts_code IS NOT NULL THEN 1 ELSE 0 END as has_a2
            FROM stocks s
            JOIN tier_assignments t ON s.ts_code = t.ts_code AND t.tier IN ({','.join(tiers)})
            JOIN composite_scores cs ON s.ts_code = cs.ts_code
                AND cs.calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
                AND cs.strategy = ?
            LEFT JOIN fundamental_reports fr ON s.ts_code = fr.ts_code
            WHERE cs.total_score IS NOT NULL
            GROUP BY s.ts_code
        """, (strategy, strategy)).fetchall()

        if not rows:
            logger.warning("No scored stocks found")
            return []

        if len(rows) < 2:
            logger.warning("Not enough scored stocks")
            return []

        # ── Load A2 ──
        a2_cache = {}
        all_codes = [r["ts_code"] for r in rows]
        ph_all = ",".join("?" * len(all_codes))
        for r2 in conn.execute(
            f"SELECT ts_code, report_json FROM fundamental_reports WHERE ts_code IN ({ph_all})",
            all_codes
        ).fetchall():
            try:
                a2_cache[r2["ts_code"]] = json.loads(r2["report_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Load volume + MA data ──
        vol_price, avg_vol, ma_data = _load_vol_price(conn, all_codes)

        # ── Compute accel_prev3 (per-stock: were prior 3 accel values ≤ 0?) ──
        # Approximate: if momentum_d3 just turned positive recently
        # Use d3>0 with d20 or d60 negative as proxy for "acceleration just flipped"
        accel_prev3 = {}
        for r in rows:
            code = r["ts_code"]
            d3 = r["momentum_d3"] or 0
            d20 = r["momentum_d20"] or 0
            d60 = r["momentum_d60"] or 0
            accel = r["momentum_accel"] or 0
            # accel_prev3 = True if momentum was recently negative (early reversal indicator)
            accel_prev3[code] = (accel > 0) and (d3 > 0) and (d20 <= 0 or d60 <= 0)

        # ── Score all stocks ──
        scored = []
        skipped_zero_vol = []
        for r in rows:
            rd = dict(r)
            code = rd["ts_code"]
            a2 = a2_cache.get(code)
            vp = vol_price.get(code, [])
            av = avg_vol.get(code, 0)

            # Skip stocks with no daily price/volume data (e.g. 920xxx 北交所)
            if not vp or (av <= 0 and all(a == 0 for a, _, _ in vp)):
                skipped_zero_vol.append(code)
                continue

            md = ma_data.get(code, {})
            vs = compute_value_score(rd, a2, vp, av, accel_prev3.get(code, False), md)
            ms = compute_momentum_score(rd, a2, vp, av)
            rd["value_score"] = vs
            rd["momentum_score"] = ms
            scored.append(rd)

        # ── Strategy-specific top-N selection ──
        if strategy == "long_term":
            base_pct = 0.35
            scored.sort(key=lambda r: -r["value_score"])
        else:
            base_pct = 0.35
            scored.sort(key=lambda r: -r["momentum_score"])

        # Regime multiplier
        try:
            mr = conn.execute(
                "SELECT regime FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
            ).fetchone()
            regime = mr["regime"] if mr else ""
            if "熊市" in str(regime) or "BEAR" in str(regime).upper():
                top_pct = base_pct * 0.60
            elif "牛市" in str(regime) or "BULL" in str(regime).upper():
                top_pct = base_pct * 1.00
            else:
                top_pct = base_pct * 0.80
        except Exception:
            top_pct = base_pct

        top_n = max(15, int(len(scored) * top_pct))
        selected = [s for s in scored[:top_n] if (s.get("total_score") or 0) >= MIN_TOTAL_SCORE]

        # ── Log ──
        if skipped_zero_vol:
            logger.info(f"Focus list [{strategy}]: skipped {len(skipped_zero_vol)} no-price-data stocks: {', '.join(skipped_zero_vol[:10])}{'...' if len(skipped_zero_vol) > 10 else ''}")

        if strategy == "long_term":
            avg_vs = sum(s["value_score"] for s in selected) / len(selected) if selected else 0
            logger.info(f"Focus list [{strategy}]: {len(selected)} selected, avg value_score={avg_vs:.1f}")
        else:
            avg_ms = sum(s["momentum_score"] for s in selected) / len(selected) if selected else 0
            logger.info(f"Focus list [{strategy}]: {len(selected)} selected, avg momentum_score={avg_ms:.1f}")

        # ── Persist ──
        trade_date = datetime.now().strftime("%Y-%m-%d")
        conn.execute("DELETE FROM focus_list WHERE strategy=?", (strategy,))
        for i, s in enumerate(selected):
            if strategy == "long_term":
                tier = _value_tier(s["value_score"])
            else:
                tier = _momentum_tier(s["momentum_score"])
            conn.execute(
                """INSERT INTO focus_list (ts_code, name, total_score, rank, list_date, position, tier, strategy)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (s["ts_code"], s["name"],
                 s["total_score"], s["rank"], trade_date, i + 1,
                 tier, strategy),
            )
        conn.commit()
        return selected
    finally:
        conn.close()


def get_focus_codes(conn, strategy=None):
    """Return list of ts_code in current focus list, ordered by position.

    Args:
        strategy: filter by strategy ('long_term' or 'hot_picks').
                  If None, returns all (legacy behavior).
    """
    if strategy:
        rows = conn.execute(
            "SELECT ts_code, name, total_score, position "
            "FROM focus_list WHERE strategy=? ORDER BY position",
            (strategy,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts_code, name, industry, total_score, position FROM focus_list ORDER BY position"
        ).fetchall()
    return [dict(r) for r in rows]
