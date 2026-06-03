"""
Agent 5: Multi-factor Fusion — weighted fusion of upstream Agent REPORTS.
Runs daily. Industry-neutral ranking. LLM synthesizes signal conflicts.

v2.0: Consumes upstream analysis reports (A2 fundamental_reports, A4 macro_reports),
      not raw data tables. Strategy-aware weights from config. Writes fusion_reports.
"""
import json
import math
import time
import logging
from datetime import datetime
from collections import defaultdict

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()


# ── Technical score computation ──────────────────────────────────

def _compute_tech_score(ind: dict, strategy: str = "long_term") -> float:
    """Compute technical score 0-100 from A1's raw indicator JSON.

    Strategy-aware: RSI overbought is a warning for long_term (risk),
    but neutral/positive for hot_picks (momentum continuation signal).
    """
    components = []
    weights = []

    # MACD: histogram sign and magnitude → tanh normalization
    macd = ind.get("macd", {})
    hist = macd.get("histogram", 0)
    if isinstance(hist, (int, float)):
        components.append(50 + math.tanh(hist * 10) * 40)
        weights.append(1.0)

    # RSI-14: strategy-aware handling
    rsi = ind.get("rsi_14", 50)
    if isinstance(rsi, (int, float)):
        if strategy == "hot_picks":
            # Overbought = momentum strength → don't penalize, slight bonus
            components.append(rsi)
        else:
            # Long_term: overbought = risk → cap at 70, oversold = opportunity
            if rsi > 80:
                components.append(55)  # cap penalty for extreme overbought
            elif rsi > 70:
                components.append(65)  # mild penalty
            elif rsi < 30:
                components.append(45)  # oversold is slightly positive for value entry
            else:
                components.append(rsi)
        weights.append(1.0)

    # Bollinger position: 0-1, score peaks at center (0.5)
    bb = ind.get("bollinger", {})
    bb_pos = bb.get("position", 0.5)
    if isinstance(bb_pos, (int, float)):
        components.append(100 - abs(bb_pos - 0.5) * 200)
        weights.append(0.5)

    # MA alignment: categorical → score
    ma = ind.get("ma_alignment", "mixed")
    components.append({"bullish": 75, "mixed": 50, "bearish": 25}.get(ma, 50))
    weights.append(1.0)

    # OBV trend
    obv_trend = ind.get("obv_trend", "flat")
    components.append({"rising": 65, "flat": 50, "falling": 35}.get(obv_trend, 50))
    weights.append(0.5)

    # Volume ratio: 1.0 = neutral, mild bonus for >1.0
    vr = ind.get("volume_ratio", 1.0)
    if isinstance(vr, (int, float)):
        components.append(50 + math.tanh((vr - 1.0) * 2) * 25)
        weights.append(0.3)

    if not components:
        return 50.0

    weighted = sum(c * w for c, w in zip(components, weights))
    total_w = sum(weights)
    return max(0.0, min(100.0, weighted / total_w))


# ── Fundamental score from A2 report ─────────────────────────────

def compute_fund_score_from_report(report: dict) -> float:
    """Convert A2's qualitative analysis to quantitative score 0-100.

    Design goals:
      - Wide differentiation: typical range 25-85, not all clustered at 40.
      - GQ excluded: single-quarter data makes GQ=LOW for almost everything;
        "can't compute growth" ≠ "bad growth".
      - Confidence acts as quality multiplier, not pull-to-center.
      - Red flags: non-linear penalty (threshold effects).
      - Data completeness bonus: multi-quarter reports get credit.
    """
    score = 40.0

    # ── Earnings quality (most differentiating) ──
    eq = report.get("earnings_quality", {})
    eq_rating = eq.get("rating", "UNKNOWN") if isinstance(eq, dict) else "UNKNOWN"
    score += {"HIGH": 25, "MEDIUM": 5, "LOW": -20}.get(eq_rating, 0)

    # ── Growth quality: EXCLUDED ──
    # Single-quarter snapshots always get LOW (no trend data).
    # Including it adds noise without signal.

    # ── Financial health ──
    fh = report.get("financial_health", {})
    fh_rating = fh.get("rating", "UNKNOWN") if isinstance(fh, dict) else "UNKNOWN"
    score += {"GOOD": 20, "FAIR": 0, "POOR": -25}.get(fh_rating, 0)

    # ── Valuation ──
    val = report.get("valuation", {})
    val_rating = val.get("rating", "UNKNOWN") if isinstance(val, dict) else "UNKNOWN"
    score += {"UNDERPRICED": 20, "FAIR": 5, "FAIR_VAL": 5, "OVERPRICED": -15}.get(val_rating, 0)

    # ── Red flags: non-linear threshold penalty ──
    flag_count = len(report.get("red_flags", []))
    severity_sum = 0
    for rf in report.get("red_flags", []):
        if isinstance(rf, dict):
            sev = rf.get("severity", "MEDIUM")
            severity_sum += 3 if sev == "HIGH" else 2 if sev == "MEDIUM" else 1
    if flag_count >= 5:
        score -= 30
    elif flag_count >= 4:
        score -= 22
    elif flag_count >= 3:
        score -= 14
    elif flag_count == 2:
        score -= 7
    elif flag_count == 1:
        score -= 3

    # ── Data completeness: multi-quarter reports get bonus ──
    quarters = report.get("quarter_count", 0)
    if quarters >= 8:
        score += 10
    elif quarters >= 4:
        score += 5

    # ── Confidence: quality multiplier (not pull-to-center) ──
    conf = report.get("confidence", 0.3)
    if isinstance(conf, (int, float)):
        # Scale: conf=0.15 → factor=0.3, conf=0.4 → factor=0.8
        factor = min(conf * 2.0, 1.0)
        score = 40 + (score - 40) * factor

    return max(5.0, min(100.0, round(score, 1)))


# ── Macro score from A4 report ───────────────────────────────────

# ── Momentum ─────────────────────────────────────────────────────

def calc_momentum(conn, target_codes):
    """Multi-timeframe momentum: 3/5/10/20/60/120 trading-day lookback.
    Uses trading-day count (array index), NOT calendar days.
    Detects and flags stocks with recent trading suspensions that distort momentum.

    Returns {code: {raw: {d3,d5,d10,d20,d60,d120}, acceleration: float, has_gap: bool, latest_date: str}}
    """
    from datetime import datetime, timedelta
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
            """Momentum over the last `days` trading days (array index, not calendar)."""
            if len(closes) > days and closes[days] > 0:
                return round((closes[0] - closes[days]) / closes[days] * 100, 1)
            return 0.0

        # Volume confirmation: 5d avg / 20d avg (lagged 5d to avoid overlap)
        if len(volumes) >= 25:
            vol_5d = sum(volumes[:5]) / 5
            vol_20d = sum(volumes[5:25]) / 20
            vol_ratio = round(vol_5d / vol_20d, 2) if vol_20d > 0 else 1.0
        else:
            vol_ratio = 1.0

        raw = {
            "d3": _pct(3), "d5": _pct(5), "d10": _pct(10),
            "d20": _pct(20), "d60": _pct(60) if len(closes) > 60 else _pct(20),
            "d120": _pct(120) if len(closes) > 120 else _pct(60),
            "vol_ratio": vol_ratio,
        }
        accel = round(raw["d3"] * 5/3 - raw["d5"], 1)

        # Detect recent trading suspensions: gap > 5 calendar days between consecutive rows
        has_gap = False
        for i in range(min(10, len(dates) - 1)):
            try:
                d1 = datetime.strptime(dates[i], "%Y-%m-%d")
                d2 = datetime.strptime(dates[i + 1], "%Y-%m-%d")
                if (d1 - d2).days > 5:
                    has_gap = True
                    break
            except (ValueError, TypeError):
                pass

        result[code] = {
            "raw": raw,
            "acceleration": accel,
            "has_gap": has_gap,
            "latest_date": dates[0] if dates else "",
        }
    return result


def classify_trend_type(momentum_data):
    """Classify each stock by trend direction (natural >0 boundary).
    long_term: d60>0 AND d120>0 (sustained uptrend)
    short_term: d60>0 (medium-term uptrend)
    declining: everything else
    """
    types = {}
    for code, m in momentum_data.items():
        raw = m["raw"]
        if raw["d60"] > 0 and raw["d120"] > 0:
            types[code] = "long_term"
        elif raw["d60"] > 0:
            types[code] = "short_term"
        else:
            types[code] = "declining"
    return types


def composite_momentum_score(m_raw, trend_type):
    """Blend multi-timeframe momentum into single 0-100 score."""
    if trend_type == "long_term":
        weights = {"d3": 0.05, "d5": 0.10, "d10": 0.15, "d20": 0.20, "d60": 0.25, "d120": 0.25}
    else:
        weights = {"d3": 0.25, "d5": 0.25, "d10": 0.20, "d20": 0.15, "d60": 0.10, "d120": 0.05}
    score = sum(m_raw[k] * weights[k] for k in weights)
    return round(50 + math.tanh(score / 10) * 40, 1)




# ── Load inputs from upstream reports ────────────────────────────

def load_inputs(conn, target_codes, strategy="long_term"):
    """Load all factor inputs, preferring upstream REPORTS over raw data tables.

    Returns: (tech_scores, fund_scores, macro_report, momentum_data, macro_regime)
    momentum_data is {code: {raw: {d3,d5,d10,d20,d60,d120}, acceleration: float}}
    """
    placeholders = ",".join("?" * len(target_codes)) if target_codes else "'__none__'"

    # ── Tech: compute from A1's indicator JSON (A1 has no LLM report) ──
    ind_rows = conn.execute(f"""
        SELECT ts_code, indicators_json FROM indicators
        WHERE ts_code IN ({placeholders})
          AND calc_date = (SELECT MAX(calc_date) FROM indicators)
    """, target_codes).fetchall()
    tech_scores = {}
    for r in ind_rows:
        try:
            ind = json.loads(r["indicators_json"])
            tech_scores[r["ts_code"]] = _compute_tech_score(ind, strategy)
        except (json.JSONDecodeError, TypeError):
            tech_scores[r["ts_code"]] = 50.0

    # ── Fundamental: from A2's fundamental_reports ──
    fund_rows = conn.execute(f"""
        SELECT ts_code, report_json FROM fundamental_reports
        WHERE ts_code IN ({placeholders})
          AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)
    """, target_codes).fetchall()
    fund_scores = {}
    for r in fund_rows:
        code = r["ts_code"]
        try:
            report = json.loads(r["report_json"])
            fund_scores[code] = compute_fund_score_from_report(report)
        except (json.JSONDecodeError, TypeError):
            fund_scores[code] = None  # mark for fallback

    # Fallback: stocks without A2 report → compute from financials table
    missing_fund = [c for c in target_codes if c not in fund_scores or fund_scores[c] is None]
    if missing_fund:
        logger.info(f"  {len(missing_fund)} stocks without A2 report, using raw financials fallback")
        fb_ph = ",".join("?" * len(missing_fund))
        fb_rows = conn.execute(f"""
            SELECT ts_code, roe, gross_margin, debt_ratio, revenue_yoy, fcf_ratio, pe_percentile
            FROM financials
            WHERE ts_code IN ({fb_ph})
              AND report_date = (SELECT MAX(report_date) FROM financials WHERE ts_code = financials.ts_code)
        """, missing_fund).fetchall()
        for r in fb_rows:
            score = 40.0  # below neutral: uncertainty penalty for missing A2 data
            if r["roe"]:
                score += min(25, r["roe"] * 1.5)
            if r["gross_margin"]:
                score += min(15, (r["gross_margin"] - 20) * 0.5)
            if r["revenue_yoy"]:
                score += min(10, r["revenue_yoy"] * 0.3)
            if r["debt_ratio"] and r["debt_ratio"] < 70:
                score += 10
            if r["fcf_ratio"] and r["fcf_ratio"] > 0:
                score += 10
            if r["pe_percentile"] and r["pe_percentile"] < 50:
                score += 5
            fund_scores[r["ts_code"]] = min(100, max(0, score))
        # Any still missing get 50
        for c in missing_fund:
            if c not in fund_scores or fund_scores[c] is None:
                fund_scores[c] = 40.0  # uncertainty penalty for missing A2 data

    # ── Macro: from A4's macro_reports ──
    macro_report = {}
    mrp = conn.execute(
        "SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1"
    ).fetchone()
    if mrp:
        try:
            macro_report = json.loads(mrp["report_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    macro_regime = {}
    mr = conn.execute(
        "SELECT regime FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
    ).fetchone()
    if mr:
        macro_regime = dict(mr)

    # ── Momentum ──
    momentum_data = calc_momentum(conn, target_codes)

    return tech_scores, fund_scores, macro_report, momentum_data, macro_regime


# ── Composite scoring ────────────────────────────────────────────

def zscore_normalize(scores):
    """Z-score normalize a dict of {code: score}. Returns {code: zscore}."""
    if not scores:
        return {}
    values = list(scores.values())
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {k: 0.0 for k in scores}
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    if std == 0:
        return {k: 0.0 for k in scores}
    return {k: round((v - mean) / std, 3) for k, v in scores.items()}


def compute_composite(tech, fund, momentum_data, target_codes, trend_types):
    """Composite scoring with single set of strategy-agnostic factor weights.

    FL handles strategy differentiation by selecting different categories
    (value/growth/stable for long_term, momentum/breakout/short_term for hot_picks).
    A5 only needs to rank stocks correctly — no per-strategy weights needed.
    """
    # Universal weights: fundamental > momentum > tech > relative_strength
    W = {"tech": 0.20, "fundamental": 0.35, "momentum": 0.30, "relative_strength": 0.15}

    # Market median momentum (for relative strength)
    market_d5s = [m["raw"]["d5"] for m in momentum_data.values() if m.get("raw", {}).get("d5") is not None]
    market_d20s = [m["raw"]["d20"] for m in momentum_data.values() if m.get("raw", {}).get("d20") is not None]
    med_d5 = sorted(market_d5s)[len(market_d5s)//2] if market_d5s else 0
    med_d20 = sorted(market_d20s)[len(market_d20s)//2] if market_d20s else 0

    raw = {}
    rels = {}
    vol_confs = {}
    for code in target_codes:
        tt = trend_types.get(code, "declining")
        t = tech.get(code, 50)
        f_raw = fund.get(code, 40)
        m = momentum_data.get(code, {})
        m_score = composite_momentum_score(m["raw"], tt) if m else 50

        # Relative strength: stock vs market momentum
        m_raw = m.get("raw", {}) if m else {}
        stock_d5 = m_raw.get("d5", 0) or 0
        stock_d20 = m_raw.get("d20", 0) or 0
        rel_d5 = stock_d5 - med_d5
        rel_d20 = stock_d20 - med_d20
        rel_strength = max(0, min(100, 50 + (rel_d5 * 0.3 + rel_d20 * 0.7) * 2))
        rels[code] = rel_strength

        # Volume confirmation: price-volume alignment
        vol_5d = m_raw.get("vol_ratio", 1.0) if m else 1.0
        price_up = stock_d5 > 0
        vol_expanding = vol_5d > 1.05
        if price_up and vol_expanding:
            vol_confirm = 1.0
        elif not price_up and not vol_expanding:
            vol_confirm = 0.7
        elif price_up and not vol_expanding:
            vol_confirm = 0.4
        else:
            vol_confirm = 0.3
        m_score = m_score * (0.7 + 0.3 * vol_confirm)
        vol_confs[code] = vol_confirm

        # Fund floor for short_term: lower bar on fundamentals
        if tt == "short_term":
            if f_raw >= 40:      f_effective = f_raw
            elif f_raw >= 30:    f_effective = f_raw * 0.7
            elif f_raw >= 15:    f_effective = f_raw * 0.4
            else:                f_effective = 0
        else:
            f_effective = f_raw

        raw[code] = (
            t * W["tech"]
            + f_effective * W["fundamental"]
            + m_score * W["momentum"]
            + rel_strength * W["relative_strength"]
        )

    scores = {code: max(0.0, min(100.0, round(s, 1))) for code, s in raw.items()}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked, dict(ranked), rels, vol_confs


# ── LLM Fusion Synthesis ─────────────────────────────────────────

def fusion_synthesis(top_n, macro_report, macro_regime, strategy,
                     tech_scores, fund_scores, momentum_data, conn, trend_types):
    """LLM synthesizes upstream signals into a trend-focused fusion report.
    New focus: trend quality, sustainability, and stage analysis.
    """
    cfg = settings.get_llm_config("A5")

    codes = [c for c, _ in top_n]
    ph = ",".join("?" * len(codes))

    # ── Gather per-stock rich context ──
    # A1: raw indicators (MACD/RSI/MA etc.)
    tech_detail = {}
    for r in conn.execute(
        f"SELECT ts_code, indicators_json FROM indicators "
        f"WHERE ts_code IN ({ph}) AND calc_date = (SELECT MAX(calc_date) FROM indicators)",
        codes,
    ).fetchall():
        try:
            ind = json.loads(r["indicators_json"])
            macd = ind.get("macd", {})
            tech_detail[r["ts_code"]] = {
                "macd": macd.get("signal", "?"),
                "rsi": ind.get("rsi_14", "?"),
                "ma": ind.get("ma_alignment", "?"),
                "bb_pos": ind.get("bollinger", {}).get("position", "?"),
                "obv": ind.get("obv_trend", "?"),
            }
        except (json.JSONDecodeError, TypeError):
            tech_detail[r["ts_code"]] = {}

    # A2: fundamental report with red flag severity
    fund_detail = {}
    for r in conn.execute(
        f"SELECT ts_code, report_json FROM fundamental_reports "
        f"WHERE ts_code IN ({ph}) AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)",
        codes,
    ).fetchall():
        try:
            rep = json.loads(r["report_json"])
            fund_detail[r["ts_code"]] = {
                "eq": rep.get("earnings_quality", {}).get("rating", "?") if isinstance(rep.get("earnings_quality"), dict) else "?",
                "gq": rep.get("growth_quality", {}).get("rating", "?") if isinstance(rep.get("growth_quality"), dict) else "?",
                "fh": rep.get("financial_health", {}).get("rating", "?") if isinstance(rep.get("financial_health"), dict) else "?",
                "val": rep.get("valuation", {}).get("rating", "?") if isinstance(rep.get("valuation"), dict) else "?",
                "red_flags": [f"{f.get('severity','?')}:{f.get('flag','?')}" if isinstance(f, dict) else str(f) for f in rep.get("red_flags", [])],
                "narrative": rep.get("narrative", "")[:150],
                "conf": rep.get("confidence", 0),
            }
        except (json.JSONDecodeError, TypeError):
            fund_detail[r["ts_code"]] = {}

    # ── Distribution context ──
    all_tech = [v for v in tech_scores.values()]
    all_fund = [v for v in fund_scores.values()]
    all_mom = [composite_momentum_score(m.get("raw", {}), trend_types.get(c, "short_term"))
               for c, m in momentum_data.items()]

    def _dist(vals):
        if not vals:
            return 50, 10
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        return round(mu, 1), round(var ** 0.5, 1)

    tech_mu, tech_sd = _dist(all_tech)
    fund_mu, fund_sd = _dist(all_fund)
    mom_mu, mom_sd = _dist(all_mom)

    # ── Build prompt ──
    regime_name = macro_regime.get("regime", "?") if macro_regime else "?"
    regime_summary = macro_report.get("regime_summary", regime_name) if macro_report else regime_name
    macro_narrative = macro_report.get("narrative", "无")[:250] if macro_report else "无"

    # Per-stock display (filtered to top stocks by score)
    display_n = min(len(top_n), max(20, int(len(top_n) * 0.5)))  # percentage-driven

    lines = [
        "你是趋势策略分析师。你的任务是理解排名背后的趋势结构：每只股票的上涨由什么驱动、处于什么阶段、在什么条件下有机会。",
        "你的报告帮助交易者做出明智的选股决策。",
        "",
        f"策略: {strategy} | 宏观: {regime_name}",
        f"宏观判断: {regime_summary}",
        f"A4宏观判断: {macro_narrative}",
        "",
        f"=== 全市场分布 (N={len(all_tech)}) ===",
        f"技术分: μ={tech_mu} σ={tech_sd} | 基本面分: μ={fund_mu} σ={fund_sd} | 动量分: μ={mom_mu} σ={mom_sd}",
        "",
        f"=== 排名靠前股票 ({display_n}只) ===",
        "格式: 排名. 代码 [趋势类型/阶段] 总分 | 多时间框架动量(d3/d5/d20/d60) | 加速度 | MACD/RSI | 基本面 | 红旗",
        "",
    ]

    for i, (code, score) in enumerate(top_n[:display_n], 1):
        t = tech_detail.get(code, {})
        f = fund_detail.get(code, {})
        ts = tech_scores.get(code, 50)
        fs = fund_scores.get(code, 50)
        tt = trend_types.get(code, "?")
        m = momentum_data.get(code, {})
        m_raw = m.get("raw", {}) if m else {}
        accel = m.get("acceleration", 0) if m else 0
        d3 = m_raw.get("d3", 0)
        d5 = m_raw.get("d5", 0)
        d20 = m_raw.get("d20", 0)
        d60 = m_raw.get("d60", 0)

        if accel > 3: stage = "ACCEL"
        elif accel > 0: stage = "SUSTAIN"
        elif accel > -3: stage = "DECEL"
        else: stage = "REVERSE"

        lines.append(
            f"{i:2d}. {code} [{tt}/{stage}] 总分:{score:.0f} [T:{ts:.0f} F:{fs:.0f}] | "
            f"d3:{d3:+.1f}% d5:{d5:+.1f}% d20:{d20:+.1f}% d60:{d60:+.1f}% acc:{accel:+.1f} | "
            f"MACD:{t.get('macd','?')} RSI:{t.get('rsi','?')} MA:{t.get('ma','?')} | "
            f"盈:{f.get('eq','?')} 财:{f.get('fh','?')} 估:{f.get('val','?')} | "
            f"红旗:{f.get('red_flags',[]) or '无'}"
        )

    lines.extend([
        "",
        "=== 分析要求 ===",
        "",
        "你是市场状态分析师。你的任务不是评价个股好坏（那是A7的工作），而是描述当前排名呈现的整体格局。",
        "",
        "分析维度:",
        "1. **排名结构**: 前列股票主要由什么因子驱动（动量/基本面/多因子共振）？当前宏观环境是否支持这个驱动力？",
        "2. **趋势阶段**: 为每只展示的股票标注趋势阶段——ACCELERATING/SUSTAINING/DECELERATING/REVERSING。参考acc=d3*5/3-d5(短期加速度)。",
        "3. **市场叙事**: 当前市场在奖励什么类型的股票？这个格局在下一周可能如何演变？",
        "",
        "输出JSON:",
        "{",
        '  "trend_stages": [{"ts_code":"...","stage":"ACCELERATING/SUSTAINING/DECELERATING/REVERSING","confidence":0.6}],',
        '  "overall_narrative": "当前排名格局、主要驱动力、趋势阶段分布、一周展望",',
        "}",
    ])

    prompt = "\n".join(lines)
    result = llm.chat_json(prompt, model=cfg["model"], max_tokens=cfg.get("max_tokens", 3000))
    return result


# ── Main entry point ─────────────────────────────────────────────

def run(trade_date=None, strategy="long_term", extra_strategies=None):
    """Multi-factor fusion. Consumes upstream reports, writes composite_scores + fusion_reports."""
    start = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    conn = get_connection()
    logger.info(f"Agent 5 starting: strategy={strategy}")

    try:
        cursor = conn.execute("SELECT ts_code FROM tier_assignments WHERE tier IN ('HOLDING', 'FAVORED', 'NEUTRAL')")
        target_codes = [r["ts_code"] for r in cursor.fetchall()]
        logger.info(f"Target: {len(target_codes)} stocks (HOLDING + FAVORED + NEUTRAL)")

        tech, fund, macro_report, momentum_data, macro_regime = load_inputs(conn, target_codes, strategy)

        # Classify stocks by trend structure
        trend_types = classify_trend_type(momentum_data)
        n_long = sum(1 for v in trend_types.values() if v == "long_term")
        n_short = sum(1 for v in trend_types.values() if v == "short_term")
        n_decl = sum(1 for v in trend_types.values() if v == "declining")

        logger.info(
            f"Trend types: long_term={n_long}, short_term={n_short}, declining={n_decl}"
        )

        ranked, scores, rels, vol_confs = compute_composite(
            tech, fund, momentum_data, target_codes, trend_types,
        )

        # Re-rank
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        scores = dict(ranked)

        logger.info(f"Ranked {len(scores)} stocks, top: {ranked[0] if ranked else 'none'}")

        # LLM Fusion Synthesis → fusion_reports
        fusion_report_json = None
        if settings.ds_api_key and len(ranked) >= 10:
            # Retry fusion_synthesis up to 3 times, no silent fallback
            fusion_ok = False
            for fs_attempt in range(3):
                try:
                    fusion_report_json = fusion_synthesis(
                        ranked, macro_report, macro_regime, strategy,
                        tech, fund, momentum_data, conn, trend_types,
                    )
                    if fusion_report_json:
                        fusion_ok = True
                        break
                except Exception as e:
                    logger.warning(f"  Fusion synthesis attempt {fs_attempt+1}/3 failed: {e}")
                if fs_attempt < 2:
                    time.sleep(10 * (fs_attempt + 1))  # 10s, 20s backoff
            if not fusion_ok:
                logger.error("A5 fusion_synthesis LLM不可用: 3次尝试全部失败, "
                             "composite_scores 已正常写入, 仅跳过 fusion_reports")
            if fusion_report_json:
                # Persist fusion report
                conn.execute(
                    """INSERT OR REPLACE INTO fusion_reports
                       (calc_date, strategy, report_json, created_at)
                       VALUES (?, ?, ?, ?)""",
                    (trade_date, strategy,
                     json.dumps(fusion_report_json, ensure_ascii=False),
                     datetime.now().isoformat()),
                )
                logger.info("  fusion_reports written")
            else:
                logger.warning("  Fusion synthesis returned None — skipping fusion_reports")

        # Clear same-date scores for this strategy (handles re-runs), keep history
        conn.execute(
            "DELETE FROM composite_scores WHERE strategy=? AND calc_date=?",
            (strategy, trade_date),
        )

        # Persist composite scores
        cfg = settings.get_strategy_config(strategy)
        for rank_idx, (code, score) in enumerate(ranked):
            m = momentum_data.get(code, {})
            m_raw = m.get("raw", {}) if m else {}
            tt = trend_types.get(code, "declining")
            conn.execute(
                """INSERT OR REPLACE INTO composite_scores
                   (ts_code, calc_date, strategy, tech_score, fundamental_score, macro_fit, momentum, total_score, rank, tier_action,
                    trend_type, momentum_d3, momentum_d5, momentum_d20, momentum_d60, momentum_accel, relative_strength, volume_confirmation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (code, trade_date, strategy,
                 round(tech.get(code, 50), 1),
                 round(fund.get(code, 40), 1),
                 0,
                 composite_momentum_score(m_raw, tt) if m else 50,
                 score, rank_idx + 1,
                 "PROMOTE" if rank_idx < len(ranked) * cfg.get("buy_pass_rate", 0.2) else "STAY",
                 tt,
                 m_raw.get("d3"), m_raw.get("d5"), m_raw.get("d20"), m_raw.get("d60"),
                 m.get("acceleration", 0) if m else 0,
                 round(rels.get(code, 50), 1), round(vol_confs.get(code, 0.5), 2)),
            )

        conn.commit()

        # ── Also compute hot_picks if this is the long_term run ──
        for extra_strat in (extra_strategies or []):
            hp_ranked, hp_scores, hp_rels, hp_vol_confs = compute_composite(
                tech, fund, momentum_data, target_codes, trend_types,
            )
            conn.execute(
                "DELETE FROM composite_scores WHERE strategy=? AND calc_date=?",
                (extra_strat, trade_date),
            )
            hp_cfg = settings.get_strategy_config(extra_strat)
            for rank_idx, (code, score) in enumerate(hp_ranked):
                m = momentum_data.get(code, {})
                m_raw = m.get("raw", {}) if m else {}
                tt = trend_types.get(code, "declining")
                conn.execute(
                    """INSERT OR REPLACE INTO composite_scores
                       (ts_code, calc_date, strategy, tech_score, fundamental_score, macro_fit, momentum, total_score, rank, tier_action,
                        trend_type, momentum_d3, momentum_d5, momentum_d20, momentum_d60, momentum_accel, relative_strength, volume_confirmation)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (code, trade_date, extra_strat,
                     round(tech.get(code, 50), 1), round(fund.get(code, 40), 1),
                     0,
                     composite_momentum_score(m_raw, tt) if m else 50,
                     score, rank_idx + 1,
                     "PROMOTE" if rank_idx < len(hp_ranked) * hp_cfg.get("buy_pass_rate", 0.2) else "STAY",
                     tt, m_raw.get("d3"), m_raw.get("d5"), m_raw.get("d20"), m_raw.get("d60"),
                     m.get("acceleration", 0) if m else 0,
                     round(hp_rels.get(code, 50), 1), round(hp_vol_confs.get(code, 0.5), 2)),
                )
            conn.commit()
            logger.info(f"A5: also computed {extra_strat} — {len(hp_scores)} stocks, top={hp_ranked[0][0]}({hp_ranked[0][1]:.0f})")

        elapsed = time.time() - start
        top_str = f"{ranked[0][0]}({scores[ranked[0][0]]:.0f})" if ranked else "N/A"
        logger.info(f"Agent 5 complete: {len(scores)} ranked, top={top_str} in {elapsed:.1f}s")

        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, stocks_processed, duration_s, summary) "
            "VALUES (5, ?, 'SUCCESS', ?, ?, ?)",
            (trade_date, len(scores), elapsed,
             f"[{strategy}] Top: {top_str} fusion_report={'Y' if fusion_report_json else 'N'}"),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 5 failed: {e}")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (5, ?, 'FAILED', ?, ?)",
            (trade_date, time.time() - start, str(e)[:200]),
        )
        conn.commit()
        raise  # re-raise so orchestrator sees the failure
    finally:
        conn.close()


if __name__ == "__main__":
    run()
