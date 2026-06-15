"""
Agent 0: Universe Manager — stock classification and strategy-specific universe.

Classification tiers:
  EXCLUDED — hard exclusion (ST/delist/low liquidity/new listing/persistent decline)
  HOLDING  — current portfolio (always kept in universe)
  FAVORED  — bullish trend, worth deep analysis
  NEUTRAL  — direction unclear, watch
  AVOID    — bearish trend, not worth analysis resources

Decision tree (priority order):
  1. EXCLUDED — ST/退市, 60d avg turnover <10M, listed <60d (pure rules)
  2. HOLDING  — in portfolio (pure rule)
  3. AVOID    — d5<0 & d20<0 & d30<0 (all timeframes falling)
  4. AVOID    — persistent decline: MA60 below ≥70% & bearish MA & near 20d low
  5. classify_stocks (remaining):
     Path 1  rule_direct — FAVORED: chg_20d>0 & chg_60d>0 | AVOID: illiquidity/extreme vol
     Path 2  LLM         — 30% random sample, batch=30
     Path 3  rule_default — FAVORED: bullish & vs_ma60>0 | AVOID: bearish & vs_ma60<0 | NEUTRAL: rest

Runs weekly. Cold start = full classification; warm run = incremental.
"""
import json
import random
import time
import logging
from datetime import datetime

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()

# Valid tier values
TIER_EXCLUDED = "EXCLUDED"
TIER_HOLDING  = "HOLDING"
TIER_FAVORED  = "FAVORED"
TIER_NEUTRAL  = "NEUTRAL"
TIER_AVOID    = "AVOID"

LLM_SAMPLE_RATE         = 0.30   # % of NEUTRAL stocks to LLM-review

# Exclusion thresholds
MIN_DAILY_AMOUNT = 10_000_000     # 60-day avg turnover < 10M → excluded
MIN_LISTING_DAYS = 60            # listed < 60 calendar days → excluded


def check_excluded(conn, trade_date):
    """Rule-based hard exclusion. Returns dict {ts_code: reason}."""
    excluded = {}

    # ST / delist / risk-warning
    for r in conn.execute(
        """SELECT ts_code, name FROM stocks
           WHERE is_st = 1 OR ts_code LIKE '%ST%' OR name LIKE '%ST%' OR name LIKE '%退%'"""
    ).fetchall():
        excluded[r["ts_code"]] = "ST/退市/风险警示"

    # Low liquidity: 60-day avg turnover < 10M
    for r in conn.execute(
        """SELECT ts_code, AVG(amount) as avg_amount FROM daily_quotes
           WHERE trade_date >= date(?, '-60 days')
           GROUP BY ts_code HAVING AVG(amount) < ?""",
        (trade_date, MIN_DAILY_AMOUNT),
    ).fetchall():
        if r["ts_code"] not in excluded:
            excluded[r["ts_code"]] = f"日均成交额<{MIN_DAILY_AMOUNT/1e4:.0f}万(流动性枯竭)"

    # Listed < 60 calendar days
    for r in conn.execute(
        """SELECT ts_code FROM stocks
           WHERE list_date IS NOT NULL AND list_date > date(?, '-60 days')""",
        (trade_date,),
    ).fetchall():
        if r["ts_code"] not in excluded:
            excluded[r["ts_code"]] = "上市不足60日"

    return excluded


def check_persistent_decline(conn, trade_date, lookback_days=40):
    """3-signal qualitative trend detection for persistent decline.

    1. Price below MA60 for >= 70% of recent 20 days
    2. Bearish MA alignment: MA5 < MA10 < MA20
    3. Latest close in bottom 25% of 20-day range
    """
    declined = {}
    candidates = [r["ts_code"] for r in conn.execute(
        """SELECT ts_code FROM daily_quotes
           WHERE trade_date >= date(?, ?)
           GROUP BY ts_code HAVING COUNT(*) >= 30""",
        (trade_date, f'-{lookback_days} days'),
    ).fetchall()]

    for code in candidates:
        rows = conn.execute(
            "SELECT close FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT ?",
            (code, lookback_days),
        ).fetchall()
        if len(rows) < 30:
            continue
        closes = [r["close"] for r in reversed(rows)]
        n = len(closes)

        ma60 = sum(closes[-60:]) / min(60, n) if n >= 20 else sum(closes) / n
        days_below = sum(1 for c in closes[-20:] if c < ma60)
        below_ratio = days_below / min(20, n)

        ma5  = sum(closes[-5:]) / 5 if n >= 5 else closes[-1]
        ma10 = sum(closes[-10:]) / 10 if n >= 10 else ma5
        ma20 = sum(closes[-20:]) / 20 if n >= 20 else ma10
        bearish_align = ma5 < ma10 < ma20

        period_high = max(closes[-20:])
        period_low  = min(closes[-20:])
        price_range = period_high - period_low
        near_low = price_range > 0 and (closes[-1] - period_low) / price_range < 0.25

        if below_ratio >= 0.70 and bearish_align and near_low:
            declined[code] = f"持续阴跌(MA空头排列,低于MA60占比{below_ratio:.0%})"

    return declined


def check_momentum_avoid(conn, trade_date):
    """Momentum-based AVOID: d5<0 AND d20<0 AND d30<0 → all timeframes falling.

    Coarse filter — A5/A7 do precise momentum analysis. A0 just catches obvious downtrends.
    """
    avoid_list = {}
    codes = [r["ts_code"] for r in conn.execute(
        """SELECT ts_code FROM daily_quotes
           WHERE trade_date >= date(?, '-40 days')
           GROUP BY ts_code HAVING COUNT(*) >= 30""",
        (trade_date,),
    ).fetchall()]

    for code in codes:
        rows = conn.execute(
            "SELECT change_pct FROM daily_quotes WHERE ts_code=? AND change_pct IS NOT NULL ORDER BY trade_date DESC LIMIT 35",
            (code,),
        ).fetchall()
        if len(rows) < 30:
            continue

        changes = [r["change_pct"] for r in rows]
        # Compound returns — more accurate than simple sum for multi-day periods
        def _compound(cs, n):
            prod = 1.0
            for c in cs[:n]:
                prod *= (1.0 + c / 100.0)
            return (prod - 1.0) * 100.0
        d5 = _compound(changes, 5)
        d20 = _compound(changes, 20)
        d30 = _compound(changes, 30)

        if d5 < 0 and d20 < 0 and d30 < 0:
            avoid_list[code] = f"全周期下跌趋势(d5:{d5:+.1f}% d20:{d20:+.1f}% d30:{d30:+.1f}%)"

    return avoid_list


def check_holdings(conn):
    """Current portfolio holdings — always Tier HOLDING."""
    return {r["ts_code"]: "当前持仓" for r in conn.execute(
        "SELECT ts_code FROM portfolio WHERE status='HOLD'"
    ).fetchall()}


def get_stock_features(conn, codes):
    """Get enriched features per stock for classification."""
    features = {}
    for code in codes:
        rows = conn.execute(
            "SELECT close, change_pct, volume, amount FROM daily_quotes "
            "WHERE ts_code=? ORDER BY trade_date DESC LIMIT 120",
            (code,),
        ).fetchall()
        if len(rows) < 60:
            continue

        closes = [r["close"] for r in rows]
        volumes = [r["volume"] for r in rows]
        n = len(closes)

        ma5  = sum(closes[:5]) / 5
        ma10 = sum(closes[:10]) / 10
        ma60 = sum(closes[:60]) / 60
        ma120 = sum(closes) / n if n >= 120 else ma60

        # Trend & momentum
        chg_20d = (closes[0] / closes[19] - 1) * 100 if n >= 20 and closes[19] > 0 else 0
        chg_60d = (closes[0] / closes[59] - 1) * 100 if n >= 60 and closes[59] > 0 else 0
        chg_30d = (closes[0] / closes[29] - 1) * 100 if n >= 30 and closes[29] > 0 else 0
        vs_ma60_pct = (closes[0] / ma60 - 1) * 100

        # MA alignment
        if ma5 > ma10 > ma60:
            alignment = "bullish"
        elif ma5 < ma10 < ma60:
            alignment = "bearish"
        else:
            alignment = "mixed"

        # Volume relationship (5-day price direction vs volume direction)
        vol_5d_avg = sum(volumes[:5]) / 5 if n >= 5 else 0
        vol_20d_avg = sum(volumes[5:25]) / 20 if n >= 25 else vol_5d_avg
        vol_ratio = vol_5d_avg / vol_20d_avg if vol_20d_avg > 0 else 1.0
        price_5d_dir = closes[0] > closes[4] if n >= 5 else True
        if price_5d_dir and vol_ratio > 1.1:
            volume_relation = "价涨量增"
        elif price_5d_dir and vol_ratio < 0.9:
            volume_relation = "价涨量缩"
        elif not price_5d_dir and vol_ratio > 1.1:
            volume_relation = "价跌量增"
        elif not price_5d_dir and vol_ratio < 0.9:
            volume_relation = "价跌量缩"
        else:
            volume_relation = "量价平稳"

        # 20-day amplitude (volatility proxy)
        if n >= 20:
            high_20 = max(r["close"] for r in rows[:20])
            low_20  = min(r["close"] for r in rows[:20])
            amplitude = (high_20 - low_20) / ((high_20 + low_20) / 2) * 100 if (high_20 + low_20) > 0 else 0
        else:
            amplitude = 0

        # Size proxy (average daily turnover)
        avg_amount = sum(r["amount"] for r in rows[:20]) / min(20, n) if rows else 0
        if avg_amount > 5e8:
            size = "大盘"
        elif avg_amount > 1e8:
            size = "中盘"
        else:
            size = "小盘"

        features[code] = {
            "close": closes[0],
            "ma5": ma5, "ma10": ma10, "ma60": ma60, "ma120": ma120,
            "vs_ma60_pct": round(vs_ma60_pct, 1),
            "chg_20d": round(chg_20d, 1),
            "chg_30d": round(chg_30d, 1),
            "chg_60d": round(chg_60d, 1),
            "alignment": alignment,
            "volume_relation": volume_relation,
            "amplitude_20d": round(amplitude, 1),
            "avg_amount": avg_amount,
            "size": size,
        }
    return features


def classify_rule_direct(features):
    """Rule-direct classification using multi-timeframe momentum + safety floors.

    FAVORED: chg_20d > 0 AND chg_60d > 0 → short+medium both up, trend confirmed
    AVOID:   extreme illiquidity (<30M daily) OR extreme volatility (>80% amplitude)
    No size bias — small/mid caps with trend confirmation pass through.
    """
    results = {}
    for code, f in features.items():
        amplitude = f.get("amplitude_20d", 0)
        avg_amount = f.get("avg_amount", 0)
        chg_20d = f.get("chg_20d", 0)
        chg_60d = f.get("chg_60d", 0)

        # Direct AVOID: extreme illiquidity
        if avg_amount < 3e7:
            results[code] = {
                "tier": TIER_AVOID,
                "reason": f"流动性不足(日均{avg_amount/1e4:.0f}万)",
                "confidence": 0.90,
                "source": "rule_direct",
            }
        # Direct AVOID: extreme volatility NOT explained by directional trend
        # High amp + small net change = whipsaw/pump-dump. High amp + large net change = strong trend.
        elif amplitude > 80 and abs(chg_20d) < amplitude * 0.5:
            results[code] = {
                "tier": TIER_AVOID,
                "reason": f"20日振幅{amplitude:.0f}%过高且趋势不足(异常波动)",
                "confidence": 0.85,
                "source": "rule_direct",
            }
        # Direct FAVORED: short + medium momentum both positive
        elif chg_20d > 0 and chg_60d > 0:
            results[code] = {
                "tier": TIER_FAVORED,
                "reason": f"短中双周期上涨(20日{chg_20d:+.1f}% 60日{chg_60d:+.1f}%)",
                "confidence": 0.80,
                "source": "rule_direct",
            }

    return results


def classify_llm(conn, codes, features):
    """LLM classification for ambiguous stocks. Batch 30, JSON output."""
    if not settings.ds_api_key:
        return {}

    results = {}
    codes_list = list(codes)
    random.shuffle(codes_list)  # no order bias per batch
    batch_size = 30
    total_batches = (len(codes_list) + batch_size - 1) // batch_size
    batch_num = 0
    logger.info(f"  LLM starting: {len(codes_list)} stocks in {total_batches} batches (batch_size={batch_size})")

    for i in range(0, len(codes_list), batch_size):
        batch_num += 1
        batch_codes = codes_list[i:i + batch_size]

        # Build enriched prompt
        lines = [
            "你是A股行业研究员。对以下股票做投资价值分类。",
            "",
            "分类标准：",
            f"  {TIER_FAVORED} — 趋势向好，技术面积极，值得深入分析",
            f"  {TIER_NEUTRAL} — 方向不明，信号矛盾，观望为主",
            f"  {TIER_AVOID}   — 趋势明确且持续恶化。极其谨慎使用：一旦标记AVOID，该股票将永久失去后续分析机会。",
            f"                 不确定时倾向NEUTRAL——宁可放过，不可误杀。",
            "",
            "分析维度：",
            "  - 价格与MA60偏离程度 + 趋势结构（MA多头/空头排列）",
            "  - 量价匹配程度：标注价格与成交量方向不一致的情况",
            "  - 20日 vs 60日动量对比：短期与中期趋势是否一致",
            "  - 对信号矛盾的股票（如价格强但量弱），倾向于NEUTRAL并说明原因",
            "  - 只有多维度一致指向趋势恶化（如MA空头+量价背离+动量全负），才考虑AVOID",
            "",
            "=== 股票数据 ===",
            f"{'代码':<12s} {'价格':>8s} {'vsMA60':>7s} {'MA排列':<8s} {'量价':<10s} {'20日':>7s} {'30日':>7s} {'60日':>7s} {'振幅':>6s} {'规模':<4s}",
        ]
        for code in batch_codes:
            f = features.get(code)
            if not f:
                continue
            lines.append(
                f"{code:<12s} {f['close']:>8.2f} {f['vs_ma60_pct']:>+6.1f}% "
                f"{str(f['alignment']):<8s} {str(f['volume_relation']):<10s} "
                f"{f['chg_20d']:>+6.1f}% {f.get('chg_30d',0):>+6.1f}% {f['chg_60d']:>+6.1f}% "
                f"{f['amplitude_20d']:>5.1f}% {str(f.get('size','?')):<4s}"
            )

        lines.append("")
        lines.append("输出JSON数组，每只股票都要有结果：")
        lines.append('[{"ts_code":"...","tier":"FAVORED/NEUTRAL/AVOID","confidence":0.0-1.0,"reason":"..."}]')
        lines.append("")
        lines.append("要求：confidence>=0.7才给FAVORED或AVOID；信号矛盾时给NEUTRAL并说明原因。")

        result = llm.chat_json("\n".join(lines))
        if result:
            # Handle both array and wrapped-object responses
            items = result if isinstance(result, list) else result.get("classifications", [])
            for item in items:
                code = item.get("ts_code", "")
                if code in batch_codes:
                    tier_raw = str(item.get("tier", "")).upper()
                    # Normalize to valid tiers
                    if tier_raw in (TIER_FAVORED, TIER_NEUTRAL, TIER_AVOID):
                        tier = tier_raw
                    elif "FAVOR" in tier_raw:
                        tier = TIER_FAVORED
                    elif "AVOID" in tier_raw:
                        tier = TIER_AVOID
                    else:
                        tier = TIER_NEUTRAL

                    conf = item.get("confidence", 0.5)
                    if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
                        conf = 0.5

                    reason = item.get("reason", f"LLM分类: {tier}")

                    results[code] = {
                        "tier": tier,
                        "reason": reason,
                        "confidence": round(conf, 2),
                        "source": "llm",
                    }

        # Any codes in this batch not classified → rule fallback for this batch
        missing = [c for c in batch_codes if c not in results and c in features]
        for code in missing:
            f = features[code]
            results[code] = rule_neutral_classify(code, f, source="rule_fallback_llm_fail")

        # Progress: log every 5 batches or every batch when total <= 10
        if batch_num % 5 == 0 or batch_num == total_batches:
            classified = len([c for c in batch_codes if c in results and results[c].get("source") != "rule_fallback_llm_fail"])
            logger.info(f"  LLM batch {batch_num}/{total_batches}: {classified}/{len(batch_codes)} classified, "
                       f"total_so_far={len(results)}")

    logger.info(f"  LLM complete: {len(results)} total classified")
    return results


def rule_neutral_classify(code, f, source="rule_default"):
    """Rule-based classification for a single stock (used as fallback and NEUTRAL default).

    Coarse multi-signal rules: alignment + vs_ma60 direction must agree.
    """
    vs_ma = f["vs_ma60_pct"]
    alignment = f.get("alignment", "mixed")

    if alignment == "bullish" and vs_ma > 0:
        return {
            "tier": TIER_FAVORED,
            "reason": f"多头排列且站上MA60({vs_ma:+.0f}%) (规则兜底)",
            "confidence": 0.55,
            "source": source,
        }
    elif alignment == "bearish" and vs_ma < 0:
        return {
            "tier": TIER_AVOID,
            "reason": f"空头排列且低于MA60({vs_ma:+.0f}%) (规则兜底)",
            "confidence": 0.55,
            "source": source,
        }
    else:
        return {
            "tier": TIER_NEUTRAL,
            "reason": f"均线{('多头' if alignment=='bullish' else '空头' if alignment=='bearish' else '杂乱')} 价格vsMA60{vs_ma:+.0f}% (规则兜底)",
            "confidence": 0.5,
            "source": source,
        }


def classify_stocks(conn, codes, trade_date):
    """Main classification pipeline with three paths.

    Returns dict {ts_code: {tier, reason, confidence, source}}.
    """
    features = get_stock_features(conn, codes)
    if not features:
        return {}

    all_results = {}
    feated_codes = set(features.keys())

    # Path 1: Rule-direct for obvious trends
    direct = classify_rule_direct(features)
    all_results.update(direct)
    directed_codes = set(direct.keys())
    logger.info(f"  Rule-direct: {len(direct)} ({len(directed_codes & {c for c in direct if direct[c]['tier']==TIER_FAVORED})} FAVORED, "
                f"{len(directed_codes & {c for c in direct if direct[c]['tier']==TIER_AVOID})} AVOID)")

    # Remaining: stocks not classified by rule-direct
    remaining = [c for c in feated_codes if c not in directed_codes]

    # Path 2: Sample 30% of remaining for LLM
    llm_candidates = random.sample(remaining, max(1, int(len(remaining) * LLM_SAMPLE_RATE))) if remaining else []
    if llm_candidates:
        llm_results = classify_llm(conn, llm_candidates, features)
        all_results.update(llm_results)
        logger.info(f"  LLM sampled: {len(llm_candidates)} → {len(llm_results)} classified")

    # Path 3: Rest → rule-default NEUTRAL
    llm_codes = set(llm_results.keys()) if llm_candidates else set()
    rest = [c for c in remaining if c not in llm_codes]
    for code in rest:
        all_results[code] = rule_neutral_classify(code, features[code])
    if rest:
        logger.info(f"  Rule-default NEUTRAL: {len(rest)}")

    return all_results


def run(strategy="long_term", trade_date=None, mode="weekly"):
    """Classify stocks and build strategy-specific universe."""
    start = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Validate strategy
    if strategy not in ("long_term", "hot_picks"):
        logger.error(f"Invalid strategy: {strategy}")
        return

    conn = get_connection()
    logger.info(f"Agent 0 starting: strategy={strategy} mode={mode} date={trade_date}")

    try:
        all_codes = [r["ts_code"] for r in conn.execute("SELECT ts_code FROM stocks").fetchall()]

        # Full classification every run — incremental warm runs missed too many changes
        target_codes = all_codes
        logger.info(f"Classifying all {len(all_codes)} stocks")

        # Step 1: Hard exclusions (pure rules)
        excluded = check_excluded(conn, trade_date)
        holdings = check_holdings(conn)

        # Momentum AVOID: all timeframes falling → low priority
        momentum_avoid = check_momentum_avoid(conn, trade_date)
        logger.info(f"  Momentum AVOID: {len(momentum_avoid)}")

        # Trend gate: 3-signal persistent decline detection (MA60 below ratio + MA bearish + near low)
        persistent_decline = check_persistent_decline(conn, trade_date)
        logger.info(f"  Persistent decline AVOID: {len(persistent_decline)}")

        logger.info(f"  EXCLUDED: {len(excluded)}, HOLDING: {len(holdings)}")

        # Step 2: Classify remaining stocks
        remaining = [c for c in target_codes if c not in excluded and c not in holdings]
        tier_results = classify_stocks(conn, remaining, trade_date)

        # Step 3: Write tier_assignments with new fields
        for code in target_codes:
            if code in excluded:
                tier, reason, confidence, source = TIER_EXCLUDED, excluded[code], 0.95, "rule_direct"
            elif code in holdings:
                tier, reason, confidence, source = TIER_HOLDING, holdings[code], 0.95, "rule_direct"
            elif code in momentum_avoid:
                tier, reason, confidence, source = TIER_AVOID, momentum_avoid[code], 0.85, "rule_direct"
            elif code in persistent_decline:
                tier, reason, confidence, source = TIER_AVOID, persistent_decline[code], 0.80, "trend_gate"
            elif code in tier_results:
                r = tier_results[code]
                tier, reason, confidence, source = r["tier"], r["reason"], r["confidence"], r["source"]
            else:
                tier, reason, confidence, source = TIER_NEUTRAL, "默认中性(无足够数据)", 0.3, "rule_default"

            conn.execute("DELETE FROM tier_assignments WHERE ts_code=?", (code,))
            conn.execute(
                """INSERT INTO tier_assignments
                   (ts_code, tier, reason, confidence, source, updated_at)
                   VALUES (?,?,?,?,?,?)""",
                (code, tier, reason, confidence, source, trade_date),
            )

        conn.commit()
        elapsed = time.time() - start

        # Distribution summary
        dist = {t: 0 for t in (TIER_EXCLUDED, TIER_HOLDING, TIER_FAVORED, TIER_NEUTRAL, TIER_AVOID)}
        for r in tier_results.values():
            dist[r["tier"]] = dist.get(r["tier"], 0) + 1
        dist[TIER_EXCLUDED] = len(excluded)
        dist[TIER_HOLDING] = len(holdings)

        summary = (f"{strategy}: EXCLUDED:{dist[TIER_EXCLUDED]} HOLDING:{dist[TIER_HOLDING]} "
                   f"FAVORED:{dist[TIER_FAVORED]} NEUTRAL:{dist[TIER_NEUTRAL]} AVOID:{dist[TIER_AVOID]}")
        logger.info(f"Agent 0 complete: {summary} in {elapsed:.1f}s")

        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, stocks_processed, duration_s, summary) "
            "VALUES (0, ?, 'SUCCESS', ?, ?, ?)",
            (trade_date, len(target_codes), elapsed, summary),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 0 failed: {e}")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (0, ?, 'FAILED', ?, ?)",
            (trade_date, time.time() - start, str(e)[:200]),
        )
        conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
