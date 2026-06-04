"""
Agent 7: Portfolio Constructor — conviction-weighted, LLM-decided, factor-aware.

Architecture (v4.0):
  Layer 1: Conviction scoring (zero LLM, strategy-differentiated weights)
  Layer 2: LLM portfolio construction (bounded authority, conviction-gated)
  Layer 3: Hard constraint enforcement (zero LLM)

Dual strategy:
  long_term — value+growth+stable, 2-4 week hold, fundamental-weighted conviction
  hot_picks — momentum+breakout+short_term, 3-5 day hold, momentum-weighted conviction

LLM authority (strategy-differentiated):
  long_term:  STRONG≥0.50 (must include), MODERATE≥0.30 (optional), WEAK<0.30 (reject)
  hot_picks:  STRONG≥0.45 (must include), MODERATE≥0.25 (optional), WEAK<0.25 (reject)

Downstream: A6 (risk review)
"""
import json
import math
import re
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

RAPID_SELL_DAYS = 7

# ══════════════════════════════════════════════════════════════════
# Layer 1: Conviction Scoring (zero LLM)
# ══════════════════════════════════════════════════════════════════

def compute_conviction(code, a5_scores, a2_report, fusion_fragility, macro_regime,
                       momentum_d3=0, momentum_d5=0, momentum_d20=0, momentum_d60=0,
                       momentum_accel=0, strategy="long_term"):
    """Per-stock conviction score 0-1. Pure math, strategy-differentiated weights.

    long_term: fundamental-heavy (mom 25%, a2 15%) — quality + sustainability
    hot_picks: momentum-heavy (mom 35%, a2 8%) — short-term upside potential
    """
    # Base: A5 score normalized to 0-1
    base = a5_scores.get("total_score", 50) / 100.0

    # A2 data quality: reduced weight (25% → 15%)
    a2_conf = a2_report.get("confidence", 0) if a2_report else 0
    a2_factor = 0.4 + 0.6 * a2_conf

    # Fragility from A5 fusion analysis
    fragility_map = {
        "STABLE": 1.0, "MEDIUM": 0.7, "REVERSING": 0.7,
        "HIGH": 0.5, "NO_DATA": 0.5,
    }
    fragility = fragility_map.get(fusion_fragility, 0.5) if fusion_fragility else 0.5
    if not fusion_fragility:
        fragility = 0.5 + 0.5 * a2_conf

    # Momentum quality (25%): multi-timeframe signal assessment, v3
    # Priority: acceleration + d3 early signal > duration of trend
    # d3>0 is the earliest reversal indicator — strongly rewarded
    if momentum_d3 > 0 and momentum_d5 > 0 and momentum_accel > 3:
        momentum_quality = 1.0   # confirmed upturn + accelerating — strongest
    elif momentum_d5 > 0 and momentum_accel > 3:
        momentum_quality = 0.90  # rising + accelerating
    elif momentum_d3 > 0 and momentum_d5 > 0:
        momentum_quality = 0.80  # short+mid term confirmed upturn
    elif momentum_d3 > 0 and momentum_accel > 3:
        momentum_quality = 0.70  # d3 just turned positive + acc strong — early reversal signal
    elif momentum_d3 > 0 and momentum_accel > 0:
        momentum_quality = 0.60  # d3 positive + trend improving — early stage
    elif momentum_d5 > 0 and momentum_accel > 0:
        momentum_quality = 0.55  # rising, not decelerating
    elif momentum_d5 > 0:
        momentum_quality = 0.45  # basic uptrend
    elif momentum_d5 < 0 and momentum_accel > 3:
        momentum_quality = 0.30  # still falling, slowing fast (not yet reversal)
    elif momentum_d5 < 0 and momentum_accel > 0:
        momentum_quality = 0.20  # falling but slowing slightly
    elif momentum_accel < -5:
        momentum_quality = 0.10  # strong deceleration
    elif momentum_accel < -3:
        momentum_quality = 0.15  # moderate deceleration
    else:
        momentum_quality = 0.35  # neutral / indeterminate

    # Red flag penalty
    rf_penalty = 0.0
    if a2_report:
        for rf in a2_report.get("red_flags", []):
            sev = rf.get("severity", "MEDIUM") if isinstance(rf, dict) else "MEDIUM"
            rf_penalty += 0.15 if sev == "HIGH" else 0.08 if sev == "MEDIUM" else 0.03
    rf_penalty = min(0.5, rf_penalty)

    # Macro regime reference (qualitative only, no numeric factor)

    # d3/d5 dual-negative penalty (replaces hard REJECT rule)
    d3_d5_penalty = 0.15 if (momentum_d3 < 0 and momentum_d5 < 0) else 0.0

    # Strategy-differentiated conviction weights
    if strategy == "hot_picks":
        conviction = (
            base * 0.25 +
            momentum_quality * 0.35 +    # primary: momentum is everything
            a2_factor * 0.08 +           # fundamentals: data quality only
            fragility * 0.12 +            # trend health
            (1.0 - rf_penalty - d3_d5_penalty) * 0.20
        )
    else:  # long_term
        conviction = (
            base * 0.25 +
            momentum_quality * 0.25 +    # balanced: quality + sustainability
            a2_factor * 0.15 +           # fundamentals matter more
            fragility * 0.15 +            # trend stability matters more
            (1.0 - rf_penalty - d3_d5_penalty) * 0.20
        )
    return round(max(0.05, min(1.0, conviction)), 3)


def get_conviction_tier(conviction, strategy="long_term"):
    """Map conviction score to tier dict. Strategy-differentiated thresholds."""
    if strategy == "hot_picks":
        if conviction >= 0.45:     return {"min": 0.45, "max_weight": 0.25, "label": "高确信(热)"}
        elif conviction >= 0.25:   return {"min": 0.25, "max_weight": 0.15, "label": "中等确信(热)"}
        else:                      return {"min": 0.0,  "max_weight": 0.0,  "label": "低确信(自动剔除)"}
    else:
        if conviction >= 0.50:     return {"min": 0.50, "max_weight": 0.25, "label": "高确信"}
        elif conviction >= 0.30:   return {"min": 0.30, "max_weight": 0.15, "label": "中等确信"}
        else:                      return {"min": 0.0,  "max_weight": 0.0,  "label": "低确信(自动剔除)"}


# ══════════════════════════════════════════════════════════════════
# Layer 2: LLM Portfolio Construction (flash, bounded authority)
# ══════════════════════════════════════════════════════════════════

def llm_construct_portfolio(candidates, holdings_sells, macro_report, fusion_report,
                            macro_regime, strategy, cfg, conn):
    """LLM makes the final portfolio construction decisions.

    Authority:
      - STRONG tier (conv>=0.5): full discretion, 8%-25%
      - MODERATE tier (conv 0.3-0.5): limited, 8%-15%
      - WEAK tier (conv<0.3): auto-rejected, LLM CANNOT include

    LLM decides:
      - Which STRONG/MODERATE stocks to include (≤ max_holdings)
      - What weight to assign each
      - Which to reject (with reasons)
      - Factor exposure narrative
    """
    if not settings.ds_api_key or not candidates:
        return None

    # ── Per-stock rich context ──
    codes = [c["ts_code"] for c in candidates]
    ph = ",".join("?" * len(codes))

    # A2 fundamental summaries
    a2_data = {}
    for r in conn.execute(
        f"SELECT ts_code, report_json FROM fundamental_reports "
        f"WHERE ts_code IN ({ph}) AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)",
        codes,
    ).fetchall():
        try:
            rep = json.loads(r["report_json"])
            a2_data[r["ts_code"]] = {
                "eq": rep.get("earnings_quality", {}).get("rating", "?"),
                "gq": rep.get("growth_quality", {}).get("rating", "?"),
                "fh": rep.get("financial_health", {}).get("rating", "?"),
                "val": rep.get("valuation", {}).get("rating", "?"),
                "rfs": [f"{rf.get('severity','?')}:{rf.get('flag','?')}" for rf in rep.get("red_flags", [])],
                "conf": rep.get("confidence", 0),
            }
        except (json.JSONDecodeError, TypeError):
            pass

    # Industry map
    code_industry = {}
    for r in conn.execute(
        f"SELECT ts_code, industry FROM stocks WHERE ts_code IN ({ph})", codes
    ).fetchall():
        code_industry[r["ts_code"]] = r["industry"] or "其他"

    # ── Market background news (moderate weight, context only) ──
    market_news = []
    for r in conn.execute(
        "SELECT category, sentiment, impact, summary FROM news_feed "
        "WHERE published_at >= datetime('now','-3 days') AND impact IN ('HIGH','MEDIUM') "
        "ORDER BY impact='HIGH' DESC, published_at DESC LIMIT 8"
    ).fetchall():
        sent = {"positive": "➕", "negative": "➖", "neutral": "➡️"}.get(r["sentiment"], "")
        market_news.append(f"  {sent}[{r['impact']}] {r['category']}: {r['summary'][:80]}")
    market_news_str = "\n".join(market_news) if market_news else "（近3天无重要新闻）"

    # ── Factor exposure summary ──
    factor_counts = defaultdict(list)
    for c in candidates:
        tag = c.get("driver_tag", "平衡")
        factor_counts[tag].append(c["ts_code"])
    factor_summary = "; ".join(f"{k}:{len(v)}只" for k, v in sorted(factor_counts.items(), key=lambda x: -len(x[1])))

    # ── Industry summary ──
    ind_counts = defaultdict(list)
    for c in candidates:
        ind = code_industry.get(c["ts_code"], "其他")
        ind_counts[ind].append(c["ts_code"])
    ind_summary = "; ".join(f"{k}:{len(v)}只" for k, v in sorted(ind_counts.items(), key=lambda x: -len(x[1]))[:12])

    # ── Effective max holdings from regime ──

    if not candidates:
        logger.warning("A7: candidates empty, no candidates to construct portfolio")
        return {}

    # ── Build segmented prompt ──
    regime_str = macro_regime.get("regime", "?") if macro_regime else "?"
    regime_summary = macro_report.get("regime_summary", regime_str) if macro_report else regime_str
    macro_narrative = str(macro_report.get("narrative", "无"))[:150] if macro_report else "无"
    sector_view = str(macro_report.get("sector_view", "")) if macro_report else ""
    fusion_narrative = str(fusion_report.get("overall_narrative", "无"))[:200] if fusion_report else "无"


    # ── Percentage-based inclusion boundary, 7-20% per batch ──
    include_pct = 0.125 if strategy == "hot_picks" else 0.10
    max_include = max(5, int(len(candidates) * include_pct))

    # Strategy-differentiated role descriptions
    if strategy == "hot_picks":
        role_lines = [
            "=== 你的角色 ===",
            "你是热点猎手。持仓3-5天，快进快出。",
            "你的任务是找出「动量最强、短期最可能快速上涨」的股票。",
            "不关心长期价值——只关心未来几天能不能涨。",
            "",
            "选股优先级: 动量强度 > 量价配合 > 基本面(仅查硬伤)",
            "硬伤: fh=POOR且eq=LOW → 建议REJECT(LLM可酌情纳入,需明确反驳理由)。其余基本面瑕疵(ROE低/负债高/估值高/数据缺失)不重要。",
            "",
            "对每只候选股回答: 1.动量够不够强? 2.什么情况下判断错误? 3.3-5天内空间够不够?",
        ]
    else:
        role_lines = [
            "=== 你的角色 ===",
            "你是长期价值侦察兵。持仓2-4周，不追短期热点。",
            "你的任务是找到「基本面扎实+趋势健康、未来2-4周有持续上涨空间」的股票。",
            "",
            "选股优先级: 趋势质量 > 基本面支撑 > 估值合理性",
            "HIGH红旗 → 建议REJECT(LLM可酌情纳入,需明确反驳理由)",
            "",
            "对每只候选股回答: 1.趋势驱动力是什么? 2.基本面是否支撑这个趋势? 3.风险边界在哪?",
        ]

    lines = role_lines + [
        "",
        f"=== 决策指引 ({strategy}) ===",
        f"候选池按确信度排序, 分批处理。每批次独立评估, 选出该批次中最值得纳入的标的。",
        "",
        # Strategy-differentiated rejection criteria
        *(
            ["[hot_picks] 你是做热点的, 不是做价值投资! 你在选'最可能涨的股票', 不是'最好的公司'。",
             "  唯一硬规则: d3<0 且 d5<-5 且 acc<0 → 必须REJECT",
             "  ROE低/负债率高/毛利率下降/估值高/利润异常/数据缺失 —— 这些都不重要! 动量够强就纳入。",
             "  一个d5=+40%的股票, 只要财务不恶化, 就算利润为负(亏损收窄), 也值得推荐。不要用long_term的标准衡量hot_picks。"]
            if strategy == "hot_picks" else
            ["[long_term] 选股指引(强烈建议, 非硬规则):",
             "  1. d3<0 且 d5<0 → 短期仍在下跌, 强烈建议REJECT(LLM可酌情纳入,需在rationale中明确说明)",
             "  2. HIGH红旗 → 高风险警告, 建议REJECT(LLM可酌情纳入,需明确反驳理由)",
             "  以下可酌情: acc:0~-3且d5>0=正常盘整可纳, acc:-3~-5但d20/d60>0=降权, acc<-5=谨慎"]
        ),
        "",
        "通用:",
        "- 反转判断: d3>0且acc>0 = 短期势头已转向上。",
        "- 不要凑数, 也不强制压缩。按标准来。",
        "",
        f"权重指引 (最低{cfg['min_single_weight']:.0%}, 最高{cfg['max_single_weight']:.0%}): 所有纳入必须≥{cfg['min_single_weight']:.0%}, 否则自动裁掉.",
        f"- 高确信(≥0.7)+趋势加速 → {max(0.15, cfg['min_single_weight']):.0%}-{cfg['max_single_weight']:.0%}",
        f"- 高确信+趋势持续 → {max(0.10, cfg['min_single_weight']):.0%}-{min(0.18, cfg['max_single_weight']):.0%}",
        f"- 确信中等或趋势减速 → {cfg['min_single_weight']:.0%}-{max(0.12, cfg['min_single_weight']):.0%}",
        "- 所有纳入的权重必须不低于最低值，否则会被系统自动裁掉",
        "",
        "趋势阶段参考：",
        "ACCELERATING（加速初期）= 最佳窗口，积极配置",
        "SUSTAINING（持续中）= 可参与，适度配置",
        "DECELERATING（减速中）= 降权重但不一定拒绝",
        "REVERSING（反转中）= 建议回避，可参与但需谨慎",
        "",
        "=== 上下文 ===",
        f"当前宏观: {regime_str} | 策略: {strategy}",
        f"宏观判断: {regime_summary}",
        f"候选池: {len(candidates)}只, 按确信度排序, 分批独立评估 (每批按自身质量选取)",
        f"权重范围: {cfg['min_single_weight']:.0%}-{cfg['max_single_weight']:.0%}",
        f"A5趋势分析: {fusion_narrative}",
        f"",
        f"=== 板块资金偏好（来自宏观分析，选股参考，非硬性约束） ===",
        f"{sector_view if sector_view else '板块数据暂不可用'}",
        "",
        "强势板块有资金支撑，弱势板块需更强个股逻辑才能纳入。",
        f"",
        f"=== 市场背景新闻（近3天参考，非指令）===",
        f"{market_news_str}",
        "",
        "注：新闻权重适中，在评估相关股票时可作为背景参考，但不作为主要决策依据。",
        "",
    ]

    # Pool summary (not full listing — each batch gets its own candidates)
    lines.append(f"=== 候选池概览 ({len(candidates)}只) ===")
    lines.append(f"因子分布: {factor_summary}")
    lines.append(f"行业分布: {ind_summary}")
    lines.append("")

    # Sell/Hold review
    if holdings_sells:
        lines.append(f"\n=== 持仓审查 ({len(holdings_sells)}只) ===")
        for h in holdings_sells:
            lines.append(
                f"  {h['ts_code']} {h['action']} | 排名{h.get('rank','?')} | "
                f"盈亏{h.get('pnl_pct',0):.0f}% | 持有{h.get('hold_days',0)}天 | {h.get('reason','')}"
            )

    # ── Footer template (per-batch values filled in loop) ──
    def build_footer(batch_size):
        n_min = max(1, int(batch_size * 0.07))
        n_max = max(1, int(batch_size * 0.20))
        return [
            "",
            f"=== 本批次选择指引 ({batch_size}只) ===",
            f"从以上列出的{batch_size}只候选股中, 按7-20%比例选出{n_min}~{n_max}只纳入(至少1只)。",
            f"其余输出REJECT。每只都要有决策——遗漏自动视为REJECT。",
            "",
            "输出JSON (rationale和risk_boundary均不超过180字):",
            "{",
            '  "a7_decisions": [',
            '    {"ts_code":"XXXX01.SZ","a7_recommendation":"INCLUDE","weight":0.15,',
            '     "rationale":"日线三连阳突破MA60,d5=+8%d20=+15%趋势加速acc=7,量价配合良好,新能车板块回暖,短期有继续上行空间",',
            '     "risk_boundary":"若跌破MA20或单日跌幅>5%则止损"},',
            '    {"ts_code":"XXXX02.SH","a7_recommendation":"REJECT",',
            '     "rationale":"d3=-2%d5=-4%仍在下跌,红旗:净利润连续3季为负+负债率>80%,技术面和基本面都未出现反转信号"},',
            '    ...为上述每只候选股输出一条决策(INCLUDE或REJECT),具体说清选/不选的理由...',
            '  ],',
            '  "a7_batch_cash": 0.0,',
            '  "a7_batch_narrative": "本批次小结"',
            "}",
        ]

    # ── Batch candidates and call LLM ──
    A7_BATCH_SIZE = 15
    all_a7_decisions = {}
    batches = [candidates[i:i + A7_BATCH_SIZE] for i in range(0, len(candidates), A7_BATCH_SIZE)]
    llm_cfg = settings.get_llm_config("A7")

    # Wait for API rate limit cooldown (A5 just finished its LLM call)
    logger.info(f"A7 LLM: cooling down 3s, then {len(batches)} batches of ≤{A7_BATCH_SIZE}...")
    time.sleep(3)

    for batch_idx, batch in enumerate(batches):
        batch_start = batch_idx * A7_BATCH_SIZE + 1
        batch_lines = list(lines)  # copy header

        # Candidate display for this batch — unified format, no trend_type split
        batch_lines.append(f"=== 候选池 批次{batch_idx+1}/{len(batches)} ({len(batch)}只, 按确信度排序) ===")
        for i, c in enumerate(batch):
            code = c["ts_code"]; a2 = a2_data.get(code, {}); ind = code_industry.get(code, "?")
            tier = get_conviction_tier(c.get("conviction", 0), strategy)
            batch_lines.append(
                f"{batch_start+i:2d}. {code} [{c.get('conviction',0):.2f}/{tier['label']}] "
                f"A5#{c.get('a5_rank','?')}/{c.get('a5_score',0):.0f} "
                f"[T:{c.get('tech_score',0):.0f} F:{c.get('fund_score',0):.0f}] "
                f"d3:{c.get('momentum_d3',0):+.1f}% d5:{c.get('momentum_d5',0):+.1f}% "
                f"d20:{c.get('momentum_d20',0):+.1f}% d60:{c.get('momentum_d60',0):+.1f}% acc:{c.get('momentum_accel',0):+.1f}"
            )
            if a2:
                batch_lines.append(f"    A2: 盈{a2.get('eq','?')} 财{a2.get('fh','?')} 估{a2.get('val','?')} | 红旗:{a2.get('rfs',[]) or '无'} | 置信:{a2.get('conf',0):.0%}")
            else:
                batch_lines.append(f"    A2: 缺失")
            batch_lines.append(f"    现价:{c.get('price',0):.2f}")

        # Holdings review (first batch only)
        if batch_idx == 0 and holdings_sells:
            batch_lines.append(f"\n=== 持仓审查 ({len(holdings_sells)}只) ===")
            for h in holdings_sells:
                batch_lines.append(
                    f"  {h['ts_code']} {h['action']} | 排名{h.get('rank','?')} | "
                    f"盈亏{h.get('pnl_pct',0):.0f}% | 持有{h.get('hold_days',0)}天 | {h.get('reason','')}"
                )

        batch_lines.extend(build_footer(len(batch)))
        batch_prompt = "\n".join(batch_lines)

        # Call LLM with retries
        batch_result = None
        backoffs = [5, 15, 30, 60, 120]
        for attempt in range(5):
            if attempt > 0:
                wait = backoffs[min(attempt - 1, len(backoffs) - 1)]
                logger.warning(f"A7 batch {batch_idx+1}/{len(batches)} attempt {attempt+1}/5 failed, retrying in {wait}s...")
                time.sleep(wait)

            raw = llm.chat(batch_prompt, model=llm_cfg["model"], max_tokens=llm_cfg["max_tokens"])
            if raw and len(raw.strip()) > 10:
                parsed = None
                if "```" in raw:
                    blocks = raw.split("```")
                    for bi, block in enumerate(blocks):
                        if bi % 2 == 1:
                            block = block.strip()
                            if block.startswith("json"): block = block[4:].strip()
                            try: parsed = json.loads(block); break
                            except (json.JSONDecodeError, TypeError): continue
                if not parsed:
                    try: parsed = json.loads(raw)
                    except (json.JSONDecodeError, TypeError): pass
                if not parsed:
                    for bo, bc in [("{", "}"), ("[", "]")]:
                        s = raw.find(bo); e = raw.rfind(bc)
                        if s != -1 and e > s:
                            try:
                                parsed = json.loads(raw[s:e + 1])
                                if isinstance(parsed, list): parsed = {"a7_decisions": parsed}
                                break
                            except (json.JSONDecodeError, TypeError): continue
                if parsed:
                    logger.info(f"A7 batch {batch_idx+1}/{len(batches)} OK (attempt {attempt+1}, {len(raw)} chars)")
                    batch_result = parsed
                    break
                else:
                    logger.warning(f"A7 batch {batch_idx+1}: {len(raw)} chars but unparseable")

            result2 = llm.chat_json(batch_prompt, model=llm_cfg["model"], max_tokens=llm_cfg["max_tokens"])
            if result2:
                logger.info(f"A7 batch {batch_idx+1}/{len(batches)} OK (attempt {attempt+1}, chat_json)")
                batch_result = result2
                break

        if batch_result is None:
            raise RuntimeError(f"A7 LLM 不可用: batch {batch_idx+1}/{len(batches)} 5次尝试全部失败")

        # Collect decisions from this batch
        for d in batch_result.get("a7_decisions", []):
            if isinstance(d, dict) and d.get("ts_code"):
                all_a7_decisions[d["ts_code"]] = d

        n_entries = len(batch_result.get("a7_decisions", []))
        logger.info(f"A7 batch {batch_idx+1}/{len(batches)}: {n_entries}/{len(batch)} entries, {len(raw) if raw else 0} chars (coverage {n_entries/len(batch):.0%})")

    # ── Build merged result ──
    merged = {
        "a7_decisions": list(all_a7_decisions.values()),
        "a7_strategic_cash": 0.0,
        "a7_cash_rationale": "",
        "a7_holdings_count_rationale": "",
        "a7_portfolio_narrative": "",
        "a7_factor_exposure": {},
    }
    logger.info(f"A7 all {len(batches)} batches done: {len(all_a7_decisions)} decisions")
    return merged


# ══════════════════════════════════════════════════════════════════
# Layer 3: Validation & Persistence
# ══════════════════════════════════════════════════════════════════

def validate_and_enforce(portfolio, candidates, cfg):
    """Hard constraint enforcement. Clips weights, removes violations. Returns (clean, violations)."""
    violations = []
    clean = {}

    # Sort by weight descending
    sorted_items = sorted(portfolio.items(), key=lambda x: -x[1]) if isinstance(portfolio, dict) else sorted(
        [(p["ts_code"], p["weight"]) for p in (portfolio if isinstance(portfolio, list) else [])],
        key=lambda x: -x[1],
    )

    if isinstance(portfolio, list):
        portfolio = {p["ts_code"]: p["weight"] for p in portfolio}

    # Safety cap (prevents LLM from going wild, not a strategy constraint)
    SAFETY_CAP = 25
    if len(portfolio) > SAFETY_CAP:
        excess = len(portfolio) - SAFETY_CAP
        violations.append(f"LLM选了{len(portfolio)}只 > {SAFETY_CAP}安全上限, 裁掉末{excess}只")
        sorted_items = sorted_items[:SAFETY_CAP]

    # Max single weight
    for code, w in sorted_items:
        if w > cfg["max_single_weight"]:
            violations.append(f"{code} 权重{w:.0%}>{cfg['max_single_weight']:.0%}, clip")
            w = cfg["max_single_weight"]
        clean[code] = w

    # Min single weight
    clean = {c: w for c, w in clean.items() if w >= cfg["min_single_weight"]}

    # Cash constraint
    total_w = sum(clean.values())
    max_total = 1.0 - cfg["min_cash"]
    if total_w > max_total:
        scale = max_total / total_w
        clean = {c: round(w * scale, 3) for c, w in clean.items()}
        violations.append(f"总仓位{total_w:.0%}>{max_total:.0%}, scale to {max_total:.0%}")

    return clean, violations


# ══════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════

def run(mode="daily", trade_date=None, strategy="long_term"):
    start = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    cfg = settings.get_strategy_config(strategy)
    logger.info(f"Agent 7 (v3): mode={mode} strategy={strategy}")

    try:
        # ── Load holdings ──
        holdings = [dict(h) for h in conn.execute("""
            SELECT p.*, s.industry, CAST((julianday(?) - julianday(p.entry_date)) AS INTEGER) as hold_days
            FROM portfolio p JOIN stocks s ON p.ts_code = s.ts_code
            WHERE p.status = 'HOLD' AND p.strategy = ?
        """, (trade_date, strategy)).fetchall()]

        latest_price = {r["ts_code"]: r["close"] for r in conn.execute(
            "SELECT ts_code, close FROM daily_quotes WHERE trade_date = (SELECT MAX(trade_date) FROM daily_quotes)"
        ).fetchall()}
        for h in holdings:
            h["current_price"] = latest_price.get(h["ts_code"])
            h["pnl_pct"] = ((h["current_price"] - h["entry_price"]) / h["entry_price"] * 100
                            if h["current_price"] and h["entry_price"] else 0)

        # ── Load A5 scores ──
        a5_scores = {}
        for r in conn.execute(
            "SELECT ts_code, total_score, tech_score, fundamental_score, macro_fit, momentum, rank, "
            "trend_type, momentum_d3, momentum_d5, momentum_d20, momentum_d60, momentum_accel "
            "FROM composite_scores "
            "WHERE calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?) "
            "AND strategy=?",
            (strategy, strategy),
        ).fetchall():
            a5_scores[r["ts_code"]] = dict(r)

        # ── Load macro ──
        macro_regime = dict(conn.execute(
            "SELECT regime FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
        ).fetchone()) if conn.execute("SELECT COUNT(*) FROM macro_regime").fetchone()[0] > 0 else {
            "regime": "CONSOLIDATION_UP"}

        macro_report = {}
        mr = conn.execute("SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1").fetchone()
        if mr:
            try:
                macro_report = json.loads(mr["report_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        fusion_report = {}
        fr = conn.execute(
            "SELECT report_json FROM fusion_reports WHERE strategy=? ORDER BY calc_date DESC LIMIT 1",
            (strategy,),
        ).fetchone()
        if fr:
            try:
                fusion_report = json.loads(fr["report_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Load A2 data ──
        a2_reports = {}
        for r in conn.execute(
            "SELECT ts_code, report_json FROM fundamental_reports fr "
            "WHERE fr.calc_date = (SELECT MAX(calc_date) FROM fundamental_reports WHERE ts_code=fr.ts_code)"
        ).fetchall():
            try:
                a2_reports[r["ts_code"]] = json.loads(r["report_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Load A5 fusion trend stages per stock ──
        fusion_fragility = {}
        if fusion_report:
            # New format: trend_stages with stage field
            for ts in fusion_report.get("trend_stages", []):
                if isinstance(ts, dict):
                    stage = ts.get("stage", "")
                    # Map trend stage to fragility tier
                    if stage in ("ACCELERATING", "SUSTAINING"):
                        fusion_fragility[ts.get("ts_code", "")] = "STABLE"
                    elif stage == "DECELERATING":
                        fusion_fragility[ts.get("ts_code", "")] = "MEDIUM"
                    elif stage == "REVERSING":
                        fusion_fragility[ts.get("ts_code", "")] = "REVERSING"
                    else:
                        fusion_fragility[ts.get("ts_code", "")] = "HIGH"
            # Fallback: old format factor_attribution
            if not fusion_fragility:
                for fa in fusion_report.get("factor_attribution", []):
                    if isinstance(fa, dict):
                        fusion_fragility[fa.get("ts_code", "")] = fa.get("fragility")

        # ── Load focus list ──
        from backend.focus_list import get_focus_codes
        focus = get_focus_codes(conn, strategy=strategy)
        focus_codes = {f["ts_code"] for f in focus}
        held_codes = {h["ts_code"] for h in holdings}

        # Industry map
        code_industry = {}
        for r in conn.execute(
            f"SELECT ts_code, industry FROM stocks WHERE ts_code IN "
            f"({','.join('?'*len(focus_codes))})",
            list(focus_codes),
        ).fetchall():
            code_industry[r["ts_code"]] = r["industry"] or "其他"

        # Factor driver tag from A5 scores
        def _driver_tag(code):
            sc = a5_scores.get(code, {})
            t, f, m = sc.get("tech_score", 50), sc.get("fundamental_score", 50), sc.get("momentum", 50)
            tags = []
            if m >= 90: tags.append("M↑↑")
            elif m >= 70: tags.append("M↑")
            if f >= 80: tags.append("F↑")
            elif f < 30: tags.append("F↓")
            if t >= 65: tags.append("T↑")
            elif t < 35: tags.append("T↓")
            return ",".join(tags) if tags else "平衡"

        logger.info(f"A7: regime={macro_regime.get('regime','?')}")

        # ── BUILD CANDIDATE POOL sorted by conviction ──
        focus_convictions = []
        for code in focus_codes:
            if code not in a5_scores or code in held_codes:
                continue
            sc = a5_scores.get(code, {})
            conv = compute_conviction(
                code, sc, a2_reports.get(code),
                fusion_fragility.get(code), macro_regime,
                strategy=strategy,
                momentum_d3=sc.get("momentum_d3") or 0,
                momentum_d5=sc.get("momentum_d5") or 0,
                momentum_d20=sc.get("momentum_d20") or 0,
                momentum_d60=sc.get("momentum_d60") or 0,
                momentum_accel=sc.get("momentum_accel") or 0,
            )
            tier = get_conviction_tier(conv, strategy)
            if tier["label"].startswith("低确信"):
                continue
            focus_convictions.append((code, conv))

        # Sort by conviction, take enough for LLM to choose from
        focus_convictions.sort(key=lambda x: -x[1])
        sample_pct = 0.50 if strategy == "long_term" else 0.80
        max_candidates = max(15, int(len(focus_convictions) * sample_pct))
        pool_codes = [c for c, _ in focus_convictions[:max_candidates]]

        # Build candidate list in conviction order (no trend-type segmentation)
        candidates = []
        for code in pool_codes:
            sc = a5_scores.get(code, {})
            price = latest_price.get(code)
            if not price or price <= 0:
                price = 10
            conv = dict(focus_convictions).get(code, 0.3)
            tier = get_conviction_tier(conv, strategy)
            tt = sc.get("trend_type", "short_term")

            # Position sizing
            buy_weight = cfg["min_single_weight"]
            available_cash = 1_000_000
            if holdings:
                pv = sum((h.get("current_price") or h.get("entry_price") or 10) * max(h.get("shares") or 100, 100)
                        for h in holdings)
                available_cash = pv * (1.0 - sum(h.get("weight", 0) for h in holdings) - cfg["min_cash"])
            shares = max(100, int(available_cash * buy_weight / price / 100) * 100)

            cand = {
                "ts_code": code,
                "conviction": conv,
                "tier": tier["label"],
                "trend_type": tt,
                "driver_tag": _driver_tag(code),
                "a5_rank": sc.get("rank", "?"),
                "a5_score": sc.get("total_score", 0),
                "tech_score": sc.get("tech_score", 50),
                "fund_score": sc.get("fundamental_score", 50),
                "momentum": sc.get("momentum", 50),
                "momentum_d3": sc.get("momentum_d3") or 0,
                "momentum_d5": sc.get("momentum_d5") or 0,
                "momentum_d20": sc.get("momentum_d20") or 0,
                "momentum_d60": sc.get("momentum_d60") or 0,
                "momentum_accel": sc.get("momentum_accel") or 0,
                "price": price,
                "suggested_shares": shares,
                "industry": code_industry.get(code, "其他"),
            }
            candidates.append(cand)
        logger.info(f"A7 candidates: {len(candidates)} total")

        # ── Sell/Hold decisions for current holdings ──
        holdings_decisions = []
        n_scores = len(a5_scores)
        for h in holdings:
            code = h["ts_code"]
            rank = a5_scores.get(code, {}).get("rank", 999)
            pnl = h.get("pnl_pct", 0)
            days = h.get("hold_days", 0)

            if strategy == "hot_picks":
                # Hot picks: aggressive — momentum decay is sell signal
                if rank > n_scores * 0.6:
                    action, reason = "SELL", f"排名{rank}(>{n_scores*0.6:.0f})→清仓"
                elif rank > n_scores * 0.4 and (pnl < -5 or days > 5):
                    action, reason = "REDUCE", f"排名{rank}下降+动量衰减→减仓"
                elif days > 5 and pnl < 0:
                    action, reason = "REDUCE", f"持有{days}日+亏损{pnl:.0f}%→时间止损"
                else:
                    action, reason = "HOLD", f"排名{rank}继续跟踪"
            else:
                # Long term: conservative — only sell on significant rank drop
                if rank > n_scores * 0.5:
                    if pnl < -10:
                        action, reason = "SELL", f"排名{rank}(>{n_scores//2})+亏损{pnl:.0f}%→止损"
                    elif pnl > 15:
                        action, reason = "REDUCE", f"排名{rank}(>{n_scores//2})+盈利{pnl:.0f}%→减仓锁定"
                    else:
                        action, reason = "REDUCE", f"排名{rank}(>{n_scores//2})→减仓观望"
                elif rank > n_scores * 0.3:
                    action, reason = ("REDUCE" if pnl < 0 else "HOLD"), f"排名{rank}中等"
                else:
                    action, reason = "HOLD", f"排名{rank}继续持有"

            d = {"ts_code": code, "action": action, "reason": reason,
                 "rank": rank, "pnl_pct": round(pnl, 2), "hold_days": days,
                 "type": "holding_review"}
            if action in ("SELL", "REDUCE") and days < RAPID_SELL_DAYS:
                d["rapid_sell_flag"] = True
            holdings_decisions.append(d)

        # ── LLM constructs portfolio ──
        llm_result = llm_construct_portfolio(
            candidates, holdings_decisions, macro_report, fusion_report,
            macro_regime, strategy, cfg, conn,
        )

        # ── Process LLM output → per-stock A7 tags ──
        # New format: unified a7_decisions array. Fallback: old portfolio+rejected format.
        a7_decisions = {}     # {ts_code: {a7_recommendation, weight, rationale}}
        a7_strategic_cash = cfg["min_cash"]
        a7_cash_rationale = ""
        a7_holdings_count_rationale = ""
        a7_portfolio_narrative = ""
        a7_factor_exposure = {}
        sell_confirmed = []
        sell_rejected = []

        if llm_result:
            # Preferred: unified a7_decisions array
            raw_decisions = llm_result.get("a7_decisions", [])
            if not raw_decisions:
                # Fallback: old portfolio+rejected format
                for item in llm_result.get("portfolio", []):
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = {
                            "a7_recommendation": "INCLUDE",
                            "weight": item.get("weight", 0),
                            "rationale": item.get("rationale", ""),
                        }
                for item in llm_result.get("rejected", []):
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = {
                            "a7_recommendation": "REJECT",
                            "weight": 0,
                            "rationale": item.get("reason", ""),
                        }
            else:
                for item in raw_decisions:
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = {
                            "a7_recommendation": item.get("a7_recommendation", "REJECT"),
                            "weight": item.get("weight", 0),
                            "rationale": item.get("rationale", ""),
                        }

            a7_strategic_cash = llm_result.get("a7_strategic_cash", llm_result.get("strategic_cash", cfg["min_cash"]))
            a7_cash_rationale = llm_result.get("a7_cash_rationale", llm_result.get("cash_reason", ""))
            a7_holdings_count_rationale = llm_result.get("a7_holdings_count_rationale", llm_result.get("holdings_count_reason", ""))
            a7_portfolio_narrative = llm_result.get("a7_portfolio_narrative", llm_result.get("portfolio_narrative", ""))
            a7_factor_exposure = llm_result.get("a7_factor_exposure", llm_result.get("factor_exposure", {}))
            sell_confirmed = llm_result.get("sell_confirm", [])
            sell_rejected = llm_result.get("sell_reject", [])
        else:
            # Fallback: mechanical — top by conviction, percentage-driven
            logger.warning("LLM unavailable — using mechanical fallback")
            top_n = max(5, int(len(candidates) * 0.20))
            sorted_cands = sorted(candidates, key=lambda c: -c["conviction"])
            for i, cand in enumerate(sorted_cands):
                if i < top_n:
                    a7_decisions[cand["ts_code"]] = {
                        "a7_recommendation": "INCLUDE",
                        "weight": cfg["min_single_weight"],
                        "rationale": f"机械fallback: 确信度#{i+1}/{len(candidates)}, 自动纳入",
                    }
                else:
                    a7_decisions[cand["ts_code"]] = {
                        "a7_recommendation": "REJECT",
                        "weight": 0,
                        "rationale": f"机械fallback: 确信度#{i+1}/{len(candidates)}, 超出top 20%",
                    }

        # ── Build final_portfolio from A7 decisions (shared by LLM and fallback) ──
        final_portfolio = {}
        for code, dec in a7_decisions.items():
            if dec["a7_recommendation"] == "INCLUDE":
                final_portfolio[code] = dec["weight"]

        # SAFEGUARD: log STRONG stocks that got REJECT
        strong_codes = {c["ts_code"] for c in candidates
                        if get_conviction_tier(c["conviction"], strategy)["label"].startswith("高确信")}
        missing_strong = strong_codes - set(final_portfolio.keys())
        if missing_strong:
            logger.warning(f"A7 rejected {len(missing_strong)} STRONG stocks (cash={a7_strategic_cash:.0%})")

        # ── Validate + enforce safety constraints ──
        clean_portfolio, violations = validate_and_enforce(final_portfolio, candidates, cfg)

        # ── Persist: ALL candidates as PENDING with A7 tags ──
        for cand in candidates:
            code = cand["ts_code"]
            a7 = a7_decisions.get(code, {
                "a7_recommendation": "REJECT",
                "weight": 0,
                "rationale": "LLM未覆盖(输出遗漏)",
            })

            rec = a7["a7_recommendation"]
            weight = a7.get("weight", 0)
            rationale = a7.get("rationale", "")

            review = {
                "a7": {
                    "recommendation": rec,
                    "conviction": cand["conviction"],
                    "tier": cand["tier"],
                    "weight": weight,
                    "rationale": rationale,
                    "driver_tag": cand.get("driver_tag", ""),
                },
                "industry": cand.get("industry", ""),
                "type": "buy_candidate",
            }

            conn.execute(
                """INSERT OR IGNORE INTO portfolio_decisions
                   (ts_code, calc_date, strategy, action, reason, weight_change,
                    suggested_shares, current_price, rapid_sell_flag,
                    macro_regime, risk_budget, status, review_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (code, trade_date, strategy,
                 "BUY" if rec == "INCLUDE" else "REJECT",
                 f"[A7:{rec}] {rationale[:80]}",
                 weight, cand.get("suggested_shares", 0), cand.get("price", 0),
                 0, macro_regime.get("regime", "?"), None,
                 "PENDING",
                 json.dumps(review, ensure_ascii=False),
                 datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
            )

        # Holdings decisions
        for d in holdings_decisions:
            action = d["action"]
            # Long term: LLM can override SELL (rescue good stocks)
            # Hot picks: LLM cannot override mechanical SELL (momentum decay = must sell)
            if action in ("SELL", "REDUCE"):
                if strategy == "long_term":
                    sell_overrides = {s.get("ts_code", ""): s for s in (sell_rejected or []) if isinstance(s, dict)}
                    if d["ts_code"] in sell_overrides:
                        action = "HOLD"
                        d["reason"] = f"LLM拒绝卖出: {sell_overrides[d['ts_code']].get('reason','')}"
            if action == "REDUCE":
                d["original_action"] = "REDUCE"
                action = "SELL"

            conn.execute(
                """INSERT OR IGNORE INTO portfolio_decisions
                   (ts_code, calc_date, strategy, action, reason, weight_change,
                    macro_regime, risk_budget, status, review_json, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (d["ts_code"], trade_date, strategy, action, d["reason"],
                 -0.02 if d.get("original_action") == "REDUCE" else 0,
                 macro_regime.get("regime", "?"), None,
                 "PENDING",
                 json.dumps({"type": "holding_review", "pnl_pct": d.get("pnl_pct", 0),
                             "hold_days": d.get("hold_days", 0)}),
                 datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
            )

        # ── Write portfolio_reports ──
        buy_industries = list(set(
            code_industry.get(code, "其他") for code in clean_portfolio
        ))
        report = {
            "date": trade_date, "strategy": strategy, "mode": mode,
            "macro_regime": macro_regime.get("regime", "?"),
            "risk_budget": None,
            "a7": {
                "effective_max_holdings": None,
                "strategic_cash": a7_strategic_cash,
                "cash_rationale": a7_cash_rationale,
                "holdings_count_rationale": a7_holdings_count_rationale,
                "portfolio_narrative": a7_portfolio_narrative,
                "factor_exposure": a7_factor_exposure,
                "include_count": sum(1 for d in a7_decisions.values() if d["a7_recommendation"] == "INCLUDE"),
                "reject_count": sum(1 for d in a7_decisions.values() if d["a7_recommendation"] == "REJECT"),
            },
            "conviction_tier_counts": {
                "STRONG": sum(1 for c in candidates if get_conviction_tier(c["conviction"], strategy)["label"].startswith("高确信")),
                "MODERATE": sum(1 for c in candidates if get_conviction_tier(c["conviction"], strategy)["label"].startswith("中等确信")),
                "WEAK_AUTO_REJECTED": sum(1 for c in candidates if get_conviction_tier(c["conviction"], strategy)["label"].startswith("低确信")),
            },
            "holdings_count": len(holdings),
            "buy_industries": buy_industries,
            "constraint_violations": violations,
        }
        conn.execute(
            "INSERT OR REPLACE INTO portfolio_reports (calc_date, strategy, report_json) VALUES (?,?,?)",
            (trade_date, strategy, json.dumps(report, ensure_ascii=False)),
        )

        conn.commit()
        elapsed = time.time() - start
        n_buy = len(clean_portfolio)
        n_reject = sum(1 for d in a7_decisions.values() if d["a7_recommendation"] == "REJECT")
        n_sell = sum(1 for d in holdings_decisions if d["action"] in ("SELL", "REDUCE"))
        logger.info(
            f"Agent 7 complete: {n_buy} buys, {n_reject} rejected, {n_sell} sells, "
            f"violations={len(violations)} in {elapsed:.1f}s"
        )
        conn.execute(
            "INSERT INTO agent_logs (agent_id,run_date,status,duration_s,summary) VALUES (7,?,'SUCCESS',?,?)",
            (trade_date, elapsed,
             f"[{strategy}] {mode}: {n_buy}B/{n_reject}R/{n_sell}S violations={len(violations)}"),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 7 failed: {e}", exc_info=True)
        conn.execute(
            "INSERT INTO agent_logs (agent_id,run_date,status,duration_s,summary) VALUES (7,?,'FAILED',?,?)",
            (trade_date, time.time() - start, str(e)[:200]))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    p.add_argument("--strategy", choices=["long_term", "hot_picks"], default="long_term")
    args = p.parse_args()
    run(mode=args.mode, strategy=args.strategy)
