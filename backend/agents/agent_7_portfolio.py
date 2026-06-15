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

def compute_conviction(code, fl_score, a2_report, strategy="long_term"):
    """Conviction 0-1 from FL score + A2 red flag fine-tuning.

    FL already handles: momentum quality, acceleration state, trend direction, volume health.
    A7 only adds: red flag fine-tuning (FL gates only check extremes).
    No duplicate computation of FL factors.
    """
    base = (fl_score or 50) / 100.0

    # Red flag fine-tuning: FL Gate 1 only rejects extremes (fh=POOR+eq=LOW, rfs≥5).
    # A7 adds granular penalty for remaining flags that passed the gate.
    rf_penalty = 0.0
    if a2_report:
        for rf in a2_report.get("red_flags", []):
            sev = rf.get("severity", "MEDIUM") if isinstance(rf, dict) else "MEDIUM"
            rf_penalty += 0.10 if sev == "HIGH" else 0.05 if sev == "MEDIUM" else 0.02
    rf_penalty = min(0.25, rf_penalty)

    # FL score dominates (85%), red flags fine-tune (15%)
    conviction = base * 0.85 + (1.0 - rf_penalty) * 0.15
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
                "narrative": rep.get("narrative", ""),
                "fundamental_score": rep.get("fundamental_score"),
                "score_rationale": rep.get("score_rationale", ""),
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
    # Unified: includes both general news and industry-tagged news from A3
    market_news = []
    for r in conn.execute(
        "SELECT category, sentiment, impact, summary, quantitative_info FROM news_feed "
        "WHERE published_at >= datetime('now','-3 days') AND impact IN ('HIGH','MEDIUM') "
        "ORDER BY impact='HIGH' DESC, published_at DESC LIMIT 12"
    ).fetchall():
        sent = {"positive": "➕", "negative": "➖", "neutral": "➡️"}.get(r["sentiment"], "")
        industries_hint = ""
        try:
            qi = json.loads(r["quantitative_info"] or "{}")
            affected = qi.get("affected_industries", [])
            if affected:
                industries_hint = " → " + "、".join(affected[:4])
        except (json.JSONDecodeError, TypeError):
            pass
        market_news.append(f"  {sent}[{r['impact']}] {r['category']}{industries_hint}: {r['summary'][:80]}")
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
    fusion_narrative = str(fusion_report.get("overall_narrative", "无"))[:100] if fusion_report else "无"

    if strategy == "hot_picks":
        role_lines = [
            "=== 你的角色 ===",
            "你是热点猎手。持仓3-5天，快进快出。",
            "你的核心任务是找出「刚启动、还在加速初期」的股票，而非「已经涨了很多」的股票。",
            "早期动量的价值 >> 已经兑现的涨幅。",
            "",
            "选股优先级: 加速度(是否刚启动) > 量价配合(资金是否在进场) > 方向健康度(是否还没透支)",
            "基本面信号(fh/eq/红旗)作为参考——3-5天持仓周期内，动量质量比财务质量更直接影响股价。",
            "但如果fh=POOR且eq=LOW同时出现，说明基本面全面崩溃，这类标的的上涨往往缺乏持续性。",
            "",
            "=== 上涨阶段判断（关键）===",
            "请结合每只股票的动量数据(d3/d5/d20/d60/acc)、RSI、量价序列，判断它当前处于哪个阶段:",
            "早期信号: acc>0且刚转正、放量、d20<20%尚未透支、RSI 40-65。空间最充裕，但也可能未确认。",
            "中期信号: acc>0、量平价升、d20 20-40%、RSI 65-75。趋势已确认，需判断剩余空间。",
            "后期信号: acc<0减速、缩量上涨、d20>40%已兑现较多、RSI>80。空间收窄，接盘风险上升。",
            "当多个后期信号叠加(如缩量+RSI>80+acc<-5)，上涨进入末期的概率显著增大。",
            "",
            "量价判断: 放量+资金流入=资金进场，趋势有支撑。缩量+价格上涨=可能是主力控盘或买盘衰竭，需结合位置判断。缩量回调后放量反弹=抛压释放后的资金回流，通常是最佳入场时机。",
            "",
            "对每只候选股回答: 1.处于上涨的哪个阶段? 2.量价是否支持继续涨? 3.3-5天内空间够不够?",
        ]
    else:
        role_lines = [
            "=== 你的角色 ===",
            "你是长期价值侦察兵。持仓2-4周，不追短期热点。",
            "你的任务是找到「基本面扎实+趋势健康、未来2-4周有持续上涨空间」的股票。",
            "",
            "选股优先级: 趋势质量 > 基本面支撑 > 估值合理性",
            "HIGH红旗代表A2深度分析发现的基本面风险信号，如果纳入需要明确的趋势或估值理由来平衡。",
            "",
            "对每只候选股回答: 1.趋势驱动力是什么? 2.基本面是否支撑这个趋势? 3.风险边界在哪?",
        ]

    if strategy == "hot_picks":
        rejection_rules = [
            "[hot_picks] 你是做热点的, 不是做价值投资!",
            "  以下信号供你参考，但不是绝对规则——结合具体情境做独立判断:",
            "  · 缩量 + RSI>80 + acc<-5: 上涨末期特征，趋势可能接近尾声",
            "  · d3<0且d5<-5且acc<0: 短期仍在走弱，需要更强的反转信号才能看多",
            "  · d20>60%: 中期涨幅已较大，即使d5还在涨，剩余空间可能有限",
            "  · ROE低/负债高/估值高/数据缺失: 在3-5天尺度上影响有限，动量质量更关键",
        ]
    else:
        rejection_rules = [
            "[long_term] 趋势阶段判断（2-4周持仓，关注早期反转）:",
            "  以下阶段描述供你参考，结合每只股票的具体数据做判断:",
            "  早期反转: d3>0+acc>3且d20或d60尚未翻正。趋势刚启动，空间最充裕但确认度最低。",
            "  趋势持续: d3>0+d5>0+d60>0且d20<40%。趋势已确认，在未透支的前提下仍有空间。",
            "  上涨后期: d20>40%或d60>80%且acc<0。中期涨幅已充分，加速度衰减，继续上行需要更强的催化剂。",
            "  d3<0且d5<0: 短期势头偏弱，2-4周持仓下需要更有力的趋势或基本面信号来平衡。",
            "  HIGH红旗: 基本面的明确风险点，纳入时请说明为什么趋势或估值可以弥补这些风险。",
            "",
            "  量价参考:",
            "  放量+资金流入=资金认可趋势。缩量上涨=控盘或买盘衰竭，结合位置。",
            "  缩量回调+放量反弹=抛压释放后的回流，通常是最佳加仓点。放量滞涨=警惕派发。",
            "  量价配合好的趋势更可靠，量价背离的趋势即使方向对也要考虑降权。",
        ]

    lines = role_lines + [
        "",
        f"=== 决策指引 ({strategy}) ===",
        f"候选池按确信度排序, 分批处理。每批次独立评估, 选出该批次中最值得纳入的标的。",
        "",
    ] + rejection_rules + [
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
        "",
        "=== 上下文 ===",
        f"当前宏观: {regime_str} | 策略: {strategy}",
        f"宏观判断: {regime_summary}",
        f"候选池: {len(candidates)}只, 按确信度排序, 分批独立评估 (每批按自身质量选取)",
        f"权重范围: {cfg['min_single_weight']:.0%}-{cfg['max_single_weight']:.0%}",
        f"A5趋势分析: {fusion_narrative}",
        f"",
        f"=== 板块资金偏好（来自宏观分析） ===",
        f"{sector_view if sector_view else '板块数据暂不可用'}",
        "",
        f"{'资金流向是短期交易的核心依据，应作为重要决策因子：' if strategy == 'hot_picks' else '强势板块有资金支撑，弱势板块需更强个股逻辑才能纳入。'}"
        f"{'\\n- 资金流入板块 → 降低基本面门槛，顺势而为\\n- 资金流出板块 → 即使技术面尚可，警惕板块拖累，提高纳入门槛\\n- 板块无明确方向 → 依靠个股自身动能' if strategy == 'hot_picks' else ''}",
        f"",
        f"=== 近期市场动态（来自新闻，弱参考，勿过度依赖）===",
        f"{market_news_str}",
        "",
        "注：以上新闻来自A3实时抓取。其中「→ 行业关键词」部分由LLM从新闻内容中提取，",
        "表示该新闻可能影响这些行业。请自行判断候选股与这些行业动态的关联——",
        "行业关键词与候选股的行业分类可能不完全一致，以你的独立判断为准。",
        "新闻信息权重适中，作为理解市场热点和风险方向的背景参考，不作为主要决策依据。",
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
            "输出JSON (weight必须是数字不是字符串, rationale/risk_boundary均不超过180字):",
            "{",
            '  "a7_decisions": [',
            '    {"ts_code":"XXXX01.SZ","a7_recommendation":"INCLUDE","weight":0.15,  // weight是数字,不要加引号',
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
    n_candidates = len(candidates)
    n_batches = max(1, (n_candidates + A7_BATCH_SIZE - 1) // A7_BATCH_SIZE)
    # Equidistant sampling: each batch gets a representative slice across
    # the conviction spectrum, avoiding batch-level quality bias.
    batches = []
    for bi in range(n_batches):
        batch = []
        for j in range(bi, n_candidates, n_batches):
            batch.append(candidates[j])
            if len(batch) >= A7_BATCH_SIZE:
                break
        batches.append(batch)
    llm_cfg = settings.get_llm_config("A7")

    # Wait for API rate limit cooldown (A5 just finished its LLM call)
    logger.info(f"A7 LLM: cooling down 3s, then {len(batches)} batches of ≤{A7_BATCH_SIZE}...")
    time.sleep(3)

    for batch_idx, batch in enumerate(batches):
        batch_lines = list(lines)  # copy header

        # Candidate display for this batch — equidistant across conviction spectrum
        batch_lines.append(f"=== 候选池 批次{batch_idx+1}/{len(batches)} ({len(batch)}只) ===")
        for i, c in enumerate(batch):
            code = c["ts_code"]; a2 = a2_data.get(code, {}); ind = code_industry.get(code, "?")
            tier = get_conviction_tier(c.get("conviction", 0), strategy)
            batch_lines.append(
                f"{i+1:2d}. {code} [{c.get('conviction',0):.2f}/{tier['label']}] "
                f"核心:{c.get('fl_core_label','?')}={c.get('fl_core_score',0):.0f} | "
                f"d3:{c.get('momentum_d3',0):+.1f}% d5:{c.get('momentum_d5',0):+.1f}% "
                f"d20:{c.get('momentum_d20',0):+.1f}% d60:{c.get('momentum_d60',0):+.1f}% acc:{c.get('momentum_accel',0):+.1f}"
            )
            if a2:
                fs_a2 = a2.get('fundamental_score')
                fs_rationale = a2.get('score_rationale', '') if isinstance(a2.get('score_rationale'), str) else ''
                a2_narrative = a2.get('narrative', '') if isinstance(a2.get('narrative'), str) else ''
                narrative_preview = a2_narrative[:120].replace('\n', ' ') if a2_narrative else ''
                fs_str = f"基本面={fs_a2:.0f}" if fs_a2 is not None else "基本面=?"
                rfs_list = a2.get('rfs', [])
                rfs_str = ', '.join(rfs_list) if rfs_list else '无'
                batch_lines.append(
                    f"    A2: {fs_str} (置信{a2.get('conf',0):.0%}) "
                    f"盈{a2.get('eq','?')} 财{a2.get('fh','?')} 估{a2.get('val','?')} "
                    f"| 红旗:{rfs_str}"
                )
                if fs_rationale:
                    batch_lines.append(f"    A2评分理由: {fs_rationale}")
                if narrative_preview and narrative_preview != fs_rationale:
                    batch_lines.append(f"    A2分析: {narrative_preview}")
            else:
                batch_lines.append(f"    A2: 缺失 (无LLM基本面分析)")
            vol_seq = list(reversed(c.get('vol_5d', [])))   # oldest → newest
            close_seq = list(reversed(c.get('close_5d', [])))
            vol_str = " → ".join(f"{v:.1f}亿" for v in vol_seq) if vol_seq else "无数据"
            close_str = " → ".join(f"{p:.2f}" for p in close_seq) if close_seq else ""
            vol_trend = ""
            if len(vol_seq) >= 2:
                if vol_seq[-1] > vol_seq[0] * 1.3: vol_trend = ">放量"
                elif vol_seq[0] > vol_seq[-1] * 1.3: vol_trend = "<缩量"
                else: vol_trend = "→持平"
            batch_lines.append(
                f"    MACD:{c.get('macd','?')} RSI:{c.get('rsi','?')} MA:{c.get('ma','?')} "
                f"现价:{c.get('price',0):.2f}"
            )
            batch_lines.append(f"    量价: {c.get('vol_analysis','?')} | 价序:{close_str}")

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
    sorted_items = sorted(portfolio.items(), key=lambda x: -float(x[1])) if isinstance(portfolio, dict) else sorted(
        [(p["ts_code"], float(p["weight"])) for p in (portfolio if isinstance(portfolio, list) else [])],
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

        # ── Load A1 indicators (volume info, MACD, RSI) ──
        a1_indicators = {}
        for r in conn.execute(
            "SELECT ts_code, indicators_json FROM indicators "
            "WHERE calc_date = (SELECT MAX(calc_date) FROM indicators)"
        ).fetchall():
            try:
                ind = json.loads(r["indicators_json"])
                a1_indicators[r["ts_code"]] = {
                    "vol_ratio": ind.get("volume_ratio", "?"),
                    "obv": ind.get("obv_trend", "?"),
                    "macd": ind.get("macd", {}).get("signal", "?"),
                    "rsi": round(ind.get("rsi_14", 50)),
                    "ma": ind.get("ma_alignment", "?"),
                }
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Load 5-day volume + close price series per stock ──
        vol_price_5d = {}
        for r in conn.execute(
            "SELECT ts_code, trade_date, amount, close FROM daily_quotes "
            "WHERE trade_date >= date('now', '-10 days') "
            "ORDER BY ts_code, trade_date DESC"
        ).fetchall():
            code = r["ts_code"]
            if code not in vol_price_5d:
                vol_price_5d[code] = {"amounts": [], "closes": []}
            if len(vol_price_5d[code]["amounts"]) < 5:
                vol_price_5d[code]["amounts"].append(round(r["amount"] / 1e8, 1))
                vol_price_5d[code]["closes"].append(round(r["close"], 2))

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

        # ── Load focus list ──
        from backend.focus_list import get_focus_codes
        focus = get_focus_codes(conn, strategy=strategy)
        focus_codes = {f["ts_code"] for f in focus}
        fl_scores = {f["ts_code"]: {"value": f.get("value_score"), "momentum": f.get("momentum_score")} for f in focus}
        fl_positions = {f["ts_code"]: f.get("position", 999) for f in focus}
        held_codes = {h["ts_code"] for h in holdings}

        # Industry map
        code_industry = {}
        for r in conn.execute(
            f"SELECT ts_code, industry FROM stocks WHERE ts_code IN "
            f"({','.join('?'*len(focus_codes))})",
            list(focus_codes),
        ).fetchall():
            code_industry[r["ts_code"]] = r["industry"] or "其他"

        # ── Volume analysis helper ──
        def _vol_analysis(amounts, closes):
            """Return detailed volume-price context for LLM reference (not a score)."""
            if not amounts or len(amounts) < 3:
                return "量价数据不足"
            n = len(amounts)
            vol_avg = sum(amounts) / n
            # Volume trajectory
            if len(amounts) >= 3:
                half = n // 2
                recent_vol = sum(amounts[-half:]) / half
                older_vol = sum(amounts[:half]) / half if half > 0 else vol_avg
                if older_vol > 0 and recent_vol > older_vol * 1.2:
                    vol_traj = "递增"
                elif older_vol > 0 and recent_vol < older_vol * 0.8:
                    vol_traj = "递减"
                else:
                    vol_traj = "平稳"
            else:
                vol_traj = "?"
            # Volume level vs latest
            if amounts[-1] > vol_avg * 1.3:
                vol_level = "放量"
            elif amounts[-1] < vol_avg * 0.7:
                vol_level = "缩量"
            else:
                vol_level = "量平"
            # Buy/sell pressure
            up_vols, dn_vols = 0, 0
            for i in range(1, n):
                if closes[i] > closes[i-1]: up_vols += amounts[i]
                elif closes[i] < closes[i-1]: dn_vols += amounts[i]
            pressure = "?"
            if up_vols + dn_vols > 0:
                ratio = up_vols / (up_vols + dn_vols) if (up_vols + dn_vols) > 0 else 0.5
                if ratio > 0.6: pressure = "资金流入"
                elif ratio < 0.4: pressure = "资金流出"
                else: pressure = "资金均衡"
            # Price summary
            if closes and closes[0] and closes[0] > 0 and closes[-1] and closes[-1] > 0:
                p_chg = (closes[-1] / closes[0] - 1) * 100
                high = max(closes)
                low = min(closes)
                p_range = (high / low - 1) * 100 if low > 0 else 0
                p_str = f"价{p_chg:+.1f}% 波动{p_range:.1f}%"
            else:
                p_str = "?"
            return f"{vol_traj}→{vol_level} {pressure} | {p_str}"

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
            fs = fl_scores.get(code, {})
            # Use FL's multi-factor total_score for conviction — aligned with FL ranking
            fl_score = sc.get("total_score", fs.get("value") if strategy == "long_term" else fs.get("momentum"))
            conv = compute_conviction(code, fl_score, a2_reports.get(code), strategy=strategy)
            tier = get_conviction_tier(conv, strategy)
            if tier["label"].startswith("低确信"):
                continue
            focus_convictions.append((code, conv))

        # Sort by conviction, take all FL candidates (FL already narrowed to 200)
        focus_convictions.sort(key=lambda x: -x[1])
        sample_pct = 1.0
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

            fs = fl_scores.get(code, {})
            # FL综合分 is the multi-factor total_score from composite_scores,
            # now computed as weighted combination of primary driver + supporting factors.
            core_label = "FL综合分"
            core_score = sc.get("total_score", 0)  # multi-factor composite from FL engine
            vol_analysis = _vol_analysis(
                list(reversed(vol_price_5d.get(code, {}).get("amounts", []))),
                list(reversed(vol_price_5d.get(code, {}).get("closes", []))),
            ) if vol_price_5d.get(code) else "量价数据不足"
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
                "vol_ratio": a1_indicators.get(code, {}).get("vol_ratio", "?"),
                "obv": a1_indicators.get(code, {}).get("obv", "?"),
                "macd": a1_indicators.get(code, {}).get("macd", "?"),
                "rsi": a1_indicators.get(code, {}).get("rsi", "?"),
                "ma": a1_indicators.get(code, {}).get("ma", "?"),
                "vol_5d": vol_price_5d.get(code, {}).get("amounts", []),
                "close_5d": vol_price_5d.get(code, {}).get("closes", []),
                "vol_analysis": vol_analysis,
                "fl_score": f"{core_label}={core_score:.0f}",
                "fl_core_score": core_score,
                "fl_core_label": core_label,
                "suggested_shares": shares,
                "industry": code_industry.get(code, "其他"),
            }
            candidates.append(cand)
        logger.info(f"A7 candidates: {len(candidates)} total")

        # ── Sell/Hold decisions for current holdings ──
        holdings_decisions = []
        fl_total = len(focus_codes)
        for h in holdings:
            code = h["ts_code"]
            fl_pos = fl_positions.get(code, 999)  # FL position (1-based, lower = better)
            fl_score = (fl_scores.get(code, {}).get("value") if strategy == "long_term"
                        else fl_scores.get(code, {}).get("momentum"))
            pnl = h.get("pnl_pct", 0)
            days = h.get("hold_days", 0)

            if strategy == "hot_picks":
                # Hot picks: aggressive — momentum decay or FL drop is sell signal
                not_in_fl = fl_pos == 999
                bottom_40 = fl_pos > fl_total * 0.6
                if not_in_fl:
                    action, reason = "SELL", f"不在FL→清仓"
                elif bottom_40 and (pnl < -5 or days > 5):
                    action, reason = "SELL", f"FL#{fl_pos}(后40%)+动量衰减→清仓"
                elif days > 5 and pnl < 0:
                    action, reason = "REDUCE", f"持有{days}日+亏损{pnl:.0f}%→时间止损"
                elif bottom_40:
                    action, reason = "REDUCE", f"FL#{fl_pos}(后40%)→减仓"
                else:
                    action, reason = "HOLD", f"FL#{fl_pos} 继续跟踪"
            else:
                # Long term: conservative — only sell on significant FL drop
                not_in_fl = fl_pos == 999
                bottom_50 = fl_pos > fl_total * 0.5
                if not_in_fl:
                    if pnl < -10:
                        action, reason = "SELL", f"不在FL+亏损{pnl:.0f}%→止损"
                    elif pnl > 15:
                        action, reason = "REDUCE", f"不在FL+盈利{pnl:.0f}%→减仓锁定"
                    else:
                        action, reason = "REDUCE", f"不在FL→减仓观望"
                elif bottom_50:
                    if pnl < -10:
                        action, reason = "SELL", f"FL#{fl_pos}(后50%)+亏损{pnl:.0f}%→止损"
                    elif pnl > 15:
                        action, reason = "REDUCE", f"FL#{fl_pos}(后50%)+盈利{pnl:.0f}%→减仓锁定"
                    else:
                        action, reason = "REDUCE", f"FL#{fl_pos}(后50%)→减仓观望"
                elif fl_pos > fl_total * 0.3 and pnl < 0:
                    action, reason = "REDUCE", f"FL#{fl_pos}中等+微亏→减仓"
                else:
                    action, reason = "HOLD", f"FL#{fl_pos} 继续持有"

            d = {"ts_code": code, "action": action, "reason": reason,
                 "rank": fl_pos, "pnl_pct": round(pnl, 2), "hold_days": days,
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
            # Normalize LLM output types (LLM may return weight as string)
            def _norm(item):
                w = item.get("weight", 0)
                return {
                    "a7_recommendation": item.get("a7_recommendation", "REJECT"),
                    "weight": float(w) if w else 0.0,
                    "rationale": item.get("rationale", ""),
                }

            # Preferred: unified a7_decisions array
            raw_decisions = llm_result.get("a7_decisions", [])
            if not raw_decisions:
                # Fallback: old portfolio+rejected format
                for item in llm_result.get("portfolio", []):
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = _norm(item)
                for item in llm_result.get("rejected", []):
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = {
                            "a7_recommendation": "REJECT",
                            "weight": 0.0,
                            "rationale": item.get("reason", ""),
                        }
            else:
                for item in raw_decisions:
                    if isinstance(item, dict):
                        a7_decisions[item.get("ts_code", "")] = _norm(item)

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
