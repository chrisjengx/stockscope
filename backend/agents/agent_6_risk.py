"""
Agent 6: Risk Officer — adversarial review of A7 portfolio recommendations.

Architecture:
  Layer 1 (no LLM): rule_based_checks() — liquidity, concentration, holdings limit
  Layer 2 (pro LLM): score_decisions() — reviews upstream agent reports for logical flaws
  Layer 3 (no LLM): _apply_adversarial_verdicts() — ranking-based VETO/APPROVE

Verdict: risk_score 1-5 → sort → top 60% APPROVED (floor 5, ceiling 15) + risk>=4 forced VETO.
Uses llm.reason() with deepseek-v4-pro for strict adversarial review.
Terminal stage of pipeline — no downstream agent.
"""
import json, time, logging
from datetime import datetime
from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm, extract_json
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()



def rule_based_checks(decision, holdings, conn, a5_scores=None, fund_reports=None):
    """Programmatic checks. Returns list of {dim, severity, detail}.

    Extended (v2): valuation extreme, momentum exhaustion, signal conflict.
    """
    code = decision["ts_code"]
    checks = []
    # Liquidity
    avg = conn.execute("SELECT AVG(amount) as a FROM daily_quotes WHERE ts_code=? AND trade_date >= date('now','-30 days')",
                       (code,)).fetchone()
    turnover = avg["a"] if avg else 0
    checks.append({"dim": "流动性", "severity": "RED" if (turnover or 0) < 50_000_000 else "GREEN",
                   "detail": f"30日均成交额 {(turnover or 0)/1e8:.1f}亿"})
    # Industry concentration
    ind = conn.execute("SELECT industry FROM stocks WHERE ts_code=?", (code,)).fetchone()
    ind = ind["industry"] if ind else "未知"
    same = [h for h in holdings if h.get("industry") == ind]
    checks.append({"dim": "行业集中", "severity": "AMBER" if len(same) >= 2 else "GREEN",
                   "detail": f"同行业已持{len(same)}只"})
    # Holdings limit
    checks.append({"dim": "持仓上限", "severity": "RED" if len(holdings) >= 8 and decision["action"] == "BUY" else "GREEN",
                   "detail": f"当前{len(holdings)}/8"})
    if decision.get("weight_change", 0) > 0.1:
        checks.append({"dim": "仓位变化", "severity": "AMBER", "detail": f"单次{decision.get('weight_change',0):.0%}>10%"})
    if decision.get("rapid_sell_flag"):
        checks.append({"dim": "短期卖出", "severity": "AMBER", "detail": f"持有{decision.get('hold_days','?')}天，需排除处置效应"})

    # ── Valuation extreme: PE>200 and revenue not growing ──
    row = conn.execute(
        "SELECT pe_ttm, revenue_yoy FROM financials WHERE ts_code=? "
        "ORDER BY report_date DESC LIMIT 1", (code,)
    ).fetchone()
    if row and (row["pe_ttm"] or 0) > 200 and (row["revenue_yoy"] or 0) <= 0:
        checks.append({"dim": "估值极端", "severity": "RED",
                       "detail": f"PE={row['pe_ttm']:.0f} + 收入无增长, 纯炒作风险"})

    # ── Momentum exhaustion: d5 sharply negative ──
    sc = (a5_scores or {}).get(code, {})
    d5 = sc.get("momentum_d5", 0)
    if d5 < -5:
        checks.append({"dim": "动量衰竭", "severity": "AMBER",
                       "detail": f"5日动量{d5:.1f}%, 短期趋势显著走弱"})

    # ── Signal conflict: strong technical but many fundamental red flags ──
    tech = sc.get("tech_score", 50)
    fr_r = (fund_reports or {}).get(code, {})
    rfs = len(fr_r.get("red_flags", [])) if isinstance(fr_r, dict) else 0
    if tech > 70 and rfs >= 3:
        checks.append({"dim": "信号冲突", "severity": "AMBER",
                       "detail": f"技术面强(tech={tech:.0f})但基本面红旗{rfs}个, 需交叉验证"})

    return checks


def score_decisions(decisions, holdings, macro, conn, strategy="long_term"):
    """LLM: review A7's portfolio recommendations by examining UPSTREAM AGENT REPORTS.

    KEY ARCHITECTURAL CHANGE (v3.0): A6 no longer re-queries raw data tables.
    Instead, it reads the structured analysis REPORTS produced by A1, A2, A3, A4, A5.
    A6's job is to find logical flaws in upstream analysis, not to redo it.
    """
    if not decisions or not settings.ds_api_key:
        return None

    # Load upstream data for ALL decisions once (before batching)
    codes = [d["ts_code"] for d in decisions]
    ph = ",".join("?" * len(codes))

    # A1 Technical data: read from indicators table (A1's native output)
    tech_reports = {}
    for r in conn.execute(
        f"SELECT ts_code, indicators_json FROM indicators "
        f"WHERE ts_code IN ({ph}) AND calc_date = (SELECT MAX(calc_date) FROM indicators)",
        codes,
    ).fetchall():
        try:
            ind = json.loads(r["indicators_json"])
            macd = ind.get("macd", {})
            tech_reports[r["ts_code"]] = {
                "overall_assessment": (
                    f"MACD:{macd.get('signal','?')} RSI:{ind.get('rsi_14','?')} "
                    f"MA:{ind.get('ma_alignment','?')} BB_pos:{ind.get('bollinger',{}).get('position','?')} "
                    f"OBV:{ind.get('obv_trend','?')} vol_ratio:{ind.get('volume_ratio','?')}"
                ),
                "confidence": 0.6,
                "_source": "indicators",
            }
        except (json.JSONDecodeError, TypeError):
            tech_reports[r["ts_code"]] = {"overall_assessment": "无", "confidence": 0}
    # Any missing codes
    for c in codes:
        if c not in tech_reports:
            tech_reports[c] = {"overall_assessment": "无", "confidence": 0}

    # A2 Fundamental Reports (earnings_quality, red_flags, growth_quality, etc.)
    fund_reports = {}
    for r in conn.execute(
        f"SELECT ts_code, report_json, overall_score FROM fundamental_reports "
        f"WHERE ts_code IN ({ph}) AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)",
        codes,
    ).fetchall():
        try:
            fund_reports[r["ts_code"]] = json.loads(r["report_json"])
        except (json.JSONDecodeError, TypeError):
            fund_reports[r["ts_code"]] = {"narrative": "无", "confidence": 0, "red_flags": []}

    # A3 News Data (now with validated related_stocks_json)
    news = {}
    for r in conn.execute(
        "SELECT related_stocks_json, sentiment, impact, summary FROM news_feed "
        "WHERE published_at >= datetime('now','-7 days') AND impact IN ('HIGH','MEDIUM')",
    ).fetchall():
        entities = []
        try:
            entities = json.loads(r["related_stocks_json"] or "[]")
        except json.JSONDecodeError:
            pass
        for e in entities:
            c = e.get("code", "")
            if c in codes:
                news[c] = news.get(c, []) + [f"{r['sentiment']}({r['impact']}): {r['summary'][:60]}"]

    # A4 Macro Report
    macro_report = {}
    mr = conn.execute("SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1").fetchone()
    if mr:
        try:
            macro_report = json.loads(mr["report_json"])
        except json.JSONDecodeError:
            pass

    # A5 Fusion Report (signal conflicts, top stocks)
    fusion_report = {}
    fr = conn.execute(
        "SELECT report_json FROM fusion_reports WHERE strategy=? ORDER BY calc_date DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    if fr:
        try:
            fusion_report = json.loads(fr["report_json"])
        except json.JSONDecodeError:
            pass

    # ── A5 composite scores (factor breakdown per stock) ──
    a5_scores = {}
    for r in conn.execute(
        f"SELECT ts_code, tech_score, fundamental_score, macro_fit, momentum, total_score, rank "
        f"FROM composite_scores WHERE ts_code IN ({ph}) AND strategy=? "
        f"AND calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)",
        codes + [strategy, strategy],
    ).fetchall():
        a5_scores[r["ts_code"]] = dict(r)

    # ── Industry per stock ──
    code_industry = {}
    for r in conn.execute(
        f"SELECT ts_code, industry FROM stocks WHERE ts_code IN ({ph})", codes
    ).fetchall():
        code_industry[r["ts_code"]] = r["industry"] or "其他"

    # ── A7 recommendations (from portfolio_decisions.review_json) ──
    a7_tags = {}
    a7_include_count = 0
    for d in decisions:
        try:
            rv = json.loads(d.get("review_json") or "{}")
            a7 = rv.get("a7", {})
            if a7:
                a7_tags[d["ts_code"]] = a7
                if a7.get("recommendation") == "INCLUDE":
                    a7_include_count += 1
        except (json.JSONDecodeError, TypeError):
            pass

    # ── A7 portfolio-level recommendation ──
    a7_portfolio = {}
    pr = conn.execute(
        "SELECT report_json FROM portfolio_reports WHERE strategy=? ORDER BY calc_date DESC LIMIT 1",
        (strategy,),
    ).fetchone()
    if pr:
        try:
            a7_portfolio = json.loads(pr["report_json"]).get("a7", {})
        except (json.JSONDecodeError, TypeError):
            pass

    # ── Data coverage stats ──
    a1_covered = sum(1 for c in codes if tech_reports.get(c, {}).get("confidence", 0) > 0)
    a2_covered = sum(1 for c in codes if fund_reports.get(c, {}).get("confidence", 0) > 0)
    a3_covered = sum(1 for c in codes if c in news)

    # ── A5 factor attribution + ranking concerns for A6 to reference ──
    fa_list = fusion_report.get("factor_attribution", []) if fusion_report else []
    rc_list = fusion_report.get("ranking_concerns", []) if fusion_report else []
    fa_str = "; ".join(
        f"{f.get('ts_code','?')}:{f.get('primary_driver','?')}({f.get('fragility','?')})"
        for f in fa_list[:10]
    ) if fa_list else "无"
    rc_str = "; ".join(
        f"{r.get('ts_code','?')}:{r.get('issue','?')[:60]}"
        for r in rc_list[:5]
    ) if rc_list else "无"

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

    # ── Build prompt ──
    lines = [
        f"宏观: {macro.get('regime','?')} | 持仓: {len(holdings)}只",
        f"数据覆盖: A1={a1_covered}/{len(decisions)} A2={a2_covered}/{len(decisions)} A3={a3_covered}/{len(decisions)}",
        f"",
        f"=== 市场背景新闻（近3天参考，非指令，权重适中）===",
        f"{market_news_str}",
        f"",
        f"A5因子归因: {fa_str}",
        f"A5排名质疑: {rc_str}",
        f"",
        "=== 待审股票 ===",
        "",
    ]

    # ── Compute hints for batch prompts ──
    if a7_include_count < len(decisions) * 0.2:
        a7_hint = f"A7仅推荐{a7_include_count}/{len(decisions)}只纳入(高度保守)。"
    else:
        a7_hint = f"A7推荐{a7_include_count}只纳入。"

    if a2_covered < len(decisions) * 0.3:
        budget_hint = f"A2仅覆盖{a2_covered}/{len(decisions)}只, 侧重技术面+宏观维度。"
    else:
        budget_hint = "A2数据覆盖充分。"

    # ── Pre-compute rule-based risk flags per stock (inject as hard context) ──
    rule_flags = {}
    for d in decisions:
        code = d["ts_code"]
        checks = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
        reds = [c["detail"] for c in checks if c["severity"] == "RED"]
        ambers = [c["detail"] for c in checks if c["severity"] == "AMBER"]
        rule_flags[code] = (reds, ambers)

    # ── Per-batch LLM call ──
    A6_BATCH_SIZE = 15  # matches A7, pro model
    all_parsed = []
    batch_decisions = [decisions[i:i + A6_BATCH_SIZE] for i in range(0, len(decisions), A6_BATCH_SIZE)]
    for batch_idx, batch in enumerate(batch_decisions):
        batch_codes = [d["ts_code"] for d in batch]

        # Build per-batch prompt lines (only the decision display section differs)
        batch_lines = list(lines)  # copy header
        for d in batch:
            code = d["ts_code"]
            tr = tech_reports.get(code, {})
            fr_r = fund_reports.get(code, {})
            n = news.get(code, [])
            sc = a5_scores.get(code, {})
            ind = code_industry.get(code, "?")
            has_a2 = fr_r.get("confidence", 0) > 0
            a7 = a7_tags.get(code, {})
            a7_rec = a7.get("recommendation", "?")
            a7_tag = f"[A7:{a7_rec}]"
            driver_tag = a7.get("driver_tag", "") or "平衡"
            data_tag = "[A2✓]" if has_a2 else "[仅技术面]"
            rank = sc.get("rank", "?"); total = sc.get("total_score", "?")
            t = sc.get("tech_score", 50); f = sc.get("fundamental_score", 50); m = sc.get("momentum", 50)

            batch_lines.append(
                f"{code} {a7_tag} [{driver_tag}] {data_tag} A5#{rank}/{total:.0f} "
                f"[T:{t:.0f} F:{f:.0f} M:{m:.0f}] {ind}"
            )
            batch_lines.append(f"  A1: {tr.get('overall_assessment','无')[:130]}")
            if has_a2:
                rfs = [f"{rf.get('severity','?')}:{rf.get('flag','?')}" if isinstance(rf, dict) else str(rf) for rf in fr_r.get('red_flags', [])]
                batch_lines.append(f"  A2: {fr_r.get('narrative','无')[:180]} | 红旗:{rfs if rfs else '无'} | 置信度:{fr_r.get('confidence',0):.0%}")
            else:
                batch_lines.append(f"  A2: 缺失")
            a7_weight = a7.get("weight", 0)
            a7_weight_str = f" 权重{a7_weight:.0%}" if a7_weight > 0 else ""
            batch_lines.append(f"  A7: {a7_rec}{a7_weight_str} (确信度{a7.get('conviction','?')}) — {a7.get('rationale','无')[:120]}")
            if n:
                batch_lines.append(f"  新闻: {'; '.join(n[:2])}")
            # Hard risk flags from rule-based checks — LLM must address these
            reds, ambers = rule_flags.get(code, ([], []))
            if reds or ambers:
                flags = []
                if reds: flags.append(f"RED: {'; '.join(reds)}")
                if ambers: flags.append(f"AMBER: {'; '.join(ambers)}")
                batch_lines.append(f"  ⚠ 规则检测: {' | '.join(flags)}")

        # A4 systemic risk + sector context
        risk_alerts = macro_report.get("risk_alerts", [])
        if risk_alerts:
            batch_lines.append(f"\nA4系统性风险: {'; '.join(risk_alerts[:5])}")
        sector_view = macro_report.get("sector_view", "")
        if sector_view:
            batch_lines.append(f"A4板块强弱: {sector_view[:200]}")
        batch_lines.append(f"A4宏观: {macro_report.get('narrative','无')[:150]}")
        batch_lines.append(f"A5融合: {fusion_report.get('overall_narrative','无')[:200]}")
        if a7_portfolio:
            batch_lines.append(
                f"A7组合: 纳入{a7_portfolio.get('include_count','?')}只 | "
                f"{a7_portfolio.get('strategic_cash',0):.0%}现金 | "
                f"{a7_portfolio.get('portfolio_narrative','')[:200]}"
            )

        # Strategy-specific scoring rubric
        if strategy == "hot_picks":
            score_rubric = [
                "风险评分(1-5) — hot_picks动量标的, 波动大, 评分应从严:",
                "  1=数据齐全+趋势加速+无红旗+无规则警告 → 极罕见, 需明确举证",
                "  2=趋势健康, 个别维度有疑虑但动量可合理解释 → 需说明理由",
                "  3=存在明确风险信号(规则AMBER/RED/基本面恶化/动量透支) → 默认给3",
                "  4=多维度风险叠加或规则RED → 建议VETO",
                "  5=致命风险(低流动性/基本面崩溃/动量逆转) → 强制VETO",
            ]
        else:
            score_rubric = [
                "风险评分(1-5):",
                "  1=无可见风险信号,数据一致,趋势健康,各维度无隐患",
                "  2=A7判断合理,个别维度有轻微疑虑但不影响大局",
                "  3=存在需要关注的风险(单一维度明显偏弱/红旗未完全消化/行业逆风)",
                "  4=显著风险(多维度偏弱/逻辑矛盾/基本面恶化)",
                "  5=致命风险(低流动性/极高估值无增长/多HIGH红旗/动量崩溃)",
            ]

        # Role + output format footer
        batch_lines += [
            "", a7_hint, budget_hint, "",
            "=== 你的角色 ===",
            "你是对抗性风险审查官。你的任务不是确认A7的判断，而是找出被忽略或低估的风险。",
            "如果一只股票有规则警告(⚠标注)但A7仍然推荐BUY，你必须解释为什么这些警告不影响风险判断。",
            "默认心态: 倾向给更高风险分——不确定时向上取整(3而非2, 4而非3)。",
            "",
            *score_rubric,
            "",
            "必须为每只待审股输出一条review——万一遗漏将使用规则兜底审查:",
            "输出JSON:",
            '{"reviews": [{"ts_code":"...","risk_score":1-5,',
            '"recommendation":"PREFER/CAUTION/AVOID",',
            '"reasoning":"具体风险分析,必须回应⚠标记",',
            '"risk_dimensions":["..."],"conditions":["..."],"confidence":0.0-1.0}],',
            '"cash_assessment":"...",',
            '"portfolio_notes":"..."}',
            "",
            "recommendation: PREFER=风险可控优先关注, CAUTION=有疑虑建议小仓, AVOID=风险显著建议回避",
        ]

        cfg = settings.get_llm_config("A6")
        result = llm.reason("\n".join(batch_lines), model=cfg["model"],
                            max_tokens=cfg["max_tokens"])
        if not result:
            logger.error(f"A6 batch {batch_idx+1}/{len(batch_decisions)} LLM返回None, 规则兜底本批")
            for d in batch:
                rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
                red_count = sum(1 for c in rule if c["severity"] == "RED")
                amber_count = sum(1 for c in rule if c["severity"] == "AMBER")
                risk = min(5, 1 + red_count * 2 + amber_count)
                all_parsed.append({
                    "ts_code": d["ts_code"], "risk_score": risk,
                    "recommendation": "CAUTION" if risk >= 3 else "PREFER",
                    "reasoning": f"规则兜底(LLM不可用): RED={red_count} AMBER={amber_count}",
                    "risk_dimensions": [c["dim"] for c in rule if c["severity"] != "GREEN"],
                    "conditions": [], "confidence": 0.3,
                })
            continue

        parsed = extract_json(result)
        if not parsed:
            logger.error(f"A6 batch {batch_idx+1}/{len(batch_decisions)} JSON解析失败, 规则兜底本批")
            for d in batch:
                rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
                red_count = sum(1 for c in rule if c["severity"] == "RED")
                amber_count = sum(1 for c in rule if c["severity"] == "AMBER")
                risk = min(5, 1 + red_count * 2 + amber_count)
                all_parsed.append({
                    "ts_code": d["ts_code"], "risk_score": risk,
                    "recommendation": "CAUTION" if risk >= 3 else "PREFER",
                    "reasoning": f"规则兜底(JSON解析失败): RED={red_count} AMBER={amber_count}",
                    "risk_dimensions": [c["dim"] for c in rule if c["severity"] != "GREEN"],
                    "conditions": [], "confidence": 0.3,
                })
            continue

        cash_assessment = ""; portfolio_notes = ""
        if isinstance(parsed, dict):
            cash_assessment = str(parsed.get("cash_assessment", ""))
            portfolio_notes = str(parsed.get("portfolio_notes", ""))
            # Explicitly pick 'reviews' — avoid picking wrong list if LLM changes key order
            if "reviews" in parsed and isinstance(parsed["reviews"], list):
                parsed = parsed["reviews"]
            else:
                for key in parsed:
                    if key != "risk_dimensions" and isinstance(parsed[key], list):
                        parsed = parsed[key]; break
        if not isinstance(parsed, list):
            logger.error(f"A6 batch {batch_idx+1} 输出格式错误, 规则兜底本批")
            for d in batch:
                rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
                red_count = sum(1 for c in rule if c["severity"] == "RED")
                amber_count = sum(1 for c in rule if c["severity"] == "AMBER")
                risk = min(5, 1 + red_count * 2 + amber_count)
                all_parsed.append({
                    "ts_code": d["ts_code"], "risk_score": risk,
                    "recommendation": "CAUTION" if risk >= 3 else "PREFER",
                    "reasoning": f"规则兜底(格式错误): RED={red_count} AMBER={amber_count}",
                    "risk_dimensions": [c["dim"] for c in rule if c["severity"] != "GREEN"],
                    "conditions": [], "confidence": 0.3,
                })
            continue

        input_codes = {d["ts_code"] for d in batch}
        output_codes = {item.get("ts_code", "") for item in parsed}
        if input_codes != output_codes:
            logger.error(f"A6 batch {batch_idx+1} code mismatch: missing={input_codes-output_codes} extra={output_codes-input_codes}, 规则兜底本批")
            for d in batch:
                rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
                red_count = sum(1 for c in rule if c["severity"] == "RED")
                amber_count = sum(1 for c in rule if c["severity"] == "AMBER")
                risk = min(5, 1 + red_count * 2 + amber_count)
                all_parsed.append({
                    "ts_code": d["ts_code"], "risk_score": risk,
                    "recommendation": "CAUTION" if risk >= 3 else "PREFER",
                    "reasoning": f"规则兜底(代码不匹配): RED={red_count} AMBER={amber_count}",
                    "risk_dimensions": [c["dim"] for c in rule if c["severity"] != "GREEN"],
                    "conditions": [], "confidence": 0.3,
                })
            continue

        for item in parsed:
            s = item.get("risk_score", 0)
            if not isinstance(s, (int, float)) or s < 1 or s > 5:
                item["risk_score"] = 3  # default to medium risk
                logger.warning(f"A6: invalid risk_score={s} for {item.get('ts_code','?')}, defaulted to 3")
            if not item.get("reasoning") or len(str(item.get("reasoning", "")).strip()) < 5:
                item["reasoning"] = item.get("reasoning", "") + " (分析不完整)"

        if parsed and cash_assessment: parsed[0]["cash_assessment"] = cash_assessment
        if parsed and portfolio_notes: parsed[0]["portfolio_notes"] = portfolio_notes

        all_parsed.extend(parsed)
        logger.info(f"A6 batch {batch_idx+1}/{len(batch_decisions)} OK: {len(parsed)}/{len(batch)} reviews ({len(parsed)/len(batch):.0%} coverage), {len(result)} chars")

    return all_parsed


def _apply_adversarial_verdicts(decisions, scores):
    """Layer 3: ranking-based adversarial elimination.

    - BUY: sort by risk_score → top 60% APPROVED (floor 3, ceiling 12)
      - risk_score >= 4 → forced VETOED regardless of rank
    - SELL/HOLD/REDUCE: all APPROVED (don't block risk management)
    """
    buys = [(d, s) for d, s in zip(decisions, scores) if d.get("action") == "BUY"]
    others = [(d, s) for d, s in zip(decisions, scores) if d.get("action") != "BUY"]

    buys.sort(key=lambda x: x[1].get("risk_score", 3))
    n = len(buys)
    cutoff = max(min(3, n), min(12, int(n * 0.6)))  # floor min(3,n), ceiling 12

    for i, (d, s) in enumerate(buys):
        risk = s.get("risk_score", 3)
        if risk >= 4:
            s["final_verdict"] = "VETOED"
            s["veto_reason"] = f"risk_score={risk}>=4, 强制否决"
        elif i < cutoff:
            s["final_verdict"] = "APPROVED"
        else:
            s["final_verdict"] = "VETOED"
            s["veto_reason"] = f"排名{i+1}/{n}, 超出组合容量(前{cutoff}只)"

    # SELL/HOLD/REDUCE: all APPROVED
    for d, s in others:
        s["final_verdict"] = "APPROVED"

    approved = sum(1 for _, s in buys if s.get("final_verdict") == "APPROVED")
    vetoed = sum(1 for _, s in buys if s.get("final_verdict") == "VETOED")
    logger.info(f"A6 verdict: {approved} APPROVED + {vetoed} VETOED (from {n} BUY, cutoff={cutoff})")

    return scores


def run(trade_date=None, strategy="long_term"):
    """Review A7 portfolio decisions. strategy controls veto thresholds."""
    start = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    logger.info(f"Agent 6 ({strategy}): reviewing A7 decisions for {trade_date}")

    try:
        decisions = conn.execute("""
            SELECT pd.* FROM portfolio_decisions pd
            WHERE pd.calc_date = (SELECT MAX(calc_date) FROM portfolio_decisions)
            AND pd.status = 'PENDING' ORDER BY pd.action, pd.ts_code
        """).fetchall()
        if not decisions:
            logger.info("No pending decisions from A7")
            return

        decisions = [dict(d) for d in decisions]
        import random as _random
        buy_decisions = [d for d in decisions if d.get("action") == "BUY"]
        reject_decisions = [d for d in decisions if d.get("action") == "REJECT"]
        other_decisions = [d for d in decisions if d.get("action") not in ("BUY", "REJECT")]
        # 10% random sample of REJECT for false-negative catch (reduced from 30% — BATCH=15 gives better LLM coverage)
        reject_sample = _random.sample(reject_decisions, max(1, int(len(reject_decisions) * 0.1))) if reject_decisions else []
        review_decisions = buy_decisions + reject_sample
        logger.info(f"A6 input: {len(decisions)} total → review {len(review_decisions)} "
                    f"({len(buy_decisions)} BUY + {len(reject_sample)}/{len(reject_decisions)} REJECT sampled)")

        holdings = [dict(h) for h in conn.execute(
            "SELECT p.*, s.industry FROM portfolio p JOIN stocks s ON p.ts_code=s.ts_code WHERE p.status='HOLD'"
        ).fetchall()]
        macro = conn.execute("SELECT * FROM macro_regime ORDER BY calc_date DESC LIMIT 1").fetchone()
        macro = dict(macro) if macro else {"regime": "CONSOLIDATION_UP"}

        # Pre-load A5 scores + A2 reports for rule_based_checks (all decisions)
        codes = [d["ts_code"] for d in decisions]
        ph = ",".join("?" * len(codes))
        a5_scores = {}
        for r in conn.execute(
            f"SELECT ts_code, tech_score, momentum_d5 FROM composite_scores "
            f"WHERE ts_code IN ({ph}) AND calc_date=(SELECT MAX(calc_date) FROM composite_scores)",
            codes,
        ).fetchall():
            a5_scores[r["ts_code"]] = dict(r)
        fund_reports = {}
        for r in conn.execute(
            f"SELECT ts_code, report_json FROM fundamental_reports "
            f"WHERE ts_code IN ({ph}) AND calc_date=(SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code=fundamental_reports.ts_code)",
            codes,
        ).fetchall():
            try:
                fund_reports[r["ts_code"]] = json.loads(r["report_json"])
            except (json.JSONDecodeError, TypeError):
                fund_reports[r["ts_code"]] = {}

        # Score via LLM — only review_decisions, retry 3 times
        reviews = None
        if review_decisions:
            for a6_attempt in range(3):
                try:
                    reviews = score_decisions(review_decisions, holdings, macro, conn, strategy)
                    if reviews is not None:
                        break
                except Exception as e:
                    logger.warning(f"A6 LLM attempt {a6_attempt+1}/3 failed: {e}")
                if a6_attempt < 2:
                    time.sleep(5 * (a6_attempt + 1))

        # Build score map: LLM-reviewed + rule-only for others
        review_map = {}
        if reviews:
            for d, s in zip(review_decisions, reviews):
                review_map[d["ts_code"]] = s

        scores = []
        for d in decisions:
            if d["ts_code"] in review_map:
                scores.append(review_map[d["ts_code"]])
            else:
                rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
                red_count = sum(1 for c in rule if c["severity"] == "RED")
                amber_count = sum(1 for c in rule if c["severity"] == "AMBER")
                risk = min(5, 1 + red_count * 2 + amber_count)
                scores.append({
                    "risk_score": risk,
                    "recommendation": "CAUTION" if risk >= 3 else "PREFER",
                    "reasoning": f"仅规则审查: RED={red_count} AMBER={amber_count}",
                    "risk_dimensions": [c["dim"] for c in rule if c["severity"] != "GREEN"],
                    "conditions": [],
                    "confidence": 0.3,
                    "final_verdict": "APPROVED",
                })

        # Apply adversarial verdicts
        scores = _apply_adversarial_verdicts(decisions, scores)

        # ── REJECT override check: flag low-risk REJECT for manual review ──
        for d, s in zip(decisions, scores):
            if d.get("action") == "REJECT" and s.get("confidence", 0) > 0.3:
                if s.get("risk_score", 3) <= 2:
                    s["final_verdict"] = "OVERRIDE_RECOMMENDED"
                    s["override_reason"] = "低风险REJECT(A6审查通过)——建议人工复核是否纳入"
                    logger.info(f"A6 OVERRIDE: {d['ts_code']} REJECT→OVERRIDE (risk={s.get('risk_score')}, conf={s.get('confidence',0):.0%})")

        # Persist
        for d, s in zip(decisions, scores):
            rule = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
            existing = {}
            try:
                existing = json.loads(d.get("review_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                pass
            review = {
                "a7": existing.get("a7", {}),
                "a6": {
                    "risk_score": s.get("risk_score", 3),
                    "recommendation": s.get("recommendation", "CAUTION"),
                    "reasoning": s.get("reasoning", ""),
                    "rule_checks": [{"dim": c["dim"], "severity": c["severity"], "detail": c["detail"]} for c in rule],
                    "conditions": s.get("conditions", []),
                    "llm_confidence": s.get("confidence", 0),
                    "final_verdict": s.get("final_verdict", "APPROVED"),
                    "veto_reason": s.get("veto_reason", ""),
                },
            }
            if scores and scores[0].get("cash_assessment"):
                review["cash_assessment"] = scores[0]["cash_assessment"]
            if scores and scores[0].get("portfolio_notes"):
                review["portfolio_notes"] = scores[0]["portfolio_notes"]

            status = s.get("final_verdict", "APPROVED")
            conn.execute("UPDATE portfolio_decisions SET status=?, review_json=? WHERE id=?",
                         (status, json.dumps(review, ensure_ascii=False), d["id"]))
            for c in rule:
                conn.execute("INSERT INTO risk_flags (ts_code, flag_date, severity, question) VALUES (?,?,?,?)",
                             (d["ts_code"], trade_date, c["severity"], c["detail"]))

        conn.commit()
        elapsed = time.time() - start
        n_reviewed = len(decisions)
        approved = sum(1 for s in scores if s.get("final_verdict") == "APPROVED")
        vetoed = sum(1 for s in scores if s.get("final_verdict") == "VETOED")
        logger.info(f"A6 complete: {n_reviewed} reviewed → {approved} APPROVED + {vetoed} VETOED in {elapsed:.1f}s")
        conn.execute("INSERT INTO agent_logs (agent_id,run_date,status,duration_s,summary) VALUES (6,?,'SUCCESS',?,?)",
                     (trade_date, elapsed, f"Reviewed:{n_reviewed} Approved:{approved} Vetoed:{vetoed}"))
        conn.commit()
    except Exception as e:
        logger.error(f"A6 failed: {e}")
        conn.execute("INSERT INTO agent_logs (agent_id,run_date,status,duration_s,summary) VALUES (6,?,'FAILED',?,?)",
                     (trade_date, time.time()-start, str(e)[:200]))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    run()
