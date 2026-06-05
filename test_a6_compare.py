"""Compare A6 prompts for long_term vs hot_picks to find root cause."""
import json, random
from backend.data.schema import get_connection
from backend.agents.agent_6_risk import rule_based_checks

conn = get_connection()

for strategy in ["long_term", "hot_picks"]:
    decisions = [dict(d) for d in conn.execute(
        "SELECT pd.* FROM portfolio_decisions pd "
        "WHERE pd.strategy=? AND pd.calc_date = ("
        "  SELECT MAX(calc_date) FROM portfolio_decisions WHERE strategy=?) "
        "ORDER BY pd.action, pd.ts_code",
        (strategy, strategy),
    ).fetchall()]

    buys = [d for d in decisions if d["action"] == "BUY"]
    rejects = [d for d in decisions if d["action"] == "REJECT"]
    random.seed(42)
    reject_sample = random.sample(rejects, max(1, int(len(rejects) * 0.1)))
    review = buys + reject_sample
    batch = review[:15]
    codes = [d["ts_code"] for d in batch]
    ph = ",".join("?" * len(codes))

    # Load same data as A6
    holdings = [dict(h) for h in conn.execute(
        "SELECT p.*, s.industry FROM portfolio p JOIN stocks s ON p.ts_code=s.ts_code WHERE p.status='HOLD'"
    ).fetchall()]

    # A5 scores
    a5_scores = {}
    for r in conn.execute(
        f"SELECT ts_code, tech_score, momentum_d5, total_score, rank FROM composite_scores "
        f"WHERE ts_code IN ({ph}) AND strategy=? "
        f"AND calc_date=(SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)",
        codes + [strategy, strategy],
    ).fetchall():
        a5_scores[r["ts_code"]] = dict(r)

    # A2
    fund_reports = {}
    for r in conn.execute(
        f"SELECT ts_code, report_json FROM fundamental_reports "
        f"WHERE ts_code IN ({ph}) "
        f"AND calc_date = (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = fundamental_reports.ts_code)",
        codes,
    ).fetchall():
        fund_reports[r["ts_code"]] = json.loads(r["report_json"])

    # A1
    tech_reports = {}
    for r in conn.execute(
        f"SELECT ts_code, indicators_json FROM indicators WHERE ts_code IN ({ph}) "
        f"AND calc_date = (SELECT MAX(calc_date) FROM indicators)",
        codes,
    ).fetchall():
        try:
            ind = json.loads(r["indicators_json"])
            macd = ind.get("macd", {})
            tech_reports[r["ts_code"]] = {
                "overall_assessment": (
                    f"MACD:{macd.get('signal','?')} RSI:{ind.get('rsi_14','?')} "
                    f"MA:{ind.get('ma_alignment','?')} "
                    f"OBV:{ind.get('obv_trend','?')} vol_ratio:{ind.get('volume_ratio','?')}"
                ),
            }
        except: pass

    # A7 tags
    a7_tags = {}
    for d in decisions:
        try:
            rv = json.loads(d.get("review_json") or "{}")
            a7 = rv.get("a7", {})
            if a7:
                a7_tags[d["ts_code"]] = a7
        except: pass

    # Pre-compute rule flags
    rule_flags = {}
    for d in batch:
        checks = rule_based_checks(d, holdings, conn, a5_scores, fund_reports)
        reds = [c["detail"] for c in checks if c["severity"] == "RED"]
        ambers = [c["detail"] for c in checks if c["severity"] == "AMBER"]
        rule_flags[d["ts_code"]] = (reds, ambers)

    # Build batch lines
    lines = []
    for d in batch:
        code = d["ts_code"]
        tr = tech_reports.get(code, {})
        fr_r = fund_reports.get(code, {})
        a7 = a7_tags.get(code, {})
        has_a2 = fr_r.get("confidence", 0) > 0
        sc = a5_scores.get(code, {})

        lines.append(
            f"{code} [{d['action']}] T:{sc.get('tech_score',50):.0f} F:{sc.get('fundamental_score',50):.0f} "
            f"A5#{sc.get('rank','?')}/{sc.get('total_score',0):.0f}"
        )
        lines.append(f"  A1: {tr.get('overall_assessment','无')[:130]}")
        if has_a2:
            rfs = [f"{rf.get('severity','?')}:{rf.get('flag','?')}" for rf in fr_r.get('red_flags', [])]
            lines.append(f"  A2: {fr_r.get('narrative','无')[:180]} | 红旗:{rfs if rfs else '无'}")
        else:
            lines.append(f"  A2: 缺失")
        a7_weight = a7.get("weight", 0)
        lines.append(f"  A7: {a7.get('recommendation','?')} 权重{a7_weight:.0%} (确信度{a7.get('conviction','?')}) — {a7.get('rationale','无')[:120]}")
        reds, ambers = rule_flags.get(code, ([], []))
        if reds or ambers:
            flags = []
            if reds: flags.append(f"RED: {'; '.join(reds)}")
            if ambers: flags.append(f"AMBER: {'; '.join(ambers)}")
            lines.append(f"  RULE: {' | '.join(flags)}")

    batch_prompt = "\n".join(lines)
    avg_per_stock = len(batch_prompt) / len(batch)

    print(f"\n=== {strategy} batch 1 ===")
    print(f"  BUY={len([d for d in batch if d['action']=='BUY'])} REJECT={len([d for d in batch if d['action']=='REJECT'])}")
    print(f"  Batch prompt: {len(batch_prompt)} chars ({avg_per_stock:.0f} per stock)")
    print(f"  Stocks with A2: {sum(1 for d in batch if fund_reports.get(d['ts_code'],{}).get('confidence',0)>0)}/15")
    print(f"  Stocks with rule flags: {sum(1 for d in batch if rule_flags.get(d['ts_code'],([],[]))[0] or rule_flags.get(d['ts_code'],([],[]))[1])}/15")

conn.close()
