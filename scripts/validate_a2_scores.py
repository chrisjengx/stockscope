"""
Validate A2 fundamental_score quality.

Checks: distribution width, internal consistency, vs fallback differentiation,
and LLM spot-check of extreme values. Run after A2 worker produces new scores.
"""
import json
import sys
import math
from datetime import datetime
from collections import Counter

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.focus_list import _fundamental_fallback

settings = get_settings()
llm = get_llm()


def load_stocks(conn):
    """Load all stocks with A2 fundamental_score (not None)."""
    rows = conn.execute("""
        SELECT fr.ts_code, s.name, fr.report_json
        FROM fundamental_reports fr
        JOIN stocks s ON fr.ts_code = s.ts_code
        WHERE fr.calc_date = (SELECT MAX(calc_date) FROM fundamental_reports WHERE ts_code = fr.ts_code)
    """).fetchall()

    stocks = []
    for r in rows:
        try:
            rep = json.loads(r["report_json"])
        except (json.JSONDecodeError, TypeError):
            continue
        fs = rep.get("fundamental_score")
        if fs is None:
            continue
        stocks.append({
            "ts_code": r["ts_code"],
            "name": r["name"],
            "fund_score": fs,
            "score_rationale": rep.get("score_rationale", ""),
            "eq": (rep.get("earnings_quality", {}) or {}).get("rating", "?"),
            "fh": (rep.get("financial_health", {}) or {}).get("rating", "?"),
            "val": (rep.get("valuation", {}) or {}).get("rating", "?"),
            "rfs": len(rep.get("red_flags", [])),
            "narrative": rep.get("narrative", ""),
            "confidence": rep.get("confidence", 0),
        })
    return stocks


# ═══════════════════════════════════════════════
# Check 1: Distribution width
# ═══════════════════════════════════════════════

def check_distribution(scores):
    n = len(scores)
    mu = sum(scores) / n
    var = sum((s - mu) ** 2 for s in scores) / n
    sigma = round(var ** 0.5, 1)

    # Max concentration in any 20-point band
    band_counts = Counter(int(s // 20) * 20 for s in scores)
    band_max = max(band_counts.values()) / n * 100

    if sigma >= 15:
        status = "PASS"
    elif sigma >= 10 and band_max < 50:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "n": n, "mean": round(mu, 1), "std": sigma,
        "p10": sorted(scores)[n // 10],
        "p25": sorted(scores)[n // 4],
        "p50": sorted(scores)[n // 2],
        "p75": sorted(scores)[3 * n // 4],
        "p90": sorted(scores)[9 * n // 10],
        "min": min(scores), "max": max(scores),
        "band_max_pct": round(band_max, 1),
        "status": status,
    }


# ═══════════════════════════════════════════════
# Check 2: Internal consistency
# ═══════════════════════════════════════════════

def check_consistency(stocks):
    # Positive: fh=GOOD AND eq=HIGH → fs >= 70
    pos_condition = [s for s in stocks if s["fh"] == "GOOD" and s["eq"] == "HIGH"]
    pos_ok = sum(1 for s in pos_condition if s["fund_score"] >= 70)
    pos_pct = round(pos_ok / len(pos_condition) * 100, 1) if pos_condition else None

    # Negative: fh=POOR AND eq=LOW → fs <= 35
    neg_condition = [s for s in stocks if s["fh"] == "POOR" and s["eq"] == "LOW"]
    neg_ok = sum(1 for s in neg_condition if s["fund_score"] <= 35)
    neg_pct = round(neg_ok / len(neg_condition) * 100, 1) if neg_condition else None

    # Rank correlation with composite rating
    def _rating_score(s):
        fh_map = {"GOOD": 3, "FAIR": 2, "POOR": 1}
        eq_map = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        return fh_map.get(s["fh"], 2) + eq_map.get(s["eq"], 2)

    n = len(stocks)
    fs_ranks = _rank([s["fund_score"] for s in stocks])
    rating_ranks = _rank([_rating_score(s) for s in stocks])
    r = _spearman_r(fs_ranks, rating_ranks, n)

    # Status
    issues = []
    if pos_pct is not None and pos_pct < 70:
        issues.append(f"正向一致性={pos_pct}%")
    if neg_pct is not None and neg_pct < 70:
        issues.append(f"负向一致性={neg_pct}%")
    if r < 0.3:
        issues.append(f"相关性r={r:.2f}")

    if not issues:
        status = "PASS"
    elif len(issues) <= 1 and r >= 0.3:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "pos_n": len(pos_condition), "pos_ok_pct": pos_pct,
        "neg_n": len(neg_condition), "neg_ok_pct": neg_pct,
        "rank_corr": round(r, 3),
        "status": status,
        "issues": issues,
    }


# ═══════════════════════════════════════════════
# Check 3: A2 vs Fallback
# ═══════════════════════════════════════════════

def check_vs_fallback(conn, stocks):
    a2_scores = [s["fund_score"] for s in stocks]
    fb_scores = []
    for s in stocks:
        fb, _ = _fundamental_fallback(conn, s["ts_code"])
        fb_scores.append(fb)

    n = len(stocks)
    a2_std = _std(a2_scores)
    fb_std = _std(fb_scores)
    ratio = round(a2_std / fb_std, 2) if fb_std > 0 else 999

    # Differences
    diffs = [abs(a2_scores[i] - fb_scores[i]) for i in range(n)]
    big_diff_pct = round(sum(1 for d in diffs if d > 30) / n * 100, 1)

    # Rank correlation
    a2_ranks = _rank(a2_scores)
    fb_ranks = _rank(fb_scores)
    r = _spearman_r(a2_ranks, fb_ranks, n)

    issues = []
    if ratio < 1.2:
        issues.append(f"std_ratio={ratio}(<1.2)")
    if big_diff_pct >= 20:
        issues.append(f"big_diff={big_diff_pct}%(≥20%)")
    if r > 0.8:
        issues.append(f"r={r:.2f}(>0.8, too similar)")
    elif r < 0.2:
        issues.append(f"r={r:.2f}(<0.2, too different)")

    if not issues:
        status = "PASS"
    elif len(issues) == 1 and ratio >= 1.2:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "a2_std": round(a2_std, 1), "fb_std": round(fb_std, 1),
        "std_ratio": ratio,
        "big_diff_pct": big_diff_pct,
        "mean_diff": round(sum(diffs) / n, 1),
        "rank_corr": round(r, 3),
        "status": status,
        "issues": issues,
    }


# ═══════════════════════════════════════════════
# Check 4: Extreme value spot check (LLM)
# ═══════════════════════════════════════════════

def spot_check_extremes(stocks):
    if len(stocks) < 20:
        return {"status": "SKIP", "reason": f"样本不足 ({len(stocks)}<20)"}

    sorted_stocks = sorted(stocks, key=lambda s: s["fund_score"])
    sample = sorted_stocks[:5] + sorted_stocks[-5:]

    cfg = settings.get_llm_config("A2")  # reuse A2 config, lightweight
    confirmed = 0
    details = []

    for s in sample:
        rfs_summary = f"{s['rfs']}个" if s['rfs'] > 0 else "无"
        prompt = f"""你是财务分析审计师。检查A2给出的一项评分是否合理。

股票: {s['ts_code']} {s['name']}
fundamental_score: {s['fund_score']}
评分理由: {s['score_rationale']}
盈利质量: {s['eq']}  财务健康: {s['fh']}  估值: {s['val']}
红旗: {rfs_summary}
分析摘要: {s['narrative'][:200]}

请判断: fundamental_score 是否与上述分析一致?
- CONFIRMED: 分数合理
- TOO_HIGH: 分数偏高 (给出建议分数)
- TOO_LOW: 分数偏低 (给出建议分数)

输出JSON: {{"verdict":"CONFIRMED/TOO_HIGH/TOO_LOW","suggested_score":0-100,"reason":"..."}}"""

        try:
            result = llm.chat_json(prompt, model=cfg.get("model", "deepseek-v4-flash"),
                                   max_tokens=300)
            if result:
                verdict = result.get("verdict", "?")
                if verdict == "CONFIRMED":
                    confirmed += 1
                details.append({
                    "ts_code": s["ts_code"],
                    "fs": s["fund_score"],
                    "verdict": verdict,
                    "suggested": result.get("suggested_score"),
                    "reason": result.get("reason", "")[:100],
                })
        except Exception as e:
            details.append({
                "ts_code": s["ts_code"], "fs": s["fund_score"],
                "verdict": "ERROR", "suggested": None, "reason": str(e)[:100],
            })

    total = len(details)
    if total == 0:
        return {"status": "SKIP", "reason": "LLM调用全部失败"}

    confirm_pct = confirmed / total * 100
    if confirm_pct >= 70:
        status = "PASS"
    elif confirm_pct >= 50:
        status = "WARN"
    else:
        status = "FAIL"

    return {
        "n": total, "confirmed": confirmed,
        "confirm_pct": round(confirm_pct, 1),
        "status": status,
        "details": details,
    }


# ═══════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════

def _std(vals):
    n = len(vals)
    mu = sum(vals) / n
    return (sum((v - mu) ** 2 for v in vals) / n) ** 0.5


def _rank(vals):
    """Return 1-based ranks (1=highest)."""
    indexed = sorted(enumerate(vals), key=lambda x: -x[1])
    ranks = [0] * len(vals)
    for rank, (idx, _) in enumerate(indexed, 1):
        ranks[idx] = rank
    return ranks


def _spearman_r(ranks1, ranks2, n):
    d2 = sum((ranks1[i] - ranks2[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3)


# ═══════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════

def validate():
    conn = get_connection()
    stocks = load_stocks(conn)

    if len(stocks) < 50:
        print(f"A2 fundamental_score 验证: 样本不足 ({len(stocks)}<50)，跳过")
        conn.close()
        return

    print(f"A2 fundamental_score 验证报告")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}  样本: N={len(stocks)}")
    print()

    # Check 1
    scores = [s["fund_score"] for s in stocks]
    c1 = check_distribution(scores)
    print(f"Check 1 分布:   {c1['status']:4s}   "
          f"σ={c1['std']}  range={c1['min']}-{c1['max']}  "
          f"p25={c1['p25']} p50={c1['p50']} p75={c1['p75']}  "
          f"max_band={c1['band_max_pct']}%")

    # Check 2
    c2 = check_consistency(stocks)
    print(f"Check 2 自洽:   {c2['status']:4s}   "
          f"正向={c2['pos_ok_pct']}%({c2['pos_n']}只) "
          f"负向={c2['neg_ok_pct']}%({c2['neg_n']}只) "
          f"r={c2['rank_corr']}")
    if c2["issues"]:
        for issue in c2["issues"]:
            print(f"         ⚠️  {issue}")

    # Check 3
    c3 = check_vs_fallback(conn, stocks)
    print(f"Check 3 vsFB:   {c3['status']:4s}   "
          f"σ(A2)={c3['a2_std']} σ(FB)={c3['fb_std']} "
          f"ratio={c3['std_ratio']}x  "
          f"big_diff={c3['big_diff_pct']}%  r={c3['rank_corr']}")
    if c3["issues"]:
        for issue in c3["issues"]:
            print(f"         ⚠️  {issue}")

    # Check 4
    if settings.ds_api_key:
        c4 = spot_check_extremes(stocks)
        if c4["status"] == "SKIP":
            print(f"Check 4 极端:   SKIP  ({c4['reason']})")
        else:
            print(f"Check 4 极端:   {c4['status']:4s}   "
                  f"{c4['confirmed']}/{c4['n']} CONFIRMED ({c4['confirm_pct']}%)")
            for d in c4.get("details", []):
                print(f"         {d['ts_code']} fs={d['fs']} {d['verdict']} {d.get('suggested','')} {d['reason']}")
    else:
        c4 = {"status": "SKIP"}
        print(f"Check 4 极端:   SKIP  (无 API key)")

    # Summary
    results = [c1, c2, c3, c4]
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_warn = sum(1 for r in results if r["status"] == "WARN")

    print()
    if n_fail == 0 and n_pass >= 3:
        print("结论: ✅ 分数可用")
    elif n_fail <= 1:
        print(f"结论: ⚠️ 分数有风险 ({n_fail} FAIL, {n_warn} WARN)，人工检查后决定")
    else:
        print(f"结论: 🔴 分数不可用 ({n_fail} FAIL)，需调整 A2 prompt")

    conn.close()


if __name__ == "__main__":
    validate()
