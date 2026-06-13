"""
Agent 5: Market Narrative — LLM synthesizes macro + FL distribution into context.

No longer scores individual stocks (that's FL's job). One lightweight LLM call
to produce a market narrative that helps A7 and A6 understand "what the market
is rewarding right now."

Writes fusion_reports for downstream consumers (A7 prompt, A6 risk review, API).
"""
import json
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


def market_narrative(conn, strategy, trade_date):
    """Generate market narrative from FL distribution + macro context.

    Reads FL output (composite_scores distribution), macro_reports,
    and sector data. One LLM call — no per-stock analysis.

    Returns: dict with overall_narrative, factor_attribution, ranking_concerns, key_risks
    """
    cfg = settings.get_llm_config("A5")

    # ── FL distribution stats ──
    dist = {}
    for col in ["tech_score", "fundamental_score", "momentum"]:
        rows = conn.execute(f"""
            SELECT {col} FROM composite_scores
            WHERE strategy=? AND calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
        """, (strategy, strategy)).fetchall()
        vals = [r[col] for r in rows if r[col] is not None]
        if vals:
            mu = sum(vals) / len(vals)
            var = sum((v - mu) ** 2 for v in vals) / len(vals)
            dist[col] = {"mean": round(mu, 1), "std": round(var ** 0.5, 1), "n": len(vals)}

    # Trend type distribution
    trend_dist = defaultdict(int)
    for r in conn.execute(f"""
        SELECT trend_type, COUNT(*) as n FROM composite_scores
        WHERE strategy=? AND calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
        GROUP BY trend_type
    """, (strategy, strategy)).fetchall():
        trend_dist[r["trend_type"] or "unknown"] = r["n"]

    # Top industries in FL
    top_industries = conn.execute(f"""
        SELECT s.industry, COUNT(*) as n, ROUND(AVG(cs.total_score), 1) as avg_score
        FROM composite_scores cs
        JOIN stocks s ON cs.ts_code = s.ts_code
        WHERE cs.strategy=? AND cs.calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
        GROUP BY s.industry HAVING n >= 3
        ORDER BY AVG(cs.total_score) DESC LIMIT 10
    """, (strategy, strategy)).fetchall()

    # Top 20 stocks (for narrative context, not individual analysis)
    top20 = conn.execute(f"""
        SELECT cs.ts_code, s.name, s.industry, cs.total_score, cs.trend_type,
               cs.momentum_d5, cs.momentum_d20, cs.momentum_d60
        FROM composite_scores cs
        JOIN stocks s ON cs.ts_code = s.ts_code
        WHERE cs.strategy=? AND cs.calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
        ORDER BY cs.total_score DESC LIMIT 20
    """, (strategy, strategy)).fetchall()

    # ── Macro context ──
    macro_report = {}
    mrp = conn.execute(
        "SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1"
    ).fetchone()
    if mrp:
        try:
            macro_report = json.loads(mrp["report_json"])
        except (json.JSONDecodeError, TypeError):
            pass

    regime_name = ""
    mr = conn.execute("SELECT regime FROM macro_regime ORDER BY calc_date DESC LIMIT 1").fetchone()
    if mr:
        regime_name = mr["regime"]

    regime_summary = macro_report.get("regime_summary", regime_name) if macro_report else regime_name
    macro_narrative = str(macro_report.get("narrative", "无"))[:300] if macro_report else "无"
    sector_view = str(macro_report.get("sector_view", ""))[:200] if macro_report else ""

    # ── Build LLM prompt ──
    lines = [
        "你是A股市场策略分析师。你的任务是用一段话描述当前市场的整体格局——不需要评价个股。",
        "",
        f"策略: {strategy} | 宏观定调: {regime_name}",
        f"宏观判断: {regime_summary}",
        f"A4宏观分析: {macro_narrative}",
        "",
        f"=== FL排名分布 (N={dist.get('momentum',{}).get('n','?')}) ===",
        f"技术分: μ={dist.get('tech_score',{}).get('mean','?')} σ={dist.get('tech_score',{}).get('std','?')}",
        f"基本面分: μ={dist.get('fundamental_score',{}).get('mean','?')} σ={dist.get('fundamental_score',{}).get('std','?')}",
        f"动量分: μ={dist.get('momentum',{}).get('mean','?')} σ={dist.get('momentum',{}).get('std','?')}",
        f"趋势分布: {dict(trend_dist)}",
        "",
        f"=== 强势行业 (按FL均分) ===",
    ]
    for ind in top_industries:
        lines.append(f"  {ind['industry']}: {ind['n']}只, 均分{ind['avg_score']}")

    lines.append("")
    lines.append(f"=== 排名前20 (供参考格局，不逐只点评) ===")
    for i, r in enumerate(top20, 1):
        lines.append(
            f"  {i:2d}. {r['ts_code']} {r['name']} [{r['industry']}] "
            f"得分{r['total_score']:.0f} {r['trend_type']} "
            f"d5:{r['momentum_d5']:+.1f}% d20:{r['momentum_d20']:+.1f}% d60:{r['momentum_d60']:+.1f}%"
        )

    lines.extend([
        "",
        f"=== 板块资金偏好 ===",
        f"{sector_view if sector_view else '板块数据暂不可用'}",
        "",
        "=== 分析要求 ===",
        "写一段200-400字的市场叙事，回答以下问题：",
        "1. 当前市场在奖励什么类型的股票？（动量驱动 vs 基本面驱动 vs 多因子共振）",
        "2. 哪些行业/板块有资金聚集？这个格局能否持续？",
        "3. FL排名的整体质量如何？有没有值得关注的结构性特征？",
        "4. 宏观环境是否支持当前的市场风格？",
        "",
        "输出JSON:",
        "{",
        f'  "overall_narrative": "200-400字的市场叙事，自然段落，不要列表格式",',
        '  "ranking_concerns": "1-2句话，FL排名中值得注意的结构性问题或偏差（如某一因子过度主导、某板块被系统性低估），无问题则写无明显结构性问题",',
        '  "key_risks": "1-2句话，当前市场环境下投资者最应该关注的风险点",',
        '  "signal_consensus": "BULLISH/MIXED/BEARISH (多空信号一致度)",',
        '  "signal_conflicts": "信号矛盾描述（如技术面偏多但基本面偏弱），无矛盾则写无明显冲突",',
        '  "macro_aligned": true/false (宏观环境是否支持当前FL排名逻辑)",',
        '  "confidence": 0.0-1.0',
        "}",
    ])

    prompt = "\n".join(lines)

    # Retry loop
    for attempt in range(3):
        try:
            result = llm.chat_json(prompt, model=cfg["model"], max_tokens=cfg.get("max_tokens", 2000))
            if result:
                logger.info(f"A5 market_narrative OK (attempt {attempt+1}, {len(prompt)} chars prompt)")
                return result
        except Exception as e:
            logger.warning(f"A5 market_narrative attempt {attempt+1}/3 failed: {e}")
        if attempt < 2:
            time.sleep(10 * (attempt + 1))

    logger.error("A5 market_narrative: all 3 attempts failed")
    return None


def run(trade_date=None, strategy="long_term"):
    """Generate market narrative from FL output. No individual stock scoring."""
    start = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    conn = get_connection()
    logger.info(f"Agent 5 (market narrative): strategy={strategy}")

    try:
        # Check FL has run
        fl_count = conn.execute(
            "SELECT COUNT(*) FROM composite_scores WHERE strategy=? AND calc_date=?",
            (strategy, trade_date)
        ).fetchone()[0]

        if fl_count == 0:
            # Try latest calc_date
            latest = conn.execute(
                "SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?",
                (strategy,)
            ).fetchone()[0]
            if latest:
                logger.info(f"  Using latest FL data: {latest}")
                trade_date = latest
            else:
                logger.warning("No FL data found — skipping market narrative")
                return

        result = market_narrative(conn, strategy, trade_date)

        if result:
            conn.execute(
                """INSERT OR REPLACE INTO fusion_reports
                   (calc_date, strategy, report_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (trade_date, strategy,
                 json.dumps(result, ensure_ascii=False),
                 datetime.now().isoformat()),
            )
            logger.info("  fusion_reports written")

        elapsed = time.time() - start
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, stocks_processed, duration_s, summary) "
            "VALUES (5, ?, 'SUCCESS', ?, ?, ?)",
            (trade_date, fl_count, elapsed,
             f"[{strategy}] market_narrative={'Y' if result else 'N'}"),
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
    finally:
        conn.close()


if __name__ == "__main__":
    run()
