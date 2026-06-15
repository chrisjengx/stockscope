"""
Agent 4: Macro Strategist — daily market state engine.

Computes: multi-timeframe trends (30/20/10/5), breadth, sector rotation (THS),
volume, volatility, market valuation (PE/PB), index data.

LLM synthesizes a macro assessment with risk alerts, sector view, position advice.
Output: macro_regime (downstream A5/A7), macro_reports (A6 review, frontend).
"""
import json
import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()

# ═══════════════════════════════════════════════════════════════
# Data loading — market breadth, trends, volume
# ═══════════════════════════════════════════════════════════════

def load_market_data(conn):
    """Aggregate market data: daily trends, breadth, volume. 30/20/10/5 windows."""
    rows = conn.execute("""
        SELECT trade_date,
               AVG(change_pct) as avg_change,
               SUM(CASE WHEN change_pct > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as breadth_pct,
               SUM(amount) as total_amount,
               COUNT(*) as stock_count
        FROM daily_quotes
        WHERE trade_date >= date('now', '-45 days') AND change_pct IS NOT NULL
        GROUP BY trade_date ORDER BY trade_date
    """).fetchall()

    if not rows:
        return _empty_data()

    dates    = [r["trade_date"] for r in rows]
    changes  = [r["avg_change"] or 0 for r in rows]
    breadths = [r["breadth_pct"] or 50 for r in rows]
    amounts  = [r["total_amount"] or 0 for r in rows]
    n = len(changes)

    def _trend(w):  return round(sum(changes[-w:]), 2) if n >= w else round(sum(changes), 2)
    def _breadth(w): return round(sum(breadths[-w:]) / min(w, n), 1)
    def _amt(w):    return sum(amounts[-w:]) / min(w, n)

    vol_20d = round(float(np.std(changes[-20:])), 2) if n >= 20 else None
    amt_5d = _amt(5); amt_20d = _amt(20)
    vol_ratio = round(amt_5d / amt_20d, 2) if amt_20d > 0 else 1.0
    latest_breadth = round(breadths[-1], 1) if breadths else 50

    # 20-day rolling series for LLM table
    daily_series = []
    for i in range(max(0, n-20), n):
        daily_series.append({
            "date": str(dates[i])[:10],
            "day_chg_pct": round(changes[i], 2),
            "breadth_pct": round(breadths[i], 1),
            "amount_billion": round(amounts[i] / 1e8, 1),
        })

    stocks_data = _load_ma20_breadth(conn)

    return {
        "trend_30d": _trend(30), "trend_20d": _trend(20),
        "trend_10d": _trend(10), "trend_5d": _trend(5),
        "volatility_20d": vol_20d,
        "breadth_latest": latest_breadth,
        "breadth_30d_avg": _breadth(30), "breadth_20d_avg": _breadth(20),
        "breadth_10d_avg": _breadth(10), "breadth_5d_avg": _breadth(5),
        "volume_5d_avg_billion": round(amt_5d / 1e8, 1),
        "volume_20d_avg_billion": round(amt_20d / 1e8, 1),
        "volume_ratio": vol_ratio,
        "breadth_ma20_above": stocks_data.get("ma20_above_pct"),
        "new_high_low_ratio": stocks_data.get("nh_nl_ratio"),
        "daily_series": daily_series,
    }


def _load_ma20_breadth(conn):
    """MA20 above ratio + NH/NL ratio. Single-pass for MA20, sampled for NH/NL."""
    rows = conn.execute("""
        SELECT dq.ts_code, dq.close
        FROM daily_quotes dq
        WHERE dq.trade_date = (SELECT MAX(trade_date) FROM daily_quotes) AND dq.close > 0
    """).fetchall()

    ma20_above = 0; total = 0
    for r in rows:
        code, close = r["ts_code"], r["close"]
        total += 1
        ma_row = conn.execute(
            "SELECT AVG(close) FROM (SELECT close FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT 20)",
            (code,),
        ).fetchone()
        if ma_row and ma_row[0] and close > ma_row[0]:
            ma20_above += 1

    # NH/NL ratio (sample 500)
    nh = nl = 0
    for r in rows[:500]:
        code, close = r["ts_code"], r["close"]
        hilo = conn.execute(
            "SELECT MAX(close), MIN(close) FROM (SELECT close FROM daily_quotes WHERE ts_code=? ORDER BY trade_date DESC LIMIT 20)",
            (code,),
        ).fetchone()
        if hilo and hilo[0] and hilo[1] and hilo[0] > hilo[1]:
            pct = (close - hilo[1]) / (hilo[0] - hilo[1])
            if pct > 0.95: nh += 1
            elif pct < 0.05: nl += 1
    nh_nl = round(nh / nl, 2) if nl > 0 else (nh if nh > 0 else 1.0)

    return {
        "ma20_above_pct": round(ma20_above/total*100, 1) if total else 50,
        "nh_nl_ratio": nh_nl,
    }


def _empty_data():
    return {"trend_30d":0,"trend_20d":0,"trend_10d":0,"trend_5d":0,
            "volatility_20d":None,
            "breadth_latest":50,
            "breadth_30d_avg":50,"breadth_20d_avg":50,"breadth_10d_avg":50,"breadth_5d_avg":50,
            "volume_5d_avg_billion":0,"volume_20d_avg_billion":0,"volume_ratio":1.0,
            "breadth_ma20_above":50,"new_high_low_ratio":1.0,"daily_series":[]}


# ═══════════════════════════════════════════════════════════════
# Market fundamentals, index data, sector heat
# ═══════════════════════════════════════════════════════════════

def load_market_fundamentals():
    import akshare as ak
    result = {"pe": None, "pb": None, "total_mcap": None, "pb_quantile_10y": None}
    try:
        df = ak.stock_market_pe_lg(symbol="上证A股")
        if df is not None and not df.empty:
            r = df.iloc[-1]
            result["pe"] = float(r["市盈率"]) if r.get("市盈率") else None
            result["total_mcap"] = float(r["总市值"]) if r.get("总市值") else None
    except Exception: pass
    try:
        df = ak.stock_a_all_pb()
        if df is not None and not df.empty:
            r = df.iloc[-1]
            result["pb"] = float(r["middlePB"]) if r.get("middlePB") else None
            result["pb_quantile_10y"] = float(r.get("quantileInRecent10YearsMiddlePB", 0)) if r.get("quantileInRecent10YearsMiddlePB") else None
    except Exception: pass
    logger.info(f"  fundamentals: PE={result['pe']} PB={result['pb']} mcap={result['total_mcap']}")
    return result


def load_index_data():
    """Load 上证指数 daily K-line from akshare. Multi-timeframe: 30/20/10/5."""
    import akshare as ak
    result = {"sh_close": None, "sh_20d_high": None, "sh_20d_low": None,
              "sh_30d_trend": None, "sh_20d_trend": None,
              "sh_10d_trend": None, "sh_5d_trend": None,
              "sh_volume_ratio": None}
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df is not None and not df.empty:
            closes = df["close"].values[-35:]
            vols = df["volume"].values[-35:]
            n = len(closes)
            result["sh_close"] = float(closes[-1])
            result["sh_20d_high"] = float(max(closes[-20:])) if n >= 20 else float(max(closes))
            result["sh_20d_low"] = float(min(closes[-20:])) if n >= 20 else float(min(closes))
            for w, key in [(31, "sh_30d_trend"), (21, "sh_20d_trend"), (11, "sh_10d_trend"), (6, "sh_5d_trend")]:
                if n >= w and closes[-w] > 0:
                    result[key] = round((closes[-1] / closes[-w] - 1) * 100, 2)
            if len(vols) >= 25:
                v5 = sum(vols[-5:])/5; v20 = sum(vols[-25:-5])/20
                result["sh_volume_ratio"] = round(v5/v20, 2) if v20 > 0 else 1.0
            logger.info(f"  index: 上证{result['sh_close']:.0f} 30d{result.get('sh_30d_trend',0):+.1f}% 5d{result.get('sh_5d_trend',0):+.1f}%")
    except Exception as e:
        logger.warning(f"  index data unavailable: {e}")
    return result


def load_sector_heat():
    """Load sector heat from THS (同花顺): snapshot + K-line trends for top/bottom.

    Snapshot: all 87 sectors, chg% + net inflow + breadth.
    K-line: top 10 + bottom 10 sectors, adds 5d/3d/latest trends.
    Filters out sectors with <10 stocks.
    """
    import akshare as ak
    logger.info("  loading sector heat from THS...")
    try:
        df = ak.stock_board_industry_summary_ths()
    except Exception as e:
        logger.warning(f"  THS sector data unavailable: {e}")
        return []

    if df is None or df.empty:
        logger.warning("  THS sector data: empty")
        return []

    # Parse snapshot
    sectors = []
    for _, r in df.iterrows():
        stock_count = int(r["上涨家数"]) + int(r["下跌家数"])
        if stock_count < 10:
            continue
        sectors.append({
            "industry": str(r["板块"]),
            "stock_count": stock_count,
            "chg_snapshot": round(float(r["涨跌幅"]), 2),
            "net_inflow": round(float(r["净流入"]), 1),
            "up_count": int(r["上涨家数"]),
            "down_count": int(r["下跌家数"]),
            "chg_5d": None, "chg_3d": None, "chg_latest": None,
        })
    sectors.sort(key=lambda x: -x["chg_snapshot"])

    # Add K-line trends for top 10 + bottom 10
    top10 = [s["industry"] for s in sectors[:10]]
    bot10 = [s["industry"] for s in sectors[-10:]]
    target_names = set(top10 + bot10)
    _add_sector_trends(sectors, target_names)

    up = sum(1 for s in sectors if s["chg_latest"] is not None and s["chg_latest"] > 0
             or s["chg_latest"] is None and s["chg_snapshot"] > 0)
    down = sum(1 for s in sectors if s["chg_latest"] is not None and s["chg_latest"] < 0
               or s["chg_latest"] is None and s["chg_snapshot"] < 0)
    total_inflow = sum(s["net_inflow"] for s in sectors)
    total_stocks = sum(s["stock_count"] for s in sectors)
    with_trends = sum(1 for s in sectors if s["chg_latest"] is not None)

    logger.info(f"  sector heat: {len(sectors)} industries ({with_trends} with trends), "
                f"{up}↑{down}↓, {total_stocks} stocks, net flow {total_inflow:+.0f}亿")
    return sectors


def _add_sector_trends(sectors, target_names):
    """Fetch K-line for target sectors and compute 5d/3d/latest chg%."""
    import akshare as ak
    name_to_sector = {s["industry"]: s for s in sectors}
    fetched = 0
    for name in target_names:
        if name not in name_to_sector:
            continue
        try:
            from datetime import date as _date, timedelta as _td
            end_d = _date.today().strftime('%Y%m%d')
            start_d = (_date.today() - _td(days=45)).strftime('%Y%m%d')
            df = ak.stock_board_industry_index_ths(symbol=name, start_date=start_d, end_date=end_d)
            if df is not None and len(df) >= 6:
                closes = df["收盘价"].values
                n = len(closes)
                s = name_to_sector[name]
                s["chg_latest"] = round((closes[-1] / closes[-2] - 1) * 100, 2) if n >= 2 else None
                s["chg_3d"] = round((closes[-1] / closes[-4] - 1) * 100, 2) if n >= 4 and closes[-4] > 0 else None
                s["chg_5d"] = round((closes[-1] / closes[-6] - 1) * 100, 2) if n >= 6 and closes[-6] > 0 else None
                fetched += 1
        except Exception as e:
            logger.warning(f"  sector K-line {name}: {e}")
    logger.info(f"  sector trends: {fetched}/{len(target_names)} fetched")


# ═══════════════════════════════════════════════════════════════
# Deterministic regime classification (no LLM)
# ═══════════════════════════════════════════════════════════════

def classify_regime(md):
    """Rule-based regime classification. Two dimensions: direction + participation.

    Dimension 1 — direction (20-trading-day cumulative avg change):
      5% threshold:  monthly gain >5% is a strong month in A-shares (~10-15% of months)
      0% threshold:  up/down boundary
      -5% threshold: monthly loss >5% is clearly weak

    Dimension 2 — participation (20-day avg % of stocks rising):
      55% threshold: majority rising → broad participation
      45% threshold: participation is adequate, not terrible
      35% threshold: only a third rising → broad decay

    Cross-table (corners → certain, middle → neutral):
                b20>55%     45-55%      35-45%      <35%
      t20>5%    牛市上升      牛市整理      中性        中性
      t20 0~5%  牛市整理      牛市整理      中性        中性
      t20 0~-5%  中性          中性       熊市整理    熊市整理
      t20<-5%    中性        熊市整理     熊市下行    熊市下行

    NH/NL, volatility, volume ratio → added as qualitative notes for LLM,
    NOT used to change the regime label.
    """
    t20 = md.get("trend_20d", 0) or 0
    b20 = md.get("breadth_20d_avg", 50) or 50
    vol = md.get("volatility_20d", 20) or 20
    vr  = md.get("volume_ratio", 1.0) or 1.0
    nh  = md.get("new_high_low_ratio", 0.5) or 0.5

    # 2×2 cross: top-left = bull, bottom-left = bear, rest = neutral
    if t20 > 5 and b20 > 55:
        regime = "牛市上升"          # strong up + broad participation
    elif t20 > 0 and b20 > 45:
        regime = "牛市整理"          # up + decent participation
    elif t20 < -5 and b20 < 35:
        regime = "熊市下行"          # sharp down + broad decay
    elif t20 < 0 and b20 < 45:
        regime = "熊市整理"          # down + weak participation
    else:
        regime = "中性震荡"          # mismatch or middle ground → uncertain

    # Build qualitative summary from all dimensions
    parts = [regime]
    parts.append(f"20日趋势{t20:+.1f}%,10日{md.get('trend_10d',0):+.1f}%,5日{md.get('trend_5d',0):+.1f}%")
    b30 = md.get('breadth_30d_avg', 50) or 50
    b10 = md.get('breadth_10d_avg', 50) or 50
    parts.append(f"宽度30日{b30:.0f}%,20日{b20:.0f}%,10日{b10:.0f}%,MA20站上{md.get('breadth_ma20_above','?'):}%")
    parts.append(f"波动率{vol:.0f}%,量比{vr:.1f},NH/NL={nh:.2f}")

    # Add interpretive notes
    notes = []
    if nh < 0.3: notes.append("NH/NL极端,空头主导")
    if vr < 0.7: notes.append("缩量明显,流动性下降")
    elif vr > 1.3: notes.append("放量明显,资金活跃")
    if vol > 2.0: notes.append("波动加剧")
    if b20 < 30: notes.append("宽度极差,个股普跌")

    # ── Near-term vs medium-term divergence ──
    # 20-day trend is inherently lagging. A market that was "牛市上升"
    # for the past 3 weeks may have already turned in the last 5 days.
    # Flag divergence so A7/A6 LLM gets a complete picture, not stale context.
    t5_val = md.get("trend_5d", 0) or 0
    t20_val = md.get("trend_20d", 0) or 0
    if t20_val > 2 and t5_val < -1:
        notes.append(f"短期背离: 中期上行(t20={t20_val:+.1f}%)但近5日走弱(t5={t5_val:+.1f}%), 警惕趋势转折")
    elif t20_val < -2 and t5_val > 1:
        notes.append(f"短期背离: 中期下行(t20={t20_val:+.1f}%)但近5日反弹(t5={t5_val:+.1f}%), 关注底部确认")

    if notes:
        parts.append("; ".join(notes))

    regime_summary = " — ".join(parts[:1]) + ": " + ", ".join(parts[1:])
    return regime, regime_summary


# ═══════════════════════════════════════════════════════════════
# LLM macro narrative
# ═══════════════════════════════════════════════════════════════

def _fmt_sectors(sectors):
    """Format THS sector data for LLM: multi-timeframe top/bottom + capital flows."""
    if not sectors:
        return "板块数据暂不可用"

    lines = []

    # Top 10 (with trends where available)
    top10 = sectors[:10]
    parts = []
    for s in top10:
        trend = ""
        if s.get("chg_latest") is not None:
            trend = f"5d{s['chg_5d']:+.1f}% 3d{s['chg_3d']:+.1f}% 最新{s['chg_latest']:+.1f}%"
        else:
            trend = f"快照{s['chg_snapshot']:+.1f}%"
        parts.append(f"{s['industry']} {trend}({s['stock_count']}只,涨{s['up_count']}/{s['down_count']},净{'入' if s['net_inflow']>=0 else '出'}{abs(s['net_inflow']):.0f}亿)")
    lines.append("领涨 Top 10:\n  " + "\n  ".join(parts))

    # Bottom 10
    bot10 = sectors[-10:]
    parts = []
    for s in bot10:
        trend = ""
        if s.get("chg_latest") is not None:
            trend = f"5d{s['chg_5d']:+.1f}% 3d{s['chg_3d']:+.1f}% 最新{s['chg_latest']:+.1f}%"
        else:
            trend = f"快照{s['chg_snapshot']:+.1f}%"
        parts.append(f"{s['industry']} {trend}({s['stock_count']}只,涨{s['up_count']}/{s['down_count']},净{'入' if s['net_inflow']>=0 else '出'}{abs(s['net_inflow']):.0f}亿)")
    lines.append("领跌 Bottom 10:\n  " + "\n  ".join(parts))

    # Top 5 net inflow
    by_inflow = sorted(sectors, key=lambda x: -x["net_inflow"])
    parts = [f"{s['industry']} 净入{s['net_inflow']:+.0f}亿" for s in by_inflow[:5]]
    lines.append("资金流入 Top 5: " + " | ".join(parts))

    # Summary
    up = sum(1 for s in sectors if (s.get("chg_latest") or s["chg_snapshot"]) > 0)
    down = sum(1 for s in sectors if (s.get("chg_latest") or s["chg_snapshot"]) < 0)
    total_inflow = sum(s["net_inflow"] for s in sectors)
    lines.append(f"板块结构: {up}涨/{down}跌, 全市场净流入{total_inflow:+.0f}亿")

    return "\n".join(lines)


def macro_narrative(md, index_data=None, sectors=None):
    cfg = settings.get_llm_config("A4")
    if not settings.ds_api_key: return None

    daily = md.get("daily_series", [])
    tbl = "日期         | 当日涨跌 | 上涨比例 | 成交额(亿)\n" + "-" * 50 + "\n"
    for d in daily:
        tbl += f"{d['date']} | {d['day_chg_pct']:>+7.2f}% | {d['breadth_pct']:>7.1f}% | {d['amount_billion']:>9.0f}\n"

    idx = index_data or {}
    prompt = f"""你是A股宏观策略师。以下是近20个交易日的客观市场数据。请仔细分析，不要只看最近几天的剧烈波动——要把短期变化放在中期趋势中理解。

=== 近20个交易日 ===
{tbl}
=== 指数 ===
上证: {idx.get('sh_close','?')} | 20日区间: {idx.get('sh_20d_high','?')} - {idx.get('sh_20d_low','?')}
30日涨跌: {idx.get('sh_30d_trend','?')}% | 20日涨跌: {idx.get('sh_20d_trend','?')}% | 10日涨跌: {idx.get('sh_10d_trend','?')}% | 5日涨跌: {idx.get('sh_5d_trend','?')}%
量比: {idx.get('sh_volume_ratio','?')}

=== 市场结构 ===
趋势: 30日{md['trend_30d']}% | 20日{md['trend_20d']}% | 10日{md['trend_10d']}% | 5日{md['trend_5d']}%
宽度: 30日{md['breadth_30d_avg']}% | 20日{md['breadth_20d_avg']}% | 10日{md['breadth_10d_avg']}% | 5日{md['breadth_5d_avg']}% | 最新{md['breadth_latest']}%
站上MA20: {md.get('breadth_ma20_above','?')}% | 20日波动率: {md.get('volatility_20d','?')}% | 新高/新低比: {md.get('new_high_low_ratio','?')}
成交: 5日均{md['volume_5d_avg_billion']}亿 / 20日均{md['volume_20d_avg_billion']}亿 (量比:{md['volume_ratio']})

=== 板块 (5日/3日/最新) ===
{_fmt_sectors(sectors)}

=== 估值 ===
PE={md.get('market_pe','?')} | PB中位数={md.get('market_pb','?')} | PB 10年历史分位={md.get('market_pb_quantile','?')} | 总市值={md.get('total_mcap_billion','?')}千亿

输出JSON:
{{"narrative":"短中期趋势分析(30d/20d/10d/5d), 宽度扩散/收缩, 量价关系, 近期波动定性",
 "risk_alerts":["具体风险信号"],
 "sector_view":"当前板块轮动特征(哪些有资金支撑,哪些在失血,轮动有无主线,5d/3d趋势方向)",
 "position_advice":"仓位建议 + 操作纪律 + 优势板块方向"}}"""
    return llm.chat_json(prompt, model=cfg["model"], max_tokens=cfg["max_tokens"])


# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def run(trade_date=None):
    t0 = time.time()
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    logger.info("Agent 4 starting")

    try:
        md = load_market_data(conn)
        mkt_fund = load_market_fundamentals()
        index_data = load_index_data()
        sectors = load_sector_heat()
        md["market_pe"] = mkt_fund["pe"]
        md["market_pb"] = mkt_fund["pb"]
        md["market_pb_quantile"] = mkt_fund["pb_quantile_10y"]
        md["total_mcap_billion"] = round(mkt_fund["total_mcap"]/100, 1) if mkt_fund["total_mcap"] else None

        regime, regime_summary = classify_regime(md)
        llm_report = macro_narrative(md, index_data, sectors)

        conn.execute("INSERT OR REPLACE INTO macro_regime (calc_date, regime, risk_budget, detail_json) VALUES (?,?,?,?)",
                     (trade_date, regime, None, json.dumps(md, ensure_ascii=False)))

        if llm_report:
            report = {
                "calc_date": trade_date, "regime": regime, "regime_summary": regime_summary,
                "narrative": llm_report.get("narrative",""),
                "risk_alerts": llm_report.get("risk_alerts",[]),
                "sector_view": llm_report.get("sector_view",""),
                "position_advice": llm_report.get("position_advice",""),
                "key_indicators": {
                    "trend_30d": md["trend_30d"], "trend_20d": md["trend_20d"],
                    "breadth_latest": md["breadth_latest"],
                    "ma20_above": md["breadth_ma20_above"],
                    "market_pe": md.get("market_pe"), "market_pb": md.get("market_pb"),
                    "market_pb_quantile": md.get("market_pb_quantile"),
                },
            }
            conn.execute("INSERT OR REPLACE INTO macro_reports (calc_date, report_json) VALUES (?,?)",
                         (trade_date, json.dumps(report, ensure_ascii=False)))

        conn.commit()
        elapsed = time.time() - t0
        logger.info(f"Agent 4 complete: {regime} in {elapsed:.1f}s")
        conn.execute("INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) VALUES (4,?,'SUCCESS',?,?)",
                     (trade_date, elapsed, f"{regime}: 上证{index_data.get('sh_close','?')} 板块{len(sectors)}"))
        conn.commit()
    except Exception as e:
        logger.error(f"Agent 4 failed: {e}")
        conn.execute("INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) VALUES (4,?,'FAILED',?,?)",
                     (trade_date, time.time()-t0, str(e)[:200]))
        conn.commit()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    run()
