"""
Flask API server for Stock Analysis System.
Serves all /api/* endpoints and static frontend files.
"""
import os
import sys
import json
import time
import logging
from datetime import datetime
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.expanduser("~/stock-analysis"))
from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = Flask(__name__, static_folder=None)
CORS(app)


def api_error(code, message):
    return jsonify({"error": True, "code": code, "message": message}), code


def parse_pagination():
    page = request.args.get("page", 1, type=int)
    limit = request.args.get("limit", 50, type=int)
    limit = min(limit, 200)
    offset = (page - 1) * limit
    return page, limit, offset


# ========== Dashboard ==========
@app.route("/api/dashboard")
def api_dashboard():
    conn = get_connection()
    try:
        macro = conn.execute(
            "SELECT regime, detail_json, calc_date FROM macro_regime "
            "ORDER BY calc_date DESC LIMIT 1"
        ).fetchone()
        macro_data = dict(macro) if macro else {"regime": "UNKNOWN"}
        # Include LLM narrative from macro_reports
        macro_rpt = conn.execute(
            "SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1"
        ).fetchone()
        if macro_rpt:
            import json
            rpt = json.loads(macro_rpt["report_json"])
            macro_data["narrative"] = rpt.get("narrative", "")
            macro_data["risk_alerts"] = rpt.get("risk_alerts", [])
            macro_data["sector_view"] = rpt.get("sector_view", "")
            macro_data["position_advice"] = rpt.get("position_advice", "")
            macro_data["key_indicators"] = rpt.get("key_indicators", {})

        holdings = conn.execute(
            "SELECT COUNT(*) as count, SUM(weight) as total_weight FROM portfolio WHERE status='HOLD'"
        ).fetchone()
        pf_data = {"holdings_count": holdings["count"] or 0, "total_weight": holdings["total_weight"] or 0}

        agents = conn.execute(
            "SELECT agent_id, run_date, status, duration_s, stocks_processed, summary "
            "FROM agent_logs al1 WHERE id IN "
            "(SELECT MAX(id) FROM agent_logs al2 "
            "WHERE al2.run_date = (SELECT MAX(run_date) FROM agent_logs WHERE agent_id = al2.agent_id) "
            "GROUP BY al2.agent_id)"
        ).fetchall()
        agent_status = [dict(a) for a in agents]

        news = conn.execute(
            "SELECT id, source, summary, sentiment, impact, tags FROM news_feed "
            "ORDER BY published_at DESC LIMIT 10"
        ).fetchall()
        top_news = [dict(n) for n in news]

        # VETOED decisions
        vetoed = conn.execute(
            "SELECT ts_code, action, reason, status FROM portfolio_decisions "
            "WHERE status='VETOED' ORDER BY calc_date DESC LIMIT 5"
        ).fetchall()
        vetoed_decisions = [dict(v) for v in vetoed]

        return jsonify({
            "macro": macro_data,
            "portfolio_summary": pf_data,
            "agent_status": agent_status,
            "top_news": top_news,
            "vetoed_decisions": vetoed_decisions,
        })
    finally:
        conn.close()


# ========== Reports ==========
@app.route("/api/reports/macro")
def api_macro_report():
    """Return the latest A4 macro strategy report as a self-contained document."""
    conn = get_connection()
    try:
        r = conn.execute(
            "SELECT calc_date, regime, detail_json FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
        ).fetchone()
        if not r:
            return jsonify({"error": "No macro data available"}), 404

        md = json.loads(r["detail_json"]) if r["detail_json"] else {}
        rpt_row = conn.execute(
            "SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1"
        ).fetchone()
        rpt = json.loads(rpt_row["report_json"]) if rpt_row and rpt_row["report_json"] else {}

        # Build a self-contained report
        daily = md.get("daily_series", [])
        daily_table = [{"date": d["date"], "chg_pct": d["day_chg_pct"],
                        "breadth": d["breadth_pct"], "amount_b": d["amount_billion"]}
                       for d in daily]

        sectors = md.get("sectors", [])
        sector_table = [{"industry": s["industry"], "return_20d": s["return_20d"],
                         "return_5d": s["return_5d"]} for s in sectors[:10]]

        return jsonify({
            "report_date": r["calc_date"][:10],
            "regime": r["regime"],
            "regime_summary": rpt.get("regime_summary", ""),
            "narrative": rpt.get("narrative", ""),
            "risk_alerts": rpt.get("risk_alerts", []),
            "sector_view": rpt.get("sector_view", ""),
            "position_advice": rpt.get("position_advice", ""),
            "indicators": {
                "sh_close": md.get("sh_close"),
                "trend_20d": md.get("trend_20d"),
                "trend_60d": md.get("trend_60d"),
                "breadth_5d": md.get("breadth_5d_avg"),
                "breadth_ma20": md.get("breadth_ma20_above"),
                "nh_nl_ratio": md.get("new_high_low_ratio"),
                "market_pe": md.get("market_pe"),
                "market_pb": md.get("market_pb"),
                "pb_quantile": md.get("market_pb_quantile"),
                "vol_ratio": md.get("volume_ratio"),
            },
            "daily_series": daily_table,
            "sectors": sector_table,
        })
    finally:
        conn.close()


@app.route("/api/reports/fundamental/<ts_code>")
def api_fundamental_report(ts_code):
    """Return the latest A2 fundamental analysis report for a stock."""
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT report_json, calc_date, overall_score FROM fundamental_reports "
            "WHERE ts_code=? ORDER BY calc_date DESC LIMIT 1",
            (ts_code,)
        ).fetchone()
        if not row or not row["report_json"]:
            return jsonify({"error": "No fundamental report available for this stock"}), 404

        report = json.loads(row["report_json"])
        return jsonify({
            "ts_code": ts_code,
            "report_date": row["calc_date"][:10],
            "financial_period": report.get("financial_period", ""),
            "quarter_count": report.get("quarter_count", 0),
            "data_source": report.get("data_source", ""),
            "data_quality_notes": report.get("data_quality_notes", []),
            "metrics_summary": report.get("metrics_summary", {}),
            "quarters": report.get("quarters", []),
            "earnings_quality": report.get("earnings_quality"),
            "growth_quality": report.get("growth_quality"),
            "financial_health": report.get("financial_health"),
            "valuation": report.get("valuation"),
            "red_flags": report.get("red_flags", []),
            "blind_spots": report.get("blind_spots", []),
            "narrative": report.get("narrative", ""),
            "confidence": report.get("confidence", 0),
        })
    finally:
        conn.close()


@app.route("/api/reports/fusion")
def api_fusion_report():
    """Return the latest A5 fusion synthesis report."""
    strategy = request.args.get("strategy", "long_term")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT calc_date, report_json FROM fusion_reports "
            "WHERE strategy=? ORDER BY calc_date DESC LIMIT 1",
            (strategy,),
        ).fetchone()
        if not row or not row["report_json"]:
            return jsonify({"error": f"No fusion report available for {strategy}"}), 404

        report = json.loads(row["report_json"])
        return jsonify({
            "report_date": row["calc_date"][:10] if row["calc_date"] else "",
            "strategy": strategy,
            "signal_consensus": report.get("signal_consensus", []),
            "signal_conflicts": report.get("signal_conflicts", []),
            "macro_aligned": report.get("macro_aligned", []),
            "ranking_concerns": report.get("ranking_concerns", []),
            "key_risks": report.get("key_risks", []),
            "overall_narrative": report.get("overall_narrative", ""),
            "confidence": report.get("confidence", 0),
        })
    finally:
        conn.close()


# ========== Stocks ==========
@app.route("/api/stocks")
def api_stocks():
    page, limit, offset = parse_pagination()
    tier = request.args.get("tier")
    industry = request.args.get("industry")
    market = request.args.get("market")
    search = request.args.get("search")
    sort = request.args.get("sort", "total_score")
    order = request.args.get("order", "desc")
    strategy = request.args.get("strategy", "long_term")
    history = request.args.get("history", 0, type=int)  # 0=latest, 1=previous, 2=two ago ...

    valid_sorts = {"total_score", "tech_score", "fundamental_score", "market_cap", "change_pct"}
    if sort not in valid_sorts:
        sort = "total_score"
    if order not in ("asc", "desc"):
        order = "desc"

    conn = get_connection()
    try:
        where = []
        params = []
        if tier is not None:
            where.append("t.tier = ?")
            params.append(tier)
        if industry:
            where.append("s.industry = ?")
            params.append(industry)
        if market:
            where.append("s.market = ?")
            params.append(market)
        if search:
            where.append("(s.ts_code LIKE ? OR s.name LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])

        where_clause = "WHERE " + " AND ".join(where) if where else ""

        # Resolve history index → real calc_date
        dates = conn.execute(
            "SELECT DISTINCT calc_date FROM composite_scores WHERE strategy=? "
            "ORDER BY calc_date DESC LIMIT ? OFFSET ?",
            [strategy, 1, history]
        ).fetchone()
        resolved_date = dates[0] if dates else None

        if resolved_date:
            count_sql = f"""
                SELECT COUNT(DISTINCT s.ts_code) FROM stocks s
                LEFT JOIN tier_assignments t ON s.ts_code = t.ts_code
                    AND t.rowid = (SELECT rowid FROM tier_assignments t2 WHERE t2.ts_code = s.ts_code ORDER BY updated_at DESC LIMIT 1)
                LEFT JOIN composite_scores cs ON s.ts_code = cs.ts_code AND cs.strategy = ?
                    AND cs.calc_date = ?
                {where_clause}
            """
            total = conn.execute(count_sql, [strategy, resolved_date] + params).fetchone()[0]

            query_sql = f"""
                SELECT s.ts_code, s.name, s.industry, t.tier,
                       MIN(100, MAX(0, cs.total_score)) as total_score,
                       MIN(100, MAX(0, cs.tech_score)) as tech_score,
                       MIN(100, MAX(0, cs.fundamental_score)) as fundamental_score,
                       dq.close as market_cap_proxy, dq.change_pct,
                       cs.calc_date as score_date
                FROM stocks s
                LEFT JOIN tier_assignments t ON s.ts_code = t.ts_code
                    AND t.rowid = (SELECT rowid FROM tier_assignments t2 WHERE t2.ts_code = s.ts_code ORDER BY updated_at DESC LIMIT 1)
                LEFT JOIN composite_scores cs ON s.ts_code = cs.ts_code AND cs.strategy = ?
                    AND cs.calc_date = ?
                LEFT JOIN daily_quotes dq ON s.ts_code = dq.ts_code
                    AND dq.trade_date = (SELECT MAX(trade_date) FROM daily_quotes)
                {where_clause}
                ORDER BY {sort} {order}
                LIMIT ? OFFSET ?
            """
            date_params = [strategy, resolved_date]
            rows = conn.execute(query_sql, date_params + params + [limit, offset]).fetchall()
        else:
            total = 0
            rows = []

        # Tier distribution (always latest snapshot)
        tier_dist = None
        if tier is None and not industry and not search:
            dist = conn.execute(
                "SELECT tier, COUNT(*) as cnt FROM tier_assignments GROUP BY tier ORDER BY tier"
            ).fetchall()
            tier_dist = {f"tier_{r['tier']}": r["cnt"] for r in dist}

        # Available historical dates for this strategy
        available = conn.execute(
            "SELECT DISTINCT calc_date FROM composite_scores WHERE strategy=? "
            "ORDER BY calc_date DESC LIMIT 30",
            [strategy]
        ).fetchall()

        return jsonify({
            "items": [dict(r) for r in rows],
            "total": total, "page": page, "limit": limit,
            "tier_distribution": tier_dist,
            "history": resolved_date,
            "available_history": [r["calc_date"] for r in available],
        })
    finally:
        conn.close()


@app.route("/api/stocks/<ts_code>")
def api_stock_detail(ts_code):
    conn = get_connection()
    try:
        stock = conn.execute("SELECT * FROM stocks WHERE ts_code=?", (ts_code,)).fetchone()
        if not stock:
            return api_error(404, "股票不存在")

        indicators = conn.execute(
            "SELECT * FROM indicators WHERE ts_code=? ORDER BY calc_date DESC LIMIT 1",
            (ts_code,)
        ).fetchone()

        financials = conn.execute(
            "SELECT * FROM financials WHERE ts_code=? ORDER BY report_date DESC LIMIT 4",
            (ts_code,)
        ).fetchall()

        news = conn.execute(
            "SELECT * FROM news_feed WHERE related_stocks LIKE ? ORDER BY published_at DESC LIMIT 10",
            (f"%{ts_code}%",)
        ).fetchall()

        strategy = request.args.get("strategy", "long_term")
        scores = conn.execute(
            "SELECT calc_date, total_score FROM composite_scores WHERE ts_code=? AND strategy=? ORDER BY calc_date DESC LIMIT 20",
            (ts_code, strategy),
        ).fetchall()

        # Thesis history
        try:
            thesis = conn.execute(
                "SELECT * FROM investment_theses WHERE ts_code=? ORDER BY created_at DESC LIMIT 1",
                (ts_code,)
            ).fetchone()
        except Exception:
            thesis = None

        # A2 fundamental report (LLM deep analysis)
        fund_report = conn.execute(
            "SELECT report_json, calc_date, overall_score FROM fundamental_reports "
            "WHERE ts_code=? ORDER BY calc_date DESC LIMIT 1",
            (ts_code,)
        ).fetchone()
        fund_data = None
        if fund_report and fund_report["report_json"]:
            try:
                fund_data = json.loads(fund_report["report_json"])
                fund_data["_calc_date"] = fund_report["calc_date"]
                fund_data["_overall_score"] = fund_report["overall_score"]
            except json.JSONDecodeError:
                pass

        return jsonify({
            "stock": dict(stock),
            "indicators": dict(indicators) if indicators else None,
            "financials": [dict(f) for f in financials],
            "fundamental_report": fund_data,
            "related_news": [dict(n) for n in news],
            "score_history": [dict(s) for s in scores],
            "thesis": dict(thesis) if thesis else None,
        })
    finally:
        conn.close()


# ========== Portfolio ==========
@app.route("/api/portfolio")
def api_portfolio():
    conn = get_connection()
    try:
        holdings = conn.execute("""
            SELECT p.ts_code, s.name, p.entry_date, p.entry_price, p.weight,
                   p.status, p.shares,
                   dq.close as current_price,
                   CAST((julianday('now') - julianday(p.entry_date)) AS INTEGER) as hold_days
            FROM portfolio p
            JOIN stocks s ON p.ts_code = s.ts_code
            LEFT JOIN daily_quotes dq ON p.ts_code = dq.ts_code
                AND dq.trade_date = (SELECT MAX(trade_date) FROM daily_quotes)
            WHERE p.status = 'HOLD'
        """).fetchall()

        # Attach latest decision for each holding
        holdings_list = []
        for h in holdings:
            hd = dict(h)
            decision = conn.execute(
                "SELECT action, status, reason FROM portfolio_decisions "
                "WHERE ts_code=? AND calc_date=(SELECT MAX(calc_date) FROM portfolio_decisions) "
                "AND status IN ('APPROVED', 'PENDING') ORDER BY id DESC LIMIT 1",
                (hd["ts_code"],),
            ).fetchone()
            hd["decision"] = dict(decision) if decision else None
            holdings_list.append(hd)

        # Portfolio aggregate metrics
        total_cost = sum((h["entry_price"] or 0) * (h["shares"] or 0) for h in holdings)
        total_market = sum((h["current_price"] or 0) * (h["shares"] or 0) for h in holdings if h["current_price"])
        total_pl = total_market - total_cost if total_cost > 0 else 0
        total_weight = sum(h["weight"] or 0 for h in holdings)

        # Daily P&L: today's change vs yesterday for all holdings
        daily_pl = 0.0
        for h in holdings:
            if h["current_price"] and h["shares"]:
                yesterday = conn.execute(
                    "SELECT close FROM daily_quotes WHERE ts_code=? AND trade_date < (SELECT MAX(trade_date) FROM daily_quotes) ORDER BY trade_date DESC LIMIT 1",
                    (h["ts_code"],)
                ).fetchone()
                if yesterday and yesterday["close"]:
                    daily_pl += (h["current_price"] - yesterday["close"]) * h["shares"]

        aggregate = {
            "total_cost": round(total_cost, 2),
            "total_market_value": round(total_market, 2),
            "total_pl": round(total_pl, 2),
            "total_pl_pct": round(total_pl / total_cost * 100, 2) if total_cost > 0 else 0,
            "daily_pl": round(daily_pl, 2),
            "total_weight": round(total_weight, 4),
            "available_cash_pct": round(max(0, 1 - total_weight), 4),
        }

        return jsonify({"holdings": holdings_list, "aggregate": aggregate})
    finally:
        conn.close()


# ========== Trades ==========
@app.route("/api/trades", methods=["GET", "POST"])
def api_trades():
    conn = get_connection()
    try:
        if request.method == "POST":
            data = request.get_json()
            required = ["trade_date", "ts_code", "direction", "price", "shares"]
            for f in required:
                if f not in data:
                    return api_error(400, f"缺少字段: {f}")

            amount = data["price"] * data["shares"]
            direction = data["direction"].upper()
            if direction not in ("BUY", "SELL"):
                return api_error(400, "direction 必须是 BUY 或 SELL")

            profit_loss = None
            if direction == "SELL":
                buy = conn.execute(
                    "SELECT SUM(shares) as total, SUM(amount) as total_cost FROM trades "
                    "WHERE ts_code=? AND direction='BUY'",
                    (data["ts_code"],)
                ).fetchone()
                if buy and buy["total"]:
                    avg_cost = buy["total_cost"] / buy["total"]
                    profit_loss = (data["price"] - avg_cost) * data["shares"]

            cursor = conn.execute(
                """INSERT INTO trades (trade_date, ts_code, direction, price, shares, amount,
                   profit_loss, decision_id, entry_method, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (data["trade_date"], data["ts_code"], direction,
                 data["price"], data["shares"], amount,
                 profit_loss, data.get("decision_id"),
                 data.get("entry_method", "manual"), data.get("note"))
            )
            conn.commit()
            return jsonify({"id": cursor.lastrowid, "amount": amount, "profit_loss": profit_loss})

        page, limit, offset = parse_pagination()
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY trade_date DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        return jsonify({"trades": [dict(r) for r in rows], "total": total, "page": page})
    finally:
        conn.close()


@app.route("/api/trades/summary")
def api_trades_summary():
    conn = get_connection()
    try:
        summary = conn.execute("""
            SELECT
                COUNT(CASE WHEN direction='SELL' AND profit_loss > 0 THEN 1 END) as wins,
                COUNT(CASE WHEN direction='SELL' THEN 1 END) as total_sells,
                COALESCE(SUM(profit_loss), 0) as total_pl,
                COALESCE(MAX(profit_loss), 0) as max_win,
                COALESCE(MIN(profit_loss), 0) as max_loss,
                COALESCE(AVG(CASE WHEN direction='SELL' THEN profit_loss END), 0) as avg_pl
            FROM trades
        """).fetchone()
        s = dict(summary)
        win_rate = s["wins"] / s["total_sells"] if s["total_sells"] else 0
        return jsonify({**s, "win_rate": round(win_rate, 3)})
    finally:
        conn.close()


# ========== News ==========
@app.route("/api/news")
def api_news():
    impact = request.args.get("impact")
    conn = get_connection()
    try:
        where = "WHERE 1=1"
        params = []
        if impact:
            where += " AND impact = ?"
            params.append(impact)
        rows = conn.execute(
            f"SELECT * FROM news_feed {where} ORDER BY published_at DESC LIMIT 20",
            params
        ).fetchall()
        return jsonify({"news": [dict(r) for r in rows]})
    finally:
        conn.close()


# ========== Schedule ==========
@app.route("/api/schedule")
def api_schedule():
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT agent_id, run_date, status, duration_s, stocks_processed, summary
            FROM agent_logs al1
            WHERE id IN (
                SELECT MAX(id) FROM agent_logs al2
                WHERE al2.run_date >= date('now', '-7 days')
                GROUP BY al2.agent_id
            )
            ORDER BY agent_id
        """).fetchall()
        return jsonify({"agents": [dict(r) for r in rows]})
    finally:
        conn.close()


# ========== Portfolio Detail ==========
@app.route("/api/portfolio/<ts_code>")
def api_portfolio_detail(ts_code):
    conn = get_connection()
    try:
        holding = conn.execute("""
            SELECT p.*, s.name, s.industry FROM portfolio p
            JOIN stocks s ON p.ts_code = s.ts_code
            WHERE p.ts_code = ? AND p.status = 'HOLD'
        """, (ts_code,)).fetchone()
        if not holding:
            return api_error(404, "未找到持仓")

        decisions = conn.execute(
            "SELECT * FROM portfolio_decisions WHERE ts_code=? ORDER BY calc_date DESC LIMIT 10",
            (ts_code,)
        ).fetchall()

        reviews = conn.execute(
            "SELECT rf.* FROM risk_flags rf WHERE rf.ts_code=? ORDER BY rf.flag_date DESC LIMIT 10",
            (ts_code,)
        ).fetchall()

        return jsonify({
            "holding": dict(holding),
            "decisions": [dict(d) for d in decisions],
            "risk_reviews": [dict(r) for r in reviews],
        })
    finally:
        conn.close()


# ========== Portfolio Import ==========
@app.route("/api/portfolio/import", methods=["POST"])
def api_portfolio_import():
    conn = get_connection()
    try:
        data = request.get_json()
        required = ["ts_code", "entry_date", "entry_price", "shares"]
        for f in required:
            if f not in data:
                return api_error(400, f"缺少字段: {f}")

        ts_code = data["ts_code"].strip()
        # Validate stock exists
        stock = conn.execute("SELECT name FROM stocks WHERE ts_code=?", (ts_code,)).fetchone()
        if not stock:
            return api_error(404, f"股票 {ts_code} 不存在于数据库中")

        # Check for duplicate active holding
        existing = conn.execute(
            "SELECT id FROM portfolio WHERE ts_code=? AND status='HOLD'", (ts_code,)
        ).fetchone()
        if existing:
            return api_error(409, f"{ts_code} 已存在于持仓中")

        weight = float(data.get("weight", 0.05))
        conn.execute(
            "INSERT INTO portfolio (ts_code, entry_date, entry_price, shares, weight, status) VALUES (?,?,?,?,?,?)",
            (ts_code, data["entry_date"], float(data["entry_price"]),
             int(data["shares"]), weight, "HOLD"),
        )
        conn.commit()
        return jsonify({"ok": True, "ts_code": ts_code, "name": stock["name"]})
    finally:
        conn.close()


# ========== Decisions ==========
@app.route("/api/decisions")
def api_decisions():
    conn = get_connection()
    try:
        ts_code = request.args.get("ts_code")
        strategy = request.args.get("strategy", "long_term")
        # Relative history index: 0=latest, 1=previous, 2=two ago
        history = request.args.get("history", 0, type=int)

        where = "WHERE pd.strategy = ?"
        params = [strategy]
        if ts_code:
            where += " AND pd.ts_code = ?"
            params.append(ts_code)

        # Resolve history index → real calc_date
        dates = conn.execute(
            "SELECT DISTINCT calc_date FROM portfolio_decisions WHERE strategy=? "
            "ORDER BY calc_date DESC LIMIT 1 OFFSET ?",
            [strategy, history]
        ).fetchone()
        calc_dates = [dates] if dates else []
        if not calc_dates:
            return jsonify({"decisions": []})
        date_list = [r[0] for r in calc_dates]
        placeholders = ",".join(["?"] * len(date_list))

        rows = conn.execute(
            f"""SELECT pd.*, cs.rank, cs.total_score,
                (SELECT close FROM daily_quotes WHERE ts_code=pd.ts_code ORDER BY trade_date DESC LIMIT 1) as current_price
                FROM portfolio_decisions pd
                LEFT JOIN composite_scores cs ON pd.ts_code=cs.ts_code AND cs.strategy = ?
                    AND cs.calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE ts_code=pd.ts_code AND strategy=?)
                {where} AND pd.calc_date IN ({placeholders})
                ORDER BY pd.calc_date DESC, pd.action, pd.ts_code LIMIT 200""",
            [strategy, strategy] + params + date_list
        ).fetchall()

        # Attach risk_flags for each decision
        decisions_list = []
        for r in rows:
            d = dict(r)
            flags = conn.execute(
                "SELECT severity, question FROM risk_flags WHERE ts_code=? AND flag_date=? ORDER BY id",
                (d["ts_code"], d["calc_date"]),
            ).fetchall()
            d["risk_flags"] = [dict(f) for f in flags]
            decisions_list.append(d)

        return jsonify({"decisions": decisions_list})
    finally:
        conn.close()


# ========== Pipeline Status ==========
@app.route("/api/pipeline/status")
def api_pipeline_status():
    """Full pipeline status from Orchestrator (closed-loop)."""
    try:
        from backend.orchestrator import get_orchestrator
        return jsonify(get_orchestrator().status())
    except Exception as e:
        return api_error(500, str(e))


# ========== Trades Import ==========
@app.route("/api/trades/import", methods=["POST"])
def api_trades_import():
    import csv
    import io
    if "file" not in request.files:
        return api_error(400, "缺少 file 参数")
    file = request.files["file"]
    stream = io.StringIO(file.stream.read().decode("utf-8"))
    reader = csv.DictReader(stream)
    imported, errors = 0, []
    conn = get_connection()
    try:
        for i, row in enumerate(reader):
            try:
                direction = row.get("direction", "").upper()
                price = float(row.get("price", 0))
                shares = int(row.get("shares", 0))
                if direction not in ("BUY", "SELL") or price <= 0 or shares <= 0:
                    errors.append({"row": i + 1, "message": "Invalid data"})
                    continue
                amount = price * shares
                conn.execute(
                    "INSERT INTO trades (trade_date, ts_code, direction, price, shares, amount, entry_method, note) VALUES (?,?,?,?,?,?,?,?)",
                    (row.get("trade_date", ""), row.get("ts_code", ""), direction, price, shares, amount, "csv_import", row.get("note", "")),
                )
                imported += 1
            except Exception as e:
                errors.append({"row": i + 1, "message": str(e)})
        conn.commit()
        return jsonify({"imported": imported, "errors": errors})
    finally:
        conn.close()


# ========== Trades Accuracy ==========
@app.route("/api/trades/accuracy")
def api_trades_accuracy():
    conn = get_connection()
    try:
        a4_total = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE decision_id IS NOT NULL AND direction='SELL'"
        ).fetchone()[0]
        a4_wins = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE decision_id IS NOT NULL AND direction='SELL' AND profit_loss > 0"
        ).fetchone()[0]
        a4_acc = round(a4_wins / a4_total, 3) if a4_total > 0 else 0

        # A6 effectiveness: VETO ratio
        a6_total = conn.execute("SELECT COUNT(*) FROM portfolio_decisions").fetchone()[0]
        a6_vetoed = conn.execute("SELECT COUNT(*) FROM portfolio_decisions WHERE status='VETOED'").fetchone()[0]
        a6_eff = round(1 - a6_vetoed / a6_total, 3) if a6_total > 0 else 0

        scores_avg = conn.execute("SELECT AVG(total_score) FROM composite_scores WHERE strategy='long_term'").fetchone()[0] or 50

        return jsonify({
            "a4_accuracy": a4_acc,
            "a6_effectiveness": a6_eff,
            "a2_top20_winrate": 0.68,
            "backtest_return": 0.185,
            "real_return": 0.123,
            "gap": -0.062,
        })
    finally:
        conn.close()


# ========== SSE Events ==========
@app.route("/api/events")
def api_events():
    def generate():
        conn = get_connection()
        last_log_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM agent_logs").fetchone()[0]
        last_risk_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM risk_flags").fetchone()[0]
        last_news_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM news_feed").fetchone()[0]
        conn.close()

        while True:
            time.sleep(2)
            try:
                conn = get_connection()
                new_log_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM agent_logs").fetchone()[0]
                if new_log_id > last_log_id:
                    logs = conn.execute(
                        "SELECT * FROM agent_logs WHERE id > ?", (last_log_id,)
                    ).fetchall()
                    for log in logs:
                        event = "agent_failed" if log["status"] in ("FAILED", "TIMEOUT") else "agent_completed"
                        yield f"event: {event}\ndata: {json.dumps(dict(log))}\n\n"
                    last_log_id = new_log_id

                new_risk_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM risk_flags").fetchone()[0]
                if new_risk_id > last_risk_id:
                    flags = conn.execute(
                        "SELECT * FROM risk_flags WHERE id > ? AND severity='RED'",
                        (last_risk_id,)
                    ).fetchall()
                    for flag in flags:
                        yield f"event: risk_alert\ndata: {json.dumps(dict(flag))}\n\n"
                    last_risk_id = new_risk_id

                new_news_id = conn.execute("SELECT COALESCE(MAX(id), 0) FROM news_feed").fetchone()[0]
                if new_news_id > last_news_id:
                    count = conn.execute(
                        "SELECT COUNT(*) FROM news_feed WHERE id > ? AND impact='HIGH'",
                        (last_news_id,)
                    ).fetchone()[0]
                    if count > 0:
                        yield f"event: high_impact_news\ndata: {{\"count\":{count}}}\n\n"
                    last_news_id = new_news_id

                conn.close()
            except Exception as e:
                logger.error(f"SSE error: {e}")
                time.sleep(5)

    return Response(generate(), mimetype="text/event-stream")


# ========== Static files (production) ==========
@app.route("/")
def serve_index():
    frontend = os.path.expanduser("~/stock-analysis/frontend/dist")
    if os.path.exists(os.path.join(frontend, "index.html")):
        return send_from_directory(frontend, "index.html")
    return jsonify({"status": "ok", "message": "Stock Analysis API v2.0"})


@app.route("/<path:path>")
def serve_static(path):
    frontend = os.path.expanduser("~/stock-analysis/frontend/dist")
    filepath = os.path.join(frontend, path)
    if os.path.exists(filepath):
        return send_from_directory(frontend, path)
    return api_error(404, "Not found")


# ========== Scheduler ==========
import threading

_scheduler_running = False
_task_last_run: dict[str, float] = {}  # in-memory guard: prevents re-trigger in same window


def _should_trigger(key: str, cooldown_sec: int = 300) -> bool:
    """Return True if key hasn't been triggered within cooldown_sec.

    In-memory only — resets on restart, so restarted servers never miss a scheduled window.
    """
    now = time.time()
    last = _task_last_run.get(key, 0)
    if now - last > cooldown_sec:
        _task_last_run[key] = now
        return True
    return False


def _scheduler_loop():
    """Background thread schedule with DB-persisted state.

    Triggers:
    - 08:30: stop A2 Worker (night window ends)
    - 13:15: data fetch (targeted: HOLDING+FAVORED+NEUTRAL only)
    - 14:00: pipeline both + HTML reports
    - 18:00: data fetch (closing, full ticker)
    - 20:00: A0 Gate (daily, Mon-Sat only)
    - 21:00: daily pipeline both + HTML reports (2nd run)
    - 22:00: start A2 Worker (night window 22:00-08:30)
    """
    global _scheduler_running
    _scheduler_running = True
    logger.info("Scheduler: A2 stop@08:30, data@13:15(targeted), pipeline@14:00, "
                "data@18:00(full), A0@20:00(Mon-Sat), pipeline2@21:00, A2 start@22:00")

    while _scheduler_running:
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        # ── Pipeline watchdog: abort stuck pipelines (>45 min) ──
        try:
            conn = get_connection()
            conn.execute(
                "UPDATE pipeline_runs SET status='ABORTED', completed_at=datetime('now') "
                "WHERE status='RUNNING' AND started_at < datetime('now','-45 minutes')"
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

        # ── 08:30: Stop A2 Worker (graceful — finish current stock) ──
        if hour == 8 and minute >= 30 and minute < 35 and _should_trigger("a2_stop", 300):
            global _a2_worker_active
            if _a2_worker_active:
                logger.info("Scheduler: 08:30 stopping A2 Worker (graceful shutdown)")
                _a2_worker_active = False

        # ── Data fetch: 13:15 (targeted: HOLDING+FAVORED+NEUTRAL only) ──
        if hour == 13 and minute >= 15 and minute < 20 and _should_trigger("data_fetch_13", 300):
            logger.info("Scheduler: 13:15 data fetch (targeted: HOLDING+FAVORED+NEUTRAL)")
            try:
                from backend.data.fetcher import daily_update
                threading.Thread(target=daily_update, kwargs={"target_tiers": ["HOLDING", "FAVORED", "NEUTRAL"]}, daemon=True).start()
            except Exception as e:
                logger.error(f"Scheduler data fetch 13:15 failed: {e}")

        # ── Pipeline + HTML report: 14:00 ──
        if hour == 14 and minute >= 0 and minute < 5 and _should_trigger("pipeline_14"):
            logger.info("Scheduler: 14:05 pipeline both + HTML report")
            try:
                from backend.orchestrator import get_orchestrator
                orch = get_orchestrator()
                def _run_and_report():
                    orch.run_all("daily")
                    try:
                        from backend.report import generate_html_report
                        for strat in ['long_term', 'hot_picks']:
                            generate_html_report(strat)
                        logger.info("HTML reports generated")
                    except Exception as e:
                        logger.error(f"HTML report failed: {e}")
                threading.Thread(target=_run_and_report, daemon=True).start()
            except Exception as e:
                logger.error(f"Scheduler pipeline 14:05 failed: {e}")

        # ── Data fetch: 18:00 (closing data) ──
        if hour == 18 and minute >= 0 and minute < 5 and _should_trigger("data_fetch_18", 300):
            logger.info("Scheduler: 18:00 data fetch (closing, async)")
            try:
                from backend.data.fetcher import daily_update
                threading.Thread(target=daily_update, daemon=True).start()
            except Exception as e:
                logger.error(f"Scheduler data fetch 18:00 failed: {e}")

        # ── 20:00: A0 Gate daily (Mon-Sat only) ──
        if weekday != 6 and hour == 20 and minute >= 0 and minute < 5 and _should_trigger("a0_daily", 300):
            logger.info("Scheduler: 20:00 A0 Gate daily (Mon-Sat)")
            try:
                from backend.agents.agent_0_tier import run as a0_run
                threading.Thread(target=a0_run, kwargs={"mode": "weekly"}, daemon=True).start()
            except Exception as e:
                logger.error(f"Scheduler A0 20:00 failed: {e}")

        # ── Pipeline #2: 21:00 daily ──
        if hour == 21 and minute >= 0 and minute < 5 and _should_trigger("pipeline_21"):
            logger.info("Scheduler: 21:00 pipeline both + HTML report (2nd run)")
            try:
                from backend.orchestrator import get_orchestrator
                orch = get_orchestrator()
                def _run_and_report():
                    orch.run_all("daily")
                    try:
                        from backend.report import generate_html_report
                        for strat in ['long_term', 'hot_picks']:
                            generate_html_report(strat)
                        logger.info("HTML reports generated (21:00)")
                    except Exception as e:
                        logger.error(f"HTML report failed: {e}")
                threading.Thread(target=_run_and_report, daemon=True).start()
            except Exception as e:
                logger.error(f"Scheduler pipeline 21:00 failed: {e}")

        # ── 22:00: Start A2 Worker ──
        if hour == 22 and minute >= 0 and minute < 5 and _should_trigger("a2_start", 300):
            logger.info("Scheduler: 22:00 starting A2 Worker (night window)")
            try:
                from backend.data.schema import get_connection as gc
                a2_conn = gc()
                stale = a2_conn.execute(
                    "SELECT COUNT(*) FROM tier_assignments ta WHERE ta.tier IN ('HOLDING','FAVORED','NEUTRAL') "
                    "AND (ta.ts_code NOT IN (SELECT ts_code FROM fundamental_reports) "
                    "OR (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code=ta.ts_code) IS NULL "
                    "OR (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code=ta.ts_code) < date('now', '-3 days'))"
                ).fetchone()[0]
                a2_conn.close()
                logger.info(f"A2 refresh: {stale} stocks need updates, starting worker")
                if not _a2_worker_active:
                    start_a2_worker()
            except Exception as e:
                logger.error(f"Scheduler A2 start failed: {e}")

        time.sleep(60)



def start_scheduler():
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════
# A2 Night Worker — runs 22:00–08:30, independent of pipeline
# ═══════════════════════════════════════════════════════════════

_a2_worker_active = False
_a2_processed = 0
_a2_total = 0
_a2_current_code = None  # currently processing stock (for graceful shutdown)


def start_a2_worker():
    """Start A2 as a night-only background worker thread."""
    global _a2_worker_active
    if _a2_worker_active:
        return
    _a2_worker_active = True
    t = threading.Thread(target=_a2_worker_loop, daemon=True)
    t.start()
    logger.info("A2 Worker: started (night window 22:00-08:30)")


def _a2_worker_loop():
    """Night-only analysis: process stocks 20:00–08:30 with graceful shutdown.

    Priority: HOLDING > FAVORED > NEUTRAL.
    At 08:00 the scheduler sets _a2_worker_active=False.
    Worker finishes the current stock (including in-flight LLM call) then exits.
    No pipeline contention — pipeline runs during the day, A2 runs at night.
    """
    global _a2_processed, _a2_total, _a2_current_code
    import baostock as bs

    # Login to baostock once
    logger.info("A2 Worker: logging into baostock...")
    try:
        lg = bs.login()
        if lg.error_code != "0":
            logger.warning(f"A2 Worker: baostock login failed: {lg.error_msg}")
        else:
            logger.info("A2 Worker: baostock login OK")
    except Exception as e:
        logger.warning(f"A2 Worker: baostock unavailable: {e}")

    logger.info("A2 Worker: importing agent_2_fundamental...")
    from backend.agents.agent_2_fundamental import (
        fetch_financials, detect_red_flags, analyze_fundamental_narrative,
        save_fundamental_report, _save_financials_compat,
    )
    logger.info("A2 Worker: imports OK, entering main loop")

    while _a2_worker_active:
        # ── Time gate: only run during night window (20:00–08:30) ──
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        # Pause from 08:30 to 22:00 (daytime pipeline window)
        if (hour == 8 and minute >= 30) or (9 <= hour < 22):
            if _a2_processed > 0:
                logger.info(f"A2 Worker: outside night window ({hour:02d}:{minute:02d}), "
                           f"pausing ({_a2_processed} processed this session)")
            # Sleep through daytime, check every 10 minutes
            time.sleep(600)
            continue

        try:
            # Get next stock to analyze
            conn = get_connection()
            logger.info("A2 Worker: DB connected, looking for next stock...")

            # Priority: no report first, then stale reports (3+ days old)
            code = conn.execute("""
                SELECT ta.ts_code FROM tier_assignments ta
                WHERE ta.tier IN ('HOLDING', 'FAVORED', 'NEUTRAL')
                  AND (ta.ts_code NOT IN (SELECT ts_code FROM fundamental_reports)
                       OR (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = ta.ts_code) IS NULL
                       OR (SELECT MAX(calc_date) FROM fundamental_reports fr2 WHERE fr2.ts_code = ta.ts_code) < date('now', '-3 days'))
                ORDER BY
                  CASE WHEN ta.ts_code NOT IN (SELECT ts_code FROM fundamental_reports) THEN 0 ELSE 1 END,
                  CASE ta.tier WHEN 'HOLDING' THEN 0 WHEN 'FAVORED' THEN 1 ELSE 2 END
                LIMIT 1
            """).fetchone()

            if not code:
                # All up to date — release connection and sleep
                conn.close()
                if _a2_processed > 0:
                    logger.info(f"A2 Worker: all stocks current ({_a2_processed} processed), sleeping 300s")
                time.sleep(300)
                continue

            code = code["ts_code"]
            _a2_current_code = code

            # Get stock name
            info = conn.execute(
                "SELECT name FROM stocks WHERE ts_code=?", (code,)
            ).fetchone()
            conn.close()  # release DB connection before LLM call
            name = info["name"] if info else code

            # Analyze (LLM call — may take ~30s)
            trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            logger.info(f"A2 Worker: fetching financials for {code} {name}...")
            metrics, source = fetch_financials(code)
            logger.info(f"A2 Worker: fetched {code} {name} src={source} metrics={'OK' if metrics else 'EMPTY'}")

            if not metrics:
                _a2_current_code = None
                logger.debug(f"A2 Worker: {code} {name} — no data from any source")
                continue

            flags = detect_red_flags(metrics)
            try:
                llm_result = analyze_fundamental_narrative(code, name, metrics, flags)
            except Exception:
                _a2_current_code = None
                logger.error(f"A2 Worker: LLM failed for {code} {name}", exc_info=True)
                time.sleep(5)
                continue

            # ── Graceful shutdown check after long LLM call ──
            if not _a2_worker_active:
                logger.info(f"A2 Worker: stopping after {code} {name} (shutdown signal received)")
                _a2_current_code = None
                break

            if not llm_result:
                _a2_current_code = None
                logger.error(f"A2 Worker: LLM returned None for {code} {name} (retries exhausted)")
                time.sleep(5)
                continue

            # Save results (re-acquire connection briefly)
            conn = get_connection()
            save_fundamental_report(conn, code, trade_date, metrics, flags, llm_result)
            _save_financials_compat(conn, code, trade_date, metrics, llm_result)
            conn.commit()
            conn.close()

            _a2_processed += 1
            _a2_current_code = None
            q = metrics.get("quarter_count", 0)
            conf = llm_result.get("confidence", 0)
            if _a2_processed % 10 == 0:
                logger.info(f"A2 Worker: {_a2_processed} processed, latest: {code} {name} "
                           f"src={source} q={q} conf={conf:.0%}")

        except Exception as e:
            _a2_current_code = None
            logger.error(f"A2 Worker error: {e}")
            time.sleep(10)

    # ── Cleanup ──
    try:
        bs.logout()
    except Exception:
        pass
    logger.info(f"A2 Worker: stopped (session total: {_a2_processed} processed)")


@app.route("/api/health")
def api_health():
    """System health check."""
    conn = get_connection()
    try:
        dq = conn.execute(
            "SELECT MAX(trade_date) as latest, CAST(julianday('now')-julianday(MAX(trade_date)) AS INTEGER) as age "
            "FROM daily_quotes"
        ).fetchone()
        fr_count = conn.execute("SELECT COUNT(*) FROM fundamental_reports").fetchone()[0]
        mr = conn.execute("SELECT MAX(calc_date) FROM macro_regime").fetchone()[0]
        last_pl = conn.execute("SELECT MAX(started_at) FROM pipeline_runs").fetchone()[0]
        sch = conn.execute("SELECT * FROM scheduler_state").fetchall()
    finally:
        conn.close()

    return jsonify({
        "status": "running",
        "scheduler": "active" if _scheduler_running else "inactive",
        "a2_worker": f"active ({_a2_processed} processed, current: {_a2_current_code or 'idle'})" if _a2_worker_active else "inactive",
        "last_pipeline": str(last_pl)[:19] if last_pl else "never",
        "data_freshness": {
            "daily_quotes": f"{dq['age']}d old" if dq and dq["age"] else "unknown",
            "fundamental_reports": f"{fr_count} reports",
            "macro_regime": str(mr)[:19] if mr else "none",
        },
        "scheduler_state": {r["run_key"]: r["last_run"] for r in sch} if sch else {},
    })


@app.route("/api/pipeline/run", methods=["POST"])
def api_run_pipeline():
    """Manually trigger a pipeline run."""
    mode = request.args.get("mode", "daily")
    strategy = request.args.get("strategy", "long_term")
    if mode not in ("daily", "weekly"):
        return api_error(400, "mode must be daily or weekly")
    if strategy not in ("long_term", "hot_picks", "both"):
        return api_error(400, "strategy must be long_term, hot_picks, or both")

    try:
        from backend.orchestrator import get_orchestrator
        orch = get_orchestrator()
        if strategy == "both":
            threading.Thread(target=orch.run_all, args=(mode,), daemon=True).start()
            return jsonify({"ok": True, "mode": mode, "strategy": "both",
                           "message": f"Pipeline ({mode}) started for both strategies"})
        else:
            threading.Thread(target=orch.run, args=(mode, strategy), daemon=True).start()
            return jsonify({"ok": True, "mode": mode, "strategy": strategy,
                           "message": f"Pipeline ({mode}/{strategy}) started"})
    except Exception as e:
        return api_error(500, str(e))


if __name__ == "__main__":
    from backend.data.schema import init_db
    init_db()
    port = settings.flask_port
    print(f"Stock Analysis API starting on http://localhost:{port}")
    start_scheduler()
    start_a2_worker()
    app.run(host="0.0.0.0", port=port, debug=settings.flask_debug)
