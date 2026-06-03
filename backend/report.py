"""HTML report generator for pipeline results."""
import json
import os
from datetime import datetime
from backend.data.schema import get_connection
from backend.lib.logging import setup_logging

setup_logging()
import logging
logger = logging.getLogger(__name__)

REPORT_DIR = os.path.expanduser("~/stock-analysis/report")

CSS = """
body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }
h1 { color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 10px; }
h2 { color: #16213e; margin-top: 30px; }
.section { background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { background: #1a1a2e; color: white; padding: 8px 6px; text-align: left; position: sticky; top: 0; }
td { padding: 6px; border-bottom: 1px solid #eee; }
tr:hover { background: #f0f4ff; }
.pos { color: #d32f2f; font-weight: bold; }
.neg { color: #2e7d32; }
.accel { background: #e8f5e9; }
.sustain { background: #fff3e0; }
.decel { background: #fce4ec; }
.reverse { background: #ffebee; }
.reject { color: #c62828; }
.include { color: #2e7d32; font-weight: bold; }
.macro { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 8px; }
.narrative { color: #555; font-style: italic; margin: 10px 0; padding: 10px; background: #fafafa; border-left: 3px solid #667eea; }
.stats { display: flex; gap: 15px; flex-wrap: wrap; }
.stat { background: #1a1a2e; color: white; padding: 10px 15px; border-radius: 6px; min-width: 80px; text-align: center; }
.stat .val { font-size: 24px; font-weight: bold; }
.stat .lbl { font-size: 11px; opacity: 0.8; }
"""


def generate_html_report(strategy="long_term", trade_date=None):
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ── Macro ──
    mr = conn.execute("SELECT * FROM macro_regime ORDER BY calc_date DESC LIMIT 1").fetchone()
    macro = dict(mr) if mr else {}
    macro_r = {}
    mrp = conn.execute("SELECT report_json FROM macro_reports ORDER BY calc_date DESC LIMIT 1").fetchone()
    if mrp:
        try: macro_r = json.loads(mrp["report_json"])
        except: pass

    # ── A5 Fusion ──
    fr = conn.execute("SELECT report_json FROM fusion_reports WHERE strategy=? ORDER BY calc_date DESC LIMIT 1", (strategy,)).fetchone()
    fusion = json.loads(fr["report_json"]) if fr else {}

    # ── Focus List ──
    fl_rows = conn.execute("SELECT fl.ts_code, fl.name, s.industry, fl.total_score, fl.rank, fl.tier FROM focus_list fl LEFT JOIN stocks s ON fl.ts_code=s.ts_code WHERE fl.strategy=? ORDER BY fl.position", (strategy,)).fetchall()
    fl_cats = conn.execute("SELECT tier, COUNT(*) n FROM focus_list WHERE strategy=? GROUP BY tier ORDER BY n DESC", (strategy,)).fetchall()

    # ── A5 Top 50: Focus List stocks sorted by total_score (what enters the pipeline) ──
    max_cs = conn.execute("SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?", (strategy,)).fetchone()[0]
    fl_codes = [r['ts_code'] for r in fl_rows]
    if fl_codes:
        ph = ",".join("?" * len(fl_codes))
        a5_top = conn.execute(
            f"SELECT s.name, cs.* FROM composite_scores cs JOIN stocks s ON cs.ts_code=s.ts_code "
            f"WHERE cs.strategy=? AND cs.calc_date=? AND cs.ts_code IN ({ph}) "
            f"ORDER BY cs.total_score DESC LIMIT 50",
            [strategy, max_cs] + fl_codes).fetchall()
    else:
        a5_top = []

    # ── A7 Decisions (BUY + REJECT) ──
    max_pd = conn.execute("SELECT MAX(calc_date) FROM portfolio_decisions WHERE strategy=?", (strategy,)).fetchone()[0]
    a7_all = conn.execute(
        "SELECT pd.ts_code, s.name, pd.action, pd.reason, pd.review_json FROM portfolio_decisions pd "
        "JOIN stocks s ON pd.ts_code=s.ts_code "
        "WHERE pd.strategy=? AND pd.calc_date=? AND pd.action IN ('BUY','REJECT') "
        "ORDER BY CAST(json_extract(pd.review_json, '$.a7.conviction') AS REAL) DESC",
        (strategy, max_pd)).fetchall()

    a7_report = {}
    pr = conn.execute("SELECT report_json FROM portfolio_reports WHERE strategy=? ORDER BY calc_date DESC LIMIT 1", (strategy,)).fetchone()
    if pr:
        try: a7_report = json.loads(pr["report_json"])
        except: pass

    conn.close()

    includes = [r for r in a7_all if _a7(r, "recommendation") == "INCLUDE"]
    rejects = [r for r in a7_all if _a7(r, "recommendation") == "REJECT"]

    # ── Render HTML ──
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{strategy} Pipeline Report — {trade_date[:16]}</title>
<style>{CSS}</style></head><body>
<h1>Stock Analysis Pipeline Report</h1>
<p>Strategy: <b>{strategy}</b> | Generated: {trade_date[:16]}</p>

<div class="macro section">
<h2>Macro Environment</h2>
<div class="stats">
<div class="stat"><div class="val">{macro.get('regime','?')}</div><div class="lbl">Regime</div></div>
<div class="stat"><div class="val" style="font-size:11px">{macro.get('regime_summary','')[:40]}</div><div class="lbl">宏观判断</div></div>
<div class="stat"><div class="val">{fl_cats[0][1] if fl_cats else 0}</div><div class="lbl">Focus List</div></div>
<div class="stat"><div class="val">{len(includes)}</div><div class="lbl">Included</div></div>
<div class="stat"><div class="val">{len(rejects)}</div><div class="lbl">Rejected</div></div>
</div>
<div class="narrative">{str(macro_r.get('narrative',''))[:500]}</div>
</div>
"""

    # ── A5 Section: Top 50 FL stocks by score (these feed into A7) ──
    html += f"""<div class="section">
<h2>A5 Multi-Factor — Focus List Top 50 by Score</h2>
<p class="narrative">{str(fusion.get('overall_narrative',''))[:500]}</p>
<p>From the {len(fl_rows)}-stock Focus List, sorted by A5 total_score. These are the top-ranked candidates that enter A7.</p>
<table><tr>
<th>#</th><th>Code</th><th>Name</th><th>Score</th><th>Type</th><th>Cat</th>
<th>T</th><th>F</th><th>M</th>
<th>d3%</th><th>d5%</th><th>d20%</th><th>d60%</th><th>Acc</th>
<th>Acc</th></tr>"""
    for i, r in enumerate(a5_top, 1):
        acc = r['momentum_accel'] or 0
        row_cls = 'accel' if acc > 5 else 'sustain' if acc > -8 else 'decel' if acc > -20 else 'reverse'
        fl_cat = next((fr['tier'] for fr in fl_rows if fr['ts_code'] == r['ts_code']), '?')
        html += f"""<tr class="{row_cls}">
<td>{i}</td><td>{r['ts_code']}</td><td>{r['name']}</td><td><b>{r['total_score']:.0f}</b></td>
<td>{r['trend_type'] or '?'}</td><td>{fl_cat}</td>
<td>{r['tech_score']:.0f}</td><td>{r['fundamental_score']:.0f}</td><td>{_moms(r)}</td>
<td class="{'pos' if (r['momentum_d3'] or 0)>0 else 'neg'}">{r['momentum_d3']:+.1f}</td>
<td class="{'pos' if (r['momentum_d5'] or 0)>0 else 'neg'}">{r['momentum_d5']:+.1f}</td>
<td class="{'pos' if (r['momentum_d20'] or 0)>0 else 'neg'}">{r['momentum_d20']:+.1f}</td>
<td class="{'pos' if (r['momentum_d60'] or 0)>0 else 'neg'}">{r['momentum_d60']:+.1f}</td>
<td><b>{acc:+.1f}</b></td></tr>"""
    html += "</table></div>"

    # ── Focus List: full breakdown ──
    html += f"""<div class="section"><h2>Focus List — {len(fl_rows)} stocks / {len(fl_cats)} categories</h2>
<div class="stats">"""
    for cat, n in fl_cats:
        html += f'<div class="stat"><div class="val">{n}</div><div class="lbl">{cat}</div></div>'
    html += """</div>
<p>The Focus List is the candidate pool that feeds into A7. Stocks are classified by A2 quality + momentum.</p>
<table><tr><th>#</th><th>Code</th><th>Name</th><th>Category</th><th>Score</th></tr>"""
    for i, r in enumerate(fl_rows, 1):
        html += f"<tr><td>{i}</td><td>{r['ts_code']}</td><td>{r['name']}</td><td><b>{r['tier']}</b></td><td>{r['total_score']:.0f}</td></tr>"
    html += "</table></div>"

    # ── A7 Included + A6 review ──
    html += f"""<div class="section"><h2>A7 Portfolio — {len(includes)} Included</h2>
<p class="narrative">{a7_report.get('a7',{}).get('portfolio_narrative','')[:400]}</p>
<table><tr><th>Code</th><th>Name</th><th>Conv</th><th>Weight</th><th>Tier</th><th>Rationale</th><th>A6 Risk</th><th>A6 Verdict</th><th>A6 Reasoning</th></tr>"""
    for r in includes:
        a7 = json.loads(r['review_json']).get('a7', {})
        a6 = json.loads(r['review_json']).get('a6', {})
        risk = a6.get('risk_score', '?')
        verdict = a6.get('final_verdict', '?')
        a6_reason = a6.get('reasoning', '')[:150]
        verdict_cls = 'include' if verdict == 'APPROVED' else 'reject'
        html += f"""<tr>
<td>{r['ts_code']}</td><td>{r['name']}</td>
<td class="include">{a7.get('conviction',0):.3f}</td>
<td><b>{a7.get('weight',0):.0%}</b></td><td>{a7.get('tier','?')}</td>
<td>{a7.get('rationale','')[:200]}</td>
<td class="include">{risk}/5</td>
<td class="{verdict_cls}"><b>{verdict}</b></td>
<td>{a6_reason}</td></tr>"""
    html += "</table></div>"

    # ── A7 Rejected (top 20) ──
    html += f"""<div class="section"><h2>A7 Rejected — Top 20 by Conviction</h2>
<table><tr><th>Code</th><th>Name</th><th>Conv</th><th>Rationale</th></tr>"""
    for r in rejects[:20]:
        a7 = json.loads(r['review_json']).get('a7', {})
        html += f"""<tr class="reject">
<td>{r['ts_code']}</td><td>{r['name']}</td>
<td>{a7.get('conviction',0):.3f}</td>
<td>{a7.get('rationale','')[:200]}</td></tr>"""
    html += "</table></div>"

    # ── A6 Risk Review (adversarial-reviewed: all BUY + 30% sampled REJECT) ──
    a6_reviewed = []
    for r in a7_all:
        try:
            a6 = json.loads(r['review_json']).get('a6', {})
            # Include only decisions that went through LLM adversarial review
            # (rule-only fallback decisions have confidence=0.3 and "仅规则审查" reasoning)
            if a6 and a6.get('llm_confidence', 0) > 0.3:
                a6_reviewed.append((r, a6))
        except: pass

    a6_approved = [(r, a) for r, a in a6_reviewed if a.get('final_verdict') == 'APPROVED']
    a6_vetoed = [(r, a) for r, a in a6_reviewed if a.get('final_verdict') == 'VETOED']

    if a6_reviewed:
        html += f"""<div class="section"><h2>A6 Risk Review — {len(a6_approved)} Approved + {len(a6_vetoed)} Vetoed</h2>"""
        if a6_vetoed:
            html += """<h3>⚠️ Vetoed</h3><table><tr><th>Code</th><th>Name</th><th>Risk</th><th>Recommendation</th><th>Reasoning</th><th>Veto Reason</th></tr>"""
            for r, a in a6_vetoed:
                html += f"""<tr class="reject">
<td>{r['ts_code']}</td><td>{r['name']}</td>
<td>{a.get('risk_score','?')}/5</td><td>{a.get('recommendation','?')}</td>
<td>{a.get('reasoning','')[:200]}</td><td>{a.get('veto_reason','')}</td></tr>"""
            html += "</table>"
        if a6_approved:
            html += """<h3>✅ Approved</h3><table><tr><th>Code</th><th>Name</th><th>Risk</th><th>Recommendation</th><th>Reasoning</th><th>Rule Checks</th></tr>"""
            for r, a in a6_approved:
                rules = "; ".join(f"{c['severity']}/{c['dim']}" for c in a.get('rule_checks', [])[:4])
                html += f"""<tr class="include">
<td>{r['ts_code']}</td><td>{r['name']}</td>
<td>{a.get('risk_score','?')}/5</td><td>{a.get('recommendation','?')}</td>
<td>{a.get('reasoning','')[:200]}</td><td>{rules}</td></tr>"""
            html += "</table>"
        html += "</div>"

    html += "</body></html>"

    # ── Write ──
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"{REPORT_DIR}/{strategy}_{ts}.html"
    with open(filename, 'w') as f:
        f.write(html)
    logger.info(f"Report: {filename} ({len(html)} chars)")
    return filename


def _a7(row, key):
    try:
        return json.loads(row['review_json']).get('a7', {}).get(key)
    except:
        return None


def _moms(r):
    """Compute composite momentum for display"""
    d3 = r['momentum_d3'] or 0; d5 = r['momentum_d5'] or 0
    d20 = r['momentum_d20'] or 0; d60 = r['momentum_d60'] or 0
    return f"{(d3+d5+d20+d60)/4:.0f}"


if __name__ == "__main__":
    for s in ['long_term', 'hot_picks']:
        generate_html_report(s)
