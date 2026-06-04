"""
Focus List Manager — scenario-based stock classification and selection.

Classifier categories:
  长期池: 价值型 | 成长型 | 稳健型  (fundamentals-driven, moderate momentum)
  短期池: 动量型 | 突破型           (momentum-driven, quality floor)
  观望型: insufficient data or red flags

Strategy mapping:
  long_term → 长期池 (4:3:3 ratio)
  hot_picks → 短期池 (1:1 ratio)
"""
import json
import logging
from datetime import datetime
from collections import defaultdict

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

TOP_PCT = 0.12     # default, overridden per strategy in rebuild()
MIN_TOTAL_SCORE = 25  # quality floor for focus list inclusion


def classify_stock(r, a2_report=None, momentum_d3=None, momentum_d5=None, momentum_d20=None,
                   p80=None, p60=None, p50=None, p40=None, p30=None):
    """Return list of matching categories (multi-label). Strategy-specific views, not mutually exclusive.

    A stock can be both 稳健型 (for long_term) AND 动量型 (for hot_picks) simultaneously.
    """
    has_a2 = r.get("has_a2") == 1 and a2_report is not None
    momentum = r.get("momentum") or 50
    total_score = r.get("total_score") or 40
    fund_score = r.get("fundamental_score") or 40

    # Percentile-based thresholds, fallback to hardcoded defaults
    mp80 = p80 if p80 is not None else 85
    mp60 = p60 if p60 is not None else 65
    mp50 = p50 if p50 is not None else 55
    mp40 = p40 if p40 is not None else 50
    mp30 = p30 if p30 is not None else 40

    # ── Parse A2 quality ──
    eq = "?"; gq = "?"; fh = "?"; val = "?"; rfs = 0
    if a2_report:
        eq = (a2_report.get("earnings_quality") or {}).get("rating", "?") if isinstance(a2_report.get("earnings_quality"), dict) else "?"
        gq = (a2_report.get("growth_quality") or {}).get("rating", "?") if isinstance(a2_report.get("growth_quality"), dict) else "?"
        fh = (a2_report.get("financial_health") or {}).get("rating", "?") if isinstance(a2_report.get("financial_health"), dict) else "?"
        val = (a2_report.get("valuation") or {}).get("rating", "?") if isinstance(a2_report.get("valuation"), dict) else "?"
        rfs = len(a2_report.get("red_flags", []))

    d3 = momentum_d3 or 0; d5 = momentum_d5 or 0; d20 = momentum_d20 or 0

    # ── Hard gate: too many red flags → 观望型 only ──
    if rfs >= 5:
        return ["观望型"]
    # eq=LOW+fh=POOR no longer hard-gated — A7 LLM has discretion to include with rationale

    tags = set()

    # ── Long-term categories ──
    # 价值型: good health, reasonable valuation, moderate flags, momentum ≥ p40
    if fh == "GOOD" and val not in ("OVERPRICED", "?") and rfs <= 2 and momentum >= mp40:
        tags.add("价值型")

    # 成长型: high growth OR high earnings quality, reasonable valuation, momentum ≥ p50
    if (gq == "HIGH" or eq == "HIGH") and val != "OVERPRICED" and rfs <= 1 and momentum >= mp50:
        tags.add("成长型")

    # 稳健型: not poor health, known fh, momentum ≥ p30, rfs ≤ 2
    if fh != "POOR" and fh != "?" and momentum >= mp30 and rfs <= 2:
        tags.add("稳健型")

    # ── Hot-picks categories ──
    # 动量型: strong momentum, quality floor
    if momentum >= mp80 and eq != "LOW":
        tags.add("动量型")

    # 突破型: moderate momentum, quality floor
    if momentum >= mp60 and eq != "LOW" and rfs <= 3:
        tags.add("突破型")

    # 短期交易型: multi-timeframe momentum positive, fundamentals not terrible
    if d3 > 0 and d5 > 0 and d20 > 0:
        if fund_score >= 30 or not has_a2:
            tags.add("短期交易型")

    # ── Fallbacks if nothing matched ──
    if not tags:
        if has_a2 and fh != "POOR" and fh != "?":
            tags.add("稳健型")
        elif has_a2 and momentum >= mp60 and eq != "LOW":
            tags.add("动量型")
        elif not has_a2 and d3 > 0 and d5 > 0:
            tags.add("短期交易型")
        else:
            tags.add("观望型")

    return sorted(tags)


def rebuild(strategy="long_term"):
    """Rebuild focus list with scenario-based classification.

    long_term → 长期池: 价值型 + 成长型 + 稳健型 (4:3:3)
    hot_picks → 短期池: 动量型 + 突破型 (1:1)
    """
    conn = get_connection()
    try:
        tiers = ("'HOLDING'", "'FAVORED'", "'NEUTRAL'")

        rows = conn.execute(f"""
            SELECT s.ts_code, s.name,
                   cs.total_score, cs.rank, t.tier, t.confidence,
                   cs.momentum, cs.fundamental_score, cs.tech_score,
                   cs.trend_type, cs.momentum_d3, cs.momentum_d5, cs.momentum_d20,
                   CASE WHEN fr.ts_code IS NOT NULL THEN 1 ELSE 0 END as has_a2
            FROM stocks s
            JOIN tier_assignments t ON s.ts_code = t.ts_code AND t.tier IN ({','.join(tiers)})
            JOIN composite_scores cs ON s.ts_code = cs.ts_code
                AND cs.calc_date = (SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)
                AND cs.strategy = ?
            LEFT JOIN fundamental_reports fr ON s.ts_code = fr.ts_code
            WHERE cs.total_score IS NOT NULL
            GROUP BY s.ts_code
            ORDER BY t.tier = 'HOLDING' DESC, has_a2 DESC, cs.total_score DESC
        """, (strategy, strategy)).fetchall()

        if not rows:
            logger.warning("No scored stocks found")
            return []

        # ── Load A2 reports for classification ──
        a2_cache = {}
        for r2 in conn.execute(
            "SELECT ts_code, report_json FROM fundamental_reports"
        ).fetchall():
            try:
                a2_cache[r2["ts_code"]] = json.loads(r2["report_json"])
            except (json.JSONDecodeError, TypeError):
                pass

        # ── Compute momentum percentiles from candidate pool ──
        mom_values = sorted([(r["momentum"] or 50) for r in rows])
        n_mom = len(mom_values)
        def mpct(p):
            idx = int(n_mom * p)
            return mom_values[min(idx, n_mom - 1)]
        p80, p60, p50, p40, p30 = mpct(0.80), mpct(0.60), mpct(0.50), mpct(0.40), mpct(0.30)
        logger.info(f"Focus mom percentiles: p80={p80:.0f} p60={p60:.0f} p50={p50:.0f} p40={p40:.0f} p30={p30:.0f}")

        # ── Classify all stocks (multi-label) ──
        classified = defaultdict(list)
        stock_tags = {}  # ts_code → list of tags (for dedup & persist)
        for r in rows:
            rd = dict(r)
            a2 = a2_cache.get(rd["ts_code"])
            d3 = rd.get("momentum_d3") or 0
            d5 = rd.get("momentum_d5") or 0
            d20 = rd.get("momentum_d20") or 0
            tags = classify_stock(rd, a2, momentum_d3=d3, momentum_d5=d5, momentum_d20=d20,
                                p80=p80, p60=p60, p50=p50, p40=p40, p30=p30)
            stock_tags[rd["ts_code"]] = tags
            for tag in tags:
                classified[tag].append(rd)

        # ── Log distribution ──
        dist = {k: len(v) for k, v in sorted(classified.items())}
        logger.info(f"Classification [{strategy}]: {dist}")

        # ── Build strategy-specific pool ──
        # Both strategies see 短期交易型 (short-term trading candidates)
        if strategy == "hot_picks":
            allowed_types = {"动量型", "突破型", "短期交易型"}
        else:
            allowed_types = {"价值型", "成长型", "稳健型"}

        # Per-strategy TOP_PCT: regime-aware, strategy-differentiated base
        if strategy == "long_term":
            base_pct = 0.30
        else:
            base_pct = 0.40
        try:
            mr = conn.execute(
                "SELECT regime FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
            ).fetchone()
            regime = mr["regime"] if mr else ""
            if "熊市" in str(regime) or "BEAR" in str(regime).upper():
                top_pct = base_pct * 0.60  # BEAR: tighter
            elif "牛市" in str(regime) or "BULL" in str(regime).upper():
                top_pct = base_pct * 1.00  # BULL: full base
            else:
                top_pct = base_pct * 0.80  # NEUTRAL: moderate
        except Exception:
            top_pct = base_pct

        # Select from each category: percentage-based, no ratio caps
        seen = set()
        selected = []
        for cat in allowed_types:
            candidates = [r for r in classified.get(cat, [])
                         if (r.get("total_score") or 0) >= MIN_TOTAL_SCORE]
            # Sort by A2 presence first, then total_score
            candidates.sort(key=lambda r: (-(r.get("has_a2") or 0), -(r.get("total_score") or 0)))
            # Take top % within this category (min 12 long_term, min 8 hot_picks)
            min_take = 12 if strategy == "long_term" else 8
            take = max(min_take, int(len(candidates) * top_pct))
            for r in candidates[:take]:
                code = r["ts_code"]
                if code not in seen:
                    seen.add(code)
                    selected.append(r)

        # Log: count per category (split multi-label tier strings)
        cat_counts = {}
        for s in selected:
            for tag in stock_tags.get(s["ts_code"], []):
                cat_counts[tag] = cat_counts.get(tag, 0) + 1
        logger.info(f"Focus list [{strategy}]: {len(selected)} selected {dict(cat_counts)}")

        # ── Persist ──
        trade_date = datetime.now().strftime("%Y-%m-%d")
        conn.execute("DELETE FROM focus_list WHERE strategy=?", (strategy,))
        for i, s in enumerate(selected):
            tier = ",".join(stock_tags.get(s["ts_code"], ["观望型"]))
            conn.execute(
                """INSERT INTO focus_list (ts_code, name, total_score, rank, list_date, position, tier, strategy)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (s["ts_code"], s["name"],
                 s["total_score"], s["rank"], trade_date, i + 1,
                 tier, strategy),
            )
        conn.commit()
        return selected
    finally:
        conn.close()


def get_focus_codes(conn, strategy=None):
    """Return list of ts_code in current focus list, ordered by position.

    Args:
        strategy: filter by strategy ('long_term' or 'hot_picks').
                  If None, returns all (legacy behavior).
    """
    if strategy:
        rows = conn.execute(
            "SELECT ts_code, name, total_score, position "
            "FROM focus_list WHERE strategy=? ORDER BY position",
            (strategy,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts_code, name, industry, total_score, position FROM focus_list ORDER BY position"
        ).fetchall()
    return [dict(r) for r in rows]
