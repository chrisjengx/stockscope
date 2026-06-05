"""Reproduce A6 bug by calling actual score_decisions."""
import json, logging
from backend.data.schema import get_connection
from backend.agents.agent_6_risk import score_decisions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

conn = get_connection()

# Run against latest long_term A7 decisions (with PENDING status)
for strategy in ["long_term", "hot_picks"]:
    # Get decisions
    decisions = [dict(d) for d in conn.execute(
        "SELECT pd.* FROM portfolio_decisions pd "
        "WHERE pd.strategy=? AND pd.calc_date = ("
        "  SELECT MAX(calc_date) FROM portfolio_decisions WHERE strategy=?) "
        "ORDER BY pd.action, pd.ts_code",
        (strategy, strategy),
    ).fetchall()]

    # Mark as PENDING so score_decisions picks them up
    conn.execute("UPDATE portfolio_decisions SET status='PENDING' WHERE strategy=?", (strategy,))
    conn.commit()

    holdings = [dict(h) for h in conn.execute(
        "SELECT p.*, s.industry FROM portfolio p JOIN stocks s ON p.ts_code=s.ts_code WHERE p.status='HOLD'"
    ).fetchall()]
    macro = dict(conn.execute(
        "SELECT * FROM macro_regime ORDER BY calc_date DESC LIMIT 1"
    ).fetchone())

    print(f"\n{'='*60}")
    print(f"Running A6 score_decisions for {strategy}")
    print(f"Decisions: {len(decisions)}, BUY: {sum(1 for d in decisions if d['action']=='BUY')}, "
          f"REJECT: {sum(1 for d in decisions if d['action']=='REJECT')}")
    print(f"{'='*60}")

    scores = score_decisions(decisions, holdings, macro, conn, strategy=strategy)
    print(f"Result: {len(scores) if scores else 0} scores\n")

    # Restore
    conn.execute("UPDATE portfolio_decisions SET status='APPROVED' WHERE strategy=?", (strategy,))
    conn.commit()

conn.close()
