"""DB cleanup: remove dead tables, purge old data, VACUUM."""
import sqlite3, os, logging, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from backend.lib.logging import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

DB = os.path.expanduser("~/stock-analysis/data/stocks.db")
BACKUP_DIR = os.path.expanduser("~/stock-analysis/data/backups")

def get_size_mb():
    return os.path.getsize(DB) / 1e6

def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    size_before = get_size_mb()
    logger.info(f"DB cleanup starting — current size: {size_before:.1f} MB")

    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # ── 1. DROP empty/dead tables ──
    dead_tables = [
        'agent_io_log', 'investment_theses', 'news_events', 'news_reports',
        'performance_attribution', 'technical_reports', 'trades',
    ]
    for t in dead_tables:
        try:
            cur.execute(f"DROP TABLE IF EXISTS [{t}]")
            logger.info(f"  DROPPED empty table: {t}")
        except Exception as e:
            logger.warning(f"  Cannot drop {t}: {e}")

    # ── 2. Purge old temporary data (keep 3 days) ──
    temp_purge = [
        ("composite_scores", "calc_date", "A5 fusion scores"),
        ("indicators", "calc_date", "A1 tech indicators"),
        ("risk_flags", "flag_date", "A6 risk flags"),
        ("agent_logs", "run_date", "agent run logs"),
        ("pipeline_runs", "started_at", "pipeline run logs"),
    ]
    for table, col, desc in temp_purge:
        try:
            cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE datetime({col}) < datetime('now', '-3 days')")
            old = cur.fetchone()[0]
            if old > 0:
                cur.execute(f"DELETE FROM [{table}] WHERE datetime({col}) < datetime('now', '-3 days')")
                logger.info(f"  PURGED {old} rows from {table} ({desc})")
            else:
                logger.info(f"  SKIP {table}: nothing older than 3 days")
        except Exception as e:
            logger.warning(f"  Cannot purge {table}: {e}")

    # ── 3. Purge old investment decisions (keep 30 days) ──
    decision_purge = [
        ("portfolio_decisions", "calc_date", "A7/A6 decisions"),
        ("portfolio_reports", "calc_date", "A7 portfolio reports"),
        ("execution_orders", "created_at", "execution orders"),
    ]
    for table, col, desc in decision_purge:
        try:
            cur.execute(f"SELECT COUNT(*) FROM [{table}] WHERE datetime({col}) < datetime('now', '-30 days')")
            old = cur.fetchone()[0]
            if old > 0:
                cur.execute(f"DELETE FROM [{table}] WHERE datetime({col}) < datetime('now', '-30 days')")
                logger.info(f"  PURGED {old} rows from {table} ({desc})")
            else:
                logger.info(f"  SKIP {table}: nothing older than 30 days")
        except Exception as e:
            logger.warning(f"  Cannot purge {table}: {e}")

    # ── 4. Drop decision_reports (no code writes, old CIO agent artifact) ──
    try:
        cur.execute("SELECT COUNT(*) FROM decision_reports")
        cnt = cur.fetchone()[0]
        cur.execute("DROP TABLE IF EXISTS decision_reports")
        logger.info(f"  DROPPED decision_reports ({cnt} rows, no active code)")
    except Exception as e:
        logger.warning(f"  Cannot drop decision_reports: {e}")

    # ── 5. Drop simulator_weights (test data only) ──
    try:
        cur.execute("SELECT COUNT(*) FROM simulator_weights")
        cnt = cur.fetchone()[0]
        cur.execute("DROP TABLE IF EXISTS simulator_weights")
        logger.info(f"  DROPPED simulator_weights ({cnt} test rows)")
    except Exception as e:
        logger.warning(f"  Cannot drop simulator_weights: {e}")

    conn.commit()

    # ── 6. VACUUM ──
    logger.info("Running VACUUM (this may take 30-60s)...")
    conn.execute("VACUUM")
    conn.close()

    size_after = get_size_mb()
    reclaimed = size_before - size_after
    logger.info(f"Cleanup complete: {size_before:.1f} → {size_after:.1f} MB "
                f"({reclaimed:.1f} MB reclaimed)")

if __name__ == "__main__":
    main()
