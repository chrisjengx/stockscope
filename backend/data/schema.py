"""Database schema initialization for Stock Analysis System."""
import sqlite3
import os
import logging
from contextlib import contextmanager
from threading import Lock

from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

DB_PATH = os.path.expanduser("~/stock-analysis/data/stocks.db")

# Thread-safe connection management
_conn_lock = Lock()


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")  # 5s retry on lock
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_connection():
    """Context manager for safe connection lifecycle."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS stocks (
        ts_code TEXT PRIMARY KEY, name TEXT NOT NULL,
        market TEXT, industry TEXT, board TEXT,
        is_st INTEGER DEFAULT 0, list_date DATE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS daily_quotes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, trade_date DATE NOT NULL,
        open REAL, high REAL, low REAL, close REAL,
        volume REAL, amount REAL, turnover REAL, change_pct REAL,
        UNIQUE(ts_code, trade_date)
    );

    CREATE TABLE IF NOT EXISTS indicators (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
        tech_score REAL, indicators_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ts_code, calc_date)
    );

    CREATE TABLE IF NOT EXISTS financials (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, report_date DATE NOT NULL,
        roe REAL, gross_margin REAL, net_margin REAL,
        debt_ratio REAL, revenue_yoy REAL, fcf_ratio REAL,
        pe_ttm REAL, pb REAL, pe_percentile REAL,
        raw_json TEXT, UNIQUE(ts_code, report_date)
    );

    CREATE TABLE IF NOT EXISTS tier_assignments (
        ts_code TEXT NOT NULL, tier TEXT NOT NULL DEFAULT 'NEUTRAL',
        reason TEXT, confidence REAL DEFAULT 0.5,
        source TEXT DEFAULT 'rule_default',
        updated_at DATE NOT NULL
    );

    CREATE TABLE IF NOT EXISTS composite_scores (
        ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
        strategy TEXT NOT NULL DEFAULT 'long_term',
        tech_score REAL, fundamental_score REAL,
        macro_fit REAL, momentum REAL,
        total_score REAL, rank INTEGER, tier_action TEXT,
        UNIQUE(ts_code, calc_date, strategy)
    );

    CREATE TABLE IF NOT EXISTS portfolio (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, strategy TEXT DEFAULT 'long_term',
        entry_date DATE NOT NULL, entry_price REAL,
        shares INTEGER DEFAULT 0, weight REAL DEFAULT 0,
        exit_date DATE, exit_price REAL,
        status TEXT DEFAULT 'HOLD',
        UNIQUE(ts_code, strategy)
    );

    CREATE TABLE IF NOT EXISTS macro_regime (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        calc_date DATE NOT NULL UNIQUE,
        regime TEXT, risk_budget REAL, detail_json TEXT
    );

    CREATE TABLE IF NOT EXISTS news_feed (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL, category TEXT,
        tags TEXT, sentiment TEXT, sentiment_strength REAL,
        impact TEXT, related_stocks TEXT, related_industry TEXT,
        summary TEXT, raw_url TEXT,
        content_hash TEXT UNIQUE,
        consumer TEXT, consumed INTEGER DEFAULT 0,
        published_at TIMESTAMP,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS risk_flags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT, flag_date DATE NOT NULL,
        severity TEXT, question TEXT, response TEXT,
        resolved INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS agent_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_id INTEGER NOT NULL, run_date DATE NOT NULL,
        status TEXT, stocks_processed INTEGER,
        tokens_used INTEGER, duration_s REAL, summary TEXT
    );

    CREATE TABLE IF NOT EXISTS simulator_weights (
        stock_type TEXT NOT NULL, regime TEXT NOT NULL,
        trend_w REAL, momentum_w REAL, volatility_w REAL,
        volume_w REAL, pattern_w REAL,
        type_i_error REAL, type_ii_error REAL,
        updated_at DATE NOT NULL,
        PRIMARY KEY (stock_type, regime)
    );

    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date DATE NOT NULL, ts_code TEXT NOT NULL,
        direction TEXT NOT NULL, price REAL NOT NULL,
        shares INTEGER NOT NULL, amount REAL NOT NULL,
        profit_loss REAL, decision_id INTEGER,
        entry_method TEXT DEFAULT 'manual',
        note TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS portfolio_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
        action TEXT NOT NULL, reason TEXT,
        weight_change REAL, macro_regime TEXT,
        risk_budget REAL, status TEXT DEFAULT 'PENDING',
        review_json TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ts_code, calc_date, action)
    );
    -- Migrate existing portfolio_decisions: add columns if missing
    """)
    # Handle migration for existing portfolio_decisions table
    try:
        conn.execute("ALTER TABLE portfolio_decisions ADD COLUMN status TEXT DEFAULT 'PENDING'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE portfolio_decisions ADD COLUMN review_json TEXT")
    except sqlite3.OperationalError:
        pass
    # Strategy-aware columns (migration for existing databases)
    _migrate = [
        ("ALTER TABLE focus_list ADD COLUMN strategy TEXT DEFAULT 'long_term'"),
        ("ALTER TABLE portfolio ADD COLUMN strategy TEXT DEFAULT 'long_term'"),
        ("ALTER TABLE trades ADD COLUMN strategy TEXT DEFAULT 'long_term'"),
        ("ALTER TABLE portfolio_decisions ADD COLUMN strategy TEXT DEFAULT 'long_term'"),
        ("ALTER TABLE portfolio_decisions ADD COLUMN suggested_shares INTEGER"),
        ("ALTER TABLE portfolio_decisions ADD COLUMN current_price REAL"),
        ("ALTER TABLE portfolio_decisions ADD COLUMN rapid_sell_flag INTEGER DEFAULT 0"),
        ("ALTER TABLE news_feed ADD COLUMN body_summary TEXT"),
        ("ALTER TABLE news_feed ADD COLUMN related_stocks_json TEXT"),
        ("ALTER TABLE news_feed ADD COLUMN quantitative_info TEXT"),
        ("ALTER TABLE news_feed ADD COLUMN classification_confidence REAL"),
        ("ALTER TABLE financials ADD COLUMN beneish_m_score REAL"),
        ("ALTER TABLE financials ADD COLUMN cfo_ni_ratio REAL"),
        ("ALTER TABLE financials ADD COLUMN ar_revenue_divergence REAL"),
        ("ALTER TABLE focus_list ADD COLUMN value_score REAL"),
        ("ALTER TABLE focus_list ADD COLUMN momentum_score REAL"),
    ]
    # tier_assignments migration: tier INTEGER → TEXT, add confidence/source
    _tier_migrate = [
        "ALTER TABLE tier_assignments ADD COLUMN confidence REAL DEFAULT 0.5",
        "ALTER TABLE tier_assignments ADD COLUMN source TEXT DEFAULT 'rule_default'",
    ]
    for sql in _tier_migrate:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass
    # Migrate existing integer tiers to text (SQLite ALTER TABLE doesn't support column type change,
    # but we can update the values. The column type is flexible in SQLite.)
    try:
        conn.execute("UPDATE tier_assignments SET tier='EXCLUDED' WHERE tier IN ('0', '0.0')")
        conn.execute("UPDATE tier_assignments SET tier='HOLDING' WHERE tier IN ('1', '1.0')")
        conn.execute("UPDATE tier_assignments SET tier='FAVORED' WHERE tier IN ('2', '2.0')")
        conn.execute("UPDATE tier_assignments SET tier='NEUTRAL' WHERE tier IN ('3', '3.0')")
        conn.execute("UPDATE tier_assignments SET tier='AVOID' WHERE tier IN ('4', '4.0')")
    except sqlite3.OperationalError:
        pass

    for sql in _migrate:
        try:
            conn.execute(sql)
        except sqlite3.OperationalError:
            pass

    # fusion_reports migration: add strategy column, drop old UNIQUE on calc_date
    try:
        conn.execute("ALTER TABLE fusion_reports ADD COLUMN strategy TEXT DEFAULT 'long_term'")
    except sqlite3.OperationalError:
        pass

    # composite_scores migration: add strategy column + fix UNIQUE to include strategy
    # Only runs once — guarded by whether the strategy column already exists
    _need_cs_migration = False
    try:
        conn.execute("ALTER TABLE composite_scores ADD COLUMN strategy TEXT DEFAULT 'long_term'")
        _need_cs_migration = True
    except sqlite3.OperationalError:
        pass  # strategy column already exists, migration done previously — skip

    if _need_cs_migration:
        conn.execute("""
            CREATE TABLE composite_scores_new (
                ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
                strategy TEXT NOT NULL DEFAULT 'long_term',
                tech_score REAL, fundamental_score REAL,
                macro_fit REAL, momentum REAL,
                total_score REAL, rank INTEGER, tier_action TEXT,
                UNIQUE(ts_code, calc_date, strategy)
            )
        """)
        conn.execute("""
            INSERT OR IGNORE INTO composite_scores_new (ts_code, calc_date, strategy,
                tech_score, fundamental_score, macro_fit, momentum, total_score, rank, tier_action)
            SELECT ts_code, calc_date, 'long_term',
                tech_score, fundamental_score, macro_fit, momentum, total_score, rank, tier_action
            FROM composite_scores
        """)
        conn.execute("DROP TABLE composite_scores")
        conn.execute("ALTER TABLE composite_scores_new RENAME TO composite_scores")

    # composite_scores: v2 multi-timeframe momentum columns + A5 extra factors
    for col, col_type in [
        ("trend_type", "TEXT"),
        ("momentum_d3", "REAL"),
        ("momentum_d5", "REAL"),
        ("momentum_d20", "REAL"),
        ("momentum_d60", "REAL"),
        ("momentum_accel", "REAL"),
        ("industry_momentum", "REAL"),
        ("relative_strength", "REAL DEFAULT 50"),
        ("volume_confirmation", "REAL DEFAULT 0.5"),
    ]:
        try:
            conn.execute(f"ALTER TABLE composite_scores ADD COLUMN {col} {col_type}")
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.executescript("""

    CREATE TABLE IF NOT EXISTS fundamental_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
        report_json TEXT NOT NULL,
        overall_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ts_code, calc_date)
    );

    CREATE TABLE IF NOT EXISTS technical_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, calc_date DATE NOT NULL,
        report_json TEXT NOT NULL,
        tech_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(ts_code, calc_date)
    );

    CREATE TABLE IF NOT EXISTS pipeline_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        mode TEXT NOT NULL,                -- 'daily' | 'weekly'
        strategy TEXT NOT NULL,            -- 'long_term' | 'hot_picks'
        status TEXT DEFAULT 'RUNNING',     -- RUNNING / COMPLETED / ABORTED
        started_at TIMESTAMP NOT NULL,
        completed_at TIMESTAMP,
        agents_total INTEGER DEFAULT 0,
        agents_ok INTEGER DEFAULT 0,
        agents_failed INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS scheduler_state (
        run_key TEXT PRIMARY KEY,
        last_run TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS agent_io_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_run_id INTEGER NOT NULL,
        agent_id INTEGER NOT NULL,
        strategy TEXT NOT NULL,
        run_date TIMESTAMP NOT NULL,

        input_schema_version TEXT NOT NULL,
        input_json TEXT NOT NULL,
        output_schema_version TEXT NOT NULL,
        output_json TEXT NOT NULL,

        input_valid INTEGER NOT NULL,
        output_valid INTEGER NOT NULL,
        validation_errors TEXT,

        duration_ms INTEGER,
        llm_calls INTEGER DEFAULT 0,
        tokens_used INTEGER DEFAULT 0,
        status TEXT NOT NULL,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS investment_theses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL, created_at DATE NOT NULL,
        updated_at DATE NOT NULL, thesis_text TEXT NOT NULL,
        key_assumptions TEXT, falsification_conditions TEXT,
        expected_horizon INTEGER, status TEXT DEFAULT 'ACTIVE',
        last_review_date DATE, review_history TEXT
    );

    CREATE TABLE IF NOT EXISTS performance_attribution (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT, calc_date DATE,
        total_return REAL, market_beta_return REAL,
        alpha_return REAL, factor_contributions TEXT,
        decision_quality TEXT, thesis_accuracy TEXT,
        report_date DATE
    );

    CREATE INDEX IF NOT EXISTS idx_daily_code_date ON daily_quotes(ts_code, trade_date);
    CREATE INDEX IF NOT EXISTS idx_indicators_code_date ON indicators(ts_code, calc_date);
    CREATE INDEX IF NOT EXISTS idx_scores_date ON composite_scores(calc_date);
    CREATE INDEX IF NOT EXISTS idx_tier_tier ON tier_assignments(tier);
    CREATE INDEX IF NOT EXISTS idx_portfolio_status ON portfolio(status);
    CREATE INDEX IF NOT EXISTS idx_news_consumer ON news_feed(consumer, consumed);
    CREATE INDEX IF NOT EXISTS idx_decisions_code_date ON portfolio_decisions(ts_code, calc_date);
    CREATE INDEX IF NOT EXISTS idx_decisions_status ON portfolio_decisions(status);
    CREATE TABLE IF NOT EXISTS universe (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,
        strategy TEXT NOT NULL,
        liquidity_score REAL,
        size_score REAL,
        is_holding INTEGER DEFAULT 0,
        inclusion_reason TEXT,
        exclusion_reason TEXT,
        updated_at DATE NOT NULL,
        UNIQUE(ts_code, strategy, updated_at)
    );

    CREATE TABLE IF NOT EXISTS news_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_date DATE NOT NULL,
        title TEXT NOT NULL,
        article_count INTEGER,
        primary_category TEXT,
        sentiment_distribution TEXT,
        related_stocks TEXT,
        impact TEXT,
        summary TEXT,
        first_seen TIMESTAMP,
        last_updated TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS news_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_date DATE NOT NULL UNIQUE,
        report_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS macro_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        calc_date DATE NOT NULL UNIQUE,
        report_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS fusion_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        calc_date DATE NOT NULL,
        strategy TEXT NOT NULL DEFAULT 'long_term',
        report_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(calc_date, strategy)
    );

    CREATE TABLE IF NOT EXISTS portfolio_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        calc_date DATE NOT NULL,
        strategy TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(calc_date, strategy)
    );

    CREATE TABLE IF NOT EXISTS execution_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,
        strategy TEXT NOT NULL,
        decision_date DATE NOT NULL,
        direction TEXT NOT NULL,
        order_type TEXT DEFAULT 'LIMIT',
        price_limit REAL,
        shares INTEGER NOT NULL,
        estimated_amount REAL,
        priority INTEGER DEFAULT 5,
        status TEXT DEFAULT 'PENDING',
        a8_override INTEGER DEFAULT 0,
        override_reason TEXT,
        review_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS decision_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        decision_date DATE NOT NULL,
        strategy TEXT NOT NULL,
        report_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(decision_date, strategy)
    );

    CREATE INDEX IF NOT EXISTS idx_theses_code ON investment_theses(ts_code, status);
    CREATE INDEX IF NOT EXISTS idx_io_pipeline ON agent_io_log(pipeline_run_id);
    CREATE INDEX IF NOT EXISTS idx_io_agent ON agent_io_log(agent_id, run_date);
    CREATE INDEX IF NOT EXISTS idx_pipeline_runs_status ON pipeline_runs(status);
    CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_quotes(trade_date);
    CREATE INDEX IF NOT EXISTS idx_agent_logs_agent_date ON agent_logs(agent_id, run_date);
    CREATE INDEX IF NOT EXISTS idx_universe_strategy ON universe(strategy);
    CREATE INDEX IF NOT EXISTS idx_exec_orders_status ON execution_orders(strategy, status);
    """)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    conn.close()
    return [t[0] for t in tables]


if __name__ == "__main__":
    tables = init_db()
    print(f"✅ Database initialized: {len(tables)} tables")
    for t in tables:
        if not t.startswith("sqlite"):
            print(f"  • {t}")
