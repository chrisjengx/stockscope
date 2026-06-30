"""
Data Fetcher — standalone module (NOT an Agent).
Single entry point for all stock data. All other modules read from local SQLite.
Runs daily before any Agent.

Pipeline:
  1. stock_info_a_code_name() → stock list (weekly)
  2. stock_zh_a_daily(symbol) → daily OHLCV for Tier 1-3 stocks (daily)
"""
import os, sys, time, logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed

sys.path.insert(0, os.path.expanduser("~/stock-analysis"))
from backend.data.schema import get_connection

logger = logging.getLogger("data_fetcher")


def update_stock_list():
    """Update stock code list from SSE/SZSE."""
    import akshare as ak
    logger.info("Updating stock code list...")
    t0 = time.time()
    df = ak.stock_info_a_code_name()
    conn = get_connection()
    new_count = 0
    for _, row in df.iterrows():
        code = str(row["code"]).zfill(6)
        if code.startswith("920"):
            continue
        market = "SH" if code.startswith(("6", "5", "9")) else "SZ"
        ts_code = f"{code}.{market}"
        conn.execute(
            "INSERT OR IGNORE INTO stocks (ts_code, name, market) VALUES (?,?,?)",
            (ts_code, str(row["name"]), "A"),
        )
        new_count += 1
    conn.commit()
    total = conn.execute("SELECT COUNT(*) FROM stocks").fetchone()[0]
    conn.close()
    logger.info(f"Stock list: {new_count} new, {total} total ({time.time()-t0:.0f}s)")
    return total


def fetch_one_daily(ts_code, start_date, end_date, timeout=15):
    """Fetch daily OHLCV from akshare. Primary source. Timeout-protected."""
    import akshare as ak
    def _fetch():
        code = ts_code.split(".")[0]
        market = ts_code.split(".")[1].lower()
        symbol = f"{market}{code}"
        return ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date, adjust="")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            df = future.result(timeout=timeout)
        return ts_code, df
    except FuturesTimeoutError:
        logger.warning(f"{ts_code} akshare TIMEOUT ({timeout}s)")
        return ts_code, None
    except Exception as e:
        logger.warning(f"{ts_code} akshare ERROR: {e}")
        return ts_code, None


def fetch_daily_baostock(ts_code, start_date, end_date, timeout=15):
    """Fetch daily OHLCV from baostock. Fallback source. Timeout-protected."""
    import baostock as bs
    import pandas as pd
    def _fetch():
        code = ts_code.split(".")[0]
        market = ts_code.split(".")[1].lower()
        bs_code = f"{market}.{code}"
        rs = bs.query_history_k_data_plus(
            bs_code, "date,open,high,low,close,volume,amount,turn,pctChg",
            start_date=start_date, end_date=end_date, frequency="d", adjustflag="2"
        )
        if rs.error_code != "0":
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"])
        df["date"] = df["date"].str[:10]
        return df
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            df = future.result(timeout=timeout)
        return ts_code, df
    except FuturesTimeoutError:
        logger.warning(f"{ts_code} baostock TIMEOUT ({timeout}s)")
        return ts_code, None
    except Exception as e:
        logger.warning(f"{ts_code} baostock ERROR: {e}")
        return ts_code, None


def merge_daily_data(df_ak, df_bs):
    """Merge akshare and baostock daily data. Prefer akshare for OHLCV timeliness,
    use baostock to fill gaps. Returns merged DataFrame or best available."""
    import pandas as pd
    if df_ak is not None and not df_ak.empty:
        if df_bs is not None and not df_bs.empty:
            # Merge: akshare preferred, baostock fills missing dates
            merged = pd.concat([df_bs, df_ak]).drop_duplicates(subset=["date"], keep="last")
            return merged
        return df_ak
    return df_bs


def save_daily_data(ts_code, df, conn):
    """Save daily OHLCV data to DB. change_pct computed from close prices."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["date"] = df["date"].astype(str).str[:10]
    df = df.sort_values("date")
    closes = df["close"].astype(float).values
    prev_close = None
    count = 0
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            close = float(row.get("close", 0) or 0)
            if prev_close is not None and prev_close > 0:
                change_pct = round((close / prev_close - 1) * 100, 4)
            else:
                change_pct = 0.0
            prev_close = close
            # Normalize turnover: akshare decimal, baostock percentage
            turnover_raw = float(row.get("turnover", 0) or 0)
            if turnover_raw > 1:
                turnover_raw = turnover_raw / 100
            conn.execute(
                """INSERT OR IGNORE INTO daily_quotes
                   (ts_code, trade_date, open, high, low, close, volume, amount, turnover, change_pct)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (ts_code, str(row.get("date", ""))[:10],
                 float(row.get("open", 0) or 0), float(row.get("high", 0) or 0),
                 float(row.get("low", 0) or 0), close,
                 float(row.get("volume", 0) or 0), float(row.get("amount", 0) or 0),
                 turnover_raw, change_pct),
            )
            count += 1
        except (ValueError, KeyError):
            pass
    return count


def daily_update(target_tiers=None, max_workers=1):
    """Main daily update: fetch today's OHLCV for stocks in target tiers.

    NOTE: max_workers=1 (sequential) is the safe default. akshare is NOT
    thread-safe and crashes with >1 workers. Sequential processes ~1 stock/s,
    so 1750 stocks takes ~30 min — acceptable for once-daily operation.
    """
    if target_tiers is None:
        target_tiers = []  # empty = ALL stocks

    start = time.time()
    trade_date = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Per-stock last trade date for hole-filling (not global MAX — catches per-stock gaps)
    conn = get_connection()
    stock_last = {}
    for r in conn.execute(
        "SELECT ts_code, MAX(trade_date) as last_date FROM daily_quotes GROUP BY ts_code"
    ).fetchall():
        stock_last[r["ts_code"]] = r["last_date"]

    # Get target stocks: specific tiers or ALL stocks
    if target_tiers:
        placeholders = ",".join("?" * len(target_tiers))
        cursor = conn.execute(
            f"SELECT ts_code FROM tier_assignments WHERE tier IN ({placeholders})",
            target_tiers,
        )
        codes = [r["ts_code"] for r in cursor.fetchall()]
        tier_label = f"Tiers {target_tiers}"
    else:
        cursor = conn.execute("SELECT ts_code FROM stocks")
        codes = [r["ts_code"] for r in cursor.fetchall()]
        tier_label = "ALL stocks"
    conn.close()

    if not codes:
        logger.warning("No stocks found!")
        return 0

    logger.info(f"=== Daily OHLCV update: per-stock → {trade_date} ({tier_label}, {len(codes)} stocks) ===")

    logger.info(f"Fetching daily data for {len(codes)} stocks (dual-source, sequential)...")

    # baostock session for the batch
    bs_ok = False
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == "0":
            bs_ok = True
            logger.info("baostock session established for dual-source fetch")
    except Exception:
        pass

    total_rows = 0
    stocks_today = 0
    ak_ok = 0
    bs_fallback = 0
    conn = get_connection()
    for i, code in enumerate(codes):
        try:
            stock_start = stock_last.get(code, default_start)
            # Primary: akshare
            _, df_ak = fetch_one_daily(code, stock_start, trade_date)
            if df_ak is not None and not df_ak.empty:
                df = df_ak
                ak_ok += 1
            elif bs_ok:
                # akshare failed — log and fallback to baostock
                logger.warning(f"{code} akshare returned empty, falling back to baostock")
                _, df_bs = fetch_daily_baostock(code, stock_start, trade_date)
                df = df_bs
                if df is not None and not df.empty:
                    bs_fallback += 1
            else:
                df = None
            if df is not None and not df.empty:
                rows = save_daily_data(code, df, conn)
                total_rows += rows
                if df["date"].astype(str).str[:10].eq(trade_date).any() and rows > 0:
                    stocks_today += 1
        except Exception as e:
            logger.warning(f"{code} unexpected ERROR: {e}")
        if (i + 1) % 100 == 0:
            conn.commit()
            elapsed = time.time() - start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(codes) - i - 1) / rate if rate > 0 else 0
            logger.info(f"  {i+1}/{len(codes)} stocks, {stocks_today} today | {total_rows} rows ({rate:.1f}/s, ETA {eta/60:.0f}min)")

    conn.commit()
    conn.close()
    elapsed = time.time() - start
    logger.info(f"Daily update complete: {stocks_today}/{len(codes)} stocks ({trade_date}) "
               f"+ {total_rows} total rows (akshare:{ak_ok} baostock_fallback:{bs_fallback}) in {elapsed:.0f}s")
    return stocks_today


def cold_start(tiers=None):
    """First run: pull 2 years of history for Tiers 1-3."""
    if tiers is None:
        tiers = [1, 2, 3]

    logger.info("=== COLD START: Updating stock list ===")
    update_stock_list()

    logger.info("=== COLD START: Fetching 2-year history ===")
    # Set up initial tier assignments (all Tier 3 initially)
    conn = get_connection()
    codes = [r[0] for r in conn.execute("SELECT ts_code FROM stocks").fetchall()]
    trade_date = datetime.now().strftime("%Y-%m-%d")
    for c in codes:
        conn.execute(
            "INSERT OR IGNORE INTO tier_assignments (ts_code, tier, reason, updated_at) VALUES (?,3,?,?)",
            (c, "Cold start", trade_date),
        )
    conn.commit()
    conn.close()

    start_date = "20160501"  # 10 years ago approx
    end_date = datetime.now().strftime("%Y%m%d")

    conn = get_connection()
    total_rows = 0
    for i, code in enumerate(codes):
        ts_code, df = fetch_one_daily(code, start_date, end_date)
        if df is not None and not df.empty:
            rows = save_daily_data(ts_code, df, conn)
            total_rows += rows
        if (i + 1) % 100 == 0:
            conn.commit()
            logger.info(f"  {i+1}/{len(codes)} stocks, {total_rows} rows")

    conn.commit()
    conn.close()
    logger.info(f"✅ Cold start complete: {total_rows} rows")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["daily", "weekly", "cold"], default="daily")
    p.add_argument("--tiers", type=int, nargs="+")
    args = p.parse_args()

    if args.mode == "cold":
        cold_start()
    elif args.mode == "weekly":
        update_stock_list()
        daily_update(target_tiers=args.tiers or [1, 2, 3])
    else:
        daily_update(target_tiers=args.tiers or [1, 2, 3])
