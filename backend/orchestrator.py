"""
Orchestrator — closed-loop pipeline executor for the 9-agent system.

Responsibilities:
  1. Trading-day gate (skip non-trading days)
  2. Stage-by-stage execution with dependency gating
  3. Stage 2 parallel execution (A4/A3/A1/A2 in ThreadPoolExecutor)
  4. SQLite write retry (WAL mode + busy timeout for concurrent writes)
  5. Error categorization (DATA / LLM / TIMEOUT / UNKNOWN)
  6. Pipeline result persistence (pipeline_runs table)
  7. Concurrency guard (one pipeline per mode/strategy at a time)

Usage:
  python -m backend.orchestrator --mode daily --strategy long_term
"""

import time
import json
import logging
import argparse
import threading
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from typing import Callable, Optional

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

# ═══════════════════════════════════════════════════════════════
# Trading day detection
# ═══════════════════════════════════════════════════════════════

def is_trading_day(d: date | None = None) -> bool:
    """Check A-share trading day. Uses akshare calendar; falls back to weekday."""
    if d is None:
        d = date.today()
    if d.weekday() >= 5:
        return False
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        if cal is not None and not cal.empty:
            trade_dates = {str(x)[:10] for x in cal["trade_date"].values}
            return d.isoformat() in trade_dates
    except Exception:
        pass
    return True  # assume trading day if calendar unavailable


# ═══════════════════════════════════════════════════════════════
# Pipeline result data class
# ═══════════════════════════════════════════════════════════════

@dataclass
class StageResult:
    stage: int
    agent: str
    status: str = "PENDING"    # PENDING / OK / FAIL / SKIPPED / TIMEOUT
    elapsed: float = 0.0
    error: str | None = None
    error_category: str | None = None  # DATA / LLM / TIMEOUT / UNKNOWN


@dataclass
class PipelineResult:
    id: int = 0
    mode: str = "daily"
    strategy: str = "long_term"
    status: str = "PENDING"
    started_at: str = ""
    completed_at: str = ""
    stages: list[StageResult] = field(default_factory=list)
    is_trading_day: bool = True
    error: str | None = None

    def ok(self) -> bool:
        return self.status == "COMPLETED"

    def failed_stages(self) -> list[StageResult]:
        return [s for s in self.stages if s.status in ("FAIL", "TIMEOUT")]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "mode": self.mode, "strategy": self.strategy,
            "status": self.status, "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stages": [{"stage": s.stage, "agent": s.agent,
                        "status": s.status, "elapsed": s.elapsed,
                        "error_category": s.error_category,
                        "error": s.error} for s in self.stages],
            "is_trading_day": self.is_trading_day,
            "agents_ok": sum(1 for s in self.stages if s.status == "OK"),
            "agents_failed": len(self.failed_stages()),
        }


# ═══════════════════════════════════════════════════════════════
# SQLite write helper — retry on BUSY
# ═══════════════════════════════════════════════════════════════

def _db_execute(conn, sql, params=(), max_retries=5):
    """Execute SQL with retry on SQLITE_BUSY (for parallel agent writes)."""
    import sqlite3
    for attempt in range(max_retries):
        try:
            conn.execute(sql, params)
            return
        except sqlite3.OperationalError as e:
            if "busy" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))
            else:
                raise


# ═══════════════════════════════════════════════════════════════
# Pipeline Orchestrator (singleton)
# ═══════════════════════════════════════════════════════════════

class Orchestrator:
    """Closed-loop pipeline orchestrator with per-stage monitoring."""

    _instance: Optional["Orchestrator"] = None
    _locks: dict[str, bool] = {}
    _lock_obj = threading.Lock()
    _shared_cache: dict[str, dict] = {}  # A1/A3/A4 results for strategy reuse

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._history: list[PipelineResult] = []
            cls._instance._current: dict[str, PipelineResult] = {}
        return cls._instance

    # ── Public API ──────────────────────────────────────────

    def run(self, mode: str = "daily", strategy: str = "long_term") -> PipelineResult:
        """Run pipeline for one strategy. Blocks until complete. Thread-safe."""
        lock_key = f"{mode}/{strategy}"
        with self._lock_obj:
            if self._locks.get(lock_key):
                logger.warning(f"Pipeline {lock_key} already running — rejected")
                return PipelineResult(mode=mode, strategy=strategy, status="REJECTED",
                                      error="already_running")
            self._locks[lock_key] = True

        if not is_trading_day():
            logger.info(f"{date.today()} not a trading day — pipeline skipped")
            return PipelineResult(mode=mode, strategy=strategy, status="SKIPPED",
                                  is_trading_day=False)

        try:
            return self._execute(mode, strategy)
        finally:
            with self._lock_obj:
                self._locks.pop(lock_key, None)

    def run_all(self, mode: str = "daily") -> list[PipelineResult]:
        """Run both strategies sequentially. A0-A5 shared; A5 computes both strategies in one pass."""
        results = []
        self._shared_cache = {}  # Reset per cycle
        # Bypass trading day check — orchestrator runs when invoked
        # long_term: full pipeline, A5 computes scores for BOTH strategies
        result_lt = self._execute(mode, "long_term")
        results.append(result_lt)
        if result_lt.status == "ABORTED":
            return results
        # hot_picks: skip A0-A5 (already done), run A7+A6 only
        result_hp = self._execute(mode, "hot_picks", skip_early_stages=True)
        results.append(result_hp)
        return results

    def status(self) -> dict:
        """Current pipeline status for API."""
        return {
            "current": {k: v.to_dict() for k, v in self._current.items()
                        if v.status in ("RUNNING",)},
            "last_5": [r.to_dict() for r in self._history[-5:]],
        }

    # ── Internal execution ──────────────────────────────────

    def _execute(self, mode: str, strategy: str, skip_early_stages: bool = False) -> PipelineResult:
        """Execute the full 9-agent DAG with closed-loop monitoring."""
        from backend.agents.agent_0_tier import run as a0
        from backend.agents.agent_1_technical import run as a1
        from backend.agents.agent_2_fundamental import run as a2_fund
        from backend.agents.agent_3_news import run as a3_news
        from backend.agents.agent_4_macro import run as a4_macro
        from backend.agents.agent_5_fusion import run as a5_fusion
        from backend.agents.agent_7_portfolio import run as a7_port
        from backend.agents.agent_6_risk import run as a6_risk
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        timeout = settings.daily_timeout if mode == "daily" else settings.weekly_timeout
        lock_key = f"{mode}/{strategy}"

        logger.info(f"╔══ Pipeline START: {lock_key} @ {trade_date}")
        t0 = time.time()

        # ── Data freshness check ──
        conn = get_connection()
        freshness = conn.execute(
            "SELECT MAX(trade_date) as latest, CAST(julianday('now') - julianday(MAX(trade_date)) AS INTEGER) as age_days "
            "FROM daily_quotes"
        ).fetchone()
        conn.close()
        if freshness and freshness["age_days"] and freshness["age_days"] > 2:
            logger.warning(f"  ⚠ daily_quotes stale: {freshness['age_days']}d old (latest={freshness['latest']})")

        # Persist run start
        conn = get_connection()
        cur = conn.execute(
            "INSERT INTO pipeline_runs (mode, strategy, status, started_at, agents_total) VALUES (?,?,?,?,?)",
            (mode, strategy, "RUNNING", trade_date, 0),
        )
        result = PipelineResult(id=cur.lastrowid, mode=mode, strategy=strategy,
                                status="RUNNING", started_at=trade_date)
        if freshness and freshness["age_days"] and freshness["age_days"] > 2:
            result.error = f"STALE_DATA: daily_quotes {freshness['age_days']}d old"
        conn.commit()
        conn.close()
        self._current[lock_key] = result

        def _run(name, stage_num, fn, **kwargs) -> StageResult:
            """Run one agent with retry. Executor always shut down to prevent thread leaks."""
            sr = StageResult(stage=stage_num, agent=name, status="RUNNING")
            per_stage_timeout = {
                "A0_Universe": 120, "A1_Tech": 900, "A3_News": 300,
                "A4_Macro": 180, "A5_Fusion": 600, "A7_Portfolio": 2400,
                "A6_Risk": 2400,  # reason() adversarial review, 18 batches × 90s
            }.get(name, timeout)

            for attempt in range(settings.agent_retries):
                executor = ThreadPoolExecutor(max_workers=1)
                t1 = time.time()
                try:
                    future = executor.submit(fn, **kwargs)
                    future.result(timeout=per_stage_timeout)
                    sr.elapsed = round(time.time() - t1, 1)
                    sr.status = "OK"
                    logger.info(f"  [{name}] OK ({sr.elapsed:.0f}s)")
                    return sr
                except TimeoutError:
                    future.cancel()
                    executor.shutdown(wait=True)  # wait for thread to actually stop
                    sr.elapsed = round(time.time() - t1, 1)
                    sr.error = f"timeout after {per_stage_timeout}s"
                    sr.error_category = "TIMEOUT"
                    logger.error(f"  [{name}] TIMEOUT — not retrying (timeout is terminal)")
                    return sr  # timeout is terminal, don't retry
                except Exception as e:
                    sr.elapsed = round(time.time() - t1, 1)
                    sr.error = str(e)[:200]
                    sr.error_category = _categorize_error(e)
                    logger.error(f"  [{name}] FAILED: {sr.error[:100]}")
                finally:
                    executor.shutdown(wait=False)
                if attempt < settings.agent_retries - 1:
                    time.sleep(settings.retry_backoff ** attempt)
            sr.status = "FAIL"
            return sr

        daily_tiers = ["HOLDING", "FAVORED", "NEUTRAL"]
        weekly_tiers = ["HOLDING", "FAVORED", "NEUTRAL"]
        tiers = daily_tiers if mode == "daily" else weekly_tiers

        try:
            # ═══ Stage 0: Data Freshness Check + Auto-Fetch ═══
            dc = get_connection()
            dq = dc.execute(
                "SELECT MAX(trade_date) as latest FROM daily_quotes"
            ).fetchone()
            dc.close()
            td_gap = 0
            if dq and dq["latest"]:
                latest_date = datetime.strptime(dq["latest"][:10], "%Y-%m-%d").date()
                d = latest_date + __import__('datetime').timedelta(days=1)
                today = date.today()
                while d <= today:
                    if is_trading_day(d):
                        td_gap += 1
                    d += __import__('datetime').timedelta(days=1)

            # Only fetch if data is both stale AND insufficient for scored stocks
            need_fetch = False
            if td_gap > 0:
                dc2 = get_connection()
                scored_n = dc2.execute("SELECT COUNT(*) FROM composite_scores WHERE strategy=? AND calc_date=(SELECT MAX(calc_date) FROM composite_scores WHERE strategy=?)", (strategy, strategy)).fetchone()[0]
                covered_n = dc2.execute("SELECT COUNT(DISTINCT ts_code) FROM daily_quotes WHERE trade_date=(SELECT MAX(trade_date) FROM daily_quotes)").fetchone()[0]
                dc2.close()
                need_fetch = covered_n < scored_n  # only fetch if coverage gap

            if need_fetch:
                logger.info(f"  ⚠ daily_quotes {td_gap}d gap AND coverage {covered_n}<{scored_n} → auto-fetching...")
                try:
                    from backend.data.fetcher import daily_update
                    daily_update()
                    logger.info(f"  ✅ Data fetch complete")
                    # Invalidate A1/A4 cache — data changed, re-analysis needed
                    self._shared_cache.pop("A1_Tech", None)
                    self._shared_cache.pop("A4_Macro", None)
                    result.stages.append(StageResult(stage=0, agent="Data_Fetch",
                                                    status="OK", error=f"fetched {td_gap}d gap"))
                except Exception as e:
                    logger.warning(f"  Data fetch failed: {e}")
                    result.stages.append(StageResult(stage=0, agent="Data_Fetch",
                                                    status="OK", error=f"fetch failed"))
            elif td_gap > 0:
                logger.info(f"  ⚠ {td_gap}d stale but coverage ok ({covered_n}>={scored_n}) → skip fetch")
                result.stages.append(StageResult(stage=0, agent="Data_Freshness",
                                                status="OK", error=f"{td_gap}d stale but sufficient data"))
            else:
                result.stages.append(StageResult(stage=0, agent="Data_Freshness",
                                                status="OK", error=f"fresh (latest={dq['latest'][:10] if dq else 'none'})"))

            # ═══ Skip early stages for hot_picks: A5 already computed both strategies ═══
            if skip_early_stages:
                result.stages.append(StageResult(stage=1, agent="A0_Universe", status="SKIPPED",
                                                 error="hot_picks — A0 skipped"))
                result.stages.append(StageResult(stage=2, agent="A2_Fund", status="OK",
                                                 error="runs as independent server Worker"))
                result.stages.append(StageResult(stage=2, agent="A3_News", status="SKIPPED",
                                                 error="hot_picks — reused from long_term"))
                result.stages.append(StageResult(stage=2, agent="A1_Tech", status="SKIPPED",
                                                 error="hot_picks — reused from long_term"))
                result.stages.append(StageResult(stage=2, agent="A4_Macro", status="SKIPPED",
                                                 error="hot_picks — reused from long_term"))
                result.stages.append(StageResult(stage=3, agent="A5_Fusion", status="SKIPPED",
                                                 error="hot_picks — computed with long_term A5"))
                # Rebuild FL for this strategy
                try:
                    from backend.focus_list import rebuild as rebuild_focus
                    fl = rebuild_focus(strategy=strategy)
                    logger.info(f"  Focus list [{strategy}]: {len(fl)} stocks")
                except Exception as e:
                    logger.error(f"  Focus list rebuild failed: {e}")
                # Jump to Stage 4
                sr = _run("A7_Portfolio", 4, lambda: a7_port(mode=mode, trade_date=trade_date, strategy=strategy))
                if sr.status != "OK":
                    sr.error = (sr.error or "") + " [degraded: upstream may have issues]"
                result.stages.append(sr)
                sr = _run("A6_Risk", 5, lambda: a6_risk(strategy=strategy, trade_date=trade_date))
                if sr.status != "OK":
                    sr.error = (sr.error or "") + " [A7 decisions remain PENDING — review manually]"
                result.stages.append(sr)
                result.status = "COMPLETED" if not result.failed_stages() else "COMPLETED_WITH_ERRORS"
                result.completed_at = datetime.now().isoformat()
                _persist_result(result)
                self._history.append(result)
                return result

            # ═══ Stage 1: A0 (weekly long_term only) ═══
            if mode == "weekly" and strategy == "long_term":
                sr = _run("A0_Universe", 1, a0, strategy=strategy, trade_date=trade_date, mode=mode)
                result.stages.append(sr)
                if sr.status != "OK":
                    result.status = "ABORTED"
                    result.error = f"A0_Universe failed: {sr.error}"
                    return result
            else:
                result.stages.append(StageResult(stage=1, agent="A0_Universe",
                                                  status="SKIPPED", error="daily or hot_picks"))

            # ═══ Stage 2: A3 first (sync), then A1+A4 (parallel) ═══
            # A2 runs as independent server Worker, NOT as pipeline daemon

            result.stages.append(StageResult(stage=2, agent="A2_Fund", status="OK",
                                            error="runs as independent server Worker"))
            stage2_results = {}

            # A3 runs first (synchronous — slow, HTTP+LLM, avoid concurrent DB writes)
            sr_a3 = _run("A3_News", 2, a3_news)
            result.stages.append(sr_a3)
            stage2_results["A3_News"] = sr_a3

            stage2_tasks = {
                "A4_Macro":   (a4_macro, {"trade_date": trade_date}),
                "A1_Tech":    (a1, {"tiers": tiers, "trade_date": trade_date}),
            }
            stage2_results = {"A3_News": sr_a3}

            # Check shared cache from first strategy (hot_picks reuses long_term's A1/A3/A4)
            for name in list(stage2_tasks.keys()):
                if name in self._shared_cache:
                    sr = self._shared_cache[name]
                    result.stages.append(StageResult(stage=2, agent=name, status=sr.status,
                                                    error=f"reused from long_term ({sr.elapsed:.0f}s)"))
                    stage2_results[name] = sr
                    del stage2_tasks[name]
                    logger.info(f"  [{name}] REUSED from long_term ({sr.elapsed:.0f}s)")

            if stage2_tasks:
                stage_timeout = 900  # A1 indicator computation can be slow
                with ThreadPoolExecutor(max_workers=len(stage2_tasks)) as pool:
                    futures = {
                        pool.submit(fn, **kw): name
                        for name, (fn, kw) in stage2_tasks.items()
                    }
                    pending = dict(futures)
                    try:
                        for fut in as_completed(futures, timeout=stage_timeout):
                            name = pending.pop(fut, futures[fut])
                            t1 = time.time()
                            sr = None
                            for attempt in range(settings.agent_retries):
                                try:
                                    fut.result()  # already completed, no timeout needed
                                    sr = StageResult(stage=2, agent=name, status="OK",
                                                     elapsed=round(time.time() - t1, 1))
                                    break
                                except Exception as e:
                                    sr = StageResult(stage=2, agent=name, status="FAIL",
                                                     error=str(e)[:200],
                                                     error_category=_categorize_error(e),
                                                     elapsed=round(time.time() - t1, 1))
                                    if attempt < settings.agent_retries - 1:
                                        # Re-submit on failure
                                        time.sleep(settings.retry_backoff ** attempt)
                                        new_fut = pool.submit(fn, **kw)
                                        try:
                                            new_fut.result(timeout=stage_timeout)
                                            sr = StageResult(stage=2, agent=name, status="OK",
                                                             elapsed=round(time.time() - t1, 1))
                                            break
                                        except Exception as e2:
                                            sr = StageResult(stage=2, agent=name, status="FAIL",
                                                             error=str(e2)[:200],
                                                             error_category=_categorize_error(e2),
                                                             elapsed=round(time.time() - t1, 1))
                            if sr:
                                result.stages.append(sr)
                                stage2_results[name] = sr
                                if sr.status == "OK" and name in ("A1_Tech", "A4_Macro"):
                                    self._shared_cache[name] = sr
                                logger.info(f"  [{name}] {sr.status} ({sr.elapsed:.0f}s)")
                    except TimeoutError:
                        # as_completed timeout — some stages didn't finish
                        for name in pending.values():
                            sr = StageResult(stage=2, agent=name, status="TIMEOUT",
                                             error=f"stage timeout after {stage_timeout}s",
                                             error_category="TIMEOUT")
                            result.stages.append(sr)
                            stage2_results[name] = sr
                            logger.error(f"  [{name}] TIMEOUT (stage deadline)")

            s2_ok = all(sr.status == "OK" for sr in stage2_results.values())

            # ═══ Stage 3: A5 Fusion (needs A1+A4; A2 is background, uses whatever available) ═══
            needed = {"A1_Tech", "A4_Macro"}
            if all(stage2_results.get(n, StageResult(stage=2, agent=n, status="FAIL")).status == "OK"
                   for n in needed):
                extra = ["hot_picks"] if strategy == "long_term" else None
                sr = _run("A5_Fusion", 3, a5_fusion, trade_date=trade_date, strategy=strategy,
                          extra_strategies=extra)
                result.stages.append(sr)
                if sr.status == "OK":
                    try:
                        from backend.focus_list import rebuild as rebuild_focus
                        fl = rebuild_focus(strategy=strategy)
                        logger.info(f"  Focus list [{strategy}]: {len(fl)} stocks")
                    except Exception as e:
                        logger.error(f"  Focus list rebuild failed: {e}")
            else:
                missing = [n for n in needed if stage2_results.get(n, StageResult(stage=2, agent=n, status="FAIL")).status != "OK"]
                sr = StageResult(stage=3, agent="A5_Fusion", status="SKIPPED",
                                 error=f"prerequisites failed: {missing}",
                                 error_category="DATA")
                result.stages.append(sr)

            # ═══ Stage 4: A7 (portfolio construction + risk annotation) ═══
            sr = _run("A7_Portfolio", 4, lambda: a7_port(mode=mode, trade_date=trade_date, strategy=strategy))
            if sr.status != "OK":
                sr.error = (sr.error or "") + " [degraded: upstream may have issues]"
            result.stages.append(sr)

            # ═══ Stage 5: A6 Risk Officer (adversarial review of A7 decisions) ═══
            sr = _run("A6_Risk", 5, lambda: a6_risk(strategy=strategy, trade_date=trade_date))
            if sr.status != "OK":
                sr.error = (sr.error or "") + " [A7 decisions remain PENDING — review manually]"
            result.stages.append(sr)

            result.status = "COMPLETED" if not result.failed_stages() else "COMPLETED_WITH_ERRORS"

        except Exception as e:
            logger.error(f"Pipeline crash: {e}")
            result.status = "ABORTED"
            result.error = str(e)[:200]

        finally:
            result.completed_at = datetime.now().isoformat()
            _persist_result(result)
            self._history.append(result)
            if len(self._history) > 100:
                self._history = self._history[-100:]
            logger.info(f"╚══ Pipeline {lock_key}: {result.status} "
                        f"({sum(1 for s in result.stages if s.status=='OK')} OK / "
                        f"{len(result.failed_stages())} FAIL) "
                        f"in {time.time()-t0:.0f}s")

        return result


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _categorize_error(e: Exception) -> str:
    """Categorize an error for monitoring."""
    msg = str(e).lower()
    if any(kw in msg for kw in ("akshare", "http", "connection", "timeout",
                                  "fetch", "remote", "api")):
        return "DATA"
    if any(kw in msg for kw in ("api_key", "llm", "chat", "deepseek", "token")):
        return "LLM"
    if "timeout" in msg:
        return "TIMEOUT"
    return "UNKNOWN"


def _persist_result(r: PipelineResult):
    """Write final pipeline status to DB."""
    try:
        conn = get_connection()
        conn.execute(
            "UPDATE pipeline_runs SET status=?, completed_at=?, agents_total=?, agents_ok=?, agents_failed=? WHERE id=?",
            (r.status, r.completed_at,
             len(r.stages),
             sum(1 for s in r.stages if s.status == "OK"),
             len(r.failed_stages()),
             r.id),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to persist pipeline result: {e}")


# Singleton accessor
def get_orchestrator() -> Orchestrator:
    return Orchestrator()


# ═══════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StockScope Pipeline Orchestrator")
    parser.add_argument("--mode", choices=["daily", "weekly"], default="daily")
    parser.add_argument("--strategy", choices=["long_term", "hot_picks", "both"], default="long_term")
    args = parser.parse_args()

    orch = Orchestrator()
    if args.strategy == "both":
        results = orch.run_all(mode=args.mode)
    else:
        results = [orch.run(mode=args.mode, strategy=args.strategy)]

    for r in results:
        print(f"\n{r.mode}/{r.strategy}: {r.status}")
        for s in r.stages:
            flag = "✓" if s.status == "OK" else "✗" if s.status == "FAIL" else "·"
            print(f"  {flag} Stage {s.stage} {s.agent} ({s.status}) {s.elapsed:.0f}s")
            if s.error:
                print(f"    └─ [{s.error_category}] {s.error[:120]}")

    # Auto-generate HTML reports
    try:
        from backend.report import generate_html_report
        strategies = list(set(r.strategy for r in results))
        for s in strategies:
            generate_html_report(s)
    except Exception as e:
        print(f"Report generation failed: {e}")
