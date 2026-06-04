# StockScope — A股多因子量化选股系统

## Architecture

```
A0(排除) → A1(Tech)+A4(Macro)∥A3(News) → A5(Fusion) → FL(Classify) → A7(Portfolio) → A6(Risk)
               A2(Fund) runs 20:00-08:30 night worker
```

**Dual strategy**: long_term (value+growth+stable) and hot_picks (momentum+breakout+short_term).

## Key Decisions

- BATCH=15 for A7/A6 LLM calls (100% coverage, zero mismatch)
- Per-batch proportional selection (7-15%, min 1)
- Qualitative over quantitative: regime_summary replaces risk_budget everywhere
- A4 uses THS sector data (90 industries, 5419 stocks) — no DIY aggregation
- Sector preference (not constraint): injected as LLM context in A7/A6
- Hard REJECT rules replaced with conviction penalties
- d3 early signal strengthened in momentum_quality (0.60-0.70)
- Percentile-based FL classification (adaptive to bull/bear)
- A5 single weight set (tech20/fund35/mom30/rs15), FL handles strategy split
- A7 dual strategy: long_term (fundamental-heavy) + hot_picks (momentum-heavy)
- A7 strategy-differentiated: conviction weights, tier thresholds, LLM role, sell logic
- A6 risk_score 1-5 with calibration anchors, adversarial verdict (60% APPROVED)

## Schedule

| Time | Event |
|------|-------|
| 08:30 | A2 Worker stop |
| 13:15 | Data fetch (HOLDING+FAVORED+NEUTRAL) |
| 14:00 | Pipeline #1 + HTML report |
| 16:30 | Data fetch (full close) |
| 18:00 | A0 Gate (Mon-Sat) |
| 18:45 | Pipeline #2 + HTML report |
| 20:00 | A2 Worker start |

## Commands

```bash
# Start server
python -m backend.api.server

# Run pipeline manually
python -m backend.orchestrator --mode daily --strategy both

# Generate reports only
python -c "from backend.report import generate_html_report; generate_html_report('long_term'); generate_html_report('hot_picks')"
```

## Key Files

| File | Purpose |
|------|---------|
| backend/orchestrator.py | Pipeline DAG executor |
| backend/agents/agent_0_tier.py | Stock universe gate |
| backend/agents/agent_5_fusion.py | Multi-factor scoring |
| backend/agents/agent_7_portfolio.py | Portfolio construction (core) |
| backend/agents/agent_6_risk.py | Adversarial risk review |
| backend/focus_list.py | Stock classification + FL builder |
| backend/report.py | HTML report generator |
| backend/api/server.py | Flask API + scheduler |
| backend/config.py | All configuration |
| backend/data/schema.py | SQLite schema + migrations |
