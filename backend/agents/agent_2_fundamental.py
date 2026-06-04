"""
Agent 2: Fundamental Analyst — deep multi-quarter financial analysis.

Data: baostock (primary) → akshare (field supplement) → 10jqka (last resort)
Scope: HOLDING → FAVORED → NEUTRAL. AVOID/EXCLUDED skipped.
Mode: async background worker, triggered by A0. Incremental updates.
"""
import json
import time
import logging
from datetime import datetime

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()

ANALYSIS_TIERS = ["HOLDING", "FAVORED", "NEUTRAL"]
MAX_QUARTERS = 8

# ═══════════════════════════════════════════════════════════════
# baostock — multi-quarter data
# ═══════════════════════════════════════════════════════════════

def _to_bs_code(ts_code: str) -> str:
    parts = ts_code.split(".")
    if len(parts) != 2:
        return None
    return f"{'sz' if parts[1].upper() == 'SZ' else 'sh'}.{parts[0]}"


def _safe_float(val) -> float | None:
    if val is None or str(val).strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _latest_quarter():
    now = datetime.now()
    if now.month <= 4:
        return now.year - 1, 3
    elif now.month <= 8:
        return now.year, 1
    elif now.month <= 10:
        return now.year, 2
    else:
        return now.year, 3


def _quarter_generator(start_yr, start_q, count):
    """Generate (year, quarter) tuples walking backward from start."""
    yr, q = start_yr, start_q
    for _ in range(count):
        yield yr, q
        q -= 1
        if q == 0:
            q = 4
            yr -= 1


def fetch_from_baostock(ts_code: str, num_quarters: int = MAX_QUARTERS) -> dict | None:
    """Fetch ALL available quarters of financial data from baostock.

    Returns dict of lists keyed by category, each list is one quarter's DataFrame.
    Assumes bs.login() already called.
    """
    import baostock as bs
    bs_code = _to_bs_code(ts_code)
    if not bs_code:
        return None

    ly, lq = _latest_quarter()
    result = {"profit": [], "growth": [], "balance": [],
              "cash_flow": [], "operation": [], "dupont": []}

    queries = [
        ("profit",    bs.query_profit_data),
        ("growth",    bs.query_growth_data),
        ("balance",   bs.query_balance_data),
        ("cash_flow", bs.query_cash_flow_data),
        ("operation", bs.query_operation_data),
        ("dupont",    bs.query_dupont_data),
    ]

    try:
        for yr, q in _quarter_generator(ly, lq, num_quarters + 2):
            for cat, fn in queries:
                rs = fn(code=bs_code, year=yr, quarter=q)
                if rs.error_code == "0":
                    df = rs.get_data()
                    if not df.empty:
                        # Tag with period for later alignment
                        df = df.copy()
                        df["_period"] = f"{yr}-{str(q*3).zfill(2)}-01"
                        result[cat].append(df)
            if len(result["profit"]) >= num_quarters:
                break

        # Valuation from K-line
        rs = bs.query_history_k_data_plus(
            bs_code, "date,close,peTTM,pbMRQ,psTTM",
            start_date=f"{ly-2}-01-01",
            end_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="d", adjustflag="2",
        )
        if rs.error_code == "0":
            df = rs.get_data()
            if not df.empty:
                result["valuation"] = df
    except Exception as e:
        logger.warning(f"baostock error for {ts_code}: {e}")

    return result if result["profit"] else None


def extract_baostock_metrics(bs_data: dict, ts_code: str) -> dict:
    """Extract multi-quarter time-series metrics from baostock DataFrames.

    Returns dict with 'quarters' list (newest first) and a flat latest-period summary.
    baostock ratio values are decimals (0.015 = 1.5%) → converted to %.
    """
    _pct = lambda v: round(v * 100, 2) if v is not None else None
    _rnd = lambda v, d: round(v, d) if v is not None else None

    quarters = []
    # Align by period across categories
    periods = set()
    for cat_df in bs_data.get("profit", []):
        periods.add(str(cat_df.iloc[0].get("_period", "")))
    periods = sorted(periods, reverse=True)

    for period in periods:
        def _row(cat):
            for df in bs_data.get(cat, []):
                if str(df.iloc[0].get("_period", "")) == period:
                    return df.iloc[0]
            return None

        p = _row("profit")
        g = _row("growth")
        b = _row("balance")
        c = _row("cash_flow")
        o = _row("operation")
        d = _row("dupont")

        def _v(row, col):
            return _safe_float(row[col]) if row is not None and col in (row.index if hasattr(row, 'index') else []) else None

        q = {
            "period": period,
            "roe": _pct(_v(p, "roeAvg") or _v(d, "dupontROE")),
            "gross_margin": _pct(_v(p, "gpMargin")),
            "net_margin": _pct(_v(p, "npMargin")),
            "eps": _rnd(_v(p, "epsTTM"), 4),
            "net_profit": _v(p, "netProfit"),
            "revenue": _v(p, "MBRevenue"),
            "profit_yoy": _pct(_v(g, "YOYNI") or _v(g, "YOYPNI")),
            "equity_yoy": _pct(_v(g, "YOYEquity")),
            "asset_yoy": _pct(_v(g, "YOYAsset")),
            "debt_ratio": _pct(_v(b, "debtAssetRatio")),
            "current_ratio": _rnd(_v(b, "currentRatio"), 2),
            "quick_ratio": _rnd(_v(b, "quickRatio"), 2),
            "cfo_sales": _pct(_v(c, "CFOSales")),
            "ar_turnover": _rnd(_v(o, "ARTRate"), 2),
            "inv_turnover": _rnd(_v(o, "INVTRate"), 2),
        }
        quarters.append(q)

    # Valuation (latest day)
    val_df = bs_data.get("valuation")
    pe_ttm = _safe_float(val_df.iloc[-1]["peTTM"]) if val_df is not None and not val_df.empty else None
    pb = _safe_float(val_df.iloc[-1]["pbMRQ"]) if val_df is not None and not val_df.empty else None
    ps = _safe_float(val_df.iloc[-1]["psTTM"]) if val_df is not None and not val_df.empty else None

    latest = quarters[0] if quarters else {}
    return {
        "ts_code": ts_code,
        "quarters": quarters,  # NEW: full time-series
        "quarter_count": len(quarters),
        "latest_period": latest.get("period", ""),
        # Flat summary fields (backward compat + quick access)
        "roe": latest.get("roe"),
        "gross_margin": latest.get("gross_margin"),
        "net_margin": latest.get("net_margin"),
        "debt_ratio": latest.get("debt_ratio"),
        "eps": latest.get("eps"),
        "net_profit": latest.get("net_profit"),
        "revenue": latest.get("revenue"),
        "profit_yoy": latest.get("profit_yoy"),
        "revenue_yoy": None,
        "net_yoy": latest.get("profit_yoy"),
        "equity_yoy": latest.get("equity_yoy"),
        "asset_yoy": latest.get("asset_yoy"),
        "current_ratio": latest.get("current_ratio"),
        "quick_ratio": latest.get("quick_ratio"),
        "cfo_sales": latest.get("cfo_sales"),
        "cfo_ni_ratio": None,
        "ar_turnover": latest.get("ar_turnover"),
        "inv_turnover": latest.get("inv_turnover"),
        "pe_ttm": _rnd(pe_ttm, 2), "pb": _rnd(pb, 2), "ps": _rnd(ps, 2),
        "fcf_ratio": None, "pe_percentile": None,
        "data_source": "baostock",
        "operating_cf": None, "receivables": None, "inventory": None,
        "goodwill": None, "goodwill_ratio": None,
        "total_assets": None, "total_equity": None, "total_debt": None,
    }

# ═══════════════════════════════════════════════════════════════
# akshare supplement
# ═══════════════════════════════════════════════════════════════

def supplement_from_akshare(ts_code: str, metrics: dict) -> dict:
    """Fill missing fields using akshare. Mutates metrics in-place."""
    missing = [f for f in ["debt_ratio", "revenue_yoy", "operating_cf",
                            "receivables", "goodwill", "total_assets"]
               if metrics.get(f) is None]
    if not missing:
        return metrics

    import akshare as ak
    code = ts_code.split(".")[0]

    def _try(fn):
        try: return fn()
        except Exception: return None

    # THS abstract
    ths = _try(lambda: ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期"))
    if ths is not None and not ths.empty:
        row = ths.iloc[0]
        def _num(col):
            try:
                v = row.get(col)
                if v is None or str(v) == "nan": return None
                if isinstance(v, str):
                    if "%" in v: return float(v.replace("%", ""))
                    if "亿" in v: return float(v.replace("亿", "")) * 1e8
                    if "万" in v: return float(v.replace("万", "")) * 1e4
                return float(v)
            except (ValueError, TypeError): return None
        if metrics.get("debt_ratio") is None:
            metrics["debt_ratio"] = _num("资产负债率(%)") or _num("资产负债率")
        if metrics.get("revenue_yoy") is None:
            metrics["revenue_yoy"] = _num("营业收入同比增长率(%)")

    # EM raw statements
    try:
        bs_df = _try(lambda: ak.stock_balance_sheet_by_report_em(symbol=code))
        cf_df = _try(lambda: ak.stock_cash_flow_sheet_by_report_em(symbol=code))
        def _get(df, names):
            if df is None or df.empty: return None
            for n in names:
                if n in df.columns:
                    try:
                        v = df.iloc[0][n]
                        return float(v) if v is not None else None
                    except: pass
            return None
        if metrics.get("total_assets") is None:
            metrics["total_assets"] = _get(bs_df, ["资产总计", "资产总额"])
        if metrics.get("total_equity") is None:
            metrics["total_equity"] = _get(bs_df, ["归属于母公司所有者权益合计", "所有者权益合计"])
        if metrics.get("total_debt") is None:
            metrics["total_debt"] = _get(bs_df, ["负债合计", "负债总额"])
        if metrics.get("receivables") is None:
            metrics["receivables"] = _get(bs_df, ["应收账款", "应收票据及应收账款"])
        if metrics.get("goodwill") is None:
            metrics["goodwill"] = _get(bs_df, ["商誉"])
            if metrics["goodwill"] and metrics.get("total_equity"):
                metrics["goodwill_ratio"] = round(metrics["goodwill"] / metrics["total_equity"] * 100, 2)
        if metrics.get("operating_cf") is None:
            metrics["operating_cf"] = _get(cf_df, ["经营活动产生的现金流量净额"])
    except Exception:
        pass

    op_cf = metrics.get("operating_cf")
    np_val = metrics.get("net_profit")
    if op_cf and np_val and np_val != 0:
        metrics["cfo_ni_ratio"] = round(op_cf / np_val, 2)

    filled = [f for f in missing if metrics.get(f) is not None]
    if filled:
        metrics["data_source"] = f"baostock+akshare({','.join(filled[:3])})"
    return metrics

# ═══════════════════════════════════════════════════════════════
# 10jqka web scraping
# ═══════════════════════════════════════════════════════════════

def _scrape_10jqka(ts_code: str) -> dict | None:
    import re, requests
    code = ts_code.split(".")[0]
    try:
        r = requests.get(f"https://basic.10jqka.com.cn/{code}/finance.html", timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        })
        r.encoding = r.apparent_encoding or "gbk"
        text = r.text
    except Exception:
        return None

    result = {}
    patterns = [
        (r"本期毛利率([\d.]+)%,.*?去年同期为([\d.]+)%", lambda m: {"gross_margin": float(m.group(1))}),
        (r"本期净利率([\d.]+)%,.*?去年同期为([\d.]+)%", lambda m: {"net_margin": float(m.group(1))}),
        (r"本期净资产收益率([\d.]+)%.*?去年同期为([\d.]+)%", lambda m: {"roe": float(m.group(1))}),
        (r"本期营业收入增长率([\-\d.]+)%.*?去年同期为([\-\d.]+)%", lambda m: {"revenue_yoy": float(m.group(1))}),
        (r"本期净利润增长率([\-\d.]+)%.*?去年同期为([\-\d.]+)%", lambda m: {"profit_yoy": float(m.group(1)), "net_yoy": float(m.group(1))}),
        (r"本期资产负债率([\d.]+)%.*?去年同期为([\d.]+)%", lambda m: {"debt_ratio": float(m.group(1))}),
        (r"本期流动比率([\d.]+).*?去年同期为([\d.]+)", lambda m: {"current_ratio": float(m.group(1))}),
        (r"本期速动比率([\d.]+).*?去年同期为([\d.]+)", lambda m: {"quick_ratio": float(m.group(1))}),
        (r"本期应收账款周转率([\d.]+).*?去年同期为([\d.]+)", lambda m: {"ar_turnover": float(m.group(1))}),
        (r"本期存货周转率([\d.]+).*?去年同期为([\d.]+)", lambda m: {"inv_turnover": float(m.group(1))}),
    ]
    for pat, extractor in patterns:
        m = re.search(pat, text)
        if m:
            try: result.update(extractor(m))
            except (ValueError, AttributeError): pass
    return result if result else None


def _supplement_from_10jqka(ts_code: str, metrics: dict) -> dict:
    missing = [f for f in ["debt_ratio", "revenue_yoy", "gross_margin", "net_margin",
                            "roe", "current_ratio", "quick_ratio", "profit_yoy",
                            "ar_turnover", "inv_turnover"] if metrics.get(f) is None]
    if not missing:
        return metrics
    s = _scrape_10jqka(ts_code)
    if not s:
        return metrics
    filled = 0
    for f in missing:
        if s.get(f) is not None and metrics.get(f) is None:
            metrics[f] = s[f]
            filled += 1
    if filled and "10jqka" not in str(metrics.get("data_source", "")):
        metrics["data_source"] = f"{metrics.get('data_source','unknown')}+10jqka"
    return metrics

# ═══════════════════════════════════════════════════════════════
# Unified fetch
# ═══════════════════════════════════════════════════════════════

def fetch_financials(ts_code: str) -> tuple[dict | None, str]:
    """Multi-source, multi-quarter financial data acquisition."""
    # Primary: baostock (all quarters)
    bs_data = fetch_from_baostock(ts_code)
    if bs_data:
        metrics = extract_baostock_metrics(bs_data, ts_code)
        if metrics and metrics.get("quarters"):
            supplement_from_akshare(ts_code, metrics)
            _supplement_from_10jqka(ts_code, metrics)
            return metrics, metrics.get("data_source", "baostock")

    # Full fallback: akshare (single quarter)
    import akshare as ak
    code = ts_code.split(".")[0]
    for label, fn in [
        ("akshare_ths", lambda: ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")),
    ]:
        try:
            df = fn()
            if df is not None and not df.empty:
                m = _extract_akshare_fallback(df, ts_code, label)
                if m:
                    _supplement_from_10jqka(ts_code, m)
                    return m, label
        except Exception:
            pass

    # Last resort: 10jqka
    s = _scrape_10jqka(ts_code)
    if s and any(s.values()):
        m = {"ts_code": ts_code, "quarters": [], "quarter_count": 0,
             "latest_period": "?", "data_source": "10jqka_scrape"}
        for k in ["roe","gross_margin","net_margin","debt_ratio","revenue_yoy",
                   "profit_yoy","net_yoy","current_ratio","quick_ratio"]:
            m[k] = s.get(k)
        m["pe_ttm"] = m["pb"] = m["ps"] = m["eps"] = None
        m["net_profit"] = m["revenue"] = None
        m["fcf_ratio"] = m["pe_percentile"] = m["cfo_ni_ratio"] = None
        m["operating_cf"] = m["receivables"] = m["inventory"] = m["goodwill"] = None
        m["goodwill_ratio"] = m["total_assets"] = m["total_equity"] = m["total_debt"] = None
        return m, "10jqka_scrape"

    logger.warning(f"  {ts_code}: ALL channels failed")
    return None, "none"


def _extract_akshare_fallback(df, ts_code: str, source: str) -> dict | None:
    if df is None or df.empty: return None
    row = df.iloc[0]
    def num(col):
        try:
            v = row.get(col)
            if v is None or str(v) in ("nan", "False"): return None
            if isinstance(v, str):
                if "%" in v: return float(v.replace("%", ""))
                if "亿" in v: return float(v.replace("亿", "")) * 1e8
                if "万" in v: return float(v.replace("万", "")) * 1e4
            return float(v)
        except: return None
    return {
        "ts_code": ts_code, "quarters": [], "quarter_count": 0,
        "latest_period": str(row.get("报告期", ""))[:10],
        "roe": num("净资产收益率(%)") or num("净资产收益率"),
        "gross_margin": num("销售毛利率(%)") or num("毛利率(%)"),
        "net_margin": num("销售净利率(%)") or num("净利率(%)"),
        "debt_ratio": num("资产负债率(%)"),
        "revenue_yoy": num("营业收入同比增长率(%)"),
        "net_yoy": num("净利润同比增长率(%)"),
        "profit_yoy": None, "eps": num("基本每股收益"),
        "net_profit": None, "revenue": None,
        "equity_yoy": None, "asset_yoy": None,
        "current_ratio": None, "quick_ratio": None,
        "cfo_sales": None, "cfo_ni_ratio": None,
        "ar_turnover": None, "inv_turnover": None,
        "pe_ttm": None, "pb": None, "ps": None,
        "fcf_ratio": None, "pe_percentile": None,
        "operating_cf": None, "receivables": None, "inventory": None,
        "goodwill": None, "goodwill_ratio": None,
        "total_assets": None, "total_equity": None, "total_debt": None,
        "data_source": source,
    }

# ═══════════════════════════════════════════════════════════════
# Red-flag detection (uses multi-quarter data)
# ═══════════════════════════════════════════════════════════════

def detect_red_flags(metrics: dict) -> list[dict]:
    """Programmatic red-flag detection. Uses multi-quarter trend data if available."""
    flags = []
    qs = metrics.get("quarters", [])

    # Latest quarter red flags
    cfo_ni = metrics.get("cfo_ni_ratio")
    if cfo_ni is not None and cfo_ni < 0.5:
        flags.append({"severity": "HIGH", "flag": "经营现金流恶化",
                      "detail": f"CFO/NI = {cfo_ni:.2f}，显著低于 1.0"})

    debt = metrics.get("debt_ratio") or 0
    if debt > 80:
        flags.append({"severity": "HIGH", "flag": "负债率过高",
                      "detail": f"资产负债率 = {debt:.1f}%"})

    net_profit = metrics.get("net_profit")
    if net_profit is not None and net_profit < 0:
        flags.append({"severity": "HIGH", "flag": "净利润为负",
                      "detail": f"净利润 = {net_profit/1e8:.2f}亿"})

    roe = metrics.get("roe")
    if roe is not None and roe < 3:
        flags.append({"severity": "MEDIUM", "flag": "ROE过低",
                      "detail": f"ROE = {roe:.1f}%"})

    quick = metrics.get("quick_ratio")
    if quick is not None and quick < 0.5:
        flags.append({"severity": "MEDIUM", "flag": "速动比率过低",
                      "detail": f"速动比率 = {quick:.2f}"})

    goodwill_ratio = metrics.get("goodwill_ratio")
    if goodwill_ratio is not None and goodwill_ratio > 30:
        flags.append({"severity": "MEDIUM" if goodwill_ratio < 50 else "HIGH",
                      "flag": "商誉减值风险",
                      "detail": f"商誉/净资产 = {goodwill_ratio:.1f}%"})

    # Multi-quarter trend flags
    if len(qs) >= 4:
        # Consecutive margin decline
        gms = [q.get("gross_margin") for q in qs[:4] if q.get("gross_margin") is not None]
        if len(gms) >= 3 and all(gms[i] > gms[i+1] for i in range(len(gms)-1)):
            flags.append({"severity": "MEDIUM", "flag": "毛利率持续下滑",
                          "detail": f"近{len(gms)}季: {' → '.join(f'{g:.1f}%' for g in gms)}"})

        # Consecutive revenue decline
        revs = [q.get("revenue") for q in qs[:4] if q.get("revenue") is not None]
        if len(revs) >= 3 and all(revs[i] is not None and revs[i+1] is not None and revs[i] < revs[i+1] * 0.95
                                  for i in range(len(revs)-1)):
            flags.append({"severity": "MEDIUM", "flag": "营收持续萎缩",
                          "detail": f"近{len(revs)}季营收连续下降"})

        # ROE trend
        roes = [q.get("roe") for q in qs[:4] if q.get("roe") is not None]
        if len(roes) >= 3 and all(roes[i] < roes[i+1] for i in range(len(roes)-1)):
            flags.append({"severity": "MEDIUM", "flag": "ROE趋势下行",
                          "detail": f"近{len(roes)}季: {' → '.join(f'{r:.1f}%' for r in roes)}"})

    return flags

# ═══════════════════════════════════════════════════════════════
# LLM analysis — full multi-quarter context
# ═══════════════════════════════════════════════════════════════

def analyze_fundamental_narrative(ts_code: str, name: str, metrics: dict,
                                   red_flags: list) -> dict | None:
    """LLM deep analysis with full multi-quarter financial history."""
    if not settings.ds_api_key:
        return None

    cfg = settings.get_llm_config("A2")
    quarters = metrics.get("quarters", [])
    quarter_count = len(quarters)

    # Build a clean summary table for the prompt
    qtable = []
    for q in quarters:
        qtable.append({
            "期": q.get("period", "?")[2:].replace("-", "Q"),  # "2026-03-31" → "26Q1"
            "ROE": q.get("roe"), "毛利率": q.get("gross_margin"),
            "净利率": q.get("net_margin"), "利润YoY": q.get("profit_yoy"),
            "负债率": q.get("debt_ratio"), "速动比率": q.get("quick_ratio"),
            "营收": q.get("revenue"), "净利润": q.get("net_profit"),
        })

    latest = quarters[0] if quarters else {}
    prev = quarters[1] if len(quarters) > 1 else {}

    # Compute quarter-over-quarter changes
    qoq_lines = []
    if prev:
        rev_curr = latest.get("revenue"); rev_prev = prev.get("revenue")
        if rev_curr and rev_prev and rev_prev != 0:
            qoq_lines.append(f"营收环比: {(rev_curr/rev_prev - 1)*100:+.1f}%")
        profit_curr = latest.get("net_profit"); profit_prev = prev.get("net_profit")
        if profit_curr and profit_prev and profit_prev != 0:
            qoq_lines.append(f"利润环比: {(profit_curr/profit_prev - 1)*100:+.1f}%")
        gm_curr = latest.get("gross_margin"); gm_prev = prev.get("gross_margin")
        if gm_curr is not None and gm_prev is not None:
            qoq_lines.append(f"毛利率变化: {gm_curr - gm_prev:+.1f}个百分点")
        debt_curr = latest.get("debt_ratio"); debt_prev = prev.get("debt_ratio")
        if debt_curr is not None and debt_prev is not None:
            trend = "上升" if debt_curr > debt_prev + 2 else "下降" if debt_curr < debt_prev - 2 else "持平"
            qoq_lines.append(f"负债率趋势: {trend} ({debt_prev:.0f}%→{debt_curr:.0f}%)")
        roe_curr = latest.get("roe"); roe_prev = prev.get("roe")
        if roe_curr is not None and roe_prev is not None:
            qoq_lines.append(f"ROE变化: {roe_curr - roe_prev:+.1f}个百分点")

    qoq_str = "\n".join(qoq_lines) if qoq_lines else "无(仅有单季度数据)"

    prompt = f"""你是财务数据分析师。以下是{name}（{ts_code}）近{quarter_count}个季度的财务数据。

=== 逐季趋势（最新在上） ===
{json.dumps(qtable, ensure_ascii=False, indent=2)}

=== 最新季度快照 ===
ROE={metrics.get('roe')}%  毛利率={metrics.get('gross_margin')}%  净利率={metrics.get('net_margin')}%
负债率={metrics.get('debt_ratio')}%  速动比率={metrics.get('quick_ratio')}  流动比率={metrics.get('current_ratio')}
PE={metrics.get('pe_ttm')}  PB={metrics.get('pb')}  PS={metrics.get('ps')}
数据源={metrics.get('data_source','?')}

=== 环比变化（最新 vs 上一季度）===
{qoq_str}

=== 程序检测的风险信号 ===
{json.dumps(red_flags, ensure_ascii=False, indent=2) if red_flags else "无"}

基于以上数据做分析。以提供的数据为唯一依据，数据中未包含的信息不应假设。如数据中存在矛盾或异常，在相关维度中说明。

1. 盈利特征 (rating: HIGH/MEDIUM/LOW/UNKNOWN, positive_factors, negative_factors)
   - 从ROE水平及多季度趋势、毛利率变动方向、经营现金流与净利润的匹配程度，描述公司盈利特征
   - 重要：数据不足(如仅单季度/缺现金流/ROE异常但无法确认原因)时，使用UNKNOWN而非LOW。LOW必须是盈利明确恶化(如多季度利润持续下滑/ROE长期低于行业/经营现金流持续为负)。

2. 增长特征 (rating: HIGH/MEDIUM/LOW/UNKNOWN, organic_growth_pct, anomaly_notes)
   - 从多季度营收和利润的变动趋势，描述公司的增长模式
   - 标注数据中的异常波动（例如利润大幅变动而营收基本不变时，描述两者差异及可能的财务原因）
   - 重要：单季度数据无法判断增长趋势时，使用UNKNOWN。

3. 财务结构 (rating: GOOD/FAIR/POOR/UNKNOWN, debt_concern, liquidity_concern)
   - 从负债率趋势、流动/速动比率的变化方向、净资产的增长或收缩，描述公司的财务结构
   - 重要：POOR必须是财务结构明确恶化(如负债率>80%且持续上升/速动比率<0.5/净资产持续缩水)。数据不足时使用UNKNOWN。

4. 估值背景 (rating: OVERPRICED/FAIR/UNDERPRICED/UNKNOWN, reasoning)
   - 结合增速和盈利水平，描述当前PE/PB/PS所反映的市场估值状态；利润为负时侧重PB/PS
   - 重要：PE/PB/PS全部缺失且无替代指标时，使用UNKNOWN并注明"关键估值数据缺失"

5. 红旗核验 — 逐一核对程序检测的风险信号：数据支持则确认，数据不支持则驳回，同时可补充数据中发现的其他风险信号

6. 数据局限 — 列出本次分析受限于数据的问题（例如仅{quarter_count}个季度的时间跨度、单一公司数据无法跨公司比较、表外事项不可见等）

7. 边际变动 — 基于环比数据，描述最近季度各项指标的变动方向：哪些在改善、哪些在恶化、哪些基本稳定

8. 近期观察 — 基于现有数据（不做预测）：
   - 最近季度出现的变化模式
   - 财务数据中的矛盾点（例如利润增长而经营现金流反向变动）

9. 综合描述 — 200-500字，基于以上各维度分析，形成对该公司的整体描述

输出JSON：
{{"earnings_quality":{{"rating":"HIGH/MEDIUM/LOW/UNKNOWN","positive_factors":["..."],"negative_factors":["..."]}},"growth_quality":{{"rating":"HIGH/MEDIUM/LOW/UNKNOWN","organic_growth_pct":null,"anomaly_notes":["..."]}},"financial_health":{{"rating":"GOOD/FAIR/POOR/UNKNOWN","debt_concern":false,"liquidity_concern":false}},"valuation":{{"rating":"OVERPRICED/FAIR/UNDERPRICED/UNKNOWN","reasoning":"..."}},"red_flags":[{{"severity":"HIGH/MEDIUM/LOW","flag":"...","detail":"..."}}],"blind_spots":["..."],"marginal_change":"最近季度改善/恶化信号","short_term_notes":"短期关注点","narrative":"...","confidence":0.0-1.0}}"""
    result = llm.chat_json(prompt, model=cfg["model"], max_tokens=cfg["max_tokens"],
                           temperature=cfg.get("temperature", 0.3))

    # ── Sanitize ratings to valid enum values (safety net for LLM deviations) ──
    if result:
        VALID_RATINGS = {
            "earnings_quality": ("HIGH", "MEDIUM", "LOW", "UNKNOWN"),
            "growth_quality": ("HIGH", "MEDIUM", "LOW", "UNKNOWN"),
            "financial_health": ("GOOD", "FAIR", "POOR", "UNKNOWN"),
            "valuation": ("OVERPRICED", "FAIR", "UNDERPRICED", "UNKNOWN"),
        }
        for section, valid in VALID_RATINGS.items():
            obj = result.get(section)
            if isinstance(obj, dict) and obj.get("rating") not in valid:
                logger.warning(
                    f"A2 sanitize: {ts_code} {section}.rating={obj.get('rating')} → "
                    f"{'FAIR' if section == 'valuation' else 'MEDIUM'} (invalid, clamped)"
                )
                obj["rating"] = "FAIR" if section == "valuation" else "MEDIUM"

    return result

# ═══════════════════════════════════════════════════════════════
# Report persistence
# ═══════════════════════════════════════════════════════════════

def save_fundamental_report(conn, ts_code: str, calc_date: str, metrics: dict,
                             red_flags: list, llm_result: dict | None) -> dict:
    report = {
        "ts_code": ts_code, "calc_date": calc_date,
        "financial_period": metrics.get("latest_period", ""),
        "quarter_count": metrics.get("quarter_count", 0),
        "quarters": metrics.get("quarters", []),  # full history
        "data_source": metrics.get("data_source", "unknown"),
        "metrics_summary": {
            "roe": metrics.get("roe"), "gross_margin": metrics.get("gross_margin"),
            "net_margin": metrics.get("net_margin"), "debt_ratio": metrics.get("debt_ratio"),
            "profit_yoy": metrics.get("profit_yoy"), "revenue_yoy": metrics.get("revenue_yoy"),
            "cfo_ni_ratio": metrics.get("cfo_ni_ratio"),
            "current_ratio": metrics.get("current_ratio"),
            "quick_ratio": metrics.get("quick_ratio"),
            "pe_ttm": metrics.get("pe_ttm"), "pb": metrics.get("pb"),
        },
        "earnings_quality": (llm_result or {}).get("earnings_quality"),
        "growth_quality": (llm_result or {}).get("growth_quality"),
        "financial_health": (llm_result or {}).get("financial_health"),
        "valuation": (llm_result or {}).get("valuation"),
        "red_flags": (llm_result or {}).get("red_flags", red_flags),
        "blind_spots": (llm_result or {}).get("blind_spots", []),
        "narrative": (llm_result or {}).get("narrative", ""),
        "confidence": (llm_result or {}).get("confidence", 0),
        "data_quality_notes": _data_quality_notes(metrics),
    }
    report_json = json.dumps(report, ensure_ascii=False)
    conn.execute("""INSERT OR REPLACE INTO fundamental_reports
        (ts_code, calc_date, report_json, overall_score)
        VALUES (?,?,?,?)""", (ts_code, calc_date, report_json, None))
    return report


def _data_quality_notes(metrics: dict) -> list[str]:
    notes = []
    src = metrics.get("data_source", "")
    qc = metrics.get("quarter_count", 0)
    if qc >= 6:
        notes.append(f"{qc}个季度数据，趋势分析可靠")
    elif qc >= 3:
        notes.append(f"{qc}个季度数据，趋势可参考但不够完整")
    else:
        notes.append("季度数据不足，趋势分析受限")
    if "akshare" in src:
        notes.append("含akshare补充字段")
    if "10jqka" in src:
        notes.append("含同花顺补充字段")
    return notes

# ═══════════════════════════════════════════════════════════════
# Incremental update logic
# ═══════════════════════════════════════════════════════════════

def _needs_update(conn, ts_code: str) -> bool:
    """Check if a stock's report needs updating based on latest financial period."""
    row = conn.execute(
        "SELECT report_json, calc_date FROM fundamental_reports "
        "WHERE ts_code=? ORDER BY calc_date DESC LIMIT 1",
        (ts_code,),
    ).fetchone()
    if not row or not row["report_json"]:
        return True

    try:
        existing = json.loads(row["report_json"])
        existing_period = existing.get("financial_period", "")
        existing_date = row["calc_date"]

        ly, lq = _latest_quarter()
        latest_period = f"{ly}-{str(lq*3).zfill(2)}-01"

        if existing_period < latest_period:
            return True

        try:
            last_date = datetime.strptime(existing_date[:10], "%Y-%m-%d")
            if (datetime.now() - last_date).days > 90:
                return True
        except (ValueError, TypeError):
            return True
    except (json.JSONDecodeError, KeyError):
        return True
    return False

# ═══════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════

def run(tiers=None, trade_date=None, strategy="long_term"):
    """Analyze fundamentals for target stocks. Async background worker."""
    if tiers is None:
        tiers = ANALYSIS_TIERS
    if trade_date is None:
        trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    t0 = time.time()
    conn = get_connection()
    logger.info(f"Agent 2 ({strategy}) starting: tiers={tiers}")

    bs_session = None
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code == "0":
            bs_session = True
            logger.info("baostock session established")
        else:
            logger.warning(f"baostock login failed: {lg.error_msg}")
    except Exception as e:
        logger.warning(f"baostock unavailable: {e}")

    try:
        ph = ",".join("?" * len(tiers))
        rows = conn.execute(
            f"""SELECT ts_code, tier FROM tier_assignments
                WHERE tier IN ({ph})
                ORDER BY CASE tier WHEN 'HOLDING' THEN 0 WHEN 'FAVORED' THEN 1 WHEN 'NEUTRAL' THEN 2 ELSE 3 END""",
            tiers,
        ).fetchall()
        codes = [r["ts_code"] for r in rows]

        if codes:
            sp = ",".join("?" * len(codes))
            srows = conn.execute(
                f"SELECT ts_code, name FROM stocks WHERE ts_code IN ({sp})", codes
            ).fetchall()
            code_to_name = {r["ts_code"]: r["name"] for r in srows}
        else:
            code_to_name = {}

        need_analysis = []
        skipped = 0
        for code in codes:
            if _needs_update(conn, code):
                need_analysis.append(code)
            else:
                skipped += 1

        logger.info(f"Target: {len(codes)} → {len(need_analysis)} need update ({skipped} current)")

        processed = 0
        no_data = 0
        for i, code in enumerate(need_analysis):
            tier = next((r["tier"] for r in rows if r["ts_code"] == code), "?")
            stock_name = code_to_name.get(code, code)

            metrics, source = fetch_financials(code)
            if not metrics:
                no_data += 1
                continue

            flags = detect_red_flags(metrics)
            llm_result = None
            if settings.ds_api_key:
                llm_result = analyze_fundamental_narrative(code, stock_name, metrics, flags)

            save_fundamental_report(conn, code, trade_date, metrics, flags, llm_result)
            _save_financials_compat(conn, code, trade_date, metrics, llm_result)

            processed += 1
            if (i + 1) % 25 == 0:
                conn.commit()
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                logger.info(f"  {i+1}/{len(need_analysis)} [{tier}] {code} src={source} "
                           f"q={metrics.get('quarter_count',0)} flags={len(flags)} ({rate:.1f}/s)")

        conn.commit()
        elapsed = time.time() - t0
        logger.info(f"Agent 2 complete: {processed} reports ({no_data} no-data, {skipped} current) in {elapsed:.0f}s")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, stocks_processed, duration_s, summary) "
            "VALUES (2, ?, 'SUCCESS', ?, ?, ?)",
            (trade_date, processed, elapsed,
             f"{strategy}: {processed} new, {skipped} current, {no_data} no-data"),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 2 failed: {e}")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (2, ?, 'FAILED', ?, ?)",
            (trade_date, time.time() - t0, str(e)[:200]),
        )
        conn.commit()
        raise
    finally:
        if bs_session:
            try:
                import baostock as bs
                bs.logout()
            except Exception:
                pass
        conn.close()


def _get_prev_metrics(conn, ts_code: str) -> dict | None:
    row = conn.execute(
        "SELECT report_json FROM fundamental_reports WHERE ts_code=? ORDER BY calc_date DESC LIMIT 1",
        (ts_code,),
    ).fetchone()
    if not row or not row["report_json"]:
        return None
    try:
        return json.loads(row["report_json"]).get("metrics_summary", {})
    except (json.JSONDecodeError, KeyError):
        return None


def _save_financials_compat(conn, ts_code, trade_date, metrics, llm_result):
    conn.execute("""INSERT OR REPLACE INTO financials
        (ts_code, report_date, roe, gross_margin, net_margin, debt_ratio,
         revenue_yoy, fcf_ratio, pe_ttm, pb, raw_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (ts_code, trade_date,
         metrics.get("roe"), metrics.get("gross_margin"), metrics.get("net_margin"),
         metrics.get("debt_ratio"),
         metrics.get("revenue_yoy") or metrics.get("profit_yoy"),
         metrics.get("fcf_ratio"), metrics.get("pe_ttm"), metrics.get("pb"),
         json.dumps({"source": metrics.get("data_source"),
                     "narrative": (llm_result or {}).get("narrative", "")},
                    ensure_ascii=False)))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--tiers", type=str, nargs="+", default=ANALYSIS_TIERS)
    p.add_argument("--strategy", type=str, default="long_term")
    args = p.parse_args()
    run(tiers=args.tiers, strategy=args.strategy)
