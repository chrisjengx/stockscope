"""
Agent 3: News Intelligence — multi-source financial news with LLM entity linking.

Sources (~10/30/30/30 ratio):
  1. 新浪财经 JSON API   — 股票频道 (10%)
  2. 联合早报 HTML         — 中国财经/国际财经/中国 (30%)
  3. 华尔街见闻 live API   — real-time financial news (30%)
  4. 同花顺 news API       — stock push news (30%)

Downstream: A6 (risk review — per-stock + market context), API (frontend display + SSE)
"""
import json
import hashlib
import re
import time
import logging
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from backend.data.schema import get_connection
from backend.config import get_settings
from backend.lib.llm_client import get_llm
from backend.lib.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
llm = get_llm()

MAX_NEWS = 500
LLM_BATCH_SIZE = 15

# ── Source 1: 新浪 JSON API ──────────────────────────────────

SINA_FEEDS = [
    (2510, "股票"),  # 1 channel, ~10% of total
]


def fetch_sina():
    """Fetch from Sina finance roll JSON API. Returns list of {title, url, body_snippet}."""
    items = []
    for lid, name in SINA_FEEDS:
        try:
            r = requests.get(
                f"https://feed.mix.sina.com.cn/api/roll/get?pageid=153&lid={lid}&k=&num=20&page=1",
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
                timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                articles = data.get("result", {}).get("data", [])
                for a in articles:
                    items.append({
                        "title": a.get("title", ""),
                        "url": a.get("url", ""),
                        "body_snippet": "",  # API doesn't provide body
                        "source": f"sina_{name}",
                    })
                logger.info(f"  sina_{name}: {len(articles)} articles")
        except Exception as e:
            logger.warning(f"  sina_{name}: {e}")
    return items


# ── Source 2: 联合早报 ───────────────────────────────────────

ZAOBAO_URLS = [
    ("https://www.zaobao.com.sg/finance/china", "zaobao_china_fin"),
    ("https://www.zaobao.com.sg/finance/world", "zaobao_world_fin"),
    ("https://www.zaobao.com.sg/news/china", "zaobao_china"),
]


def fetch_zaobao():
    """Scrape headlines from 联合早报 finance pages."""
    items = []
    for url, label in ZAOBAO_URLS:
        try:
            r = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.content, "html.parser")
            count = 0
            for a in soup.find_all("a"):
                text = a.get_text(strip=True)
                href = a.get("href", "")
                if 15 < len(text) < 200 and href and ("/finance/" in href or "/news/" in href or "/story/" in href):
                    if not href.startswith("http"):
                        href = f"https://www.zaobao.com.sg{href}" if href.startswith("/") else href
                    items.append({
                        "title": text,
                        "url": href,
                        "body_snippet": "",
                        "source": label,
                    })
                    count += 1
            logger.info(f"  {label}: {count} articles")
        except Exception as e:
            logger.warning(f"  {label}: {e}")
    return items


# ── Source 3: 华尔街见闻 JSON API ────────────────────────────

def fetch_wallstreetcn():
    """Fetch real-time financial news from 华尔街见闻 live API."""
    items = []
    try:
        r = requests.get(
            "https://api-one.wallstcn.com/apiv1/content/lives?channel=global-channel&limit=40",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            articles = data.get("data", {}).get("items", []) if isinstance(data.get("data"), dict) else data.get("data", [])
            for a in articles:
                if not isinstance(a, dict):
                    continue
                title = a.get("title") or a.get("content_text", "")
                if len(title) < 10:
                    continue
                items.append({
                    "title": title[:200],
                    "url": a.get("uri", ""),
                    "body_snippet": a.get("content_text", "")[:500],
                    "source": "wallstreetcn",
                })
        logger.info(f"  wallstreetcn: {len(items)} articles")
    except Exception as e:
        logger.warning(f"  wallstreetcn: {e}")
    return items


# ── Source 4: 同花顺 news API ─────────────────────────────────

def fetch_10jqka():
    """Fetch news from 同花顺 (10jqka) stock news API."""
    items = []
    try:
        r = requests.get(
            "https://news.10jqka.com.cn/tapp/news/push/stock/?page=1&tag=",
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            # Response: {code, data: {list: [...], total: N}}
            articles = []
            d = data.get("data")
            if isinstance(d, dict):
                articles = d.get("list", [])
            elif isinstance(d, list):
                articles = d
            for a in articles:
                title = a.get("title", "") or a.get("subject", "")
                if len(title) < 10:
                    continue
                items.append({
                    "title": title[:200],
                    "url": a.get("url", "") or a.get("shareurl", ""),
                    "body_snippet": a.get("digest", "")[:500],
                    "source": "10jqka",
                })
        logger.info(f"  10jqka: {len(items)} articles")
    except Exception as e:
        logger.warning(f"  10jqka: {e}")
    return items



# ── Entity extraction ─────────────────────────────────────────

# Preload all stock codes + names at module level for fast lookup
_stock_names: dict = {}
_stock_codes: set = set()


def _preload_stocks(conn):
    """Preload stock name→code mappings for fast entity extraction."""
    global _stock_names, _stock_codes
    if _stock_names:
        return
    for r in conn.execute("SELECT ts_code, name FROM stocks WHERE name IS NOT NULL AND name != ''").fetchall():
        name_clean = r["name"].replace(" ", "").replace("(", "").replace(")", "")
        if len(name_clean) >= 3:
            _stock_names[name_clean] = r["ts_code"]
        _stock_codes.add(r["ts_code"][:6])


def extract_entities(title: str, body: str) -> list[dict]:
    """Extract stock entities from text. Returns [{code, relevance, confidence}]."""
    text = f"{title} {body}"
    entities = []
    seen = set()

    # Phase 1: Direct 6-digit stock code detection
    codes_found = set(re.findall(r'\b(\d{6})\b', text))
    for code in codes_found:
        if code in _stock_codes and code not in seen:
            suffix = "SZ" if code.startswith(("0", "3")) else "SH"
            entities.append({"code": f"{code}.{suffix}", "relevance": "PRIMARY", "confidence": 0.95})
            seen.add(code)

    # Phase 2: Company name matching against preloaded stock names
    for name_clean, ts_code in _stock_names.items():
        if ts_code[:6] not in seen and name_clean in text:
            entities.append({"code": ts_code, "relevance": "SECONDARY",
                           "confidence": 0.80 if len(name_clean) >= 4 else 0.60})
            seen.add(ts_code[:6])

    return entities


# ── Dedup ─────────────────────────────────────────────────────

def content_hash(title, url):
    return hashlib.md5(f"{title}{url}".encode()).hexdigest()


def dedup(conn, items):
    new_items = []
    for item in items:
        h = content_hash(item["title"], item.get("url", ""))
        existing = conn.execute("SELECT id FROM news_feed WHERE content_hash=?", (h,)).fetchone()
        if not existing:
            item["hash"] = h
            new_items.append(item)
    return new_items


# ── LLM classification (batch) ───────────────────────────────

def classify_batch(conn, items):
    """Classify news items in batches. LLM → keyword fallback per item."""
    cfg = settings.get_llm_config("A3")
    all_results = []

    for i in range(0, len(items), LLM_BATCH_SIZE):
        batch = items[i:i + LLM_BATCH_SIZE]
        if settings.ds_api_key:
            batch_results = _llm_classify_batch(batch, cfg)
        else:
            batch_results = None

        for j, item in enumerate(batch):
            if batch_results and j < len(batch_results):
                r = batch_results[j]
                # LLM entity linking: use LLM-provided stock codes, validated against DB
                entities = _build_entities_from_llm(r.get("related_stocks", []))
                # Affected industries: LLM free-text keywords → stored for A7 context
                affected = r.get("affected_industries", [])
                if not isinstance(affected, list):
                    affected = []
                all_results.append({
                    "category": r.get("category", "市场评论"),
                    "sentiment": r.get("sentiment", "neutral"),
                    "sentiment_intensity": r.get("sentiment_intensity", 0.5),
                    "impact": r.get("impact", "MEDIUM"),
                    "tags": ",".join(r.get("tags", [])),
                    "summary": r.get("summary", item["title"][:100]),
                    "related_stocks_json": json.dumps(entities, ensure_ascii=False),
                    "related_stocks_compat": ",".join(e["code"] for e in entities),
                    "is_breaking": 1 if r.get("is_breaking") else 0,
                    "classification_confidence": 0.8,
                    "llm_insight": r.get("insight", ""),
                    "affected_industries": affected,
                })
            else:
                # Keyword fallback: use regex + string matching entity extraction
                entities = extract_entities(item["title"], item.get("body_snippet", ""))
                cat, sent, impact = _keyword_classify(item["title"])
                all_results.append({
                    "category": cat, "sentiment": sent, "sentiment_intensity": 0.5,
                    "impact": impact, "tags": "",
                    "summary": item["title"][:100],
                    "related_stocks_json": json.dumps(entities, ensure_ascii=False),
                    "related_stocks_compat": ",".join(e["code"] for e in entities),
                    "is_breaking": 0, "classification_confidence": 0.3,
                    "llm_insight": "",
                    "affected_industries": [],
                })

    return all_results


def _build_entities_from_llm(llm_codes: list) -> list[dict]:
    """Convert LLM-provided stock codes to validated entity list."""
    if not isinstance(llm_codes, list) or not llm_codes:
        return []
    entities = []
    seen = set()
    for raw in llm_codes:
        if not isinstance(raw, str):
            continue
        code = raw.strip().upper()
        # Validate: must match 6-digit+exchange format and exist in DB
        prefix = code[:6]
        if prefix in _stock_codes and code not in seen:
            suffix = "PRIMARY" if prefix.startswith(("0", "3")) else ""
            entities.append({"code": code, "relevance": "PRIMARY", "confidence": 0.90})
            seen.add(code)
    return entities


def _llm_classify_batch(items, cfg):
    """Classify a batch of news items in one LLM call."""
    lines = [
        "你是财经新闻分析师。识别对A股市场有实质影响的新闻，忽略噪音。",
        "你的分析会影响后续的选股决策——对重大利好/利空不要遗漏。",
        "对每条新闻输出分类、情感、影响力，以及该新闻影响的行业。",
    ]
    lines.append("")
    for idx, item in enumerate(items):
        lines.append(f"[{idx}] {item['title'][:120]}")
    lines.append("")
    lines.append("输出JSON数组（每条新闻一个对象）：")
    lines.append('[{"idx":0,"category":"行业政策/公司事件/宏观数据/市场评论",')
    lines.append('"sentiment":"positive/negative/neutral","sentiment_intensity":0.0-1.0,')
    lines.append('"impact":"HIGH/MEDIUM/LOW","tags":["..."],"summary":"一句话摘要",')
    lines.append('"affected_industries":["新能源","锂电池"],  // 数组内容仅为示例，实际行业由你根据新闻自行判断')
    lines.append('"is_breaking":false,"insight":"这条新闻对A股投资的潜在影响（1-2句话，无影响则留空）"}]')

    result = llm.chat_json("\n".join(lines), model=cfg["model"], max_tokens=cfg["max_tokens"])
    if result:
        items_list = result if isinstance(result, list) else result.get("items", result.get("news", []))
        if isinstance(items_list, list):
            return items_list
    return None


def _keyword_classify(title):
    """Fast keyword-based classification fallback."""
    category = "市场评论"
    for cat, keywords in {
        "行业政策": ["政策", "监管", "发改委", "工信部", "国务院", "补贴", "规划", "央行", "美联储"],
        "公司事件": ["业绩", "公告", "并购", "重组", "减持", "增持", "分红", "回购", "立案", "处罚"],
        "宏观数据": ["GDP", "PMI", "CPI", "社融", "利率", "汇率", "通胀"],
    }.items():
        if any(kw in title for kw in keywords):
            category = cat
            break
    pos = sum(1 for kw in ["增长", "利好", "突破", "超预期", "涨停", "创新高", "中标", "获批"] if kw in title)
    neg = sum(1 for kw in ["下跌", "亏损", "暴雷", "违约", "处罚", "调查", "退市", "减持", "跌停"] if kw in title)
    sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"
    impact = "HIGH" if any(kw in title for kw in ["政策", "监管", "处罚", "调查", "重组", "并购", "立案"]) else \
             "LOW" if any(kw in title for kw in ["研报", "评级", "概念"]) else "MEDIUM"
    return category, sentiment, impact


# ── Main entry point ──────────────────────────────────────────

def run():
    """Fetch news from all sources, classify, store. Non-blocking."""
    start = time.time()
    trade_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    conn = get_connection()
    _preload_stocks(conn)
    logger.info(f"Agent 3 starting: {len(_stock_names)} stock names loaded")

    try:
        # Fetch from all sources
        all_items = fetch_sina() + fetch_zaobao() + fetch_wallstreetcn() + fetch_10jqka()
        logger.info(f"Fetched {len(all_items)} raw items from 3 sources")

        # Dedup
        new_items = dedup(conn, all_items)
        logger.info(f"After dedup: {len(new_items)} new / {len(all_items)} total")

        if not new_items:
            logger.info("No new items — skipping classification")
            return

        # Classify
        classifications = classify_batch(conn, new_items)

        # Store
        stored = 0
        for item, cls in zip(new_items, classifications):
            insight = cls.get("llm_insight", "")
            quant_info = json.dumps({
                "insight": insight,
                "is_breaking": cls.get("is_breaking", 0),
                "affected_industries": cls.get("affected_industries", []),
            }, ensure_ascii=False)
            conn.execute(
                """INSERT OR IGNORE INTO news_feed
                   (source, category, tags, sentiment, impact,
                    related_stocks, related_stocks_json,
                    summary, body_summary,
                    content_hash, consumer, published_at, raw_url,
                    quantitative_info, classification_confidence)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (item["source"], cls["category"], cls["tags"],
                 cls["sentiment"], cls["impact"],
                 cls.get("related_stocks_compat", ""),
                 cls.get("related_stocks_json", "[]"),
                 cls["summary"],
                 item.get("body_snippet", "")[:500],
                 item["hash"],
                 "agent_5,agent_7",
                 datetime.now().isoformat(),
                 item.get("url", ""),
                 quant_info,
                 cls.get("classification_confidence", 0.3)),
            )
            stored += 1

        # Trim old news: keep latest MAX_NEWS
        conn.execute(f"""
            DELETE FROM news_feed WHERE id NOT IN (
                SELECT id FROM news_feed ORDER BY published_at DESC LIMIT {MAX_NEWS}
            )
        """)
        deleted = conn.execute("SELECT changes()").fetchone()[0]

        conn.commit()
        elapsed = time.time() - start
        logger.info(f"Agent 3 complete: {stored} new, {deleted} old cleaned in {elapsed:.1f}s")

        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (3, ?, 'SUCCESS', ?, ?)",
            (trade_date, elapsed, f"{stored} news from sina+10jqka+zaobao+wallstcn"),
        )
        conn.commit()

    except Exception as e:
        logger.error(f"Agent 3 failed: {e}")
        conn.execute(
            "INSERT INTO agent_logs (agent_id, run_date, status, duration_s, summary) "
            "VALUES (3, ?, 'FAILED', ?, ?)",
            (trade_date, time.time() - start, str(e)[:200]),
        )
        conn.commit()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run()
