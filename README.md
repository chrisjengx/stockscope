# StockScope

A股多因子量化选股系统。8-Agent LLM Pipeline + 夜间基本面分析，双策略（长期价值 + 热点动量），全链路透明。

## 架构

```
A0(Gate) → A1(Tech)+A4(Macro)∥A3(News) → A5(Fusion) → FL(Classify) → A7(Portfolio) → A6(Risk)
               A2(Fund) 夜间 20:00–08:30
```

9 个 Agent 各司其职：A0 分级 → A1 技术指标 → A2 基本面（夜间）→ A3 新闻 → A4 宏观 → A5 多因子排名 → FL 品类分类 → A7 持仓构建 → A6 风险审查。

A7 和 A6 形成对抗决策：A7 是乐观的交易员推荐买入，A6 是悲观的风险官审查否决。

## 双策略

| | long_term | hot_picks |
|--|-----------|-----------|
| 周期 | 2-4 周 | 3-5 天 |
| 风格 | 价值+成长+稳健 | 动量+突破+短期 |
| 核心驱动 | 基本面+趋势质量 | 动量强度+量价配合 |

## 快速开始

### 环境要求

- Python 3.12+
- macOS / Linux
- DeepSeek API Key（LLM 调用）

### 安装

```bash
git clone https://github.com/your-username/stock-analysis.git
cd stock-analysis

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install akshare baostock flask flask-cors numpy pandas requests beautifulsoup4 lxml tqdm
```

### 配置

```bash
export StockAgent_DS_API_KEY="your-deepseek-api-key"
```

数据库自动创建（首次运行时 `schema.py` 用 `CREATE TABLE IF NOT EXISTS` 建表）。

### 首次运行

```bash
# 1. 拉取数据（全量，约10分钟）
python -c "from backend.data.fetcher import daily_update; daily_update()"

# 2. 跑首次 Pipeline（两个策略会按顺序独立运行）
python -m backend.orchestrator --mode daily --strategy both

# 3. 生成报告
python -c "from backend.report import generate_html_report; generate_html_report('long_term'); generate_html_report('hot_picks')"
```

报告在 `report/` 目录。

### 启动服务

```bash
python -m backend.api.server
```

服务启动后自动运行调度器：

| 时间 | 事件 |
|------|------|
| 08:30 | A2 夜间分析停止 |
| 13:15 | 定向数据拉取 |
| 14:05 | Pipeline #1 + HTML 报告 |
| 16:30 | 全量数据拉取 |
| 18:00 | A0 股票分级（周一到六） |
| 19:00 | Pipeline #2 + HTML 报告 |
| 20:00 | A2 夜间分析启动 |

API 服务在 `http://localhost:5001`。

### API 端点

| 端点 | 说明 |
|------|------|
| `GET /api/dashboard` | 仪表盘数据（市场状态、持仓、Agent 状态） |
| `GET /api/reports/macro` | 最新宏观分析报告 |
| `GET /api/reports/fundamental/<ts_code>` | 单股基本面报告 |
| `GET /api/reports/fusion?strategy=long_term` | A5 融合分析报告 |
| `GET /api/decisions` | 最新投资决策 |
| `POST /api/pipeline/run?mode=daily&strategy=both` | 手动触发 Pipeline（两个策略独立运行） |

## 核心设计

### A7 持仓构建（三层架构）

```
Layer 1: Conviction（纯数学）— 动量质量级联 + 基本面 + 惩罚
Layer 2: LLM 选股（分批 15 只/批）— 宏观/板块/新闻/A5 全量上下文
Layer 3: 硬约束 — 权重上下限 + 总仓位上限 + 安全上限 25 只
```

### A6 风险审查（对抗制衡）

程序化检查（6 维度）+ LLM 深度审查 + risk_score 1-5 校准锚 + 排名淘汰（前 60% APPROVED，risk≥4 强制 VETO）。

### 板块数据

A4 从同花顺获取 90 个行业板块实时数据（涨跌幅 + 净流入 + 涨跌比），同时取 Top/Bottom 板块 K 线计算 5日/3日趋势。注入 A7 和 A6 的 LLM prompt 作为选股偏好参考。

### 设计原则

- **定性替代定量** — 市场判断用语言描述而非数字约束，让 LLM 自主判断
- **数学做安全网，LLM 做增值** — Conviction 防止系统性错误，LLM 在安全边界内创造价值
- **板块是偏好不是约束** — 同一段数据 A7 找机会、A6 找风险
- **对抗决策** — A7 乐观推荐，A6 悲观审查

## 目录结构

```
stock-analysis/
├── backend/
│   ├── agents/          # 9 个 Agent
│   │   ├── agent_0_tier.py        # 股票分级
│   │   ├── agent_1_technical.py   # 技术指标
│   │   ├── agent_2_fundamental.py # 基本面（夜间）
│   │   ├── agent_3_news.py        # 新闻采集
│   │   ├── agent_4_macro.py       # 宏观判断
│   │   ├── agent_5_fusion.py      # 多因子排名
│   │   ├── agent_6_risk.py        # 风险审查
│   │   └── agent_7_portfolio.py   # 持仓构建
│   ├── api/server.py     # Flask API + 调度器
│   ├── data/schema.py    # SQLite schema
│   ├── data/fetcher.py   # 数据拉取
│   ├── orchestrator.py   # Pipeline DAG 执行器
│   ├── focus_list.py     # FL 品类分类
│   ├── report.py         # HTML 报告生成
│   ├── config.py         # 配置
│   └── lib/              # 工具库（LLM client, logging）
├── docs/
│   └── backend-arch.html # 完整架构文档
├── data/                 # SQLite 数据库（gitignore）
├── report/               # HTML 报告（gitignore）
├── logs/                 # 日志（gitignore）
├── CLAUDE.md             # AI Agent 指令
└── README.md
```

## License

MIT
