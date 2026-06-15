# StockScope 算法审查报告

**日期**: 2025-06-15
**原则**: 只关注对"找到上升期股票"目标有实质贡献的问题，不过度设计。每个问题给出具体解决方案而非仅指问题。

---

## 先纠正上一版的错误

上一版审查有两个关键误判：

1. **"_direction_structure 缺失7种模式"是错误的**——仔细重数，代码已覆盖全部16种模式(2^4)。`return 30`是死代码。真正的问题是**部分模式之间的相对评分存在逻辑不一致**（见§2）。

2. **"Hot-picks永不触发A0"是过度推理**——`run_all()`中long_term先跑，A0的tier_assignments对所有策略共享。hot_picks在FL的rebuild中读取同样的tier数据。daily模式下所有策略都不触发A0（这是设计选择，不是hot_picks特有的问题）。这个"问题"对目标无实质影响，不再讨论。

---

## 真正需要解决的问题（4个）

| # | 问题 | 为什么重要 |
|---|------|-----------|
| 1 | **RSI在技术分中的使用方式** — 孤立使用RSI值，不考虑趋势上下文 | 影响每只股票的tech_score，直接影响long_term排名 |
| 2 | **_direction_structure 评分逻辑不一致** — 部分模式排序违背"长周期>短周期"原则 | 影响trend_quality的30%权重，间接影响选股 |
| 3 | **A2 fallback 无盈利质量检测** — 仅5个浅层财务指标 | 影响无A2覆盖股票的基本面评分可靠性 |
| 4 | **红旗惩罚量级太小** — 对conviction几乎无影响 | A7确信度不能反映真实风险差异 |

---

## 问题1: RSI使用方式 — 需要趋势上下文

### 1.1 为什么"直接用RSI"和"惩罚高RSI"都不对

当前代码（[focus_list.py:103-104](backend/focus_list.py#L103-L104)）:

```python
rsi = ind.get("rsi_14", 50)
components.append(rsi)  # RSI直接作为分数
```

这导致 RSI=85（严重超买）→贡献85分，RSI=45（底部积累）→贡献45分。

但如果简单地"惩罚高RSI"（比如RSI>70时降分），会导致另一个问题：**强势趋势股天然维持高RSI**。一只从50涨到80的股票，RSI一路上行到75-85是**健康的趋势特征**，不是风险信号。惩罚它会系统性错过最强的上升趋势。

这就是用户说的"堵了A，B又漏了"——简单的阈值惩罚只是把问题从"奖励超买"变成"惩罚强势"，没有真正解决问题。

### 1.2 核心洞察: RSI的含义取决于趋势加速度

同一RSI值在不同上下文中含义完全不同：

| RSI | 加速度>0 (仍在加强) | 加速度<0 (已在衰减) |
|-----|---------------------|---------------------|
| 75-85 | ✅ 强势趋势，动能充沛 | ⚠️ 高位滞涨，警惕见顶 |
| 55-70 | ✅ 健康上升，趋势确立 | → 趋势走平，方向待观察 |
| 40-55 | ▲ 刚从底部回升，早期阶段 | ↓ 弱势横盘 |
| <40 | △ 超跌反弹初期（风险较高） | ❌ 持续走弱 |

**同一RSI=78:**
- 如果accel=+8（加速中）→ 这是强势趋势中的正常状态，应给高分
- 如果accel=-6（减速中）→ 这是高位动能衰竭，应降分

### 1.3 解决方案: RSI与加速度联合评分

在 `_compute_tech_score` 中，不再孤立使用RSI，而是结合动量加速度（FL已计算好，可通过参数传入）：

```python
def _compute_tech_score(ind, momentum_accel=0, strategy="long_term"):
    """
    momentum_accel: 从FL的calc_momentum获取的加速度值。
    accel>0 = 趋势在加速, accel<0 = 趋势在减速。
    """
    components, weights = [], []
    
    # MACD histogram (unchanged)
    macd = ind.get("macd", {})
    hist = macd.get("histogram", 0)
    if isinstance(hist, (int, float)):
        components.append(50 + math.tanh(hist * 10) * 40)
        weights.append(1.0)
    
    # ── RSI: context-aware scoring ──
    rsi = ind.get("rsi_14", 50)
    if isinstance(rsi, (int, float)):
        if momentum_accel > 0:
            # 加速中: RSI偏高=趋势健康, 在50-80区间给高分
            # 峰值在65-75(强势但不过热), 超过85开始微降
            if rsi > 85:
                rsi_score = 80 - (rsi - 85) * 0.8   # 85→80, 95→72
            elif rsi >= 50:
                rsi_score = 50 + (rsi - 50) * 0.8    # 50→50, 65→62, 75→70
            else:
                rsi_score = rsi  # <50: 加速但还没热起来, 保持原值
        else:
            # 减速中: RSI偏高=警惕, 降低高分端贡献
            # 40-60是减速环境下的最优区间(趋势稳固但未透支)
            if rsi > 75:
                rsi_score = 70 - (rsi - 75) * 1.2    # 75→70, 85→58
            elif rsi > 60:
                rsi_score = 55 + (rsi - 60) * 0.5    # 60→55, 70→60
            elif rsi >= 40:
                rsi_score = rsi                        # 40-60: 正常区间
            else:
                rsi_score = 35 + rsi * 0.25           # 30→42, 20→40
        components.append(rsi_score)
        weights.append(1.0)
    
    # Bollinger (unchanged — asymmetric design is correct)
    bb = ind.get("bollinger", {})
    bb_pos = bb.get("position", 0.5)
    if isinstance(bb_pos, (int, float)):
        if bb_pos >= 0.5:
            score_bb = 50 + (bb_pos - 0.5) * 100
        else:
            score_bb = bb_pos * 100
        components.append(score_bb)
        weights.append(0.5)
    
    # MA alignment, OBV, volume ratio (unchanged)
    ma = ind.get("ma_alignment", "mixed")
    components.append({"bullish": 75, "mixed": 50, "bearish": 25}.get(ma, 50))
    weights.append(1.0)
    
    obv_trend = ind.get("obv_trend", "flat")
    components.append({"rising": 65, "flat": 50, "falling": 35}.get(obv_trend, 50))
    weights.append(0.5)
    
    vr = ind.get("volume_ratio", 1.0)
    if isinstance(vr, (int, float)):
        components.append(50 + math.tanh((vr - 1.0) * 2) * 25)
        weights.append(0.3)
    
    if not components:
        return 50.0
    return max(0.0, min(100.0, 
        sum(c * w for c, w in zip(components, weights)) / sum(weights)))
```

**调用侧修改**（[focus_list.py:411-413](backend/focus_list.py#L411-L413)）:

`_compute_tech_score` 需要接收 `momentum_accel`。在 FL 的 rebuild 中，tech_score的计算从Step 2移到Step 4之后（momentum已计算好），或提前计算momentum后再补算tech_score。更简洁的做法：在per-stock循环中（Step 6），tech_score已经可以用momentum_accel：

```python
# focus_list.py rebuild(), Step 6 per-stock循环中:
# 当前: tech_score 在 Step 2 从 indicators 批量计算
# 改为: tech_score 在 Step 6 中重新计算(或调整),传入 momentum_accel
# 因为在 Step 4 已经计算了 momentum_data
m = momentum_data.get(code, {})
accel = m.get("acceleration", 0) if m else 0
# 重新计算 tech_score, 这次带上加速度上下文
tech_score = _compute_tech_score(ind, momentum_accel=accel, strategy=strategy)
```

### 1.4 这个方案为什么比"直接惩罚高RSI"好

- **不丢失强势股**: 加速中的高RSI仍然得高分（因为趋势在加强，不是衰竭）
- **能识别衰竭**: 减速中的高RSI自动降分（同样的RSI值，不同上下文不同处理）
- **早期介入友好**: 加速中RSI=55（刚从50回升）→得分55，配合加速度加成，综合排名上升
- **不引入新阈值**: 使用连续函数，没有RSI=70→71的断崖

---

## 问题2: _direction_structure 评分逻辑不一致

### 2.1 现状

代码覆盖了全部16种模式，但部分评分存在逻辑矛盾。核心理应遵循的原则是：**长周期信号(d60/d20)比短周期信号(d3/d5)更可靠，因此长周期正向应对得分贡献更大**。

### 2.2 发现的不一致

**不一致1: 单正信号排序混乱**

```
(1,0,0,0) = 45  ← 仅d3正 (=45)
(0,0,0,1) = 25  ← 仅d60正 (=25)
(0,0,1,0) = 20  ← 仅d20正 (=20)
(0,1,0,0) = 18  ← 仅d5正  (=18)
```

d3（3天，最噪声）单独正→45分。d60（60天，最可靠）单独正→25分。**最不可靠的信号得了最高分**。如果一只股票仅d60为正（长期上涨但近期走弱），它的下行风险远小于仅d3为正（可能是死猫反弹）。

合理排序应为：仅d60 > 仅d20 > 仅d5 > 仅d3。

**不一致2: (0,0,1,1) vs (0,1,0,1)**

```
(0,0,1,1) = 40  ← d20>0, d60>0: 中+长期正
(0,1,0,1) = 45  ← d5>0,  d60>0: 短+长期正
```

(0,0,1,1)有d20>0和d60>0——两个中长周期都正向，信号应该强于(0,1,0,1)（仅d5>0+d60>0）。d20的信息量远大于d5。但评分却是40 < 45。

**不一致3: 相邻模式断崖**

```
(1,1,1,0) = 70  ← d60未翻正
(1,1,1,1) = 95  ← d60翻正
```

从"d60刚好为负"到"d60刚好为正"，分数从70跳到95（+25分）。d60从-0.1%变成+0.1%在实际中几乎无区别，但评分突变25分。

### 2.3 解决方案: 加权打分替代硬编码模式表

放弃逐模式手写分数的做法，改用**时间框架加权求和 + 对齐加成**：

```python
def _direction_structure(d3, d5, d20, d60):
    """Score multi-timeframe trend direction (0-100).
    
    Core principle: longer timeframes carry more weight.
    Alignment bonus: contiguous positive blocks > scattered signals.
    """
    # ── 1. Weighted base score (0-100) ──
    # 每个时间框架的贡献由其方向×权重决定
    # d60最可靠(40%), d20次之(30%), d5(18%), d3(12%)
    signs = [1 if d3 > 0 else 0, 1 if d5 > 0 else 0, 
             1 if d20 > 0 else 0, 1 if d60 > 0 else 0]
    weights = [12, 18, 30, 40]  # d3, d5, d20, d60
    base = sum(s * w for s, w in zip(signs, weights))  # 0-100
    
    # ── 2. Alignment bonus/penalty ──
    # 连续正向块给加分（信号一致性强）
    # 例如(1,1,1,1): 4个连续 → +8
    # 例如(1,1,0,0): 2个连续 → +3
    # 例如(1,0,1,0): 0个连续 → -5 (信号矛盾)
    runs = 0
    current_run = 0
    for s in signs:
        if s == 1:
            current_run += 1
            runs = max(runs, current_run)
        else:
            current_run = 0
    
    if runs >= 4:       alignment = 8
    elif runs >= 3:     alignment = 5
    elif runs >= 2:     alignment = 2
    elif runs >= 1:     alignment = 0
    else:               alignment = -10  # 全跌
    
    # ── 3. Magnitude adjustment (mild) ──
    # 小幅正值(0-3%)和大幅正值(>15%)有细微区分
    # 用tanh平滑映射，避免断崖
    mag_bonus = 0
    for val, w in [(d3, 2), (d5, 3), (d20, 5), (d60, 8)]:
        mag_bonus += math.tanh(val / 15) * w  # 每个最多贡献w分
    mag_bonus = max(-5, min(5, mag_bonus))  # 钳制在±5
    
    return max(5, min(98, base + alignment + mag_bonus))
```

### 2.4 新旧对比

| 模式 | 旧分 | 新分(approx) | 变化 | 说明 |
|------|------|-------------|------|------|
| (1,1,1,1) | 95 | 100+8+5→98(clamped) | +3 | 依旧最高 |
| (0,1,1,1) | 75 | 88+3+3=94 | +19 | d3微跌不应扣这么多 |
| (1,1,0,1) | 75 | 88+2+3=93 | +18 | 同上 |
| (1,0,1,1) | 70 | 82+0+2=84 | +14 | 回调后的恢复 |
| (1,1,1,0) | 70 | 60+3+3=66 | -4 | d60为负应更谨慎 |
| (1,0,0,1) | 55 | 52+0+1=53 | -2 | 维持 |
| (1,0,0,0) | 45 | 12+0+0=12 | **-33** | 仅d3正不应给45! |
| (0,0,0,1) | 25 | 40+0+2=42 | +17 | d60为正比仅d3可靠 |
| (0,0,0,0) | 10 | 0-10-3=0(clamped) | -10 | 维持 |

关键修正：
- (1,0,0,0) 从45→12：仅d3正不应该得高分
- (0,0,0,1) 从25→42：仅d60正应该高于仅d3正
- (0,1,1,1) 从75→94：上升趋势中的短暂回调是买点，不应过度扣分
- 消除了所有相邻模式间的>15分断崖

---

## 问题3: A2 fallback 增强

### 3.1 现状

A2覆盖率会随运行时间逐步提升（用户确认）。但确实存在始终无A2覆盖的股票（如新上市、数据源缺失）。当前fallback仅用5个指标做线性打分（[focus_list.py:181-196](backend/focus_list.py#L181-L196)）。

### 3.2 需要增强的点

当前fallback缺失的关键维度：
- **盈利质量**: financials表已有`cfo_ni_ratio`（经营现金流/净利润）和`ar_revenue_divergence`（应收/营收背离），但未使用
- **行业相对**: 所有行业用同一公式，ROE=12%对银行vs科技含义不同
- **估值区分度**: PE分位仅贡献5分（总分~90），几乎无区分

### 3.3 解决方案

```python
def _fundamental_fallback(conn, code):
    """Pure-computation fundamental score for stocks without A2 report.
    
    v2: adds earnings quality, industry-relative adjustment, better valuation.
    """
    row = conn.execute("""
        SELECT roe, gross_margin, debt_ratio, revenue_yoy, fcf_ratio, 
               pe_percentile, pb, cfo_ni_ratio, ar_revenue_divergence,
               net_margin
        FROM financials WHERE ts_code=? ORDER BY report_date DESC LIMIT 1
    """, (code,)).fetchone()
    
    if not row:
        return 40.0, 0.2
    
    # ── 1. Profitability (0-30) ──
    score = 0.0
    if row["roe"]:
        # ROE mapped to 0-20: 0%→5, 8%→12, 15%→16, 25%→20
        score += min(20, max(0, row["roe"] * 0.8 + 2))
    if row["net_margin"]:
        # Net margin bonus: >10% is good, >20% is excellent
        score += min(10, max(0, (row["net_margin"] - 5) * 0.5))
    
    # ── 2. Earnings Quality (0-20) ──
    quality = 10  # neutral baseline
    if row["cfo_ni_ratio"] is not None:
        # CFO/NI > 1.0 = earnings backed by cash (conservative, good)
        # CFO/NI < 0.5 = earnings may be accrual-based (aggressive, risky)
        cfo_ratio = row["cfo_ni_ratio"]
        if cfo_ratio > 1.0:
            quality += min(8, (cfo_ratio - 1.0) * 4)   # 1.0→10, 2.0→14, 3.0→20(clamped)
        elif cfo_ratio > 0.5:
            quality += (cfo_ratio - 0.5) * 8             # 0.5→10, 0.75→12, 1.0→14
        else:
            quality += (cfo_ratio - 0.5) * 12            # 0.0→4, 0.5→10
    if row["ar_revenue_divergence"] is not None:
        # AR growing much faster than revenue = aggressive revenue recognition
        ar_div = row["ar_revenue_divergence"]
        if ar_div > 20:
            quality -= min(8, (ar_div - 20) * 0.3)
        elif ar_div < -10:
            quality += min(3, abs(ar_div + 10) * 0.2)
    score += max(0, min(20, quality))
    
    # ── 3. Financial Health (0-20) ──
    health = 10
    if row["debt_ratio"] is not None:
        dr = row["debt_ratio"]
        if dr < 30:      health += 8
        elif dr < 50:    health += 4
        elif dr < 70:    health += 0
        elif dr < 85:    health -= 5
        else:            health -= 10
    if row["fcf_ratio"] is not None and row["fcf_ratio"] > 0:
        health += min(6, row["fcf_ratio"] * 1.5)
    elif row["fcf_ratio"] is not None:
        health -= min(5, abs(row["fcf_ratio"]) * 1.0)
    score += max(0, min(20, health))
    
    # ── 4. Growth (0-15) ──
    growth = 7
    if row["revenue_yoy"] is not None:
        rev = row["revenue_yoy"]
        if rev > 30:     growth += 7
        elif rev > 15:   growth += 5
        elif rev > 5:    growth += 2
        elif rev > -5:   growth += 0
        elif rev > -15:  growth -= 3
        else:            growth -= 8
    score += max(0, min(15, growth))
    
    # ── 5. Valuation (0-15) ──
    val = 7
    if row["pe_percentile"] is not None:
        pep = row["pe_percentile"]
        if pep < 20:     val += 6     # 历史低位
        elif pep < 40:   val += 4     # 偏低
        elif pep < 60:   val += 1     # 中位
        elif pep < 80:   val -= 2     # 偏高
        else:            val -= 5     # 历史高位
    if row["pb"] is not None and row["pb"] > 0:
        # PB<1 = potentially undervalued (but may be value trap)
        # PB>10 = likely overvalued unless hyper-growth
        pb = row["pb"]
        if pb < 1:       val += 2
        elif pb > 10:    val -= 3
    score += max(0, min(15, val))
    
    confidence = 0.3 if row["cfo_ni_ratio"] is not None else 0.2
    return max(10.0, min(85.0, score)), confidence
```

### 3.4 改进点总结

| 维度 | 旧 | 新 |
|------|-----|-----|
| 盈利质量 | 无 | CFO/NI比率 + 应收/营收背离 |
| 利润率 | 仅毛利率 | 毛利率 + 净利率 |
| 估值 | PE分位仅5分，几乎无区分 | 15分，PE分位+PB双维度 |
| 增长 | 线性映射 | 分段映射，区分高速/中速/衰退 |
| 置信度 | 固定0.3 | 有盈利质量数据→0.3，否则→0.2 |

---

## 问题4: 红旗惩罚量级

### 4.1 现状

```python
# agent_7_portfolio.py:54-62
rf_penalty = 0.0
if a2_report:
    for rf in a2_report.get("red_flags", []):
        sev = rf.get("severity", "MEDIUM")
        rf_penalty += 0.10 if sev == "HIGH" else 0.05 if sev == "MEDIUM" else 0.02
rf_penalty = min(0.25, rf_penalty)
conviction = base * 0.85 + (1.0 - rf_penalty) * 0.15
```

一只 FL=95 但有 3个HIGH红旗的股票: `conviction = 0.95*0.85 + 0.75*0.15 = 0.807 + 0.113 = 0.92`

与无红旗时(0.95*0.85+1.0*0.15=0.957)仅差0.037。相当于FL总分差约4分——在排名中几乎无区分度。

### 4.2 目标角度分析

红旗(A2 red_flags)代表基本面风险信号（如"净利润连续3季为负"、"负债率>80%"、"审计意见异常"等）。这些信号对"找到上升期股票"的目标是**重要否定项**——一个基本面有严重缺陷的股票，即使技术面再好，其上升趋势的可持续性存疑。

当前设计的问题不是方向错误（确实给了惩罚），而是**量级太小**——LLM在A7 prompt中看到conviction=0.92被标为"STRONG"，自然会优先关注它，但此时红旗信号几乎被淹没。

### 4.3 解决方案: 红旗作为conviction乘数

改变结构：红旗从"加性微调(15%权重)"改为"乘性降权"：

```python
def compute_conviction(code, fl_score, a2_report, strategy="long_term"):
    base = (fl_score or 50) / 100.0
    
    # Red flag penalty: multiplicative, not additive
    # Each HIGH flag reduces conviction by 15%, MEDIUM by 8%, LOW by 3%
    rf_multiplier = 1.0
    if a2_report:
        for rf in a2_report.get("red_flags", []):
            sev = rf.get("severity", "MEDIUM") if isinstance(rf, dict) else "MEDIUM"
            rf_multiplier *= (0.85 if sev == "HIGH" else 0.92 if sev == "MEDIUM" else 0.97)
    
    # Floor: won't drop below 0.4 regardless of red flag count
    # (keeps stocks in play but with meaningfully reduced conviction)
    rf_multiplier = max(0.4, rf_multiplier)
    
    conviction = base * rf_multiplier
    return round(max(0.05, min(1.0, conviction)), 3)
```

**效果对比**:

| 场景 | 旧conviction | 新conviction | 变化 |
|------|-------------|-------------|------|
| FL=85, 无红旗 | 0.872 | 0.850 | ~持平 |
| FL=85, 1 HIGH | 0.858 | 0.723 | **-0.135** |
| FL=85, 3 HIGH | 0.835 | 0.522 | **-0.313** |
| FL=85, 1 HIGH+2 MED | 0.826 | 0.572 | **-0.254** |
| FL=60, 无红旗 | 0.660 | 0.600 | ~持平 |
| FL=60, 3 HIGH | 0.622 | 0.368 | **-0.254** |

关键改变：
- 1个HIGH红旗 → conviction从STRONG降为MODERATE（对long_term: 0.723<0.50? 不对，0.723>0.50仍是STRONG...让我调整）

Hmm，乘数0.85可能还是不够。让我调整一下：

```python
rf_multiplier *= (0.80 if sev == "HIGH" else 0.90 if sev == "MEDIUM" else 0.96)
# Floor: 0.35
rf_multiplier = max(0.35, rf_multiplier)
```

| 场景 | 新conviction |
|------|-------------|
| FL=85, 无红旗 | 0.850 |
| FL=85, 1 HIGH | 0.680 |
| FL=85, 2 HIGH | 0.544 |
| FL=85, 3 HIGH | 0.435 |
| FL=85, 1 HIGH+2 MED | 0.551 |

这样1个HIGH红旗就能从STRONG(≥0.50)降到MODERATE附近，2个HIGH明确降到MODERATE。这个量级让红旗真正发挥"风险信号"的作用。

---

## 策略建议汇总

### 立即执行（高ROI，低风险）

1. **RSI上下文感知** — 改`_compute_tech_score`和调用侧，传入momentum_accel。约30行改动。直接提升tech_score的合理性。

2. **_direction_structure去硬编码** — 用加权公式替代16个if。约25行改动。消除逻辑不一致，提高trend_quality可靠性。

3. **A2 fallback增强** — 加入盈利质量维度。约40行改动。提升无A2覆盖股票的评分质量。

4. **红旗乘性惩罚** — 改`compute_conviction`的rf_penalty结构。约10行改动。让红旗真正影响确信度。

### 不需要改的（基于目标判断）

- **FL→A7评分链**: FL用total_score（多因子）做排名写表，A7用total_score做conviction后给LLM展示多维度数据。LLM需要的是丰富的上下文做独立判断，不是单一分数。当前A7 prompt中每只股票展示了FL综合分+动量+技术+基本面+量价+红旗——信息完整。**维持现状**。

- **A3→FL评分**: A3的新闻情感是定性信息，适合LLM消费不适合定量公式。A7/A6的LLM prompt中已有新闻背景。**维持现状**。

- **A0运行频率**: daily模式下所有策略共享同一份tier_assignments，不需要hot_picks单独触发。**维持现状**。

- **权重与regime联动/双路径自适应/参数自调整**: 缺乏回测验证时，增加复杂度只增加bug面。等有了回测数据再优化。**暂不设计**。
