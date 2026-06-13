"""Unified configuration. Single source of truth for all settings."""
import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Settings:
    # Paths
    project_root: Path = field(default_factory=lambda: Path(os.path.expanduser("~/stock-analysis")))
    db_path: Path = field(default_factory=lambda: Path(os.path.expanduser("~/stock-analysis/data/stocks.db")))
    frontend_dist: Path = field(default_factory=lambda: Path(os.path.expanduser("~/stock-analysis/frontend/dist")))
    output_dir: Path = field(default_factory=lambda: Path(os.path.expanduser("~/stock-analysis/output")))

    # DeepSeek API (shared credentials)
    ds_api_key: str = ""
    ds_base_url: str = "https://api.deepseek.com/anthropic"

    # LLM 请求级共享参数
    llm_timeout: int = 180
    llm_max_retries: int = 3

    # ── 各 Agent LLM 独立配置 ──
    # A1 无 LLM，其余 Agent 各自指定模型和参数
    agent_llm: dict = field(default_factory=lambda: {
        "A0": {"model": "deepseek-v4-flash", "temperature": 0.3, "max_tokens": 4000},
        "A2": {"model": "deepseek-v4-pro", "temperature": 0.3, "max_tokens": 4000},
        "A3": {"model": "deepseek-v4-flash", "temperature": 0.3, "max_tokens": 4000},
        "A4": {"model": "deepseek-v4-flash", "temperature": 0.3, "max_tokens": 2000},
        "A5": {"model": "deepseek-v4-flash", "temperature": 0.3, "max_tokens": 8000},
        "A6": {"model": "deepseek-v4-pro", "temperature": 0.7, "max_tokens": 20000},
        "A7": {"model": "deepseek-v4-pro",  "temperature": 0.3, "max_tokens": 20000},
    })

    def get_llm_config(self, agent: str) -> dict:
        """Return LLM config for a given agent (e.g. 'A2')."""
        return self.agent_llm.get(agent, {"model": "deepseek-v4-flash", "temperature": 0.3, "max_tokens": 2000})

    # Flask
    flask_port: int = 5001
    flask_debug: bool = False

    # Agent scheduling
    daily_timeout: int = 300
    weekly_timeout: int = 1800
    agent_retries: int = 3
    retry_backoff: float = 2.0

    # Trading constraints
    max_holdings: int = 8
    max_single_weight: float = 0.25
    min_single_weight: float = 0.08
    min_cash: float = 0.10
    rapid_sell_days: int = 7

    # ── Long-term strategy (80%) ──
    long_term_max_holdings: int = 8
    long_term_max_single_weight: float = 0.25
    long_term_min_single_weight: float = 0.08
    long_term_min_cash: float = 0.10
    long_term_buy_count: int = 25
    long_term_focus_size: int = 100
    long_term_buy_pass_rate: float = 0.20
    long_term_sell_pass_rate: float = 0.80
    long_term_hold_pass_rate: float = 0.70
    long_term_absolute_veto: int = 4

    # ── Hot-picks strategy (20%) ──
    hot_picks_max_holdings: int = 5
    hot_picks_max_single_weight: float = 0.40
    hot_picks_min_single_weight: float = 0.05
    hot_picks_min_cash: float = 0.05
    hot_picks_buy_count: int = 15
    hot_picks_focus_size: int = 50
    hot_picks_buy_pass_rate: float = 0.40
    hot_picks_sell_pass_rate: float = 0.90
    hot_picks_hold_pass_rate: float = 0.60
    hot_picks_absolute_veto: int = 4

    def __post_init__(self):
        self._load_dotenv()

    def _load_dotenv(self):
        env_path = self.project_root / ".env"
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

        self.ds_api_key = os.environ.get("StockAgent_DS_API_KEY", self.ds_api_key)
        self.flask_port = int(os.environ.get("FLASK_PORT", self.flask_port))
        self.flask_debug = os.environ.get("FLASK_DEBUG", "").lower() == "true"

    def get_strategy_config(self, strategy: str) -> dict:
        if strategy == "hot_picks":
            return {
                "max_holdings": self.hot_picks_max_holdings,
                "max_single_weight": self.hot_picks_max_single_weight,
                "min_single_weight": self.hot_picks_min_single_weight,
                "min_cash": self.hot_picks_min_cash,
                "buy_count": self.hot_picks_buy_count,
                "focus_size": self.hot_picks_focus_size,
                "buy_pass_rate": self.hot_picks_buy_pass_rate,
                "sell_pass_rate": self.hot_picks_sell_pass_rate,
                "hold_pass_rate": self.hot_picks_hold_pass_rate,
                "absolute_veto_score": self.hot_picks_absolute_veto,
            }
        return {
            "max_holdings": self.long_term_max_holdings,
            "max_single_weight": self.long_term_max_single_weight,
            "min_single_weight": self.long_term_min_single_weight,
            "min_cash": self.long_term_min_cash,
            "buy_count": self.long_term_buy_count,
            "focus_size": self.long_term_focus_size,
            "buy_pass_rate": self.long_term_buy_pass_rate,
            "sell_pass_rate": self.long_term_sell_pass_rate,
            "hold_pass_rate": self.long_term_hold_pass_rate,
            "absolute_veto_score": self.long_term_absolute_veto,
        }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
