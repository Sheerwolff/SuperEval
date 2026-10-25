"""Runtime settings, all overridable from the environment (or a local .env)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    agent_model: str = os.getenv("AGENT_MODEL", "gpt-5-mini")
    judge_model: str = os.getenv("JUDGE_MODEL", "gpt-5")  # stronger than the agent (decision)
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'pricewatch.db'}")
    # the full snapshot (downloaded by app.data.fetch_snapshot) if present, else the committed sample
    snapshot_dir: Path = Path(os.getenv("SNAPSHOT_DIR") or (
        ROOT / "data" / "snapshot" if (ROOT / "data" / "snapshot" / "meta.json").exists() else ROOT / "data" / "sample"))
    daily_budget_usd: float = float(os.getenv("DAILY_BUDGET_USD", "2.0"))
    rate_limit_per_hour: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "30"))
    judge_sample_rate: float = float(os.getenv("JUDGE_SAMPLE_RATE", "0.3"))
    alert_webhook_url: str | None = os.getenv("ALERT_WEBHOOK_URL") or None
    api_url: str = os.getenv("API_URL", "http://127.0.0.1:8000")
    freshness_sla_hours: float = float(os.getenv("FRESHNESS_SLA_HOURS", "36"))
    # "Let's connect" card. WhatsApp number (international format, digits only) lives only in env/secrets.
    contact_name: str = os.getenv("CONTACT_NAME", "Sheer Wolff")
    contact_email: str = os.getenv("CONTACT_EMAIL", "sheerwolff@gmail.com")
    contact_linkedin: str = os.getenv("CONTACT_LINKEDIN", "https://www.linkedin.com/in/sheerwolff")
    contact_whatsapp: str | None = os.getenv("CONTACT_WHATSAPP") or None
    # unlocks the owner role on the Review and Golden Set pages (approving references, owner judgments)
    admin_key: str | None = os.getenv("ADMIN_KEY") or None

    @property
    def live_llm(self) -> bool:
        return self.openai_api_key is not None


settings = Settings()

# USD per 1M tokens (input, output). Used for cost tracking and the budget guard.
MODEL_PRICES = {
    "gpt-5": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "sim": (0.25, 2.0),  # simulated runs are costed like gpt-5-mini so charts stay realistic
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    base = next((m for m in sorted(MODEL_PRICES, key=len, reverse=True) if model.startswith(m)), "gpt-5-mini")
    p_in, p_out = MODEL_PRICES[base]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000
