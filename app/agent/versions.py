"""Versioned agent configs. Each trace is stamped with version + prompt hash so any
metric can be sliced by exactly what was deployed."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

PROMPTS = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class AgentVersion:
    version: str
    label: str
    description: str
    system_template: str
    temperature: float
    max_steps: int = 6

    @property
    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_template.encode()).hexdigest()[:12]

    def system_prompt(self, catalog) -> str:
        return self.system_template.format(
            chains=", ".join(catalog.chains),
            cities=", ".join(catalog.cities),
            snapshot_date=catalog.snapshot_date,
        )


@cache
def load_versions() -> dict[str, AgentVersion]:
    out = {}
    for path in sorted(PROMPTS.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        out[raw["version"]] = AgentVersion(
            version=raw["version"],
            label=raw["label"],
            description=raw["description"].strip(),
            system_template=raw["system"],
            temperature=float(raw.get("model_settings", {}).get("temperature", 0.2)),
            max_steps=int(raw.get("model_settings", {}).get("max_steps", 6)),
        )
    return out


def get_version(name: str) -> AgentVersion:
    return load_versions()[name]
