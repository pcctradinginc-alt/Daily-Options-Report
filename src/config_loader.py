"""
config_loader.py
Lädt API Keys aus config/config.yaml oder Umgebungsvariablen.
Validiert Pflichtfelder beim Start.
"""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ["anthropic_api_key"]

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config() -> dict:
    """
    Lädt Konfiguration in folgender Priorität:
    1. Umgebungsvariablen (überschreiben alles)
    2. config/config.yaml
    """
    config = {}

    # 1. YAML laden
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists() and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Fehler beim Laden von config.yaml: %s", e)

    # 2. Umgebungsvariablen überschreiben YAML
    env_map = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "TRADIER_TOKEN":     "tradier_token",
        "FINNHUB_KEY":       "finnhub_key",
        "ALPHA_VANTAGE_KEY": "alpha_vantage_key",
        "GMAIL_RECIPIENT":   "gmail_recipient",
        "SMTP_SENDER":       "smtp_sender",
        "SMTP_PASSWORD":     "smtp_password",
    }
    for env_var, key in env_map.items():
        val = os.environ.get(env_var)
        if val:
            config[key] = val.strip()

    return config


def validate_config(cfg: dict) -> bool:
    """Prüft ob Pflichtfelder vorhanden sind. Gibt False zurück wenn nicht."""
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        logger.error("Fehlende Pflicht-Keys in config: %s", missing)
        return False
    return True
