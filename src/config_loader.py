"""
config_loader.py
Lädt API Keys aus config/config.yaml oder Umgebungsvariablen.

v8:
- Tradier-Production ist Standard. Sandbox wird nur genutzt, wenn TRADIER_SANDBOX=true gesetzt ist.
- TRADIER_TOKEN ist Pflicht, weil Options-EV und konsistente Quotes auf Tradier basieren.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

REQUIRED_KEYS = ["anthropic_api_key", "tradier_token"]

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:  # pragma: no cover
    YAML_AVAILABLE = False


def _parse_bool(value, default=False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "sandbox"):
        return True
    if s in ("0", "false", "no", "n", "off", "prod", "production", "live"):
        return False
    return default


def load_config() -> dict:
    """
    Lädt Konfiguration in folgender Priorität:
    1. Umgebungsvariablen überschreiben alles.
    2. config/config.yaml.
    3. Sichere Defaults.

    Wichtig: Tradier läuft standardmäßig gegen Production, nicht Sandbox.
    """
    config = {}

    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    if config_path.exists() and YAML_AVAILABLE:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            logger.error("Fehler beim Laden von config.yaml: %s", e)

    env_map = {
        "ANTHROPIC_API_KEY": "anthropic_api_key",
        "TRADIER_TOKEN":     "tradier_token",
        "FINNHUB_KEY":       "finnhub_key",
        "ALPHA_VANTAGE_KEY": "alpha_vantage_key",
        "GMAIL_RECIPIENT":   "gmail_recipient",
        "SMTP_SENDER":       "smtp_sender",
        "SMTP_PASSWORD":     "smtp_password",
        "TRADIER_SANDBOX":   "tradier_sandbox",
        "TRADIER_ENV":       "tradier_env",
        "SEC_USER_AGENT":    "sec_user_agent",
    }
    for env_var, key in env_map.items():
        val = os.environ.get(env_var)
        if val is not None and str(val).strip() != "":
            config[key] = val.strip()

    # Production ist Default. Sandbox nur explizit.
    if "tradier_env" in config and "tradier_sandbox" not in config:
        config["tradier_sandbox"] = _parse_bool(config.get("tradier_env"), default=False)
    else:
        config["tradier_sandbox"] = _parse_bool(config.get("tradier_sandbox"), default=False)

    config["tradier_base_url"] = (
        "https://sandbox.tradier.com" if config.get("tradier_sandbox")
        else "https://api.tradier.com"
    )
    config["tradier_mode"] = "sandbox" if config.get("tradier_sandbox") else "production"

    return config


def validate_config(cfg: dict) -> bool:
    """Prüft Pflichtfelder. Gibt False zurück, wenn etwas fehlt."""
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        logger.error("Fehlende Pflicht-Keys in config: %s", missing)
        return False
    if cfg.get("tradier_sandbox"):
        logger.warning("TRADIER_SANDBOX=true — Sandboxdaten sind verzögert/Simulation. Für Production Secret auf false lassen.")
    else:
        logger.info("Tradier-Modus: PRODUCTION api.tradier.com")
    return True
