"""
finbert_sentiment.py — robuste finBERT Sentiment-Analyse

Ziel:
- FinBERT wirklich lazy laden, sobald es gebraucht wird.
- Kein falsches Blockieren durch FINBERT_AVAILABLE=False beim Import.
- Sauberer Fallback auf Keyword-Sentiment, wenn transformers/torch/Modell nicht verfügbar sind.
- Batch-Ausgabe immer längengleich zur Eingabe.

Environment:
    ENABLE_FINBERT=false   -> FinBERT komplett deaktivieren
    FINBERT_MODEL_NAME     -> anderes HuggingFace-Modell, Default: ProsusAI/finbert
    FINBERT_DEVICE         -> -1 CPU, 0 GPU. Default: -1
"""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "ProsusAI/finbert"

_pipeline = None
_load_attempted = False
_last_error: str | None = None

# Rückwärtskompatibilität: Wird nach erfolgreichem Laden True.
# Wichtig: Der Wert ist beim Import absichtlich False/unknown und darf
# in anderen Modulen NICHT als Vorbedingung für den ersten Load benutzt werden.
FINBERT_AVAILABLE = False


def is_finbert_enabled() -> bool:
    """Feature-Flag. Standard: aktiv."""
    raw = os.getenv("ENABLE_FINBERT", "true").strip().lower()
    return raw not in {"0", "false", "no", "off", "disabled"}


def get_finbert_status() -> dict[str, Any]:
    """Status für Logging/Debug."""
    return {
        "enabled": is_finbert_enabled(),
        "loaded": _pipeline is not None,
        "load_attempted": _load_attempted,
        "available": FINBERT_AVAILABLE,
        "model": os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL),
        "error": _last_error,
    }


def _parse_device() -> int:
    raw = os.getenv("FINBERT_DEVICE", "-1").strip()
    try:
        return int(raw)
    except ValueError:
        logger.warning("Ungültiger FINBERT_DEVICE=%r — nutze CPU (-1)", raw)
        return -1


def _load_model():
    """Lädt FinBERT beim ersten echten Aufruf. Danach gecacht."""
    global _pipeline, _load_attempted, _last_error, FINBERT_AVAILABLE

    if _pipeline is not None:
        return _pipeline

    if not is_finbert_enabled():
        _last_error = "FinBERT per ENABLE_FINBERT deaktiviert"
        FINBERT_AVAILABLE = False
        return None

    _load_attempted = True
    model_name = os.getenv("FINBERT_MODEL_NAME", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    device = _parse_device()

    try:
        from transformers import pipeline
    except Exception as exc:
        _last_error = f"transformers import failed: {exc}"
        FINBERT_AVAILABLE = False
        logger.warning("finBERT nicht verfügbar — transformers/torch Import fehlgeschlagen: %s", exc)
        return None

    try:
        logger.info("Lade finBERT (%s) auf %s...", model_name, "CPU" if device < 0 else f"device {device}")

        # Variante 1: Moderne Transformers-Versionen.
        try:
            _pipeline = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
                top_k=None,
                truncation=True,
                max_length=512,
                device=device,
            )
        except TypeError:
            # Variante 2: ältere Transformers-Versionen.
            _pipeline = pipeline(
                task="text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True,
                truncation=True,
                max_length=512,
                device=device,
            )

        FINBERT_AVAILABLE = True
        _last_error = None
        logger.info("finBERT geladen")
        return _pipeline

    except Exception as exc:
        _pipeline = None
        FINBERT_AVAILABLE = False
        _last_error = str(exc)
        logger.warning("finBERT konnte nicht geladen werden — Keyword-Sentiment bleibt aktiv: %s", exc)
        return None


def _flatten_pipeline_result(result: Any) -> list[dict[str, Any]]:
    """Normalisiert unterschiedliche Transformers-Pipeline-Ausgabeformen.

    Möglich sind u.a.:
    - [{'label': 'positive', 'score': ...}, ...]
    - [[{'label': 'positive', 'score': ...}, ...]]
    - [{'label': 'positive', 'score': ...}] bei top-1
    """
    if result is None:
        return []

    if isinstance(result, dict):
        return [result]

    if isinstance(result, list):
        if not result:
            return []
        if all(isinstance(x, dict) for x in result):
            return result
        if len(result) == 1 and isinstance(result[0], list):
            return _flatten_pipeline_result(result[0])

    return []


def _score_from_label_rows(rows: Iterable[dict[str, Any]]) -> float:
    """Konvertiert FinBERT Label-Scores in [-1, +1]."""
    scores: dict[str, float] = {}
    for row in rows:
        try:
            label = str(row.get("label", "")).lower().strip()
            score = float(row.get("score", 0.0))
        except Exception:
            continue
        if label:
            scores[label] = score

    # ProsusAI/finbert nutzt i.d.R. positive / negative / neutral.
    pos = scores.get("positive", scores.get("pos", 0.0))
    neg = scores.get("negative", scores.get("neg", 0.0))
    neutral = scores.get("neutral", scores.get("neu", 0.0))

    # Falls nur top-1 zurückkommt, wenigstens Richtung abbilden.
    if not pos and not neg and not neutral and scores:
        best_label = max(scores, key=scores.get)
        if "pos" in best_label:
            pos = scores[best_label]
        elif "neg" in best_label:
            neg = scores[best_label]
        elif "neu" in best_label:
            neutral = scores[best_label]

    if neutral > 0.60:
        net = (pos - neg) * 0.30
    else:
        net = pos - neg

    return round(max(-1.0, min(1.0, net)), 3)


def get_finbert_sentiment(text: str) -> float:
    """Sentiment für einen Text. 0.0 bei Fehler/Neutral/Fallback."""
    if not text or not str(text).strip():
        return 0.0

    pipe = _load_model()
    if pipe is None:
        return 0.0

    try:
        raw = pipe(str(text)[:1000])
        rows = _flatten_pipeline_result(raw)
        return _score_from_label_rows(rows)
    except Exception as exc:
        logger.debug("finBERT Inference Fehler: %s", exc)
        return 0.0


def get_finbert_sentiment_batch(texts: list[str]) -> list[float]:
    """Sentiment für mehrere Texte. Ergebnis ist immer gleich lang wie texts."""
    if not texts:
        return []

    pipe = _load_model()
    if pipe is None:
        return [0.0] * len(texts)

    # Positionen behalten, damit leere Texte nicht die Reihenfolge verschieben.
    valid_items: list[tuple[int, str]] = []
    for idx, text in enumerate(texts):
        if text and str(text).strip():
            valid_items.append((idx, str(text)[:1000]))

    scores = [0.0] * len(texts)
    if not valid_items:
        return scores

    try:
        valid_texts = [text for _, text in valid_items]
        raw_results = pipe(valid_texts)

        # Bei Batch sollte raw_results eine Liste pro Text sein. Falls die
        # Pipeline bei einem einzelnen Element anders formatiert, normalisieren.
        if len(valid_texts) == 1:
            normalized_results = [raw_results]
        else:
            normalized_results = raw_results if isinstance(raw_results, list) else []

        for (original_idx, _), raw in zip(valid_items, normalized_results):
            rows = _flatten_pipeline_result(raw)
            scores[original_idx] = _score_from_label_rows(rows)

        return scores

    except Exception as exc:
        logger.debug("finBERT Batch Fehler: %s", exc)
        return [0.0] * len(texts)
