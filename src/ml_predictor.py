"""
ml_predictor.py — Outcome-Predictor für empfohlene Options-Trades (v2, gehärtet).

Die gesamte ML-Logik ist hier gekapselt. Das Modell schätzt die Wahrscheinlichkeit,
dass ein Setup gewinnt; es wirkt nur WEICH (Conviction-Boost) und greift nie in die
harten Gates ein (Ausnahme: sehr niedrige Wahrscheinlichkeit -> Hard-Block, siehe rules).

────────────────────────────────────────────────────────────────────────────
LABEL-DEFINITION  (Risiko: Noisy Labels)
────────────────────────────────────────────────────────────────────────────
Klassifikation (is_win): kommt aus `trade_resolutions` und ist die Auflösung der ECHTEN
Exit-Regeln auf dem real empfohlenen OPTIONSKONTRAKT (nicht nur Underlying-Richtung):
  - Take-Profit  (+RULES.exit_take_profit_pct, default +50%) -> WIN
  - Stop-Loss    (-RULES.exit_stop_loss_pct,   default -30%) -> LOSS
  - Time-Stop    (nach time_stop_hours ohne Zielbewegung)    -> Close zum Mark, WIN falls Return>0
  - Expiry       -> Close, WIN falls Return>0
Dieses Label berücksichtigt damit implizit Kosten/Breakeven (TP/SL auf dem Optionspreis,
inkl. Entry-/Exit-Slippage) UND den Time-Stop — also genau die Schwächen von
"direction_return_pct > 0". `win_threshold_used` wird im Journal mitgespeichert.

Regression (optional): Ziel ist `exit_return_pct` (kontinuierlicher Options-Return beim
Exit). Wird zusätzlich trainiert und über predict_expected_return() angeboten; fließt
NICHT in die Entscheidung, dient als Diagnose/Erweiterung.

────────────────────────────────────────────────────────────────────────────
FEATURE-AUDIT  (Risiko: Feature Leakage)  — manuell geprüft, alle Decision-Zeitpunkt
────────────────────────────────────────────────────────────────────────────
Quelle aller Features ist `signals.market_json` (= process_ticker-Output ZUM
Entscheidungszeitpunkt) + `runs.vix` (bei Run-Start gesetzt) + die geparsten Signal-
Felder (direction/horizon/dte). KEIN Feld nutzt Daten aus der Zukunft:
  - score/raw_signal_score/gate_adjusted_score : zum Decision-Zeitpunkt berechnet
  - ev_pct/ev_dollars/breakeven_move_pct        : aus der zum Entscheid bewerteten Option
  - iv_to_rv/iv_rank/iv_percentile              : IV-Rank aus Journal-Historie BIS zum Snapshot
  - gap_pct/rvol/gap_volume_*/above_ma50/unusual: Tageskurs-/Volumendaten bis Decision
  - news_alpha/news_*/sentiment_*               : aus dem News-Cluster vor dem Trade
  - vix/vix_regime                              : VIX bei Run-Start
  - dir_call/horizon_num/dte_days/is_etf        : Signal-Parameter
  - data_quality_score                          : Datenqualität des Snapshots
Bewusst NICHT als Feature: alles aus `trade_resolutions`/`outcomes` (= Label/Zukunft).

CLI:
    python src/ml_predictor.py --train        # trainieren (falls genug Daten)
    python src/ml_predictor.py --info         # Status, Metriken, Kalibrierung, Importances
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Optionale ML-Stacks: XGBoost bevorzugt, sonst sklearn-Fallbacks ──────────
ML_AVAILABLE = False
_IMPORT_ERROR = None
_HAS_XGB = False
try:
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.ensemble import (
        HistGradientBoostingClassifier, HistGradientBoostingRegressor,
        RandomForestClassifier,
    )
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    ML_AVAILABLE = True
    try:
        from xgboost import XGBClassifier, XGBRegressor
        _HAS_XGB = True
    except Exception:
        _HAS_XGB = False
except Exception as exc:  # pragma: no cover
    _IMPORT_ERROR = str(exc)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = DATA_DIR / "ml_outcome_model.joblib"
FEATURES_PATH = DATA_DIR / "ml_feature_names.json"


def _artifact_dir() -> Path:
    """Artefakt-Verzeichnis zur AUFRUFZEIT (Test-Isolation via ML_DATA_DIR).

    Ohne Override: data/. Verhindert, dass ein Trainingslauf im Test je das echte
    Produktiv-Modell überschreibt."""
    override = os.environ.get("ML_DATA_DIR")
    return Path(override) if override else DATA_DIR


def _model_path() -> Path:
    return _artifact_dir() / "ml_outcome_model.joblib"


def _features_path() -> Path:
    return _artifact_dir() / "ml_feature_names.json"


NAN = float("nan")

# Vollständiges Feature-Set (extract_features liefert genau diese Keys).
ALL_FEATURES = [
    "score", "raw_signal_score", "gate_adjusted_score",
    "ev_pct", "ev_dollars", "breakeven_move_pct",
    "iv_to_rv", "iv_rank", "iv_percentile",
    "gap_pct", "rvol", "gap_volume_confirmed", "gap_volume_bonus",
    "above_ma50", "unusual",
    "sector_momentum_confirmation", "relative_to_sector_pct", "sector_vs_market_pct",
    "news_alpha", "news_sentiment_score", "sentiment_price_score_adjustment",
    "vix", "vix_regime", "dir_call", "horizon_num", "dte_days", "is_etf",
    "data_quality_score",
]
# Rückwärtskompatibler Alias.
FEATURE_COLUMNS = ALL_FEATURES

# Reduziertes, robustes Start-Set (Risiko: Overfitting bei vielen Features + wenig Daten).
# Bewusst klein gehalten und auf die ökonomisch tragfähigsten Treiber beschränkt.
CORE_FEATURES = [
    "score", "ev_pct", "breakeven_move_pct", "iv_to_rv", "iv_rank",
    "gap_pct", "rvol", "news_alpha", "sentiment_price_score_adjustment",
    "vix", "vix_regime", "dir_call", "dte_days",
]


def _num(value: Any) -> float:
    try:
        if value is None or value == "":
            return NAN
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _encode_momentum(value: Any) -> float:
    s = str(value or "").lower()
    if "confirm" in s:
        return 1.0
    if "disagree" in s or "against" in s or "conflict" in s:
        return -1.0
    return 0.0


def _vix_regime(vix: float) -> float:
    """0 = Low-Vol (<18), 1 = Mid (18-25), 2 = High-Vol (>=25). NaN wenn VIX fehlt."""
    if vix != vix:  # NaN
        return NAN
    if vix < 18.0:
        return 0.0
    if vix < 25.0:
        return 1.0
    return 2.0


def extract_features(d: dict, vix: Any, direction: Any, horizon: Any, dte_days: Any) -> dict:
    """Baut den Feature-Vektor aus einem process_ticker-Dict (live ODER aus market_json).

    Identische Funktion für Training und Inferenz -> keine Train/Serve-Drift.
    Alle Werte stammen aus dem Decision-Zeitpunkt (siehe FEATURE-AUDIT oben).
    """
    d = d or {}
    opt = d.get("options") or {}
    above = d.get("above_ma50")
    vix_num = _num(vix)
    return {
        "score": _num(d.get("score")),
        "raw_signal_score": _num(d.get("raw_signal_score")),
        "gate_adjusted_score": _num(d.get("gate_adjusted_score")),
        "ev_pct": _num(opt.get("ev_pct")),
        "ev_dollars": _num(opt.get("ev_dollars")),
        "breakeven_move_pct": _num(opt.get("breakeven_move_pct")),
        "iv_to_rv": _num(opt.get("iv_to_rv")),
        "iv_rank": _num(opt.get("iv_rank")),
        "iv_percentile": _num(opt.get("iv_percentile")),
        "gap_pct": _num(d.get("gap_pct")),
        "rvol": _num(d.get("rvol")),
        "gap_volume_confirmed": 1.0 if d.get("gap_volume_confirmed") else 0.0,
        "gap_volume_bonus": _num(d.get("gap_volume_bonus")),
        "above_ma50": 1.0 if above is True else (0.0 if above is False else NAN),
        "unusual": 1.0 if d.get("unusual") else 0.0,
        "sector_momentum_confirmation": _encode_momentum(d.get("sector_momentum_confirmation")),
        "relative_to_sector_pct": _num(d.get("relative_to_sector_pct")),
        "sector_vs_market_pct": _num(d.get("sector_vs_market_pct")),
        "news_alpha": _num(d.get("news_alpha")),
        "news_sentiment_score": _num(d.get("news_sentiment_score")),
        "sentiment_price_score_adjustment": _num(d.get("sentiment_price_score_adjustment")),
        "vix": vix_num,
        "vix_regime": _vix_regime(vix_num),
        "dir_call": 1.0 if str(direction).upper() == "CALL" else 0.0,
        "horizon_num": {"T1": 1.0, "T2": 2.0, "T3": 3.0}.get(str(horizon).upper(), NAN),
        "dte_days": _num(dte_days),
        "is_etf": 1.0 if d.get("is_etf") else 0.0,
        "data_quality_score": _num(d.get("data_quality_score")),
    }


# ── Modell-Factories (starke Regularisierung gegen Overfitting) ──────────────
def _make_classifier():
    """Bevorzugt XGBoost, sonst HistGradientBoosting, sonst RandomForest. Stark reguliert."""
    if _HAS_XGB:
        return XGBClassifier(
            max_depth=3, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.5, learning_rate=0.05, n_estimators=200,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ), "xgboost"
    try:
        return HistGradientBoostingClassifier(
            max_depth=3, max_leaf_nodes=15, learning_rate=0.05, max_iter=250,
            min_samples_leaf=10, l2_regularization=2.0, random_state=42,
        ), "hist_gbm"
    except Exception:
        return RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ), "random_forest"


def _make_regressor():
    if _HAS_XGB:
        return XGBRegressor(
            max_depth=3, min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=2.0, reg_alpha=0.5, learning_rate=0.05, n_estimators=200,
            random_state=42, n_jobs=-1,
        ), "xgboost"
    try:
        return HistGradientBoostingRegressor(
            max_depth=3, max_leaf_nodes=15, learning_rate=0.05, max_iter=250,
            min_samples_leaf=10, l2_regularization=2.0, random_state=42,
        ), "hist_gbm"
    except Exception:
        return None, None


def _calibration_buckets(y_true, p_pred, n_bins: int = 10) -> list[dict]:
    """Kalibrierung in 10%-Buckets: vorhergesagte Wahrscheinlichkeit vs. tatsächliche Win-Rate."""
    out = []
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (p_pred >= lo) & (p_pred < hi if i < n_bins - 1 else p_pred <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        out.append({
            "bucket": f"{int(lo * 100)}-{int(hi * 100)}%",
            "n": n,
            "mean_pred": round(float(p_pred[mask].mean()), 3),
            "actual_rate": round(float(y_true[mask].mean()), 3),
        })
    return out


class OutcomePredictor:
    """Kapselt Laden/Trainieren/Vorhersagen + Monitoring des Outcome-Modells."""

    def __init__(self) -> None:
        self._bundle: dict | None = None

    # ── Persistenz ────────────────────────────────────────────────────
    def load(self) -> bool:
        if not ML_AVAILABLE:
            return False
        if self._bundle is not None:
            return True
        model_path = _model_path()
        if model_path.exists():
            try:
                self._bundle = joblib.load(model_path)
                return True
            except Exception as exc:
                logger.warning("ML-Modell konnte nicht geladen werden: %s", exc)
        return False

    def is_trained(self) -> bool:
        return bool(self._bundle and self._bundle.get("model") is not None)

    def n_trades_trained(self) -> int:
        return int(self._bundle["meta"].get("n_trades", 0)) if self.is_trained() else 0

    def model_version(self) -> str | None:
        return self._bundle["meta"].get("model_version") if self.is_trained() else None

    def is_productive(self) -> bool:
        """Risiko 1: produktive Nutzung erst ab RULES.ml_reliable_min_trades (=100) Trades.

        Darunter bleibt das Modell deaktiviert (kein Boost/Gate), nur Shadow-Logging.
        """
        from rules import RULES
        return self.is_trained() and self.n_trades_trained() >= RULES.ml_reliable_min_trades

    # Rückwärtskompatibler Alias (Aufrufer nutzen teils is_reliable()).
    def is_reliable(self) -> bool:
        return self.is_productive()

    # ── Daten ─────────────────────────────────────────────────────────
    def prepare_labeled_dataset(self) -> "pd.DataFrame | None":
        """Features + Labels aus dem Journal (signals ⋈ runs ⋈ trade_resolutions), zeitlich sortiert."""
        if not ML_AVAILABLE:
            return None
        from trading_journal import connect
        con = connect()
        rows = con.execute(
            """
            SELECT s.market_json, s.direction, s.horizon, s.dte_days, r.vix,
                   tr.is_win, tr.exit_return_pct, tr.resolved_at
            FROM trade_resolutions tr
            JOIN signals s ON s.signal_id = tr.signal_id
            JOIN runs r ON r.run_id = s.run_id
            WHERE tr.status = 'resolved' AND tr.is_win IS NOT NULL
            ORDER BY tr.resolved_at ASC
            """
        ).fetchall()
        con.close()

        records = []
        for row in rows:
            try:
                d = json.loads(row["market_json"]) if row["market_json"] else {}
            except (ValueError, TypeError):
                d = {}
            feats = extract_features(d, row["vix"], row["direction"], row["horizon"], row["dte_days"])
            feats["is_win"] = int(row["is_win"])
            feats["exit_return_pct"] = _num(row["exit_return_pct"])
            feats["resolved_at"] = row["resolved_at"]
            records.append(feats)
        return pd.DataFrame.from_records(records)

    # ── Training-Helfer ────────────────────────────────────────────────
    @staticmethod
    def _fill(X: "pd.DataFrame", medians: dict) -> "pd.DataFrame":
        return X.fillna(value=medians).fillna(0.0)

    def _walk_forward_oos(self, X, y):
        """Walk-Forward-CV (Risiko: Concept Drift): expanding window, chronologisch.

        Liefert gepoolte Out-of-Sample-Vorhersagen (für AUC/Brier/Kalibrierung) + Fold-AUCs.
        Fällt bei zu wenig Daten auf einen einzelnen 75/25-Zeit-Split zurück.
        """
        n = len(X)
        y_oos, p_oos, fold_aucs = [], [], []

        def _fit_pred(tr_idx, te_idx):
            ytr = y.iloc[tr_idx]
            if ytr.nunique() < 2:
                return None
            med = {c: (float(X.iloc[tr_idx][c].median()) if X.iloc[tr_idx][c].notna().any() else 0.0)
                   for c in X.columns}
            m, _ = _make_classifier()
            m.fit(self._fill(X.iloc[tr_idx], med), ytr)
            return m.predict_proba(self._fill(X.iloc[te_idx], med))[:, 1]

        if n >= 80:
            n_splits = 4
            start = int(n * 0.5)
            bounds = np.linspace(start, n, n_splits + 1, dtype=int)
            for i in range(n_splits):
                tr_idx = list(range(0, bounds[i]))
                te_idx = list(range(bounds[i], bounds[i + 1]))
                if len(te_idx) < 5:
                    continue
                p = _fit_pred(tr_idx, te_idx)
                if p is None:
                    continue
                yte = y.iloc[te_idx]
                y_oos.extend(yte.tolist())
                p_oos.extend(p.tolist())
                if yte.nunique() >= 2:
                    try:
                        fold_aucs.append(round(float(roc_auc_score(yte, p)), 3))
                    except ValueError:
                        pass
        elif n >= 40:
            cut = int(n * 0.75)
            p = _fit_pred(list(range(cut)), list(range(cut, n)))
            if p is not None:
                y_oos = y.iloc[cut:].tolist()
                p_oos = list(p)
        return (np.array(y_oos), np.array(p_oos), fold_aucs) if y_oos else (None, None, [])

    def _select_features(self, X, y, candidate):
        """Importance-basierte Feature-Selection (Risiko: Overfitting).

        Permutation-Importance (modell-agnostisch) auf einem Zeit-Holdout; behält Features
        mit positivem Beitrag (min. 6, max. 15). Liefert (selected, importances_dict).
        """
        n = len(X)
        if n < 50:
            # Zu wenig Daten für verlässliche Selektion -> reduziertes Set unverändert.
            return candidate, {f: 0.0 for f in candidate}
        cut = int(n * 0.75)
        med = {c: (float(X.iloc[:cut][c].median()) if X.iloc[:cut][c].notna().any() else 0.0)
               for c in candidate}
        m, _ = _make_classifier()
        m.fit(self._fill(X.iloc[:cut], med), y.iloc[:cut])
        try:
            r = permutation_importance(
                m, self._fill(X.iloc[cut:], med), y.iloc[cut:],
                n_repeats=5, random_state=42, scoring="roc_auc",
            )
            imp = {f: float(v) for f, v in zip(candidate, r.importances_mean)}
        except Exception as exc:
            logger.debug("Permutation-Importance fehlgeschlagen: %s", exc)
            return candidate, {f: 0.0 for f in candidate}

        ranked = sorted(candidate, key=lambda f: imp[f], reverse=True)
        selected = [f for f in ranked if imp[f] > 0.0]
        if len(selected) < 6:
            selected = ranked[:6]
        selected = selected[:15]
        # Originalreihenfolge beibehalten (Determinismus).
        selected = [f for f in candidate if f in set(selected)]
        return selected, imp

    # ── Training ──────────────────────────────────────────────────────
    def train(self, min_trades: int = 50) -> dict | None:
        """Trainiert Klassifikator (+ optional Regressor). None, wenn zu wenig/eindeutige Daten.

        Hinweis: `min_trades` ist die Trainings-Untergrenze (Modell existiert für Monitoring).
        Produktiv wird es erst ab RULES.ml_reliable_min_trades (=100), siehe is_productive().
        """
        if not ML_AVAILABLE:
            logger.warning("ML nicht verfügbar (Import fehlgeschlagen): %s", _IMPORT_ERROR)
            return None

        df = self.prepare_labeled_dataset()
        n = 0 if df is None else len(df)
        if n < min_trades:
            logger.info("ML: zu wenig aufgelöste Trades zum Trainieren (%d < %d)", n, min_trades)
            return None

        y = df["is_win"].astype(int)
        if y.nunique() < 2:
            logger.info("ML: nur eine Klasse vorhanden — kein Training")
            return None

        candidate = [f for f in CORE_FEATURES if f in df.columns]
        Xc = df[candidate].copy()

        # Walk-Forward OOS (Drift-robuste Schätzung) + Kalibrierung daraus.
        y_oos, p_oos, fold_aucs = self._walk_forward_oos(Xc, y)
        oos: dict = {}
        calibration: list[dict] = []
        if y_oos is not None and len(y_oos) >= 10:
            try:
                oos["auc"] = round(float(roc_auc_score(y_oos, p_oos)), 3) if len(set(y_oos)) > 1 else None
            except ValueError:
                oos["auc"] = None
            oos["acc"] = round(float(accuracy_score(y_oos, (p_oos >= 0.5).astype(int))), 3)
            try:
                oos["brier"] = round(float(brier_score_loss(y_oos, p_oos)), 4)
            except ValueError:
                oos["brier"] = None
            oos["n_test"] = int(len(y_oos))
            oos["fold_aucs"] = fold_aucs
            calibration = _calibration_buckets(y_oos, p_oos)

        # Feature-Selection auf dem reduzierten Set.
        selected, importances = self._select_features(Xc, y, candidate)
        Xs = df[selected].copy()
        medians = {c: (float(Xs[c].median()) if Xs[c].notna().any() else 0.0) for c in selected}

        # Finales Produktionsmodell auf ALLEN Daten.
        model, algo = _make_classifier()
        model.fit(self._fill(Xs, medians), y)

        # Optionale Regressions-Variante auf exit_return_pct.
        regressor, reg_algo = None, None
        ret = df["exit_return_pct"]
        if ret.notna().sum() >= max(min_trades, 40) and ret.std(skipna=True) and ret.std(skipna=True) > 0:
            reg, reg_algo = _make_regressor()
            if reg is not None:
                try:
                    reg.fit(self._fill(Xs, medians), ret.fillna(ret.median()))
                    regressor = reg
                except Exception as exc:
                    logger.debug("Regressor-Training fehlgeschlagen: %s", exc)
                    regressor = None

        trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta = {
            "n_trades": int(n),
            "trained_at": trained_at,
            "model_version": f"{algo}_{trained_at}_n{n}",
            "algo": algo,
            "regressor_algo": reg_algo,
            "win_rate": round(float(y.mean()), 3),
            "features": selected,
            "n_features": len(selected),
            "oos": oos,
            "calibration": calibration,
            "min_trades": min_trades,
            "productive_min": _import_rules_min(),
            "xgboost": _HAS_XGB,
        }
        bundle = {"model": model, "regressor": regressor, "features": selected,
                  "medians": medians, "importances": importances, "meta": meta}

        artifact_dir = _artifact_dir()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, _model_path())
        _features_path().write_text(json.dumps(selected, indent=2), encoding="utf-8")
        self._bundle = bundle
        logger.info("ML-Modell trainiert: algo=%s n=%d win_rate=%.2f feats=%d oos=%s",
                    algo, n, meta["win_rate"], len(selected), oos)
        return meta

    def maybe_retrain(self, min_trades: int = 50, max_age_days: int = 7,
                      new_trades_trigger: int = 20) -> dict | None:
        """Auto-(Re)Training: erstes Modell ab min_trades, danach alle max_age_days
        oder bei >= new_trades_trigger neuen aufgelösten Trades. Fail-safe."""
        if not ML_AVAILABLE:
            return None
        self.load()
        from trading_journal import connect
        con = connect()
        n = con.execute(
            "SELECT COUNT(*) FROM trade_resolutions WHERE status='resolved' AND is_win IS NOT NULL"
        ).fetchone()[0]
        con.close()

        if not self.is_trained():
            if n >= min_trades:
                return self.train(min_trades)
            logger.info("ML: noch kein Modell — %d/%d aufgelöste Trades", n, min_trades)
            return None

        meta = self._bundle["meta"]
        trained_n = int(meta.get("n_trades", 0))
        try:
            age_days = (datetime.now(timezone.utc) - datetime.fromisoformat(meta["trained_at"])).days
        except (KeyError, ValueError):
            age_days = 999
        if (n - trained_n) >= new_trades_trigger or age_days >= max_age_days:
            logger.info("ML: Retraining (jetzt=%d, trainiert_mit=%d, alter=%dd)", n, trained_n, age_days)
            return self.train(min_trades)
        return meta

    # ── Inferenz ──────────────────────────────────────────────────────
    def _vector(self, features: dict):
        bundle = self._bundle
        medians = bundle.get("medians", {})
        vec = []
        for name in bundle["features"]:
            val = _num(features.get(name))
            if val != val:  # NaN -> Trainings-Median
                val = float(medians.get(name, 0.0))
            vec.append(val)
        return pd.DataFrame([vec], columns=bundle["features"])

    def predict_win_probability(self, features: dict) -> float:
        """Win-Wahrscheinlichkeit [0,1]. Fail-safe: 0.5 (neutral) bei fehlendem Modell/Fehler."""
        if not ML_AVAILABLE:
            return 0.5
        if not self.is_trained() and not self.load():
            return 0.5
        try:
            proba = float(self._bundle["model"].predict_proba(self._vector(features))[0, 1])
            return max(0.0, min(1.0, proba))
        except Exception as exc:
            logger.debug("ML predict fehlgeschlagen: %s", exc)
            return 0.5

    def predict_expected_return(self, features: dict) -> float | None:
        """Optionale Regressions-Vorhersage des Exit-Returns in %. None, wenn kein Regressor."""
        if not ML_AVAILABLE or (not self.is_trained() and not self.load()):
            return None
        reg = self._bundle.get("regressor")
        if reg is None:
            return None
        try:
            return round(float(reg.predict(self._vector(features))[0]), 2)
        except Exception:
            return None

    def get_feature_importance(self) -> "pd.DataFrame | None":
        if not ML_AVAILABLE:
            return None
        if not self.is_trained() and not self.load():
            return pd.DataFrame(columns=["feature", "importance"])
        imp = self._bundle.get("importances", {})
        feats = self._bundle["features"]
        df = pd.DataFrame({"feature": feats, "importance": [imp.get(f, 0.0) for f in feats]})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)

    def calibration(self) -> list[dict]:
        """Kalibrierungs-Buckets aus dem Walk-Forward-OOS (Training)."""
        if not self.is_trained() and not self.load():
            return []
        return self._bundle["meta"].get("calibration", [])

    # ── Monitoring / Drift (Risiken 3 & 8) ─────────────────────────────
    def monitor_drift(self, recent_days: int = 90) -> dict:
        """Vergleicht die tatsächliche Win-Rate der letzten `recent_days` mit älteren Daten.

        Modell-agnostisch (misst die reale Performance der aufgelösten Trades). Setzt
        `degraded=True` bei deutlichem Abfall (>=15pp, je Seite genug Stichprobe).
        """
        if not ML_AVAILABLE:
            return {"available": False}
        from trading_journal import connect
        con = connect()
        rows = con.execute(
            "SELECT is_win, resolved_at FROM trade_resolutions "
            "WHERE status='resolved' AND is_win IS NOT NULL"
        ).fetchall()
        con.close()

        cutoff = datetime.now(timezone.utc) - timedelta(days=recent_days)
        recent, older = [], []
        for r in rows:
            try:
                dt = datetime.fromisoformat(r["resolved_at"])
                dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
            (recent if dt >= cutoff else older).append(int(r["is_win"]))

        def _wr(xs):
            return round(sum(xs) / len(xs), 3) if xs else None

        rec_wr, old_wr = _wr(recent), _wr(older)
        degraded = (rec_wr is not None and old_wr is not None
                    and len(recent) >= 15 and len(older) >= 15
                    and (old_wr - rec_wr) >= 0.15)
        return {
            "available": True,
            "recent_days": recent_days,
            "recent": {"n": len(recent), "win_rate": rec_wr},
            "older": {"n": len(older), "win_rate": old_wr},
            "degraded": bool(degraded),
        }


def _import_rules_min() -> int:
    try:
        from rules import RULES
        return RULES.ml_reliable_min_trades
    except Exception:
        return 100


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="ML Outcome-Predictor")
    parser.add_argument("--train", action="store_true", help="Modell trainieren (falls genug Daten)")
    parser.add_argument("--info", action="store_true", help="Status/Metriken/Kalibrierung/Importances")
    parser.add_argument("--min-trades", type=int, default=50, help="Trainings-Untergrenze")
    args = parser.parse_args()

    if not ML_AVAILABLE:
        raise SystemExit(f"ML-Stack nicht installiert (sklearn/joblib/pandas): {_IMPORT_ERROR}")

    predictor = OutcomePredictor()
    if args.train:
        meta = predictor.train(min_trades=args.min_trades)
        if meta:
            print("Training OK:", json.dumps(meta, indent=2, default=str))
            print("\nFeature-Importances:")
            print(predictor.get_feature_importance().head(15).to_string(index=False))
            print("\nProduktiv nutzbar (>=%d Trades): %s" % (meta["productive_min"], predictor.is_productive()))
        else:
            print("Kein Modell trainiert (zu wenig Daten / nur eine Klasse / Stack fehlt).")
    elif args.info:
        if predictor.load() and predictor.is_trained():
            print("Meta:", json.dumps(predictor._bundle["meta"], indent=2, default=str))
            print("productive:", predictor.is_productive())
            print("drift:", json.dumps(predictor.monitor_drift(), indent=2, default=str))
            print("\nImportances:")
            print(predictor.get_feature_importance().head(15).to_string(index=False))
        else:
            print("Kein trainiertes Modell vorhanden.")
    else:
        parser.print_help()
