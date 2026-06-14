[README.md](https://github.com/user-attachments/files/26564692/README.md)
# Options Trading Signal Bot

Vollautomatisches tägliches Options-Trading-Signal-System.
Analysiert Finanznews, bewertet Marktdaten und verschickt
eine HTML-Email mit konkreten Handelsempfehlungen.

---

## Wie es funktioniert

```
1. News-Analyse
   ~10 RSS-Feeds (Bloomberg, CNBC, Benzinga, Yahoo, MarketWatch, WSJ, FT, SEC) parallel.
   Artikel werden je Ticker geclustert und nach Katalysator-Typ bewertet
   (FDA, Merger, 8-K/13D/Form-4, Wire) → News-Alpha 0–100.
   Claude wählt aus den Top-Clustern → handelbare Signale.
   (Earnings-Nähe ist KEIN Score-Bonus, sondern ein Risiko-Gate; siehe Schritt 2.)

2. Marktdaten
   Kurse (AlphaVantage → Yahoo → Finnhub), historische Daten
   (MA50, MA20, RelVol) und Options-Greeks (Tradier).
   Normalisierter Score 0–100 mit Trend-Alignment und Liquiditäts-Filter.

3. Report
   Claude erstellt Trade-Empfehlung mit 5-Punkte-Begründung,
   Exit-Plan und Marktstatus. Versand als HTML-Email.
```

---

## Voraussetzungen

Python 3.9+

| API | Zweck | Kosten |
|-----|-------|--------|
| [Anthropic](https://console.anthropic.com) | Claude | ~$0.01/Tag |
| [Tradier](https://developer.tradier.com) | Options-Greeks | Sandbox: kostenlos |
| [Finnhub](https://finnhub.io) | Earnings | Free Tier |
| [Alpha Vantage](https://www.alphavantage.co) | Kurse | Free: 25/Tag |

Gmail App-Passwort: Google Account → Sicherheit → 2FA → App-Passwörter

---

## Installation

```bash
git clone https://github.com/DEIN-USERNAME/options-trading-bot.git
cd options-trading-bot
pip install -r requirements.txt
cp config/config.example.yaml config/config.yaml
# config.yaml mit API Keys befüllen
```

---

## Verwendung

```bash
# Normaler Lauf (verschickt Email)
python src/main.py

# Dry-run (kein Email, Report als HTML gespeichert)
python src/main.py --dry-run

# Mit Details in der Konsole
python src/main.py --dry-run --verbose

# Einzelne Steps testen
python src/news_analyzer.py --verbose
python src/market_data.py --signals "UBER:CALL:MED:T1:21DTE"
python src/report_generator.py --summary-file market_summary.txt --dry-run
```

---

## Automatisch täglich (Cron)

```bash
# Täglich Mo–Fr um 10:30 ET (14:30 UTC)
30 14 * * 1-5 cd /pfad/zum/bot && python src/main.py >> logs/daily.log 2>&1
```

---

## GitHub Actions (automatisch in der Cloud)

Secrets setzen: Repository → Settings → Secrets and variables → Actions

```
ANTHROPIC_API_KEY
TRADIER_TOKEN
FINNHUB_KEY
ALPHA_VANTAGE_KEY
GMAIL_RECIPIENT
SMTP_SENDER
SMTP_PASSWORD
```

Dann läuft der Bot täglich Mo–Fr automatisch um 14:30 UTC.
Manueller Start: Actions → Daily Options Report → Run workflow

---

## Handelsregeln

| VIX | Einsatz | Status |
|-----|---------|--------|
| ≥ 25 | — | ❌ Kein Trade |
| 20–24.99 | 150 € | ⚠️ Reduziert |
| < 20 | 250 € | ✅ Normal |

Ausschluss wenn: Score < 50 · Δ% gegen Signal · unter MA50 · Spread > 2% · OI < 5.000

---

## Machine Learning Outcome Predictor

Optionales ML-Modul, das die Trefferrate der empfohlenen Trades verbessern soll.
Es ist bewusst **weich** und **fail-safe**: Fehlt das Modell oder sklearn, läuft der Bot
unverändert weiter.

**Faithful Label (echte Exit-Regeln).** Jeder empfohlene/bewertete Trade mit echtem
Optionskontrakt wird in `trade_resolutions` verfolgt. Bei jedem Lauf wird der aktuelle
Options-Mark (Tradier) geholt und die **Exit-Regeln der Empfehlung** aufgelöst:
Take-Profit (+50 %), Stop-Loss (−30 %), Time-Stop und Expiry. Das Label `is_win` spiegelt
also den realen Trade-Ausgang wider — nicht nur die Underlying-Bewegung. Granularität:
1 Mark pro Lauf (tagesgenau), forward-only; vergangene Trades werden nicht nachgelabelt.

**Modell.** `src/ml_predictor.py` trainiert einen `RandomForestClassifier` (flach gehalten
gegen Overfitting) auf 27 Decision-Zeitpunkt-Features (Score, EV, IV/RV, IV-Rank, Gap/RVol,
Sektor-Momentum, News-Confidence/Sentiment, VIX, Direction, Horizon, DTE …). Gespeichert als
`data/ml_outcome_model.joblib` + `data/ml_feature_names.json`. Retraining automatisch ab 50
aufgelösten Trades, danach alle 7 Tage oder bei ≥ 20 neuen Outcomes.

**Tägliche Wirkung (weich).** In Schritt 2.5 berechnet der Bot pro Ticker eine Win-
Wahrscheinlichkeit. Sie fließt als Conviction-Bonus und (erst ab `ml_reliable_min_trades`
aufgelösten Trades) als zusätzliches Gate `ml_win_prob ≥ 0.52` ein. **Harte Gates
(VIX, Liquidität, EV, Datenqualität, Earnings-IV) werden nie überschrieben.**

**Monatlicher Win-Rate-Report.** Am 1. des Monats (`monthly_winrate_report.yml`) kommt eine
E-Mail mit Gesamt-Win-Rate (inkl. Wilson-Konfidenzintervallen), Win-Rate pro Monat/Horizon/
Stärke/Sektor, **ML-gefiltert vs. ungefiltert** (forward/out-of-sample) und Feature-Importances.
Bei zu wenig Daten zeigt der Report transparent „noch zu wenig Trades" statt Schein-Signifikanz.

```bash
# Modell manuell trainieren (sobald genug Daten da sind)
python src/ml_predictor.py --train
python src/ml_predictor.py --info          # Status + Feature-Importances

# Monatsreport testen (kein Email-Versand, schreibt monthly_winrate_preview.html)
python src/monthly_winrate_report.py --dry-run
```

> Realistische Erwartung: Bei ~1 Trade/Tag dauert es Monate, bis genügend unabhängige,
> aufgelöste Trades für ein belastbares Modell vorliegen. Bis dahin ist das ML-Gate inaktiv
> und der Report weist die dünne Datenlage offen aus.

---

## Disclaimer

Dieses Projekt dient ausschließlich zu Bildungszwecken und stellt
keine Anlageberatung dar. Trading mit Optionen birgt erhebliche Risiken.
