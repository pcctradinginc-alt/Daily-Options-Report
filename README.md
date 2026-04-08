[README.md](https://github.com/user-attachments/files/26564692/README.md)
# Options Trading Signal Bot

Vollautomatisches tägliches Options-Trading-Signal-System.
Analysiert Finanznews, bewertet Marktdaten und verschickt
eine HTML-Email mit konkreten Handelsempfehlungen.

---

## Wie es funktioniert

```
1. News-Analyse
   14 RSS-Feeds (Reuters, Bloomberg, CNBC, Benzinga) parallel.
   Artikel werden geclustert und mit gewichtetem Score bewertet
   (Aktualität × Quellen-Qualität × Velocity × Earnings-Proximity).
   Claude analysiert Top-Cluster → handelbare Signale.

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

## Disclaimer

Dieses Projekt dient ausschließlich zu Bildungszwecken und stellt
keine Anlageberatung dar. Trading mit Optionen birgt erhebliche Risiken.
