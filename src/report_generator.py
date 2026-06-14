"""
report_generator.py — HTML-Report + Email-Versand (Step 3)

Fixes v2:
- call_claude() nimmt vix_direct Parameter (Fix Nr. 1+2)
- VIX aus main.py direkt genutzt — nicht aus Claude-JSON
- build_html() zeigt PUT/CALL korrekt an (Fix Nr. 7)
- _compress_summary(): Earnings-Liste auf 10 Ticker gekürzt
- max_tokens 1500, timeout 30s
- Exit-Plan: Stop-Loss -40%, Take-Profit +50%, konkrete USD-Preise
"""

import json
import logging
import smtplib
import sys
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import requests
from requests.exceptions import RequestException, Timeout

from rules import apply_vix_rules, RULES
from llm_schema import validate_report_payload, build_cancelled_report

logger = logging.getLogger(__name__)

PROMPT = """Du bist eine regelbasierte Options-KI. Antworte NUR mit JSON - kein Text, kein Markdown.

HARTE REGELN:
- VIX >= 25 -> no_trade: true, no_trade_grund: maximal 12 Woerter ohne Satzzeichen
- VIX 20-24.99 -> einsatz: 150
- VIX < 20 -> einsatz: 250
- Waehle NIEMALS einen Ticker mit Score < 65 fuer echten Trade. Score 50-64 ist nur Research.
- Waehle NIEMALS einen Ticker mit Gate=FAIL, DATA_QUALITY_OK=False, SECTOR_MARKET_OK=False, EV_OK=False, EARN_IV_OK=False oder Liquiditaets-Hinweis
- Nutze conservative_entry/Entry als Einstiegspreis, NICHT blind Midpoint
- kontrakte = floor(einsatz / (entry_price * 100))
- stop_loss_eur = 30% von einsatz
- bid/ask/midpoint/entry/ev aus Marktdaten uebernehmen, nicht schaetzen
- Sentiment darf NIEMALS einen schlechten EV, schlechte Liquiditaet oder Earnings-IV-Block ueberschreiben
- Du siehst absichtlich KEINE News-Texte. Entscheide nur anhand nackter Marktdaten, Gates, Greeks, Preis, Liquiditaet, IV/RV und Sektor.

DATENQUALITAET:
- Tradier Production ist Standard. Sandbox/Delayed-Daten nur als Dry-run-Kontext betrachten.
- Tradier-Optionsdaten mit nicht-Tradier-Underlying sind immer no_trade true. Kein Yahoo/AlphaVantage-Fallback fuer finalen EV.
- Wenn Quote-Quelle oder Optionsdaten inkonsistent sind: no_trade true.
- Wenn DATA_FLAGS auf kaputte Historie, Spike ohne News oder fehlende Basisdaten hinweisen: no_trade true.
- Wenn No-Trade-Reason im Marktdatenblock steht, diese Begruendung uebernehmen.

MARKT-/SEKTORFILTER:
- CALL braucht idealerweise Aktie > Sektor und Sektor > SPY/QQQ.
- PUT braucht idealerweise Aktie < Sektor und Sektor < SPY/QQQ.
- Gegen klaren Sektor-/Markttrend: no_trade oder Research-Only, nicht schoenrechnen.
- Relative Staerke/Schwaeche darf den Score verbessern, aber nie EV/Liquiditaet/Datenqualitaet ueberschreiben.

SENTIMENT/PREISREAKTION:
- Nutze SentPx als Divergenzfeature: bearish_news_absorbed kann CALL bestaetigen, bullish_news_not_confirmed kann PUT bestaetigen.
- SentPx ist nur Ranking/Timing, kein harter EV-Ersatz.

RICHTUNGSLOGIK:
- CALL darf positiv laufen: change_pct > 0 und ueber MA50 ist gut
- CALL ist schwach bei change_pct < 0 oder unter MA50
- PUT darf negativ laufen: change_pct < 0 und unter MA50 ist gut
- PUT ist schwach bei change_pct > 0 oder ueber MA50
- Also: change_pct < 0 oder unter MA50 ist KEIN Ausschluss fuer PUT

OPTIONS-EV UND KOSTEN:
- Bevorzuge hoechstes EV%, positives EV$, hohe FillP, niedrigen Spread, ausreichendes OI
- ExitSlip ist realer Kostenblock und muss im Risiko genannt werden
- Kein Trade wenn erwarteter Move Entry+Exit-Slippage+Theta+IV-Risiko nicht klar schlaegt
- Chance/Risiko muss Entry, Break-even-Move, EV%, EV$, FillP, ExitSlip, IV/RV, IVRank und TimeStop nennen

EARNINGS / IV-CRUSH:
- EARN_IV_OK=False ist harter Ausschluss fuer Long-Optionen
- IVRank/IVPct aus eigener Journal-Historie: bei hohem Rank/Percentile ist Long-Option zu teuer
- Cold Start: Wenn IV-Historie zu kurz ist und IV/RV >= 1.50, ist Long-Option no_trade wegen Overpricing
- Wenn Earnings nahe und IV/RV unbekannt oder zu hoch: no_trade true
- Earnings nicht nur als Score-Malus behandeln, sondern als Trade-Gate

ETF-SONDERREGEL:
- ETF nur ausgeben, wenn Optionsdaten und EV_OK vorhanden sind
- Wenn keine Optionsdaten: no_trade true

BEGRUENDUNG (begruendung_detail - 5 Felder, je max 2 Saetze, keine Anfuehrungszeichen):
- ticker_wahl: Warum dieser Ticker? Score- und EV-Vergleich.
- option_wahl: Strike, Delta, IV, IV/RV, Spread, Entry, ExitSlip, EV.
- timing: Richtungsspezifisch: CALL vs PUT, MA50, RelVol, Sektorfilter, SentPx-Divergenz.
- chance_risiko: Einsatz, Entry, Break-even, Ziel, Stop.
- risiko: Hauptrisiko inklusive Spread, Slippage, Datenqualitaet, Earnings/IV.

TIME-STOP:
- Bei 7-14 DTE: nach 24h pruefen.
- Bei 15-30 DTE: nach 48h pruefen.
- Bei >30 DTE: nach 72h pruefen.
- Wenn Underlying dann nicht mindestens 1% in Zielrichtung gelaufen ist: Exit/Close pruefen.

MARKTSTATUS: markt-Feld 2-3 Saetze. strategie-Feld 1 Satz.
TICKER_TABELLE: ALLE Ticker aus Marktdaten eintragen.
Regime NUR: LOW-VOL, TRENDING oder HIGH-VOL
regime_farbe NUR: gruen, gelb oder rot

Gib direction exakt aus den Marktdaten zurueck: CALL oder PUT.

JSON-Schema:
{"datum":"DD.MM.YYYY","vix":"WERT","regime":"TRENDING","regime_farbe":"gelb","no_trade":false,"no_trade_grund":"","vix_warnung":false,"direction":"CALL","ticker":"SYMBOL","strike":"WERT","laufzeit":"DATUM","delta":"WERT","iv":"WERT%","iv_to_rv":"WERT","bid":"WERT","ask":"WERT","midpoint":"WERT","conservative_entry":"WERT","entry_price":"WERT","exit_slippage_points":"WERT","fill_probability":"WERT","ev_pct":"WERT","ev_dollars":"WERT","breakeven_move_pct":"WERT","time_stop":"Nach 48h +1% sonst Exit pruefen","kontrakte":"N","einsatz":150,"stop_loss_eur":45,"unusual":false,"begruendung_detail":{"ticker_wahl":"...","option_wahl":"...","timing":"...","chance_risiko":"...","risiko":"..."},"markt":"...","strategie":"...","ausgeschlossen":"TICKER: GRUND","ticker_tabelle":[{"ticker":"USO","direction":"CALL","kurs":"120.89","chg":"+2.11%","ma50":"84.88","trend":"ueber MA50","sector":"XLE","rel_sector":"+0.85","sentpx":"bearish_news_absorbed","relvol":"1.99","bull":"61.3%","score":"86.65","ev_ok":true,"ev_pct":"18.4","gewinner":true,"ausgeschlossen":false,"no_trade_reason":""}]}
"""


# ══════════════════════════════════════════════════════════
# JSON REPAIR
# ══════════════════════════════════════════════════════════

def repair_json_quotes(text: str) -> str:
    result, in_str, escaped, i = [], False, False, 0
    while i < len(text):
        ch = text[i]
        if escaped:
            result.append(ch); escaped = False; i += 1; continue
        if ch == '\\':
            result.append(ch); escaped = True; i += 1; continue
        if ch == '"':
            if not in_str:
                in_str = True; result.append(ch)
            else:
                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                next_ch = text[j] if j < len(text) else ''
                if next_ch in ',}]:\n' or j >= len(text):
                    in_str = False; result.append(ch)
                else:
                    result.append('\\"')
            i += 1; continue
        if in_str and ch in '\n\r':
            result.append(' '); i += 1; continue
        result.append(ch); i += 1
    return ''.join(result)


def close_fragment(frag: str) -> str:
    in_str, i = False, 0
    while i < len(frag):
        if frag[i] == '\\' and in_str and i + 1 < len(frag):
            i += 2; continue
        if frag[i] == '"':
            in_str = not in_str
        i += 1
    if in_str:
        frag += '"'
    last = frag.rfind(",")
    if last > 5:
        frag = frag[:last]
    in_str, i = False, 0
    while i < len(frag):
        if frag[i] == '\\' and in_str and i + 1 < len(frag):
            i += 2; continue
        if frag[i] == '"':
            in_str = not in_str
        i += 1
    if in_str:
        frag += '"'
    frag += "]" * max(0, frag.count("[") - frag.count("]"))
    frag += "}" * max(0, frag.count("{") - frag.count("}"))
    return frag


def extract_json_fragment(text: str) -> str:
    start = text.find("{")
    if start == -1:
        raise ValueError("Kein öffnendes { im Claude-Response")
    end = text.rfind("}")
    if end == -1:
        logger.debug("Kein schließendes } — close_fragment wird angewendet")
        return text[start:]
    return text[start:end + 1]


# ══════════════════════════════════════════════════════════
# SUMMARY KOMPRIMIERUNG
# ══════════════════════════════════════════════════════════

def _compress_summary(summary: str) -> str:
    lines = summary.splitlines()
    result = []
    for line in lines:
        if line.startswith("EARNINGS NAECHSTE"):
            parts = line.split(": ", 1)
            if len(parts) == 2:
                tickers = [t.strip() for t in parts[1].split(",")][:10]
                line = parts[0] + ": " + ", ".join(tickers) + (" ..." if len(tickers) == 10 else "")
        result.append(line)
        if "SENTIMENT-FALLBACK" in line:
            break
    return "\n".join(result)[:4000]


# ══════════════════════════════════════════════════════════
# CLAUDE CALL
# ══════════════════════════════════════════════════════════

def _apply_mandate(result: dict, mandated_ticker: str | None,
                   mandated_direction: str | None) -> dict:
    """Erzwingt die deterministisch getroffene Auswahl im Claude-Result.

    Claude formuliert nur — die Auswahl ist gesetzt. Ohne Mandat unverändert. Selektions-
    basiertes no_trade wird aufgehoben; harte VIX-/Budget-Stops setzt apply_vix_rules danach.
    """
    if not mandated_ticker:
        return result
    result["ticker"] = mandated_ticker
    if mandated_direction:
        result["direction"] = mandated_direction
    result["no_trade"] = False
    result["no_trade_grund"] = ""
    return result


def call_claude(summary: str, api_key: str, vix_direct=None,
                mandated_ticker: str | None = None, mandated_direction: str | None = None) -> dict:
    summary = _compress_summary(summary)

    # Deterministische Auswahl: der Trade ist bereits vom System bestimmt. Claude FORMULIERT
    # nur, es ENTSCHEIDET nicht. Der Mandats-Hinweis hält den Report-Text kohärent zum
    # gewählten Ticker; die maßgebliche Erzwingung erfolgt unten am geparsten Result.
    user_content = "Marktdaten:\n" + summary
    if mandated_ticker:
        user_content += (
            f"\n\nENTSCHEIDUNG STEHT FEST (deterministisch vom System gewählt, NICHT ändern):\n"
            f"ticker = {mandated_ticker}, direction = {mandated_direction or 'aus Marktdaten'}.\n"
            f"Wähle KEINEN anderen Ticker. Schreibe Strike/Laufzeit/Greeks/EV exakt aus den "
            f"Marktdaten dieses Tickers und liefere nur die Begründung. Setze gewinner=true für "
            f"diesen Ticker in der ticker_tabelle und no_trade=false (Ausnahme: harte VIX-/"
            f"Datenprobleme darfst du als no_trade markieren)."
        )

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key":         api_key,
                "anthropic-version": "2023-06-01",
                "content-type":      "application/json",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 1500,
                "system":     PROMPT,
                "messages":   [{"role": "user", "content": user_content}],
            },
            timeout=30,
        )
        r.raise_for_status()
    except (RequestException, Timeout) as e:
        raise RuntimeError("Claude API nicht erreichbar: " + str(e)) from e

    data = r.json()
    if "content" not in data or not data["content"]:
        raise ValueError("Leerer Content in Claude-Response")

    text = data["content"][0]["text"].strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        fragment = extract_json_fragment(text)
    except ValueError as e:
        raise ValueError("JSON-Extraktion fehlgeschlagen: " + str(e)) from e

    parsers = [
        ("direkt",           lambda f: json.loads(f)),
        ("quote_repair",     lambda f: json.loads(repair_json_quotes(f))),
        ("close_fragment",   lambda f: json.loads(close_fragment(f))),
        ("beide_kombiniert", lambda f: json.loads(repair_json_quotes(close_fragment(f)))),
    ]
    last_error = None
    result     = None
    for name, parser in parsers:
        try:
            result = parser(fragment)
            if name != "direkt":
                logger.info("JSON repariert mit Methode: %s", name)
            break
        except json.JSONDecodeError as e:
            last_error = e
            logger.debug("Parse-Versuch '%s' fehlgeschlagen: %s", name, e)

    if result is None:
        raise ValueError("JSON Parse Fehler nach 4 Versuchen: " + str(last_error) +
                         " | Raw: " + text[:300])

    validated, errors = validate_report_payload(result)
    if errors:
        logger.error("Report-Pydantic-Schema-Guard: fail-closed: %s", errors[:5])
        result = build_cancelled_report("; ".join(errors[:5]), raw=text)
    else:
        # Deterministische Auswahl erzwingen: Claude darf Ticker/Richtung NICHT ändern.
        # VIX-/Budget-Regeln bleiben über apply_vix_rules (darunter) autoritativ.
        result = _apply_mandate(validated, mandated_ticker, mandated_direction)

    # Autoritativen VIX nutzen — nicht Claude-JSON-Feld
    authoritative_vix = vix_direct if vix_direct is not None else result.get("vix", "n/v")
    result = apply_vix_rules(authoritative_vix, result)
    logger.info("VIX=%s (direkt) Einsatz=%s no_trade=%s",
                authoritative_vix, result.get("einsatz","?"), result.get("no_trade"))

    return result


# ══════════════════════════════════════════════════════════
# HTML BUILDER
# ══════════════════════════════════════════════════════════

def build_html(d: dict, today: str) -> str:
    G = "#34c759"; R = "#ff3b30"; O = "#ff9500"
    GR = "#86868b"; LG = "#c7c7cc"; DK = "#1d1d1f"
    BG = "#f5f5f7"; WH = "#ffffff"; BD = "#e5e5ea"
    no_trade = d.get("no_trade", False)

    def card(icon, bg, title, content):
        return (f'<div style="background:{WH};border-radius:18px;padding:28px;'
                f'margin-bottom:16px;box-shadow:0 2px 12px rgba(0,0,0,0.07);">'
                f'<div style="display:flex;align-items:center;margin-bottom:20px;">'
                f'<div style="width:36px;height:36px;background:{bg};border-radius:10px;'
                f'text-align:center;line-height:36px;margin-right:12px;font-size:18px;">{icon}</div>'
                f'<h2 style="margin:0;font-size:18px;font-weight:700;color:{DK};">{title}</h2>'
                f'</div>{content}</div>')

    def row(label, val, col=None, last=False):
        c = col or DK
        b = "" if last else f"border-bottom:1px solid {BD};"
        return (f'<div style="display:flex;justify-content:space-between;padding:10px 0;{b}">'
                f'<span style="font-size:14px;color:{GR};">{label}</span>'
                f'<span style="font-size:14px;font-weight:600;color:{c};">{val}</span></div>')

    def section(label, html, border=True):
        b = f"border-bottom:1px solid {BD};" if border else ""
        return (f'<div style="padding:14px 0;{b}">'
                f'<p style="margin:0 0 6px 0;font-size:11px;font-weight:600;color:{GR};'
                f'text-transform:uppercase;letter-spacing:0.06em;">{label}</p>'
                f'<p style="margin:0;font-size:13px;color:{DK};line-height:1.6;">{html}</p></div>')

    # ── Trade Card ────────────────────────────────────────
    if no_trade:
        trade_card = card("❌", "#ffeaea", f'<span style="color:{R};">No Trade</span>',
                          f'<p style="margin:0 0 16px 0;font-size:14px;color:{DK};">'
                          f'{d.get("no_trade_grund","")}</p>'
                          f'<div style="background:{BG};border-radius:12px;padding:16px;">'
                          f'<p style="margin:0;font-size:13px;color:{DK};line-height:1.6;">'
                          f'Kein Trade heute — Kapitalschutz bei erhöhter Volatilität. '
                          f'Morgen läuft die Analyse erneut.</p></div>')
    else:
        einsatz   = d.get("einsatz", 150)
        stop_loss = d.get("stop_loss_eur", round(einsatz * RULES.exit_stop_loss_pct))
        sl_txt    = f"{RULES.exit_stop_loss_pct * 100:.0f}%"
        tp_txt    = f"{RULES.exit_take_profit_pct * 100:.0f}%"
        trail_txt = f"{RULES.exit_trailing_after_tp_pct * 100:.0f}%"

        # ML Win-Probability (nur wenn vorhanden)
        ml_row = ""
        ml_prob = d.get("ml_win_prob")
        if ml_prob is not None:
            try:
                mlp = float(ml_prob) * 100.0
                ml_col = G if mlp >= 55 else (O if mlp >= 50 else R)
                ml_row = row("ML Win-Prob", f"{mlp:.0f}%", ml_col)
            except (ValueError, TypeError):
                ml_row = ""

        # Richtung korrekt aus Daten lesen
        direction     = d.get("direction", "CALL")
        direction_str = "Long Call" if direction != "PUT" else "Long Put"
        direction_col = G if direction != "PUT" else O
        trade_icon    = "✅" if direction != "PUT" else "🔽"
        card_bg       = "#e8f5e9" if direction != "PUT" else "#fff3e0"

        trade_rows = (
            row("Richtung",            direction_str, direction_col) +
            row("Strike",              d.get("strike","n/v")) +
            row("Laufzeit",            d.get("laufzeit","n/v")) +
            row("Delta",               d.get("delta","n/v")) +
            row("IV",                  d.get("iv","n/v")) +
            row("Bid / Ask",           str(d.get("bid","n/v")) + " / " + str(d.get("ask","n/v"))) +
            row("Midpoint",             d.get("midpoint","n/v")) +
            row("Einstieg konservativ", d.get("entry_price", d.get("conservative_entry","n/v"))) +
            row("Fill-Wahrscheinlichkeit", d.get("fill_probability","n/v")) +
            row("Options-EV",          str(d.get("ev_pct","n/v")) + "% / " + str(d.get("ev_dollars","n/v")) + "$") +
            row("Break-even Move",     str(d.get("breakeven_move_pct","n/v")) + "%") +
            ml_row +
            row("Time-Stop",           d.get("time_stop", d.get("time_stop_rule", "48h: +1% Zielrichtung sonst Exit prüfen"))) +
            row("Kontrakte",           str(d.get("kontrakte","n/v"))) +
            row("Einsatz",             str(einsatz) + "€") +
            row("Stop-Loss",           f"–{sl_txt} = max. " + str(stop_loss) + "€", R) +
            row("Take-Profit 1",       f"+{tp_txt} → 50% verkaufen", G) +
            row("Take-Profit 2",       f"Rest mit –{trail_txt} Stop", G) +
            row("Unusual Activity",    "JA 🔥" if d.get("unusual") else "nein",
                O if d.get("unusual") else DK, last=True)
        )
        bd    = d.get("begruendung_detail", {})
        items = [
            ("🏆", "Ticker",        bd.get("ticker_wahl","n/v")),
            ("📐", "Option",        bd.get("option_wahl","n/v")),
            ("⏱",  "Timing",        bd.get("timing","n/v")),
            ("⚖️", "Chance/Risiko", bd.get("chance_risiko","n/v")),
            ("⚠️", "Hauptrisiko",   bd.get("risiko","n/v")),
        ]
        begr = ""
        for i, (icon, label, text) in enumerate(items):
            b = f"border-bottom:1px solid {BD};" if i < len(items) - 1 else ""
            begr += (f'<div style="display:flex;gap:10px;padding:10px 0;{b}">'
                     f'<span style="font-size:16px;min-width:24px;">{icon}</span>'
                     f'<div><p style="margin:0 0 2px 0;font-size:10px;font-weight:700;'
                     f'color:{GR};text-transform:uppercase;">{label}</p>'
                     f'<p style="margin:0;font-size:12px;color:{DK};line-height:1.5;">{text}</p>'
                     f'</div></div>')
        trade_card = card(
            trade_icon, card_bg,
            d.get("ticker","") +
            f' <span style="font-size:14px;color:{direction_col};">{direction_str}</span>',
            trade_rows +
            f'<div style="margin-top:20px;background:{BG};border-radius:14px;'
            f'padding:8px 16px 4px 16px;">'
            f'<p style="margin:10px 0 4px 0;font-size:10px;font-weight:700;color:{GR};'
            f'text-transform:uppercase;">Begründung</p>{begr}</div>',
        )

    # ── VIX Warnung ───────────────────────────────────────
    vix_warning = ""
    if d.get("vix_warnung") and not no_trade:
        vix_warning = (f'<div style="background:#fff9e6;border-left:4px solid {O};'
                       f'border-radius:12px;padding:14px 18px;margin-bottom:16px;">'
                       f'<span style="font-size:18px;">⚠️</span>'
                       f'<span style="font-size:13px;font-weight:600;color:{DK};margin-left:8px;">'
                       f'Erhöhte Volatilität (VIX 20–24) – Einsatz auf '
                       f'<strong>{d.get("einsatz",150)}€</strong> reduziert</span></div>')

    # ── Exit Plan mit konkreten USD-Preisen ───────────────
    exit_card = ""
    if not no_trade:
        stop_pct  = RULES.exit_stop_loss_pct
        tp1_pct   = RULES.exit_take_profit_pct
        trail_pct = RULES.exit_trailing_after_tp_pct
        sl_txt    = f"{stop_pct * 100:.0f}%"
        tp_txt    = f"{tp1_pct * 100:.0f}%"
        trail_txt = f"{trail_pct * 100:.0f}%"
        stop_e    = round(d.get("einsatz", 150) * stop_pct)

        try:
            mid_f = float(str(d.get("midpoint", "0")).replace(",", "."))
        except (ValueError, TypeError):
            mid_f = 0.0

        try:
            kontr = int(str(d.get("kontrakte", "1")).replace("n/v", "1"))
        except (ValueError, TypeError):
            kontr = 1

        if mid_f > 0:
            stop_usd   = round(mid_f * (1 - stop_pct), 2)
            tp1_usd    = round(mid_f * (1 + tp1_pct), 2)
            tp2_usd    = round(tp1_usd * (1 - trail_pct), 2)
            cost_total = round(mid_f * 100 * kontr, 2)
            cost_str   = f"Einstieg: {mid_f:.2f} USD × {kontr} Kontrakt(e) = {cost_total:.2f} USD"
            stop_str   = f"–{sl_txt} → {stop_usd:.2f} USD (max. {stop_e}€ Verlust)"
            tp1_str    = f"+{tp_txt} → {tp1_usd:.2f} USD | 50% schließen"
            tp2_str    = f"Rest mit –{trail_txt} Stop → {tp2_usd:.2f} USD"
        else:
            cost_str = "Einstieg: n/v"
            stop_str = f"–{sl_txt} = max. {stop_e}€"
            tp1_str  = f"+{tp_txt} → 50% schließen"
            tp2_str  = f"Rest mit –{trail_txt} Stop"

        exit_card = card("🎯", "#fff3e0", "Exit-Plan",
                         row("Gesamtkosten",   cost_str) +
                         row("Stop-Loss",       stop_str, R) +
                         row("Take-Profit 1",   tp1_str, G) +
                         row("Take-Profit 2",   tp2_str, G) +
                         row("Zeit-Exit",       d.get("time_stop", d.get("time_stop_rule", "48h ohne +1% Zielbewegung → Exit prüfen"))) +
                         row("Delta Rebalance", "Delta > ±0.30 → prüfen") +
                         row("Vega Exit",       "IV +20% → 50% schließen", last=True))

    # ── Marktstatus ───────────────────────────────────────
    rc    = {"gruen": G, "gelb": O, "rot": R}.get(d.get("regime_farbe","gelb"), O)
    ampel = (f'<span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
             f'background:{rc};margin-right:7px;vertical-align:middle;"></span>')
    try:
        vix_f = float(str(d.get("vix","15")).replace(",","."))
    except (ValueError, TypeError):
        vix_f = 15.0
    vix_pct   = min(100, int((vix_f / 40) * 100))
    vix_color = G if vix_f < 18 else (O if vix_f < 25 else R)

    markt_card = card("🔍", "#e8f0fe", "Marktstatus",
                      f'<div style="display:flex;justify-content:space-between;'
                      f'padding:12px 0;border-bottom:1px solid {BD};">'
                      f'<span style="font-size:14px;color:{GR};">Regime</span>'
                      f'<span style="font-size:15px;font-weight:700;color:{rc};">'
                      f'{ampel}{d.get("regime","n/v")}</span></div>'
                      f'<div style="padding:12px 0;border-bottom:1px solid {BD};">'
                      f'<div style="display:flex;justify-content:space-between;margin-bottom:6px;">'
                      f'<span style="font-size:14px;color:{GR};">VIX</span>'
                      f'<span style="font-size:16px;font-weight:700;color:{vix_color};">'
                      f'{d.get("vix","n/v")}</span></div>'
                      f'<div style="height:5px;background:#e5e5ea;border-radius:3px;">'
                      f'<div style="height:5px;width:{vix_pct}%;background:{vix_color};'
                      f'border-radius:3px;"></div></div></div>' +
                      section("Marktlage", d.get("markt","")) +
                      section("Strategie", d.get("strategie","")) +
                      row("Ausgeschlossen", d.get("ausgeschlossen","–"), last=True))

    # ── Ticker Tabelle ────────────────────────────────────
    def th(label, align="right"):
        return (f'<th style="padding:8px 6px;text-align:{align};font-size:11px;'
                f'font-weight:600;color:{GR};text-transform:uppercase;'
                f'border-bottom:2px solid {BD};">{label}</th>')

    def td(val, align="right", color=DK, bold=False):
        fw = "700" if bold else "500"
        return (f'<td style="padding:10px 6px;text-align:{align};font-size:12px;'
                f'font-weight:{fw};color:{color};border-bottom:1px solid {BD};">{val}</td>')

    rows_html = ""
    for t in d.get("ticker_tabelle", []):
        if t.get("ticker","") in ("X","","SYMBOL"):
            continue
        chg       = t.get("chg","")
        chg_col   = G if "+" in str(chg) else (R if "-" in str(chg) else DK)
        row_color = LG if t.get("ausgeschlossen") else DK
        bold      = bool(t.get("gewinner"))
        rows_html += (f'<tr {"style=background:#f0fff4;" if bold else ""}>' +
                      td(("★ " if bold else "") + t.get("ticker",""), "left",
                         G if bold else row_color, bold) +
                      td(t.get("kurs",""),   "right", row_color, bold) +
                      td(chg,               "right", chg_col,   bold) +
                      td(t.get("ma50",""),  "right", row_color) +
                      td(t.get("trend",""), "center",row_color) +
                      td(t.get("relvol",""),"right", O if t.get("unusual") else row_color) +
                      td(t.get("bull",""),  "right", row_color) +
                      td(t.get("score",""), "right", row_color, bold) + "</tr>")

    if not rows_html:
        rows_html = (f'<tr><td colspan="8" style="padding:16px;text-align:center;'
                     f'font-size:12px;color:{GR};">Keine Daten</td></tr>')

    tabelle_card = card("📋", "#f0f0f5", "Alle analysierten Titel",
                        f'<table style="width:100%;border-collapse:collapse;"><thead><tr>'
                        f'{th("Ticker","left")}{th("Kurs")}{th("Δ%")}{th("MA50")}'
                        f'{th("Trend","center")}{th("RelVol")}{th("Bull%")}{th("Score")}'
                        f'</tr></thead><tbody>{rows_html}</tbody></table>')

    # Status-Zeile zeigt korrekte Richtung
    if no_trade:
        status     = "NO TRADE"
        status_col = R
    else:
        direction  = d.get("direction", "CALL")
        status     = ("CALL · " if direction != "PUT" else "PUT · ") + d.get("ticker","")
        status_col = G if direction != "PUT" else O

    return (f'<html><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>'
            f'<body style="margin:0;padding:0;background:{BG};'
            f"font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;\">"
            f'<div style="max-width:620px;margin:0 auto;padding:32px 16px;">'
            f'<div style="text-align:center;margin-bottom:28px;">'
            f'<p style="margin:0 0 6px 0;font-size:12px;font-weight:600;color:{GR};'
            f'letter-spacing:0.08em;text-transform:uppercase;">Daily Options Report</p>'
            f'<h1 style="margin:0 0 8px 0;font-size:30px;font-weight:700;color:{DK};">'
            f'Daily Options Report</h1>'
            f'<div style="display:inline-block;background:{WH};border-radius:20px;'
            f'padding:6px 18px;box-shadow:0 1px 6px rgba(0,0,0,0.08);">'
            f'<span style="font-size:14px;color:{GR};">'
            f'{d.get("datum",today)} &nbsp;|&nbsp; '
            f'VIX <strong>{d.get("vix","n/v")}</strong> &nbsp;|&nbsp; '
            f'<strong style="color:{status_col};">{status}</strong>'
            f'</span></div></div>'
            f'{trade_card}{vix_warning}{exit_card}{markt_card}{tabelle_card}'
            f'<div style="text-align:center;padding:20px 0;'
            f'border-top:1px solid {BD};margin-top:8px;">'
            f'<p style="margin:0;font-size:12px;color:{GR};">VIX ✓ · Earnings ✓ · Greeks ✓</p>'
            f'</div></div></body></html>')


# ══════════════════════════════════════════════════════════
# EMAIL
# ══════════════════════════════════════════════════════════

def send_email(subject: str, html_content: str, cfg: dict) -> bool:
    recipient = cfg.get("gmail_recipient","")
    sender    = cfg.get("smtp_sender","")
    password  = cfg.get("smtp_password","")
    host      = cfg.get("smtp_host","smtp.gmail.com")
    port      = int(cfg.get("smtp_port", 587))

    if not all([recipient, sender, password]):
        logger.warning("SMTP nicht vollständig konfiguriert — Email nicht verschickt")
        return False

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = sender
    msg["To"]      = recipient
    msg.attach(MIMEText(html_content, "html", "utf-8"))

    try:
        with smtplib.SMTP(host, port, timeout=15) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.sendmail(sender, recipient, msg.as_string())
        logger.info("Email verschickt an %s", recipient)
        return True
    except smtplib.SMTPException as e:
        logger.error("SMTP-Fehler: %s", e)
        return False
    except OSError as e:
        logger.error("Netzwerk-Fehler beim Email-Versand: %s", e)
        return False


# ══════════════════════════════════════════════════════════
# MONATLICHER WIN-RATE REPORT
# ══════════════════════════════════════════════════════════

def _fmt_rate(block: dict) -> str:
    """'62% (n=21, CI 41–79%)' — oder '—' bei leerer Stichprobe."""
    if not block or not block.get("n"):
        return "—"
    wr = block.get("win_rate")
    if wr is None:
        return f"n={block['n']}"
    s = f"{wr * 100:.0f}% (n={block['n']}"
    lo, hi = block.get("ci_low"), block.get("ci_high")
    if lo is not None and hi is not None:
        s += f", CI {lo * 100:.0f}–{hi * 100:.0f}%"
    return s + ")"


def build_monthly_winrate_html(stats: dict) -> str:
    """Rendert den monatlichen Win-Rate-Report. Erwartet das stats-Dict aus
    monthly_winrate_report.compute_stats(). Zeigt Stichprobengrößen + Konfidenz-
    intervalle und ist ehrlich bei zu wenig Daten (keine Schein-Signifikanz)."""
    G = "#34c759"; R = "#ff3b30"; O = "#ff9500"
    GR = "#86868b"; DK = "#1d1d1f"; BG = "#f5f5f7"; WH = "#ffffff"; BD = "#e5e5ea"

    def card(icon, bg, title, content):
        return (f'<div style="background:{WH};border-radius:18px;padding:24px;margin-bottom:16px;'
                f'box-shadow:0 2px 12px rgba(0,0,0,0.07);">'
                f'<div style="display:flex;align-items:center;margin-bottom:16px;">'
                f'<div style="width:34px;height:34px;background:{bg};border-radius:10px;'
                f'text-align:center;line-height:34px;margin-right:12px;font-size:17px;">{icon}</div>'
                f'<h2 style="margin:0;font-size:17px;font-weight:700;color:{DK};">{title}</h2>'
                f'</div>{content}</div>')

    def row(label, val, col=None, last=False):
        b = "" if last else f"border-bottom:1px solid {BD};"
        return (f'<div style="display:flex;justify-content:space-between;padding:9px 0;{b}">'
                f'<span style="font-size:13px;color:{GR};">{label}</span>'
                f'<span style="font-size:13px;font-weight:600;color:{col or DK};">{val}</span></div>')

    def table(title, rows):
        if not rows:
            body = f'<tr><td colspan="2" style="padding:12px;color:{GR};font-size:12px;">Keine Daten</td></tr>'
        else:
            body = ""
            for r in rows:
                wr = r.get("win_rate")
                if wr is None:
                    wcol = DK
                else:
                    wcol = G if wr >= 0.55 else (O if wr >= 0.45 else R)
                val = r.get("display") or _fmt_rate(r)
                body += (f'<tr><td style="padding:8px 6px;font-size:12px;color:{DK};'
                         f'border-bottom:1px solid {BD};">{r.get("label","")}</td>'
                         f'<td style="padding:8px 6px;font-size:12px;text-align:right;color:{wcol};'
                         f'border-bottom:1px solid {BD};">{val}</td></tr>')
        return (f'<p style="margin:14px 0 6px 0;font-size:11px;font-weight:700;color:{GR};'
                f'text-transform:uppercase;letter-spacing:0.05em;">{title}</p>'
                f'<table style="width:100%;border-collapse:collapse;"><tbody>{body}</tbody></table>')

    month = stats.get("month_label", "")
    overall = stats.get("overall", {})
    ml = stats.get("ml", {})

    # Hinweis bei zu wenig Daten.
    notice = ""
    if stats.get("insufficient"):
        notice = (f'<div style="background:#fff9e6;border-left:4px solid {O};border-radius:12px;'
                  f'padding:14px 18px;margin-bottom:16px;font-size:13px;color:{DK};">⚠️ '
                  f'Noch zu wenig abgeschlossene Trades (n={overall.get("n",0)} < '
                  f'{stats.get("min_sample",30)}) für belastbare Aussagen. Die Zahlen sind rein '
                  f'deskriptiv — Konfidenzintervalle sind entsprechend breit.</div>')

    # Overall.
    rec_block = overall.get("recommended", {})
    overall_card = card("📊", "#e8f0fe", "Gesamt-Win-Rate",
                        row("Alle aufgelösten Trades", _fmt_rate(overall),
                            G if (overall.get("win_rate") or 0) >= 0.5 else R) +
                        row("Nur empfohlene (versendet)", _fmt_rate(rec_block)) +
                        row("Ø Return Gewinner", f'{overall.get("avg_win_ret","n/v")}%', G) +
                        row("Ø Return Verlierer", f'{overall.get("avg_loss_ret","n/v")}%', R, last=True))

    # ML-Impact.
    if not ml.get("available"):
        ml_inner = row("Status", "ML nicht verfügbar (sklearn/Modell fehlt)", GR, last=True)
    elif not ml.get("reliable"):
        ml_inner = row("Status", ml.get("note", "Modell noch nicht verlässlich"), O, last=True)
    else:
        imp = ml.get("impact_pp")
        imp_txt = "n/v" if imp is None else f'{"+" if imp >= 0 else ""}{imp:.0f}pp'
        imp_col = G if (imp or 0) > 0 else (R if (imp or 0) < 0 else GR)
        ml_inner = (
            row(f'Gefiltert (≥{int(ml.get("threshold",0.55)*100)}%)', _fmt_rate(ml.get("filtered", {})), G) +
            row("Ungefiltert (alle)", _fmt_rate(ml.get("unfiltered", {}))) +
            row("ML-Impact (Differenz)", imp_txt, imp_col) +
            row("Methodik", ml.get("note", "out-of-sample/forward"), GR, last=True))
    ml_card = card("🤖", "#f3e8ff", "ML-Mehrwert (ehrlich, out-of-sample)", ml_inner)

    # Breakdown-Tabellen.
    breakdown = card("🧭", "#f0f0f5", "Aufschlüsselung",
                     table("Pro Monat (letzte 6)", stats.get("by_month", [])) +
                     table("Pro Horizon", stats.get("by_horizon", [])) +
                     table("Pro Signal-Stärke", stats.get("by_strength", [])) +
                     table("Pro Sektor", stats.get("by_sector", [])) +
                     table("Pro VIX-Regime", stats.get("by_regime", [])) +
                     table("Exit-Gründe (Anteil aufgelöster Trades)", stats.get("exit_reasons", [])))

    # Feature-Importances.
    fi = stats.get("feature_importance", [])
    if fi:
        fi_rows = "".join(
            row(f["feature"], f'{f["importance"]*100:.1f}%') for f in fi[:10]
        )
        interp = stats.get("feature_interpretation", "")
        fi_card = card("🔬", "#e8f5e9", "Feature-Importances (Modell)",
                       fi_rows + (f'<p style="margin:12px 0 0 0;font-size:12px;color:{GR};'
                                  f'line-height:1.5;">{interp}</p>' if interp else ""))
    else:
        fi_card = card("🔬", "#e8f5e9", "Feature-Importances (Modell)",
                       row("Status", "Noch kein trainiertes Modell", GR, last=True))

    return (f'<html><head><meta charset="UTF-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1.0"></head>'
            f'<body style="margin:0;padding:0;background:{BG};'
            f"font-family:-apple-system,BlinkMacSystemFont,'Helvetica Neue',Arial,sans-serif;\">"
            f'<div style="max-width:620px;margin:0 auto;padding:32px 16px;">'
            f'<div style="text-align:center;margin-bottom:24px;">'
            f'<p style="margin:0 0 6px 0;font-size:12px;font-weight:600;color:{GR};'
            f'letter-spacing:0.08em;text-transform:uppercase;">Daily Options Report</p>'
            f'<h1 style="margin:0 0 4px 0;font-size:28px;font-weight:700;color:{DK};">'
            f'Monthly Win-Rate Report</h1>'
            f'<p style="margin:0;font-size:15px;color:{GR};">{month}</p></div>'
            f'{notice}{overall_card}{ml_card}{breakdown}{fi_card}'
            f'<div style="text-align:center;padding:18px 0;border-top:1px solid {BD};margin-top:8px;">'
            f'<p style="margin:0;font-size:11px;color:{GR};">Label: echte Exit-Regeln (TP/SL/Time-Stop) '
            f'auf Optionsebene · forward-only</p></div></div></body></html>')


def send_monthly_winrate_email(cfg: dict, stats: dict | None = None, dry_run: bool = False) -> bool:
    """Baut den Monatsreport und verschickt ihn (oder speichert ihn im Dry-run).

    Wird ohne stats lazy aus monthly_winrate_report berechnet (kein Import-Zyklus,
    da der Import erst zur Laufzeit erfolgt)."""
    if stats is None:
        from monthly_winrate_report import compute_stats
        stats = compute_stats(cfg)

    html = build_monthly_winrate_html(stats)
    ml = stats.get("ml", {})
    impact = ml.get("impact_pp")
    if ml.get("reliable") and impact is not None:
        suffix = f' | ML Impact: {"+" if impact >= 0 else ""}{impact:.0f}pp'
    elif not ml.get("available"):
        suffix = " | ML: n/v"
    else:
        suffix = " | ML: noch zu wenig Daten"
    subject = f"📈 Daily Options Report – Monthly Win-Rate {stats.get('month_label','')}{suffix}"

    if dry_run:
        with open("monthly_winrate_preview.html", "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("Dry-run: monthly_winrate_preview.html gespeichert (%s)", subject)
        return True
    return send_email(subject, html, cfg)


# ══════════════════════════════════════════════════════════
# DIREKTE AUSFÜHRUNG
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from config_loader import load_config, validate_config

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Report Generator")
    parser.add_argument("--summary",      help="Market Summary Text")
    parser.add_argument("--summary-file", help="Datei mit Market Summary")
    parser.add_argument("--output",       help="HTML-Report speichern")
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    if not validate_config(cfg):
        raise SystemExit("Konfiguration unvollständig")

    if args.summary:
        market_summary = args.summary
    elif args.summary_file:
        with open(args.summary_file) as f:
            market_summary = f.read().strip()
    else:
        market_summary = sys.stdin.read().strip()

    if not market_summary:
        raise SystemExit("Kein Market Summary angegeben")

    today   = datetime.now().strftime("%d.%m.%Y")
    subject = "Daily Options Report – " + today

    data        = call_claude(market_summary, cfg.get("anthropic_api_key",""))
    html_report = build_html(data, today)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(html_report)
        logger.info("Report gespeichert: %s", args.output)

    if not args.dry_run:
        send_email(subject, html_report, cfg)
    else:
        with open("report_preview.html", "w", encoding="utf-8") as f:
            f.write(html_report)
        logger.info("Dry-run: report_preview.html gespeichert")
