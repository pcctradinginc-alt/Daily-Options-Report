"""
payoff_sim.py — Kalibrier-/Validierungs-Harness für die Payoff-Geometrie (reine Standard-Lib).

KEIN Alpha-Beweis (die Live-Win-Rate braucht n>=30 echte Trades). Dieses Werkzeug zeigt unter
EXPLIZITEN Annahmen, wie sich die Exit-/Spread-Regeln *mechanisch* auf Win-Rate/EV/Profit-Factor
auswirken — damit Schwellen evidenzbasiert statt nach Bauchgefühl kalibriert werden.

Drei Analysen (CLI):
  python tools/payoff_sim.py compare    # OLD- vs NEW-config (Spread-Cap, spread-aware Exits, Time-Stop)
  python tools/payoff_sim.py frontier   # Breakeven: welche Signal-Güte (p, J) macht PF > 1?
  python tools/payoff_sim.py all        # beides (Default)

Modell: Black-Scholes-Fair-Value (r=0) als "Mid"; bid=mid*(1-s/2), ask=mid*(1+s/2). Entry=ask
(= conservative_entry der Live-Logik), Tages-Marks am BID (wie resolve_open_trades). Pro Tag erst
TP/SL (Bid-Return), dann Time-Stop. Underlying GBM (sigma=IV). Katalysator = Sprung (+/-J% über
3 Tage) mit Trefferquote p — weil eine Event-Strategie auf Sprünge setzt, nicht auf konstanten Drift.

Spiegelt die Live-Schwellen aus src/rules.py (spread-aware Exits, Caps) bewusst nach, ohne den
App-Graph zu importieren (torch/sklearn-frei, schnell, überall lauffähig).
"""
from __future__ import annotations

import math
import random
import statistics
import sys

STEP_H = 24
MAX_HOLD_DAYS = 10
JUMP_DAYS = 3


# ── Black-Scholes (r=0) ─────────────────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sigma, is_call):
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, (S - K) if is_call else (K - S))
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if is_call:
        return S * _norm_cdf(d1) - K * _norm_cdf(d2)
    return K * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S, K, T, sigma, is_call):
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if (is_call and S > K) else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1) if is_call else _norm_cdf(d1) - 1.0


def strike_for_delta(S, T, sigma, is_call, target_delta):
    lo, hi = S * 0.5, S * 1.5
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        d = abs(bs_delta(S, mid, T, sigma, is_call))
        if (d > target_delta) == is_call:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def spread_aware_thresholds(entry, mid, half_spread, tp=0.50, sl=0.30):
    """Spiegelt rules.spread_adjusted_exit_thresholds (bounded)."""
    if entry <= 0 or mid <= 0:
        return tp, sl
    tp_bid = ((1.0 + tp) * mid - half_spread) / entry - 1.0
    sl_bid = 1.0 - ((1.0 - sl) * mid - half_spread) / entry
    return max(0.20, min(tp, tp_bid)), max(sl, min(0.60, sl_bid))


# ── Simulation ──────────────────────────────────────────────────────────────
def build_candidates(rng, n=60):
    cands = []
    for _ in range(n):
        iv = rng.uniform(0.35, 0.75)
        dte = rng.choice([10, 14, 21, 30, 45])
        is_call = rng.random() < 0.5
        K = strike_for_delta(100.0, dte / 365.0, iv, is_call, rng.uniform(0.35, 0.55))
        cands.append({"spread_frac": rng.uniform(2.0, 10.0) / 100.0, "iv": iv,
                      "dte": dte, "is_call": is_call, "K": K})
    return cands


def simulate(cfg, cand, mu, rng, n_paths, p_correct=None, jump_pct=0.0):
    """Ein Kandidat, n_paths Pfade. mu=annualisierter Drift; optional Katalysator-Sprung."""
    S0, K, dte, sigma, is_call, s = (100.0, cand["K"], cand["dte"], cand["iv"],
                                     cand["is_call"], cand["spread_frac"])
    T0 = dte / 365.0
    mid0 = bs_price(S0, K, T0, sigma, is_call)
    if mid0 <= 0.02:
        return None
    half0 = mid0 * s / 2.0
    entry = mid0 + half0
    tp, sl = (spread_aware_thresholds(entry, mid0, half0) if cfg["spread_aware"]
              else (0.50, 0.30))
    dt = STEP_H / 24.0 / 365.0
    sig_step = sigma * math.sqrt(dt)
    base_drift = (mu - 0.5 * sigma * sigma) * dt
    n_days = min(dte, MAX_HOLD_DAYS)

    rets = []
    for _ in range(n_paths):
        correct = (rng.random() < p_correct) if p_correct is not None else None
        cat_per_day = 0.0
        if p_correct is not None:
            cat_per_day = (jump_pct if correct else -jump_pct) / JUMP_DAYS / 100.0
        S, ret, exit_ret = S0, 0.0, None
        for day in range(1, n_days + 1):
            cat = (cat_per_day if is_call else -cat_per_day) if day <= JUMP_DAYS else 0.0
            S *= math.exp(base_drift + cat + sig_step * rng.gauss(0, 1))
            T = max(1e-6, (dte - day) / 365.0)
            mid = bs_price(S, K, T, sigma, is_call)
            bid = max(0.0, mid - mid * s / 2.0)
            ret = bid / entry - 1.0
            if ret >= tp or ret <= -sl:
                exit_ret = ret; break
            if day * STEP_H >= cfg["time_stop_h"]:
                moved = ((S - S0) / S0 >= 0.01) if is_call else ((S - S0) / S0 <= -0.01)
                if not moved:
                    exit_ret = ret; break
        rets.append(exit_ret if exit_ret is not None else ret)
    return rets


def aggregate(rets):
    wins = sum(1 for r in rets if r > 0)
    gw = sum(r for r in rets if r > 0)
    gl = -sum(r for r in rets if r < 0)
    return {"win_rate": wins / len(rets), "mean_ret": statistics.mean(rets),
            "profit_factor": (gw / gl) if gl > 0 else float("inf"), "n": len(rets)}


def run_config(cfg, cands, mu, rng, n_paths, p_correct=None, jump_pct=0.0):
    admitted = [c for c in cands if c["spread_frac"] * 100.0 <= cfg["cap_pct"]]
    all_rets = []
    for c in admitted:
        r = simulate(cfg, c, mu, rng, n_paths, p_correct, jump_pct)
        if r:
            all_rets.extend(r)
    if not all_rets:
        return None
    out = aggregate(all_rets)
    out["admitted"] = len(admitted)
    return out


OLD = {"cap_pct": 8.0, "spread_aware": False, "time_stop_h": 24}
NEW = {"cap_pct": 5.0, "spread_aware": True, "time_stop_h": 36}


# ── CLI ──────────────────────────────────────────────────────────────────────
def cmd_compare(seed=12345, n_paths=4000):
    cands = build_candidates(random.Random(seed), 60)
    print("OLD vs NEW config — gleiche Population/Zufallsbasis (kein Katalysator, reiner Drift)\n")
    for mu in (0.0, 0.15, 0.30):
        o = run_config(OLD, cands, mu, random.Random(seed + 1), n_paths)
        n = run_config(NEW, cands, mu, random.Random(seed + 1), n_paths)
        print(f"── mu={mu:.2f}/Jahr {'─'*40}")
        print(f"  Trichter  OLD {o['admitted']:>3} | NEW {n['admitted']:>3}")
        print(f"  Win-Rate  {o['win_rate']*100:5.1f}% -> {n['win_rate']*100:5.1f}%  "
              f"({(n['win_rate']-o['win_rate'])*100:+.1f}pp)")
        print(f"  PF        {o['profit_factor']:.2f} -> {n['profit_factor']:.2f}  "
              f"({n['profit_factor']-o['profit_factor']:+.2f})\n")


def cmd_frontier(seed=12345, n_paths=3000):
    cands = build_candidates(random.Random(seed), 50)
    Js, ps = [3, 5, 8, 12], [0.50, 0.55, 0.60, 0.65]
    print("Breakeven-Frontier (NEW): Profit-Factor je (Trefferquote p, Katalysator-Move J%)")
    print("PF > 1.00 = profitabel -> die messbare Anforderung an die Signal-Güte für Alpha.\n")
    print("   p\\J " + "".join(f"{j:>7}%" for j in Js))
    for p in ps:
        row = [run_config(NEW, cands, 0.0, random.Random(seed + 7), n_paths, p, J)["profit_factor"]
               for J in Js]
        print(f"  {p:.2f} " + "".join(f"{v:>8.2f}" for v in row))
    print()


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd in ("compare", "all"):
        cmd_compare()
    if cmd in ("frontier", "all"):
        cmd_frontier()


if __name__ == "__main__":
    main(sys.argv)
