"""
test_payoff_sim.py — schnelle Invarianten-Checks für das Kalibrier-Harness tools/payoff_sim.py.

Kein Alpha-Test, sondern: die Mechanik verhält sich wie erwartet (BS korrekt, Frontier monoton,
spread-aware Schwellen spiegeln die Live-Regel). Kleine Pfadzahl -> schnell.

Standalone:  python tests/test_payoff_sim.py
oder:        pytest tests/test_payoff_sim.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_bs_matches_known_atm_value():
    import payoff_sim as sim
    # ATM Call, 1J, sigma=0.20, r=0 -> ~ S*(2N(0.1)-1) ≈ 7.97
    px = sim.bs_price(100.0, 100.0, 1.0, 0.20, True)
    assert abs(px - 7.97) < 0.1, px
    # Put-Call-Parität bei r=0: C - P = S - K = 0 für ATM
    put = sim.bs_price(100.0, 100.0, 1.0, 0.20, False)
    assert abs(px - put) < 1e-6


def test_spread_aware_matches_live_rule():
    import payoff_sim as sim
    from rules import RULES
    tp_sim, sl_sim = sim.spread_aware_thresholds(2.08, 2.04, 0.04)
    tp_live, sl_live = RULES.spread_adjusted_exit_thresholds(2.08, 2.04, 0.04)
    # Live-Regel rundet auf 4 Stellen, die Sim nicht -> Toleranz entsprechend.
    assert abs(tp_sim - tp_live) < 1e-3
    assert abs(sl_sim - sl_live) < 1e-3


def test_new_admits_fewer_than_old():
    import payoff_sim as sim
    cands = sim.build_candidates(random.Random(1), 200)
    old = [c for c in cands if c["spread_frac"] * 100 <= sim.OLD["cap_pct"]]
    new = [c for c in cands if c["spread_frac"] * 100 <= sim.NEW["cap_pct"]]
    assert len(new) < len(old)   # 5%-Cap ist strenger als 8%


def test_frontier_monotone_in_signal_quality():
    import payoff_sim as sim
    cands = sim.build_candidates(random.Random(1), 20)
    # Höhere Trefferquote UND größerer Katalysator-Move => höherer Profit-Factor.
    weak = sim.run_config(sim.NEW, cands, 0.0, random.Random(9), 400, p_correct=0.50, jump_pct=3)
    strong = sim.run_config(sim.NEW, cands, 0.0, random.Random(9), 400, p_correct=0.65, jump_pct=12)
    assert strong["profit_factor"] > weak["profit_factor"]
    assert strong["win_rate"] > weak["win_rate"]
    # Ohne Edge bleibt PF unter der starken Signal-Variante klar zurück.
    assert weak["profit_factor"] < strong["profit_factor"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all payoff_sim tests passed")
