"""
Einfaches Interface auf dem bestehenden, robusten TradingJournal.
"""

from trading_journal import (
    create_run,
    log_market_signals,
    log_final_decision,
    update_due_outcomes,
    get_iv_stats
)

class TradingJournal:
    """Einfaches, benutzerfreundliches Interface"""
    
    def __init__(self):
        self.run_id = None

    def start_run(self):
        """Neuen Run starten"""
        self.run_id = create_run()
        return self.run_id

    def log_signals(self, parsed_signals, market_data, clusters=None):
        """Signale + Marktdaten loggen"""
        if self.run_id is None:
            self.start_run()
        log_market_signals(self.run_id, parsed_signals, market_data, clusters)

    def log_decision(self, result: dict):
        """Finale Entscheidung (Trade oder No-Trade) loggen"""
        if self.run_id is None:
            self.start_run()
        log_final_decision(self.run_id, result)

    def update_outcomes(self, cfg):
        """Fällige Outcomes updaten"""
        return update_due_outcomes(cfg)

    def get_iv_stats(self, ticker: str, current_iv: float | None = None):
        """IV-Rank aus eigener Historie"""
        return get_iv_stats(ticker, current_iv)

    def get_run_id(self):
        return self.run_id


# Singleton für einfache Nutzung
journal = TradingJournal()
