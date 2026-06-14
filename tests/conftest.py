"""
conftest.py — gemeinsame pytest-Fixtures.

Zwei Aufgaben:
1. `src/` global auf den Importpfad legen (ersetzt das per-Datei sys.path-Hacking).
2. Test-Isolation (Fund B): JEDER Test bekommt automatisch eine frische tmp-DB und ein
   tmp-Artefakt-Verzeichnis. Damit kann kein Test je die echte data/-DB oder das
   produktive ML-Modell anfassen — auch nicht versehentlich über einen Modul-Import,
   der den DB-Pfad früher beim Import gebunden hätte.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_data(tmp_path, monkeypatch):
    """Leitet Journal-DB + ML-Artefakte je Test in ein tmp-Verzeichnis um.

    `trading_journal.resolve_db_path()` und `ml_predictor._artifact_dir()` lesen diese
    Env-Variablen zur Aufrufzeit — daher genügt das Setzen hier, ohne Modul-Reload.
    """
    monkeypatch.setenv("JOURNAL_DB_PATH", str(tmp_path / "journal.sqlite"))
    monkeypatch.setenv("ML_DATA_DIR", str(tmp_path))
    yield


@pytest.fixture
def journal_db():
    """Frische, initialisierte Journal-Verbindung in der tmp-DB (Schema bereits angelegt)."""
    import trading_journal as tj
    con = tj.connect()
    try:
        yield con
    finally:
        con.close()
