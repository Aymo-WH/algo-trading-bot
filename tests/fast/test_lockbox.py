"""Lockbox mechanics — pre-registered S4a/S4b in specs/REFEREE-SELFTEST-2026-07-08.md."""
import pytest
from cryptography.fernet import Fernet

from validation.lockbox import build_lockbox, open_lockbox

PLAINTEXT = b"date,SPY\n2022-01-03,100.0\n2022-01-04,101.5\n"


def _paths(tmp_path):
    return dict(lockbox_dir=str(tmp_path / "lb"),
                token_path=str(tmp_path / "tok.txt"),
                access_log=str(tmp_path / "access.log"))


def test_round_trip_and_no_plaintext_at_rest(tmp_path):
    kw = _paths(tmp_path)
    enc_path = build_lockbox(PLAINTEXT, **kw)
    token = (tmp_path / "tok.txt").read_text().strip()
    enc = open(enc_path, "rb").read()
    assert b"SPY" not in enc and PLAINTEXT not in enc      # encrypted at rest
    out = open_lockbox(token, lockbox_dir=kw["lockbox_dir"],
                       access_log=kw["access_log"])
    assert out == PLAINTEXT                                # byte-identical


def test_wrong_token_refused_and_audited(tmp_path):
    kw = _paths(tmp_path)
    build_lockbox(PLAINTEXT, **kw)
    for bad in (Fernet.generate_key().decode(), "garbage-token"):
        with pytest.raises(PermissionError):
            open_lockbox(bad, lockbox_dir=kw["lockbox_dir"],
                         access_log=kw["access_log"])
    log = (tmp_path / "access.log").read_text()
    assert "OPEN REFUSED" in log and "BUILT lockbox" in log


def test_rebuild_refused(tmp_path):
    kw = _paths(tmp_path)
    build_lockbox(PLAINTEXT, **kw)
    with pytest.raises(FileExistsError):
        build_lockbox(PLAINTEXT, **kw)


def test_existing_token_refused(tmp_path):
    kw = _paths(tmp_path)
    (tmp_path / "tok.txt").write_text("stale\n")
    with pytest.raises(FileExistsError):
        build_lockbox(PLAINTEXT, **kw)
