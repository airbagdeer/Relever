"""Unit tests: status/fetch/reset tolerate a vanished (404) or transiently-erroring batch id.

A single unresolvable ``batches.retrieve()`` must not abort the whole command — a 404 batch (left
behind by an interrupted ``submit`` or expired/cancelled by OpenAI) is marked ``not_found`` and can
be cleared + resubmitted; a transient API error is logged and skipped. No real OpenAI calls: the
client is a fake and module path globals are monkeypatched onto a tmp dir.

Run: uv run pytest src/create_dataset/classifying_data/batch_api/test/test_status_resilience.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIError, NotFoundError

for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents):
    if (_p / "pyproject.toml").is_file():
        sys.path.insert(0, str(_p))
        break

from src.create_dataset.classifying_data.batch_api import label_frontier_batch as lfb  # noqa: E402


# --------------------------------------------------------------------------- #
# fakes / helpers
# --------------------------------------------------------------------------- #
def _notfound() -> NotFoundError:
    resp = httpx.Response(404, request=httpx.Request("GET", "http://t"))
    return NotFoundError("no batch found", response=resp, body=None)


def _apierror() -> APIError:
    return APIError("connection reset", request=httpx.Request("GET", "http://t"), body=None)


def _fake_batch(status: str, *, output_file_id: str | None = None, total: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        status=status, output_file_id=output_file_id, error_file_id=None,
        request_counts=SimpleNamespace(total=total, completed=total, failed=0))


class _FakeClient:
    """Maps batch_id -> a fake Batch, or an Exception instance to raise on retrieve."""

    def __init__(self, mapping: dict):
        outer = self

        class _Batches:
            def retrieve(self, bid):
                v = outer.mapping[bid]
                if isinstance(v, Exception):
                    raise v
                return v
        self.mapping = mapping
        self.batches = _Batches()


def _entry(batch_id, status, **kw):
    e = {"event_id": 0, "batch_id": batch_id, "status": status, "fetched": False,
         "output_file_id": None, "n_requests": 5, "n_candidates": 5}
    e.update(kw)
    return e


@pytest.fixture()
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(lfb, "OUT_ROOT", out)
    monkeypatch.setattr(lfb, "MANIFEST_PATH", out / "_manifest.json")
    return out


def _write_manifest(queries: dict) -> None:
    lfb.MANIFEST_PATH.write_text(json.dumps(
        {"config": {"model": "gpt-5.2"}, "updated_at": None, "queries": queries}))


def _load_manifest() -> dict:
    return json.loads(lfb.MANIFEST_PATH.read_text())["queries"]


# --------------------------------------------------------------------------- #
# _safe_retrieve
# --------------------------------------------------------------------------- #
def test_safe_retrieve_success_notfound_and_error():
    good = _fake_batch("completed", output_file_id="f")
    client = _FakeClient({"good": good, "gone": _notfound(), "boom": _apierror()})

    b, err = lfb._safe_retrieve(client, "good")
    assert err is None and b is good
    b, err = lfb._safe_retrieve(client, "gone")
    assert b is None and err == "not_found"
    b, err = lfb._safe_retrieve(client, "boom")
    assert b is None and err.startswith("error:")


# --------------------------------------------------------------------------- #
# cmd_status
# --------------------------------------------------------------------------- #
def test_cmd_status_marks_404_and_keeps_going(paths, monkeypatch, capsys):
    _write_manifest({
        "0000": _entry("good", "in_progress"),
        "0001": _entry("gone", "in_progress"),   # 404 in the middle must not abort the loop
        "0002": _entry("good2", "validating"),
    })
    client = _FakeClient({
        "good": _fake_batch("completed", output_file_id="f"),
        "gone": _notfound(),
        "good2": _fake_batch("in_progress"),
    })
    monkeypatch.setattr(lfb, "_client", lambda: client)

    lfb.cmd_status(argparse.Namespace(qids=None, all=False))  # must not raise

    q = _load_manifest()
    assert q["0000"]["status"] == "completed"
    assert q["0001"]["status"] == "not_found"     # the vanished batch is flagged, not fatal
    assert q["0002"]["status"] == "in_progress"   # queries after the bad one still refreshed
    assert "not found (404)" in capsys.readouterr().out


def test_cmd_status_survives_transient_error(paths, monkeypatch):
    _write_manifest({"0000": _entry("boom", "in_progress"), "0001": _entry("good", "in_progress")})
    client = _FakeClient({"boom": _apierror(), "good": _fake_batch("completed", output_file_id="f")})
    monkeypatch.setattr(lfb, "_client", lambda: client)

    lfb.cmd_status(argparse.Namespace(qids=None, all=False))  # must not raise

    q = _load_manifest()
    assert q["0000"]["status"] == "in_progress"    # transient error: left as-is
    assert q["0001"]["status"] == "completed"


# --------------------------------------------------------------------------- #
# cmd_reset — a 404 batch is gone (no output, no cost) -> clear for resubmit
# --------------------------------------------------------------------------- #
def test_cmd_reset_clears_404_batch(paths, monkeypatch, capsys):
    _write_manifest({"0003": _entry("gone", "in_progress")})
    monkeypatch.setattr(lfb, "_client", lambda: _FakeClient({"gone": _notfound()}))

    lfb.cmd_reset(argparse.Namespace(qids=["0003"], failed=False))

    e = _load_manifest()["0003"]
    assert e["batch_id"] is None and e["status"] is None and e["fetched"] is False
    assert "cleared for resubmit" in capsys.readouterr().out


def test_cmd_reset_failed_autodetects_not_found(paths, monkeypatch):
    _write_manifest({
        "0003": _entry("gone", "not_found"),          # picked up by --failed
        "0004": _entry("live", "in_progress"),        # not terminal -> left alone
    })
    monkeypatch.setattr(lfb, "_client", lambda: _FakeClient(
        {"gone": _notfound(), "live": _fake_batch("in_progress")}))

    lfb.cmd_reset(argparse.Namespace(qids=None, failed=True))

    q = _load_manifest()
    assert q["0003"]["batch_id"] is None               # cleared
    assert q["0004"]["batch_id"] == "live"             # untouched


# --------------------------------------------------------------------------- #
# cmd_fetch — a 404 batch is skipped, not fatal
# --------------------------------------------------------------------------- #
def test_cmd_fetch_skips_404(paths, monkeypatch, capsys):
    _write_manifest({"0001": _entry("gone", "in_progress")})
    monkeypatch.setattr(lfb, "_client", lambda: _FakeClient({"gone": _notfound()}))

    lfb.cmd_fetch(argparse.Namespace(qids=None, force=False))  # must not raise

    assert _load_manifest()["0001"]["status"] == "not_found"
    assert "not found (404)" in capsys.readouterr().out
