import datetime as dt
import json
import logging

from semsearch.config import Settings
from semsearch.logging_setup import JsonFormatter
from semsearch.store import build_filter, point_id, to_rfc3339


def test_point_ids_are_stable_and_unique():
    assert point_id("a.pdf", 0) == point_id("a.pdf", 0)
    assert point_id("a.pdf", 0) != point_id("a.pdf", 1)
    assert point_id("a.pdf", 1) != point_id("b.pdf", 1)


def test_build_filter():
    assert build_filter() is None
    flt = build_filter(["a.txt", "b.txt"], dt.date(2026, 1, 1), None)
    assert len(flt.must) == 2
    assert flt.must[0].match.any == ["a.txt", "b.txt"]
    assert flt.must[1].range.lte is None


def test_dates_are_stored_as_utc_midnight():
    assert to_rfc3339(dt.date(2026, 9, 29)) == "2026-09-29T00:00:00Z"


def test_json_log_lines_include_extra_fields():
    record = logging.LogRecord("semsearch", logging.INFO, __file__, 1, "search", None, None)
    record.request_id = "abc"
    record.total_ms = 12.5
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "search" and line["level"] == "INFO"
    assert line["request_id"] == "abc" and line["total_ms"] == 12.5


def test_settings_come_from_environment(monkeypatch):
    monkeypatch.setenv("SEMSEARCH_CHUNK_SIZE", "300")
    monkeypatch.setenv("SEMSEARCH_QDRANT_URL", "http://qdrant:6333")
    settings = Settings(_env_file=None)
    assert settings.chunk_size == 300 and settings.qdrant_url == "http://qdrant:6333"


def test_bge_query_instruction_is_the_default():
    assert Settings(_env_file=None).query_prefix.startswith("Represent this sentence")
