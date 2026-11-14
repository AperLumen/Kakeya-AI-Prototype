"""可观测性 JSONL sink（事件落盘 / 轮转 / 清理 / 日志桥接）的测试。"""

import io
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from kakeya_ai.ai import observability
from kakeya_ai.ai.observability import (
    LLMObservability,
    _JsonlSink,
    _SINK_CACHE,
    configure_logging,
)

@pytest.fixture(autouse=True)
def _isolate_sink(monkeypatch, tmp_path):
    """把 OBS_JSONL_PATH 指向临时目录，并在每个测试前清空 sink 缓存。"""
    monkeypatch.setenv(observability.OBS_JSONL_PATH_ENV, str(tmp_path / "obs"))
    _SINK_CACHE.clear()
    yield
    for sink in _SINK_CACHE.values():
        sink.close()
    _SINK_CACHE.clear()


def _read_events(directory: Path):
    events = []
    for path in sorted(directory.glob("obs-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            events.append(json.loads(line))
    return events


def test_record_decision_writes_jsonl_line(tmp_path):
    obs = LLMObservability()
    obs.record_decision(source_label="chat_cascade", state_key="s1")

    events = _read_events(tmp_path / "obs")
    assert len(events) == 1
    event = events[0]
    assert event["event"] == "decision"
    assert event["source"] == "chat_cascade"
    assert event["state_key"] == "s1"
    # timestamp 必须可被 ISO 解析，以供 obs_report 的 --since 过滤
    datetime.fromisoformat(event["timestamp"])


def test_record_fallback_and_parse_failure_write_lines(tmp_path):
    obs = LLMObservability()
    obs.record_fallback(source_label="chat_deep_error", reason="boom")
    obs.record_parse_failure(stage="deep_eval", response_snippet="not json", reason="json")

    events = _read_events(tmp_path / "obs")
    assert [e["event"] for e in events] == ["fallback", "parse_failure"]
    assert events[0]["source"] == "chat_deep_error"
    assert events[0]["reason"] == "boom"
    assert events[1]["stage"] == "deep_eval"
    assert events[1]["snippet"] == "not json"


def test_record_llm_call_writes_duration_and_tokens(tmp_path):
    obs = LLMObservability()
    obs.record_llm_call(
        model="deepseek-r1:7b",
        prompt_len=120,
        duration_ms=1234.5,
        success=True,
        prompt_tokens=100,
        completion_tokens=50,
    )

    (event,) = _read_events(tmp_path / "obs")
    assert event["event"] == "llm_call"
    assert event["model"] == "deepseek-r1:7b"
    assert event["duration_ms"] == 1234.5
    assert event["success"] is True
    assert event["prompt_tokens"] == 100
    assert event["completion_tokens"] == 50


def test_no_env_means_zero_file_side_effects(monkeypatch, tmp_path):
    monkeypatch.delenv(observability.OBS_JSONL_PATH_ENV, raising=False)
    _SINK_CACHE.clear()
    obs = LLMObservability()
    obs.record_decision("x")
    obs.record_fallback("y")
    obs.record_parse_failure("z", "snippet")
    obs.record_llm_call("m", 1, 1.0, True)
    assert list(tmp_path.rglob("*.jsonl")) == []


def test_sink_rotates_daily_and_prunes_old_files(tmp_path):
    directory = tmp_path / "obs"
    directory.mkdir()
    today = date(2026, 9, 6)
    # 两个文件：一个刚在保留期内，一个已超出。
    inside = directory / "obs-2026-08-24.jsonl"  # 13 天前
    outside = directory / "obs-2026-08-01.jsonl"  # 36 天前
    inside.write_text('{"event": "decision"}\n', encoding="utf-8")
    outside.write_text('{"event": "decision"}\n', encoding="utf-8")

    sink = _JsonlSink(str(directory), today=today)
    sink.write({"event": "decision", "source": "s"})
    sink.close()

    assert (directory / "obs-2026-09-06.jsonl").exists()
    assert inside.exists()
    assert not outside.exists()

    lines = (directory / "obs-2026-09-06.jsonl").read_text(encoding="utf-8")
    assert json.loads(lines.splitlines()[0])["source"] == "s"

    # 注入一个更晚的日期再写一次，会轮转到新文件。
    nxt = _JsonlSink(str(directory), today=today + timedelta(days=1))
    nxt.write({"event": "decision", "source": "s2"})
    nxt.close()
    assert (directory / "obs-2026-09-07.jsonl").exists()


def test_configure_logging_writes_to_file(tmp_path):
    log_file = tmp_path / "ai.log"
    configure_logging(level=logging.INFO, stream=io.StringIO(), log_file=str(log_file))
    try:
        observability.get_logger().info("hello %s", "world")
        for handler in observability.get_logger().handlers:
            handler.flush()
        lines = log_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["message"] == "hello world"
        assert record["level"] == "INFO"
    finally:
        # 恢复测试前的 logger 状态（无 handler，向根 logger 传播）。
        logger = observability.get_logger()
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        logger.addHandler(logging.NullHandler())
        logger.propagate = True
