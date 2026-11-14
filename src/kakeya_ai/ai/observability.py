"""AI 推理层的轻量可观测性。

为以下内容提供结构化日志与内存指标：

- LLM 调用（模型、prompt 长度、耗时、成功/失败、token 数）
- 解析失败（JSON 分数、候选推理、自评分、深层评估）
- 决策来源与回落触发（cache / heuristic / fast / deep / beam / cascaded / error）

所有指标都汇总在一个单例 ``LLMObservability`` 实例里。本模块除 Python 标准库
外无外部依赖。

可选的持久化 JSONL 落盘：把 ``OBS_JSONL_PATH`` 环境变量设为某个目录，则每个
``record_*`` 事件（外加逐行 flush）都会追加到该目录下的
``obs-<YYYY-MM-DD>.jsonl``，按天轮转并保留 14 天。汇总时用任意 JSONL 工具（jq / pandas）读即可。
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional


DEFAULT_LOG_SNIPPET_LEN = 200

# JSONL 落盘配置（见模块 docstring）。
OBS_JSONL_PATH_ENV = "OBS_JSONL_PATH"
OBS_RETENTION_DAYS = 14
_OBS_FILE_PREFIX = "obs-"
_OBS_FILE_SUFFIX = ".jsonl"


class _JsonlSink:
    """只追加的 JSONL 落盘器，带按天轮转与有界保留期。

    每次 :meth:`write` 追加一行 JSON 并立即 flush，因此已写入的行能扛住进程退出
    / 崩溃。目标文件是 ``directory`` 下的 ``obs-<YYYY-MM-DD>.jsonl``；为新的一天
    打开文件时也会清理早于 ``OBS_RETENTION_DAYS`` 的文件。

    ``today`` 可注入为 ``datetime.date``，以便测试具有确定性。
    """

    def __init__(self, directory: str, today: Optional[date] = None) -> None:
        self._dir = Path(directory)
        self._today = today
        self._file: Optional[Any] = None
        self._file_date: Optional[date] = None
        self._dir.mkdir(parents=True, exist_ok=True)

    def _current_date(self) -> date:
        return self._today if self._today is not None else datetime.now().date()

    def write(self, event: Dict[str, Any]) -> None:
        day = self._current_date()
        if self._file is None or self._file_date != day:
            self._open(day)
        assert self._file is not None
        self._file.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._file.flush()

    def _open(self, day: date) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        path = self._dir / f"{_OBS_FILE_PREFIX}{day.isoformat()}{_OBS_FILE_SUFFIX}"
        self._file = open(path, "a", encoding="utf-8")
        self._file_date = day
        self._prune(day)

    def _prune(self, today: date) -> None:
        cutoff = today - timedelta(days=OBS_RETENTION_DAYS)
        for path in self._dir.glob(f"{_OBS_FILE_PREFIX}*{_OBS_FILE_SUFFIX}"):
            stamp = path.name[len(_OBS_FILE_PREFIX):-len(_OBS_FILE_SUFFIX)]
            try:
                day = date.fromisoformat(stamp)
            except ValueError:
                continue
            if day < cutoff:
                try:
                    path.unlink()
                except OSError:
                    pass

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._file_date = None


# 惰性创建的落盘器，以目录路径为键，这样重复的 record_* 调用不会反复打开文件。
# 仅在设置了 OBS_JSONL_PATH_ENV 时才会填充。
_SINK_CACHE: Dict[str, _JsonlSink] = {}


def _get_sink() -> Optional[_JsonlSink]:
    """返回 OBS_JSONL_PATH 对应的 JSONL 落盘器；未设置时返回 None。"""
    from ..config import raw as _env_raw

    directory = _env_raw(OBS_JSONL_PATH_ENV).strip()
    if not directory:
        return None
    sink = _SINK_CACHE.get(directory)
    if sink is None:
        try:
            sink = _JsonlSink(directory)
        except OSError:
            get_logger().warning(
                "Could not open observability JSONL dir %s", directory, exc_info=True,
            )
            return None
        _SINK_CACHE[directory] = sink
    return sink


def _write_sink_event(event: str, fields: Dict[str, Any]) -> None:
    """向 JSONL 落盘器追加一个事件；绝不抛异常。"""
    sink = _get_sink()
    if sink is None:
        return
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
    }
    record.update(fields)
    try:
        sink.write(record)
    except Exception:
        get_logger().warning("Observability JSONL sink write failed", exc_info=True)


@dataclass
class LLMObservability:
    """LLM 可观测性指标的内存累加器。

    这里刻意用普通 dataclass，方便调用方创建隔离实例用于测试。为方便起见，
    模块级默认实例通过 ``default_observability()`` 暴露。
    """

    # 计数器
    llm_calls_total: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    llm_call_duration_ms: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(float))
    )
    parse_failures_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fallback_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    decisions_total: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # 最近的失败快照（有界，供调试用）
    recent_parse_failures: list[Dict[str, Any]] = field(default_factory=list)
    max_recent_failures: int = 20

    def record_llm_call(
        self,
        model: str,
        prompt_len: int,
        duration_ms: float,
        success: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """记录一次 LLM API 调用。

        Parameters
        ----------
        model : str
            模型标签，例如 "qwen3.5:4b"。
        prompt_len : int
            prompt 的字符长度（作为规模的近似）。
        duration_ms : float
            调用的墙上时钟耗时（毫秒）。
        success : bool
            该次调用是否返回了可用响应。
        prompt_tokens : int, optional
            已知时的 prompt token 数。
        completion_tokens : int, optional
            已知时的 completion token 数。
        """
        status = "success" if success else "failure"
        self.llm_calls_total[model][status] += 1
        self.llm_call_duration_ms[model]["count"] += 1.0
        self.llm_call_duration_ms[model]["sum"] += duration_ms
        self.llm_call_duration_ms[model]["sum_squares"] += duration_ms * duration_ms
        self.llm_call_duration_ms[model]["prompt_tokens_sum"] += float(prompt_tokens)
        self.llm_call_duration_ms[model]["completion_tokens_sum"] += float(completion_tokens)
        self.llm_call_duration_ms[model]["prompt_len_sum"] += float(prompt_len)
        _write_sink_event(
            "llm_call",
            {
                "model": model,
                "prompt_len": prompt_len,
                "duration_ms": round(duration_ms, 3),
                "success": success,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        )

    def record_parse_failure(
        self,
        stage: str,
        response_snippet: str,
        reason: str = "",
    ) -> None:
        """记录一次解析失败，并附上有界的响应片段。"""
        self.parse_failures_total[stage] += 1
        snippet = response_snippet[:DEFAULT_LOG_SNIPPET_LEN]
        entry = {
            "stage": stage,
            "reason": reason,
            "snippet": snippet,
        }
        self.recent_parse_failures.append(entry)
        if len(self.recent_parse_failures) > self.max_recent_failures:
            self.recent_parse_failures.pop(0)

        logger = get_logger()
        logger.warning(
            "Parse failure at stage=%s reason=%s snippet=%r",
            stage,
            reason,
            snippet,
        )
        _write_sink_event(
            "parse_failure",
            {"stage": stage, "reason": reason, "snippet": snippet},
        )

    def record_fallback(self, source_label: str, reason: str = "") -> None:
        """记录发生了一次回落决策路径。"""
        self.fallback_total[source_label] += 1
        logger = get_logger()
        logger.info("Fallback to source=%s reason=%s", source_label, reason)
        _write_sink_event(
            "fallback", {"source": source_label, "reason": reason},
        )

    def record_decision(self, source_label: str, state_key: str = "") -> None:
        """记录一次打分/聊天请求的最终决策来源。"""
        self.decisions_total[source_label] += 1
        logger = get_logger()
        logger.info(
            "Decision source=%s state_key=%s",
            source_label,
            state_key,
        )
        _write_sink_event(
            "decision", {"source": source_label, "state_key": state_key},
        )

    def summary(self) -> Dict[str, Any]:
        """返回所有已累加指标的、可 JSON 序列化的摘要。"""
        duration_summary: Dict[str, Any] = {}
        for model, stats in self.llm_call_duration_ms.items():
            count = stats.get("count", 0.0)
            total = stats.get("sum", 0.0)
            mean = total / count if count > 0 else 0.0
            sq_sum = stats.get("sum_squares", 0.0)
            variance = (sq_sum / count - mean * mean) if count > 0 else 0.0
            variance = max(0.0, variance)
            duration_summary[model] = {
                "count": int(count),
                "total_ms": round(total, 3),
                "mean_ms": round(mean, 3),
                "std_ms": round(variance**0.5, 3),
                "prompt_tokens_total": int(stats.get("prompt_tokens_sum", 0.0)),
                "completion_tokens_total": int(stats.get("completion_tokens_sum", 0.0)),
                "prompt_len_total": int(stats.get("prompt_len_sum", 0.0)),
            }

        llm_calls_summary: Dict[str, Dict[str, int]] = {}
        for m, s in self.llm_calls_total.items():
            llm_calls_summary[m] = {
                "success": s.get("success", 0),
                "failure": s.get("failure", 0),
            }

        return {
            "llm_calls": llm_calls_summary,
            "llm_duration_ms": duration_summary,
            "parse_failures": dict(self.parse_failures_total),
            "fallbacks": dict(self.fallback_total),
            "decisions": dict(self.decisions_total),
            "recent_parse_failures": list(self.recent_parse_failures),
        }

    def reset(self) -> None:
        """清空所有已累加指标（测试中很有用）。"""
        self.llm_calls_total.clear()
        self.llm_call_duration_ms.clear()
        self.parse_failures_total.clear()
        self.fallback_total.clear()
        self.decisions_total.clear()
        self.recent_parse_failures.clear()

    def __repr__(self) -> str:
        return f"LLMObservability({self.summary()})"


# 模块级单例，便于在 AI 层各处方便地使用。
_OM_DEFAULT: Optional[LLMObservability] = None


def default_observability() -> LLMObservability:
    """返回默认的全局可观测性累加器。"""
    global _OM_DEFAULT
    if _OM_DEFAULT is None:
        _OM_DEFAULT = LLMObservability()
    return _OM_DEFAULT


def get_logger(name: str = "kakeya_ai.ai") -> logging.Logger:
    """返回 AI 推理层的 logger。

    若该 logger 没有 handler，则挂一个 NullHandler，这样不配置 logging 的库
    不会发出警告。
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


def configure_logging(
    level: int = logging.INFO,
    stream: Optional[Any] = None,
    log_file: Optional[str] = None,
) -> None:
    """配置 AI 层 logger，使其输出结构化 JSON 或纯文本。

    Parameters
    ----------
    level : int
        日志级别，例如 logging.INFO。
    stream : file-like or None
        若提供，日志写入此处；否则使用 stderr。
    log_file : str or None
        若提供，日志会额外追加到该文件（UTF-8，使用同一个 JSON formatter）。
        在后续调用中传 None 可再次移除该文件 handler。
    """
    logger = get_logger()
    logger.setLevel(level)
    # 若 configure_logging 被调用两次，避免添加重复的 handler。
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)

    handler: logging.Handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(_JSONFormatter())
    logger.addHandler(handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(_JSONFormatter())
        logger.addHandler(file_handler)
    logger.propagate = False


class _JSONFormatter(logging.Formatter):
    """把日志记录输出为单行 JSON 对象。"""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def observe_llm_call(
    model_arg: str = "model",
    prompt_arg: str = "prompt",
    observability: Optional[LLMObservability] = None,
    extract_tokens: Optional[Callable[[Any], Any]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """记录 LLM 调用指标与时长的装饰器。

    Parameters
    ----------
    model_arg : str
        被包装函数中存放模型标签的参数名。
    prompt_arg : str
        被包装函数中存放 prompt 字符串的参数名。
    observability : LLMObservability or None
        记录到的实例；默认为全局单例。
    extract_tokens : callable or None
        可选的 ``result -> (prompt_tokens, completion_tokens)``，在成功时用于
        从被包装函数的返回值中提取 token 数，以丰富所记录的调用信息。

    Example
    -------
    .. code-block:: python

        @observe_llm_call()
        def generate(model: str, prompt: str, ...) -> str:
            ...
    """
    obs = observability if observability is not None else default_observability()

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig = func.__code__.co_varnames
        model_idx = sig.index(model_arg) if model_arg in sig else None
        prompt_idx = sig.index(prompt_arg) if prompt_arg in sig else None

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            model = kwargs.get(model_arg)
            if model is None and model_idx is not None and model_idx < len(args):
                model = args[model_idx]
            model = model or "unknown"

            prompt = kwargs.get(prompt_arg)
            if prompt is None and prompt_idx is not None and prompt_idx < len(args):
                prompt = args[prompt_idx]
            prompt_len = len(prompt) if isinstance(prompt, str) else 0

            t0 = time.perf_counter()
            success = False
            prompt_tokens = 0
            completion_tokens = 0

            def _record() -> None:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                obs.record_llm_call(
                    model=model,
                    prompt_len=prompt_len,
                    duration_ms=duration_ms,
                    success=success,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

            if inspect.isgeneratorfunction(func):

                def _gen_wrapper():
                    nonlocal success
                    try:
                        yield from func(*args, **kwargs)
                        success = True
                    except Exception:
                        get_logger().exception("LLM streaming call failed model=%s", model)
                        raise
                    finally:
                        _record()

                return _gen_wrapper()

            try:
                result = func(*args, **kwargs)
                success = True
                if extract_tokens is not None:
                    try:
                        tokens = extract_tokens(result)
                        if tokens:
                            prompt_tokens = int(tokens[0] or 0)
                            completion_tokens = int(tokens[1] or 0)
                    except Exception:
                        get_logger().debug("token extraction failed", exc_info=True)
                return result
            except Exception:
                get_logger().exception("LLM call failed model=%s", model)
                raise
            finally:
                _record()

        return wrapper

    return decorator
