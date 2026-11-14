"""Ollama API 调用的 token 用量统计。

Ollama 的 ``/api/generate`` 端点会在最终响应里返回 ``prompt_eval_count`` 与
``eval_count``。本模块提供一个小型累加器，让调用方可以收集每步或每会话的
token 统计，而不必把统计值一路透传过每个返回值。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class TokenUsage:
    """prompt 与 completion token 计数的累加器。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    #: 并发保护（2026-11-06 补）：``beam_search`` 的 deep 评估并发与
    #: ``PathExecutor._execute_parallel`` 会从多个线程同时累加；裸 ``+=``
    #: 是读-改-写，竞态会丢更新。锁粒度只到累加方法，``summary``/``repr``
    #: 读旧值无害，不持锁。
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def add_from_ollama_response(self, response: Dict[str, Any]) -> None:
        """从单个 Ollama generate 响应 dict 累加 token 计数。"""
        prompt = int(response.get("prompt_eval_count") or 0)
        completion = int(response.get("eval_count") or 0)
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += prompt + completion
            self.calls += 1

    def summary(self) -> Dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls": self.calls,
        }

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, completion={self.completion_tokens}, "
            f"total={self.total_tokens}, calls={self.calls})"
        )
