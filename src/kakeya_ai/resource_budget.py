"""资源预算：超时、重试、并发的**单一真源**。

**为什么单独一个模块**：此前"一顿操作最多花多久、最多重试几次、最多同时跑几个"
散落在至少四个地方，且**互相不知道彼此存在**：

===================  ==========================================
位置                  形态
===================  ==========================================
``ai/ollama_client``  ``timeout=300``、``DEFAULT_MAX_RETRIES``、
                      ``DEFAULT_RETRY_BACKOFF`` 硬编码字面量
``ai/cascaded_reasoning``  ``fast_timeout=30.0`` / ``deep_timeout=60.0``
``ai/beam_search``     同上（**复制粘贴**，改一处不会同步另一处）
``ai/expert_router``   ``max_workers=1`` 默认参数（串行是默认，并发要显式给）
===================  ==========================================

后果与 ``config.py`` 面临的问题同源，但更严重一层：``config.py`` 解决的是
"**同一个变量**在 A/B 处解析不一致"，而这里还有"**同一件事**（如一次 fast
推理超时）在两个模块各写一个数"——改了一个，另一个不会跟着变；也没有任何
地方能回答"这个系统整体允许的最坏耗时是多少"。

本模块把"预算"收敛成一个对象 :class:`Budget`：

- **可显式构造**（不依赖环境变量），因此测试能造"1 秒超时"而不必等 300 秒；
- **可从 ``config.py`` 注册表加载**（:data:`DEFAULT_BUDGET`），环境变量是
  **唯一**的覆盖入口；
- **可审计**（:meth:`Budget.explain` 报出每一项的来源值）；
- **非法值当场拒绝**（:meth:`Budget.validate`，抛 ``InputValidationError``），
  不像 ``config.py`` 那样静默回退——预算错了会让长跑任务跑飞，必须报出来。

设计取舍
--------

1. **与 ``config.py`` 的分工**：``config`` 负责"从环境读出一个值"（含类型与
   回退），本模块负责"**这一族值合起来构成什么策略**"。所以这里**不**自己写
   ``os.environ.get``，一律走 ``config.get``。
2. **``Budget`` 是 dataclass（不可变）**：预算是共享的只读策略，不该被某个
   调用点就地改掉。要变就造新的（:meth:`Budget.overridden`）。
3. **不做全局单例**：模块级只给一个 :data:`DEFAULT_BUDGET` 常量。函数默认参数
   引用它，但调用方**总是**能传入自己的 Budget——避免"全局状态"导致的测试
   互相污染（本项目多个测试文件会改环境变量）。
4. **超时用"单次调用"口径**：``http_timeout`` 是**一次 HTTP 请求**的上限，
   不是整个任务的上限。整个任务的上限由 ``max_attempts`` × ``http_timeout``
   在预算意义上界定（见 :meth:`Budget.worst_case_seconds`），这与
   ``proof_monitor`` 的"卡住判定"配合使用：前者是**事前**约束，后者是**事后**观测。
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Iterator

from .config import as_float, as_int
from .errors import InputValidationError

__all__ = [
    "Budget",
    "DEFAULT_BUDGET",
    "TimeExceededError",
    "deadline",
]


# ---------------------------------------------------------------------------
# 环境变量名（在此集中声明；登记进 config.ENV_SPEC 由本模块负责同步）
# ---------------------------------------------------------------------------

#: 单次 HTTP 请求超时（秒）。默认 300（沿用 ``ollama_client`` 的既有值）。
ENV_HTTP_TIMEOUT = "KAKEYA_HTTP_TIMEOUT"
#: 瞬时失败的最大重试次数（每主机）。
ENV_MAX_RETRIES = "KAKEYA_MAX_RETRIES"
#: 重试退避基数（秒），实际等待 ``backoff * 2**attempt``。
ENV_RETRY_BACKOFF = "KAKEYA_RETRY_BACKOFF"
#: fast 模型推理超时（秒）。默认 30（沿用 ``cascaded_reasoning`` 既有值）。
ENV_FAST_TIMEOUT = "KAKEYA_FAST_TIMEOUT"
#: deep 模型推理超时（秒）。默认 60。
ENV_DEEP_TIMEOUT = "KAKEYA_DEEP_TIMEOUT"
#: 并发执行的工作线程上限。默认 4（沿用既有实现的取值；多数调用点仍默认串行）。
ENV_MAX_WORKERS = "KAKEYA_MAX_WORKERS"


#: 各项默认值。**集中一处**，便于审查"系统默认允许多慢/多并发"。
DEFAULT_HTTP_TIMEOUT = 300.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF = 1.0
DEFAULT_FAST_TIMEOUT = 30.0
DEFAULT_DEEP_TIMEOUT = 60.0
DEFAULT_MAX_WORKERS = 4
#: **分类/判定类短调用**的超时。比 ``DEFAULT_FAST_TIMEOUT`` 短得多，是刻意的：
#: 这类调用只是"读一眼、给个标签"（如 `nlu.intent` 的语义分类与一致性判据），
#: 慢就不值得等——等不到就退回离线兜底，比拖住交互更划算。
DEFAULT_CLASSIFY_TIMEOUT = 12.0


class TimeExceededError(Exception):
    """某个操作超过了预算允许的时间。

    刻意**不**继承 ``kakeya_ai.errors.KakeyaError`` 的语义子类：它是
    :func:`deadline` 这个工具函数抛的**控制流信号**（局部、可恢复），
    而不是"这次调用失败了"。调用方通常就地在 ``except`` 里降级，不往外冒。
    """

    def __init__(self, label: str, limit: float, elapsed: float) -> None:
        self.label = label
        self.limit = limit
        self.elapsed = elapsed
        super().__init__(
            f"{label or '操作'}超过预算：限 {limit:.3f}s，实际 {elapsed:.3f}s"
        )


@dataclass(frozen=True)
class Budget:
    """一组资源预算（不可变）。

    所有时间单位为**秒**。字段语义见各属性注释；:meth:`validate` 会做一致性
    检查，:meth:`explain` 给出人类可读的来源说明。
    """

    #: 单次 HTTP 请求的超时上限。
    http_timeout: float = DEFAULT_HTTP_TIMEOUT
    #: 瞬时失败的最大重试次数（0 = 不重试）。
    max_retries: int = DEFAULT_MAX_RETRIES
    #: 重试退避基数；第 k 次重试等待 ``retry_backoff * 2**(k-1)`` 秒。
    retry_backoff: float = DEFAULT_RETRY_BACKOFF
    #: fast 模型单次推理超时。
    fast_timeout: float = DEFAULT_FAST_TIMEOUT
    #: deep 模型单次推理超时。
    deep_timeout: float = DEFAULT_DEEP_TIMEOUT
    #: 并行工作线程上限。
    max_workers: int = DEFAULT_MAX_WORKERS
    #: 记录"哪些项来自环境覆盖"，供 :meth:`explain` 审计。**不参与相等性**
    #: 判断（否则两个数值相同但来源不同的预算会被判不等，违背直觉）。
    _sources: dict = field(default_factory=dict, compare=False, repr=False)

    # -- 校验 ---------------------------------------------------------------

    def validate(self) -> "Budget":
        """检查各字段是否在合理范围；非法即抛 ``InputValidationError``。

        **与 ``config.py`` 的策略刻意相反**：那边非法值回退默认（服务不该
        因可选配置写错而起不来）；这边直接报错。理由：预算决定长跑任务的
        上限，一个 ``-1`` 的超时会退化成"立即超时"或"永不超时"（取决于下游
        怎么用），静默回退会让人以为改了配置。宁可启动时报错。
        """
        if not (self.http_timeout > 0):
            raise InputValidationError(
                "HTTP 超时必须为正数", field="http_timeout", value=self.http_timeout
            )
        if self.max_retries < 0:
            raise InputValidationError(
                "重试次数不能为负", field="max_retries", value=self.max_retries
            )
        if self.retry_backoff < 0:
            raise InputValidationError(
                "退避基数不能为负", field="retry_backoff", value=self.retry_backoff
            )
        if not (self.fast_timeout > 0):
            raise InputValidationError(
                "fast 超时必须为正数", field="fast_timeout", value=self.fast_timeout
            )
        if not (self.deep_timeout > 0):
            raise InputValidationError(
                "deep 超时必须为正数", field="deep_timeout", value=self.deep_timeout
            )
        if self.max_workers < 1:
            raise InputValidationError(
                "并发数至少为 1", field="max_workers", value=self.max_workers
            )
        # deep 通常比 fast 慢（更多 token）；deep < fast 多半是配错了。
        # 不报错（有些部署确实如此），但不静默——见 explain()。
        return self

    # -- 派生量 -------------------------------------------------------------

    def worst_case_seconds(self) -> float:
        """本预算下**单次调用链**的最坏耗时估计（含全部重试）。

        口径：``http_timeout * (max_retries + 1)`` + 退避总等待。
        这是**预算意义**上的上界，用于容量规划与"这任务值不值得等"的判断；
        不是运行时保证（实际会受网络与队列影响）。
        """
        attempts = self.max_retries + 1
        backoff_total = sum(
            self.retry_backoff * (2 ** k) for k in range(self.max_retries)
        )
        return self.http_timeout * attempts + backoff_total

    def overridden(self, **changes: object) -> "Budget":
        """返回一份改了若干字段的新预算（原对象不变）。

        比 ``dataclasses.replace`` 多做的事：把改动记进 ``_sources``，
        便于 :meth:`explain` 说明"这个值是谁改的"。
        """
        new = replace(self, **changes)
        src = dict(self._sources)
        for k in changes:
            src[k] = "显式覆盖"
        return Budget(**{**new.__dict__, "_sources": src})

    # -- 审计 ---------------------------------------------------------------

    def explain(self) -> str:
        """多行说明各字段当前值与来源（部署自查用）。"""
        rows = [
            ("http_timeout", self.http_timeout, ENV_HTTP_TIMEOUT),
            ("max_retries", self.max_retries, ENV_MAX_RETRIES),
            ("retry_backoff", self.retry_backoff, ENV_RETRY_BACKOFF),
            ("fast_timeout", self.fast_timeout, ENV_FAST_TIMEOUT),
            ("deep_timeout", self.deep_timeout, ENV_DEEP_TIMEOUT),
            ("max_workers", self.max_workers, ENV_MAX_WORKERS),
        ]
        lines = ["resource budget:"]
        for name, value, env in rows:
            src = self._sources.get(name, "默认" if env not in _env_present() else "环境")
            lines.append(f"  {name:15s} = {value!r:>8}  [{src}: {env}]")
        lines.append(
            f"  worst_case_seconds ≈ {self.worst_case_seconds():.1f}s "
            "(单次调用链含全部重试)"
        )
        if self.deep_timeout < self.fast_timeout:
            lines.append(
                "  注意：deep_timeout < fast_timeout，通常意味着配置写反了"
            )
        return "\n".join(lines)

    def __str__(self) -> str:            # pragma: no cover - 仅调试
        return (
            f"Budget(http={self.http_timeout}s, retries={self.max_retries}, "
            f"backoff={self.retry_backoff}s, fast={self.fast_timeout}s, "
            f"deep={self.deep_timeout}s, workers={self.max_workers})"
        )


def _env_present() -> set:
    """当前已设置的相关环境变量名（用于 ``explain`` 标注来源）。

    每次调用都重新读，这样测试里 monkeypatch 环境后 ``explain`` 也准确。
    """
    import os

    return {
        ENV_HTTP_TIMEOUT, ENV_MAX_RETRIES, ENV_RETRY_BACKOFF,
        ENV_FAST_TIMEOUT, ENV_DEEP_TIMEOUT, ENV_MAX_WORKERS,
    } & set(os.environ)


def _load_default() -> Budget:
    """从 ``config`` 注册表装配默认预算。

    数值全走 ``kakeya_ai.config.as_float`` / ``as_int``（含非法值回退与
    warning 日志），因此环境变量的解析口径与系统其余部分**完全一致**。
    """
    http = as_float(ENV_HTTP_TIMEOUT, DEFAULT_HTTP_TIMEOUT)
    retries = as_int(ENV_MAX_RETRIES, DEFAULT_MAX_RETRIES)
    backoff = as_float(ENV_RETRY_BACKOFF, DEFAULT_RETRY_BACKOFF)
    fast = as_float(ENV_FAST_TIMEOUT, DEFAULT_FAST_TIMEOUT)
    deep = as_float(ENV_DEEP_TIMEOUT, DEFAULT_DEEP_TIMEOUT)
    workers = as_int(ENV_MAX_WORKERS, DEFAULT_MAX_WORKERS)
    present = _env_present()
    sources = {name: "环境" for name in (
        "http_timeout", "max_retries", "retry_backoff",
        "fast_timeout", "deep_timeout", "max_workers",
    ) if {
        "http_timeout": ENV_HTTP_TIMEOUT, "max_retries": ENV_MAX_RETRIES,
        "retry_backoff": ENV_RETRY_BACKOFF, "fast_timeout": ENV_FAST_TIMEOUT,
        "deep_timeout": ENV_DEEP_TIMEOUT, "max_workers": ENV_MAX_WORKERS,
    }[name] in present}
    budget = Budget(
        http_timeout=float(http if http is not None else DEFAULT_HTTP_TIMEOUT),
        max_retries=int(retries if retries is not None else DEFAULT_MAX_RETRIES),
        retry_backoff=float(backoff if backoff is not None else DEFAULT_RETRY_BACKOFF),
        fast_timeout=float(fast if fast is not None else DEFAULT_FAST_TIMEOUT),
        deep_timeout=float(deep if deep is not None else DEFAULT_DEEP_TIMEOUT),
        max_workers=int(workers if workers is not None else DEFAULT_MAX_WORKERS),
        _sources=sources,
    )
    return budget.validate()


#: 进程启动时装配一次的默认预算。**调用方总能传入自己的 Budget 覆盖它**。
DEFAULT_BUDGET = _load_default()


@contextlib.contextmanager
def deadline(limit: float, *, label: str = "",
             clock: Callable[[], float] = time.perf_counter) -> Iterator[None]:
    """上下文管理器：退出时若超过 ``limit`` 秒则抛 :class:`TimeExceededError`。

    这是**协作式**超时——它不打断正在执行的代码（Python 无法安全地强杀线程），
    而是在块结束时检查。适合"跑一段计算，超了就放弃结果"的场景::

        with deadline(budget.fast_timeout, label="fast 推理"):
            text = model.generate(...)
        # 到这里若已超时，上面会抛 TimeExceededError，调用方可降级

    **``clock`` 可注入**——测试传一个假时钟即可验证超时逻辑，不必真的 sleep。
    这是本项目对"时间相关代码难测"的一贯处理方式。
    """
    t0 = clock()
    yield
    elapsed = clock() - t0
    if elapsed > limit:
        raise TimeExceededError(label, limit, elapsed)
