"""统一配置层：所有环境变量的**单一注册表** + 带默认值的类型化读取。

**为什么需要它**：在这之前，十余个环境变量散落在 ``memory_guard``、
``ai/observability``、``ai/ollama_client``、``ai/cascaded_reasoning`` 等模块里，
各自 ``os.environ.get(...)``。后果有三：

1. **无从发现**：想部署的人无法列出"这个系统认哪些环境变量、默认值是什么"，
   只能翻遍源码。这是私有化部署的第一道硬门槛。
2. **语义不一致**：同一个"开关"在 A 处认 ``1/true/yes/on``，在 B 处只认 ``1``；
   "数值"在 C 处用 ``int()``、D 处用 ``int(float())``。行为差异无法预期。
3. **无校验**：写错了（如 ``CHAT_DEEP_NUM_PREDICT=abc``）只会静默走默认值，
   部署方以为自己改了配置，实际没生效。

本模块把这三件事收敛到一处：``ENV_SPEC`` 是**唯一**的变量清单（名字、类型、
默认值、说明），下面的 ``as_*`` 读取器是**唯一**的解析入口。

**解析失败一律回退默认值，不抛异常**（本机既有行为与测试都依赖这一点，见
``tests/test_gpu_volume.py::test_bad_env_falls_back_to_default``）：一个服务
不该因为某个可选变量写错就起不来。失败会记进 ``kakeya_ai.config`` 的 logger
（warning 级），因此"改了没生效"可以在日志里查到，而不是彻底静默。

**向后兼容**：各模块既有的 ``XXX_ENV`` 常量与 ``DEFAULT_XXX`` 常量保持原样
导出（测试与外部代码引用它们），只是取值改从这里来——行为逐位不变。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("kakeya_ai.config")

# ---------------------------------------------------------------------------
# 环境变量清单（唯一注册表）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvVar:
    """一个环境变量的元数据（名字、类型、默认值与说明）。"""

    name: str
    kind: str          # 该变量的取值类型，只能是 "str" | "int" | "float" | "bool" | "csv" 这五种中的一种
    default: Any
    doc: str
    #: 仅用于展示，说明该变量影响哪个子系统。
    scope: str = "core"

    def describe(self) -> str:
        default = self.default
        if isinstance(default, str) and not default:
            default = "(空)"
        return f"{self.name}  [{self.kind}, 默认 {default!r}]  {self.doc}"


#: **唯一**的环境变量清单。新增配置项时在这里立条目，不要在模块里另起
#: ``os.environ.get``。顺序即文档与 ``describe_env()`` 的输出顺序。
ENV_SPEC: Tuple[EnvVar, ...] = (
    # --- 构建期（被 cpp/build_nano.sh 读取，不经过本模块的读取函数）---
    EnvVar(
        "KAKEYA_MARCH",
        "str",
        "",
        "C++ 内核的指令集基线；空=-march=native，none=不加，其他值如 x86-64-v2。"
        "出包/CI 必须显式指定，否则产物绑定构建机 CPU",
        scope="build",
    ),
    # --- 内存 / GPU 守门 ---
    EnvVar(
        "KAKEYA_GPU_MEMORY_RESERVE",
        "bytes",
        2 * 1024**3,
        "GPU 任务前需为系统保留的空闲内存（字节，接受 2147483648 或 2e9）",
        scope="core",
    ),
    # --- AI 推理层 ---
    EnvVar(
        "OLLAMA_HOSTS",
        "csv",
        "",
        "逗号分隔的 Ollama 实例列表，轮询+故障转移；空=单实例",
        scope="ai",
    ),
    EnvVar(
        "CHAT_DEEP_NUM_PREDICT",
        "int",
        None,
        "deep 模型的 num_predict 基线；空=用内置 1024",
        scope="ai",
    ),
    # --- 可观测性 ---
    EnvVar(
        "OBS_JSONL_PATH",
        "str",
        "",
        "AI 调用链 JSONL 落盘目录；空=不落盘",
        scope="ai",
    ),
    # --- 资源预算类配置项（这些由 kakeya_ai.resource_budget 装配成 Budget 对象）---
    EnvVar(
        "KAKEYA_HTTP_TIMEOUT",
        "float",
        300.0,
        "单次 HTTP 请求超时（秒）；资源预算项之一，见 kakeya_ai.resource_budget",
        scope="resource",
    ),
    EnvVar(
        "KAKEYA_MAX_RETRIES",
        "int",
        2,
        "瞬时失败的最大重试次数（每主机）；资源预算项之一",
        scope="resource",
    ),
    EnvVar(
        "KAKEYA_RETRY_BACKOFF",
        "float",
        1.0,
        "重试退避基数（秒），实际等待 backoff * 2**attempt；资源预算项之一",
        scope="resource",
    ),
    EnvVar(
        "KAKEYA_FAST_TIMEOUT",
        "float",
        30.0,
        "fast 模型单次推理超时（秒）；资源预算项之一",
        scope="resource",
    ),
    EnvVar(
        "KAKEYA_DEEP_TIMEOUT",
        "float",
        60.0,
        "deep 模型单次推理超时（秒）；资源预算项之一",
        scope="resource",
    ),
    EnvVar(
        "KAKEYA_MAX_WORKERS",
        "int",
        4,
        "并行工作线程上限；资源预算项之一",
        scope="resource",
    ),
)

_BY_NAME: Mapping[str, EnvVar] = {var.name: var for var in ENV_SPEC}

# ---------------------------------------------------------------------------
# 读取器
# ---------------------------------------------------------------------------

_BOOL_TRUE = frozenset({"1", "true", "yes", "on"})


class _Unset:
    """哨兵：区分"未设置"与"显式设成空串"。"""

    def __repr__(self) -> str:            # pragma: no cover - 仅调试用
        return "<unset>"


UNSET = _Unset()


def _is_truthy(raw: str) -> bool:
    """``1/true/yes/on``（大小写不限）视为真。"""
    return raw.strip().lower() in _BOOL_TRUE


def raw(name: str, default: str = "") -> str:
    """读取原始字符串；未设置返回 ``default``。"""
    value = os.environ.get(name)
    return default if value is None else value


def flag(name: str, default: bool = False) -> bool:
    """读取布尔开关。

    认 ``1/true/yes/on``（大小写不限）；**未设置或空串返回 ``default``**，
    其他任何值一律 ``False``（与既有各模块语义一致）。
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return _is_truthy(value)


def as_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """读取整数；未设置或非法返回 ``default``。

    与 :func:`as_bytes` 的区别：本函数**不接受** ``"1e9"``/``"2147483648.0"``
    这类浮点写法，用于本该是整数的场景。浮点容错的场景请用 ``as_bytes``。
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        logger.warning("环境变量 %s=%r 不是整数，回退默认值 %r", name, value, default)
        return default


def as_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """读取浮点数；未设置或非法返回 ``default``。"""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value.strip())
    except ValueError:
        logger.warning("环境变量 %s=%r 不是数值，回退默认值 %r", name, value, default)
        return default


def as_bytes(name: str, default: Optional[int] = None) -> Optional[int]:
    """读取字节数；接受 ``2147483648`` 与 ``2e9`` 两种写法。

    历史实现用 ``int(float(raw))``（见 ``memory_guard.reserve_bytes``），
    为了不改行为而保留这一宽容度。
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(float(value.strip()))
    except ValueError:
        logger.warning("环境变量 %s=%r 不是字节数，回退默认值 %r", name, value, default)
        return default


def as_csv(name: str, default: Sequence[str] = ()) -> List[str]:
    """读取逗号分隔列表；自动去掉空白项与首尾空格。

    ``"a, b,,"`` → ``["a", "b"]``；未设置或全为空白 → ``list(default)``。
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


_READERS: Mapping[str, Callable[[str, Any], Any]] = {
    "str": raw,
    "int": as_int,
    "float": as_float,
    "bytes": as_bytes,
    "bool": flag,
    "csv": as_csv,
}


def get(name: str, default: Any = UNSET) -> Any:
    """按注册表读取一个变量（类型与默认值都取自 ``ENV_SPEC``）。

    未注册的名字抛 ``KeyError``——这是刻意的：配置项必须先在注册表立条目，
    否则"系统认哪些变量"这个问题又会失去唯一答案。
    """
    spec = _BY_NAME.get(name)
    if spec is None:
        raise KeyError(
            f"未注册的环境变量：{name!r}。请在 kakeya_ai/config.py 的 ENV_SPEC 中登记"
        )
    fallback = spec.default if default is UNSET else default
    return _READERS[spec.kind](spec.name, fallback)


# ---------------------------------------------------------------------------
# 快照与自描述
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigSnapshot:
    """当前生效的配置快照（全部来自注册表，未注册的变量不出现）。

    服务启动时把它写进日志/健康检查，就能回答"这台机器按什么配置在跑"。
    """

    values: Dict[str, Any] = field(default_factory=dict)
    #: 被环境变量**实际覆盖**过的键（与默认值不同），便于区分"默认"与"改过"。
    overridden: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.values)

    def summary(self) -> str:
        lines = []
        for name in sorted(self.values):
            mark = "*" if name in self.overridden else " "
            lines.append(f"{mark} {name} = {self.values[name]!r}")
        lines.append("（* = 被环境变量覆盖，其余为默认值）")
        return "\n".join(lines)


def snapshot() -> ConfigSnapshot:
    """读取所有已注册变量的当前取值。"""
    values: Dict[str, Any] = {}
    overridden: List[str] = []
    for spec in ENV_SPEC:
        values[spec.name] = get(spec.name)
        if os.environ.get(spec.name, "").strip():
            overridden.append(spec.name)
    return ConfigSnapshot(values=values, overridden=tuple(overridden))


def describe_env() -> str:
    """返回全部已注册环境变量的可读清单（用于文档与 ``--help``）。"""
    lines: List[str] = []
    current_scope = ""
    for spec in ENV_SPEC:
        if spec.scope != current_scope:
            if lines:
                lines.append("")
            lines.append(f"[{spec.scope}]")
            current_scope = spec.scope
        lines.append("  " + spec.describe())
    return "\n".join(lines)


def env_names() -> Tuple[str, ...]:
    """全部已注册变量的名字（顺序同 ``ENV_SPEC``）。"""
    return tuple(spec.name for spec in ENV_SPEC)


__all__ = [
    "ConfigSnapshot",
    "EnvVar",
    "ENV_SPEC",
    "UNSET",
    "as_bytes",
    "as_csv",
    "as_float",
    "as_int",
    "describe_env",
    "env_names",
    "flag",
    "get",
    "raw",
    "snapshot",
]
