r"""``kakeya_ai`` 的错误分类体系（异常分层）。

**为什么需要它**：此前全库 76 处 ``raise`` 里 72 处是**裸 ``ValueError``**，
唯一的自定义异常是 :class:`kakeya_ai.memory_guard.InsufficientMemoryError`。
后果不是"风格不统一"，而是**调用方无法按类型处置**：下面这些语义完全不同
的失败，在 ``except`` 时长得一模一样——

- **输入非法**（``N must be positive``）：调用方传错了参数，应改代码/改调用；
- **文件格式损坏**（``不是有效的 .kpt 文件``）：数据坏了，应重新生成或修文件；
- **前提缺失/契约违反**（``error_mode='actual' requires the backing tubes``）：
  调用方漏传了必需参数，是被调方的**契约**，应与"输入非法"分开报；
- **能力占位**（某功能尚未接入）：不是错误用法，而是"这条路本实现没有"；
- **数值不可达**（``LP infeasible``、体素精度下限）：**算过但达不到**，
  与"没算"、“输入错”都不是一回事。

这与本项目既有的纪律一脉相承：``wolff_axioms`` 的"零覆盖"事件里，
把"未测"当成"通过"（假阳性）与把"未测"当成"不通过"（假阴性）**根源相同**
——两者都没有区分"未测"与"测了"。异常层现在是同一个病：**没有区分失败类型**。

设计取舍
--------

1. **向后兼容优先**：每个子类都同时继承一个旧的内置异常
   （:class:`InvalidInputError` → ``ValueError``，
   :class:`BackendUnavailableError` → ``RuntimeError`` 等）。
   现有 ``except ValueError`` 依旧捕获得到，**迁移可以逐处进行、不必一次改完**。

2. **只做浅层树**：三个大类（输入 / 环境 / 未实现）+ 少量具体子类。
   不做"每个模块一个异常"的细粒度分类——那会变成装饰品，因为没人会去记。

3. **结构化字段优于一句消息**（照 :class:`InsufficientMemoryError` 的做法）：
   :class:`NumericalLimitError` 携带 ``shortfall``（还差多少），
   使"算过但不够"的调用方能拿到数值继续处置，而不是去解析错误消息文本。

4. **消息仍用中文**（与项目其余部分一致），并遵守"不用 emoji/装饰符号"
   的文档纪律——错误消息会进日志与实验报告。

用法
----

判断"该重试还是该改代码"时按类型分派::

    try:
        result = orchestrator.run(...)
    except InputValidationError:
        # 调用方参数错，不要重试
        raise
    except BackendUnavailableError:
        # 环境缺东西（内核没编译 / Ollama 没起），可降级或提示
        fallback()
    except NumericalLimitError as exc:
        # 算过，但没达到目标；exc.shortfall 说明差多少
        report_gap(exc.shortfall)

``except KakeyaError`` 可捕获本包抛出的**全部**自定义异常。
"""

from __future__ import annotations

__all__ = [
    "KakeyaError",
    "InputValidationError",
    "TraceFormatError",
    "ContractViolationError",
    "BackendUnavailableError",
    "NumericalLimitError",
    "NotImplementedYetError",
]


class KakeyaError(Exception):
    """本包所有自定义异常的基类。

    捕获它即可覆盖"本包主动抛出的"全部错误。**不**替代内置异常的意义：
    子类同时继承内置类型（见各子类），所以按内置类型 catch 的既有代码
    不受影响。
    """


# ---------------------------------------------------------------------------
# 一大类：调用方输入 / 数据本身有问题
# ---------------------------------------------------------------------------


class InputValidationError(KakeyaError, ValueError):
    """调用方传入的参数或几何构型不合法。

    典型场景：``N must be a positive integer``、``Shading densities must lie
    in [0,1]``、``Direction vector must be non-zero``。

    处置：**改调用代码**，不要重试——同样的输入再跑一次结果一样。

    继承 ``ValueError`` 以保持向后兼容（既有 ``except ValueError`` 仍生效）。
    """

    def __init__(self, message: str, *, field: str | None = None,
                 value: object = None) -> None:
        #: 出错的参数名（可选）。便于调用方/上层框架定位，不必解析消息文本。
        self.field = field
        #: 实际收到的值（可选）。**只用于诊断**，不要指望它可序列化。
        self.value = value
        if field is not None and "「" not in message:
            message = f"参数 {field!r} 非法：{message}"
        super().__init__(message)


class TraceFormatError(KakeyaError, ValueError):
    """``.kpt`` 证明轨迹文件格式损坏或版本不受支持。

    与 :class:`InputValidationError` 分开的原因：处置完全不同。参数非法要改
    代码；而轨迹文件坏了要**重新生成文件或修数据**——调用代码一个字都不用动。

    ``.kpt`` 的读侧有四处入口（``peek_trace``/``read_trace``/``iter_steps``/
    ``verify_trace``），它们共用这一类型，使"文件坏了"与"我传错了参数"
    在调用方可见。

    字段：``path``（出错的文件，若有）、``line``（出错行号，若已知）。
    """

    def __init__(self, message: str, *, path: str | None = None,
                 line: int | None = None) -> None:
        #: 出错的文件路径（可选）。
        self.path = path
        #: 出错的行号（1-based，可选）。
        self.line = line
        loc = ""
        if path is not None:
            loc = f"{path}" + (f":{line}" if line is not None else "")
        if loc:
            message = f"{loc}：{message}"
        super().__init__(message)


class ContractViolationError(KakeyaError, ValueError):
    """被调函数的前置条件未被满足（调用方**漏传**了必需的东西）。

    与 :class:`InputValidationError` 的区别在归责：
    ``InputValidationError`` 是"给的值不对"，本类是"该给的值没给"。
    典型：``error_mode='actual' requires the backing tubes``、
    ``move3_wider_grains requires either delta or max_local_error``。

    分开的价值：前者提示调用方检查取值，后者提示调用方检查**调用方式**
    （是不是漏了参数），排查方向不同。

    ``missing`` 列出缺失项的名字。
    """

    def __init__(self, message: str, *, missing: tuple[str, ...] = ()) -> None:
        #: 缺失的依赖项名字。
        self.missing = missing
        super().__init__(message)


# ---------------------------------------------------------------------------
# 二大类：环境/后端不可用（与代码无关，重试或降级有意义）
# ---------------------------------------------------------------------------


class BackendUnavailableError(KakeyaError, RuntimeError):
    """所需的后端/内核/外部服务不可用。

    典型：C++ 内核未编译、GPU 后端不存在、Ollama 服务未启动。

    处置：**可降级或重试**（与参数错误相反）。调用方通常有 fallback 路径
    （如 ``accel`` 的 nano → fast → numpy 调度）。

    继承 ``RuntimeError`` 兼顾既有 ``except RuntimeError``。
    """

    def __init__(self, message: str, *, backend: str | None = None) -> None:
        #: 不可用的后端名字（如 ``"nano"``、``"mlx"``、``"ollama"``）。
        self.backend = backend
        super().__init__(message)


# ---------------------------------------------------------------------------
# 三大类：算得出，但达不到目标
# ---------------------------------------------------------------------------


class NumericalLimitError(KakeyaError, RuntimeError):
    """计算已完成，但结果达不到目标（**不是**输入错，也不是环境缺）。

    典型：``LP infeasible``（约束下无解）、体素精度下限（合并到某一步后
    ``local_kt_error`` 停滞在离散化下限之上，无法达到 ``delta**(-zeta)``）。

    为何单列：本项目已多次记录"**算过但不够**"与"**无法判定**"必须可区分
    （见 ``moves._effective_tube_ratio`` 的哨兵修复）。调用方拿到本异常时
    应知道"再算下去也不会更好"，除非放宽目标或提高精度。

    ``shortfall`` 是"还差多少"的**数值**（含义由抛出方定义，如
    ``actual - target``）；有了它，调用方不必去解析消息文本。
    """

    def __init__(self, message: str, *, target: float | None = None,
                 actual: float | None = None) -> None:
        #: 目标值（若适用）。
        self.target = target
        #: 实际达到的值（若适用）。
        self.actual = actual
        super().__init__(message)

    @property
    def shortfall(self) -> float | None:
        """距目标还差多少（``actual - target``）；缺任一值时为 ``None``。

        注：正数表示"超出目标"，负数为"未达到"——沿用 ``actual - target``
        的自然含义，调用方按需取绝对值。
        """
        if self.target is None or self.actual is None:
            return None
        return self.actual - self.target


# ---------------------------------------------------------------------------
# 四大类：本实现没有这条路径
# ---------------------------------------------------------------------------


class NotImplementedYetError(KakeyaError, NotImplementedError):
    """功能在本实现中**尚未接入 / 尚未提供**（是已知缺口，不是调用方用法错误）。

    为什么不用裸 ``NotImplementedError``：它属于内置异常，任何"抽象方法未
    覆写"也会抛它，两者语义不同（一个是"子类忘实现"，一个是"本版本就没有"）。
    本类型明确表示**已知的、有登记的能力缺口**。

    **判定标准是"本实现没有这条路径"**，而不是"这个函数没写"——比如
    ``sticky_kakeya.multiscale_induction`` 本身**有可用实现**，但编排器的主
    循环从未调用它（走的是"δ 减半"占位，见 内部设计文档已知局限 #2）。
    若调用方通过某入口请求"启用真正的多尺度归纳调度"，该入口应抛本异常并
    在消息里指明缺口登记位置，而不是抛"参数非法"——否则调用方会去查参数，
    查不到任何问题。

    习惯上把缺口登记位置写在消息里（如 ``见 内部设计文档已知局限 #2``）。
    用 ``feature`` 标注缺口的功能名，便于上层聚合与统计。
    """

    def __init__(self, message: str, *, feature: str | None = None) -> None:
        #: 缺口的功能名（可选）。
        self.feature = feature
        super().__init__(message)
