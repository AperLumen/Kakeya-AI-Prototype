"""surrogate 打分模型的训练数据生成。

收集 (state, move_scores) 对。当有 Ollama 服务可用时，标签来自 LLM 打分器；
否则回落到确定性启发式打分器，并加少量高斯噪声，使训练在离线状态下仍可运行。

**特征清单是单一真源**：``FEATURE_NAMES`` 决定状态向量的维数与顺序，
``build_training_data``（训练）与 ``SurrogateScorer._state_to_vector``（推理）
都按它取特征。此前两处各自硬编码 4 个字段，扩维时极易漂移——训练用 A 组、
推理用 B 组，模型静默给出无意义的分数。现在任何一处增删特征都必须改
``FEATURE_NAMES``，并且旧模型能靠 ``feature_names`` 元数据自证是否兼容。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import ContractViolationError
from .heuristics import heuristic_move_scores
from .ollama_client import is_server_ready
from .two_stage_scorer import two_stage_score


# 几何状态特征的合理取值范围。
RHO_RANGE = (0.05, 0.95)
DELTA_RANGE = (0.001, 0.1)
LOCAL_ERROR_RANGE = (0.1, 5.0)
VOLUME_RATIO_RANGE = (0.5, 2.0)

#: 状态向量的特征名清单——**维数与顺序的唯一真源**。
#:
#: 前 4 个是历史特征（顺序与维数**不可变**：旧权重文件按它解释），
#: 之后的为扩展特征。改这份清单即改变 ``input_dim``，旧权重将无法使用
#: （加载时靠 ``feature_names`` 元数据拒绝，而不是静默算错）。
#:
#: 为什么加这 4 个：surrogate 的职责是在"该走 Move #1 还是 #2"上替代 LLM，
#: 而单靠 (rho, delta, local_error, volume_ratio) 无法区分"管族很大但每根
#: 都很细"与"管族很小但每根都很粗"——两者 (delta, local_error) 可以相同。
#: 管数、grains 数、黏性标志把这两种情形分开，是提升 top-two gap 命中率的
#: 直接来源。
FEATURE_NAMES: Tuple[str, ...] = (
    # —— 历史特征（顺序固定，勿动）——
    "rho",
    "delta",
    "local_error",
    "volume_ratio",
    # —— 扩展特征 ——
    "log_n_tubes",     # 管数的对数（跨度从个位数到 δ^-2，取对数才线性可分）
    "n_grains_norm",   # grains 数 / 管数，度量分解粒度
    "sticky",          # 是否判定为黏性（0/1），决定能否走 Sticky 分支
    "aspect",          # 长宽比 c/a，度量 grain 的扁平程度
)


#: 供 ``sample_states`` 采样用的取值范围。键必须在 ``FEATURE_NAMES`` 里；
#: ``sticky`` 是标志位，单独处理。
FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
    "rho": RHO_RANGE,
    "delta": DELTA_RANGE,
    "local_error": LOCAL_ERROR_RANGE,
    "volume_ratio": VOLUME_RATIO_RANGE,
    "log_n_tubes": (0.0, 8.0),      # 1 ~ e^8 ≈ 2981 根管
    "n_grains_norm": (0.0, 1.0),
    "aspect": (1.0, 64.0),
}


def state_to_features(
    state: Dict[str, float],
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> List[float]:
    """按 ``feature_names`` 的顺序把 state dict 摊平成特征列表。

    缺失的键按 ``_DEFAULT_FEATURE`` 回退（而不是抛异常）：state dict 由多个
    调用点拼装，容忍缺省能让"新增特征"不必一次改完所有构造点。回退值本身
    也是显式的——见 ``_DEFAULT_FEATURE``。
    """
    return [float(state.get(name, _DEFAULT_FEATURE.get(name, 0.0)))
            for name in feature_names]


#: 每个特征在 state dict 缺失时的回退值（中性值，不偏袒任何 Move）。
_DEFAULT_FEATURE: Dict[str, float] = {
    "rho": 0.5,
    "delta": 0.01,
    "local_error": 1.0,
    "volume_ratio": 1.0,
    "log_n_tubes": 0.0,
    "n_grains_norm": 0.0,
    "sticky": 0.0,
    "aspect": 1.0,
}


def sample_states(n: int, seed: Optional[int] = None) -> List[Dict[str, float]]:
    """Sample ``n`` geometric states for training.

    每个 state 含 ``FEATURE_NAMES`` 里的全部键，外加若干派生键。
    """
    rng = np.random.default_rng(seed)
    states = []
    for _ in range(n):
        row: Dict[str, float] = {
            name: float(rng.uniform(*FEATURE_RANGES[name]))
            for name in FEATURE_NAMES
            if name in FEATURE_RANGES
        }
        # 标志位不在 FEATURE_RANGES 里，单独采样。
        row["sticky"] = float(rng.integers(0, 2))
        states.append(row)
    return states


def _read_field(source: Any, name: str, default: Any) -> Any:
    """从对象或 dict 里读一个字段（**两种载体的统一入口**）。

    训练数据有两个来源，字段载体不同：
    - 内存中的 ``ProofMetrics`` —— 属性访问，浮点是 ``float``；
    - ``.kpt`` 文件读回来的 step —— dict 访问，浮点是 ``float.hex()`` 文本
      （写入端刻意用十六进制保证比特级往返，见 ``trace_format``）。

    此前只支持前者，于是"从 traces 训练"这条路有**两处静默失效**：
    dict 走到 ``getattr(dict, "rho", 0.5)`` 一律取默认值（特征全变成常量），
    而 ``step.get("action").name`` 抛 ``AttributeError`` 又被吞掉（样本数恒 0）。
    两者都不报错，训练照样"成功"。
    """
    if isinstance(source, dict):
        return source.get(name, default)
    return getattr(source, name, default)


def _to_float(value: Any, default: float) -> float:
    """把字段值统一成 float，兼容 ``.kpt`` 的十六进制文本。

    ``float.fromhex`` 是 ``float.hex()`` 的精确逆运算；不认识的文本
    （如手工编辑过的文件）走 ``float()``，仍失败则回退默认值——**但这条
    回退不再是静默的**，见 ``state_from_metrics`` 的 ``strict``。
    """
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            return float.fromhex(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return default
    return default


def _to_bool(value: Any) -> bool:
    """把字段值统一成 bool，兼容十六进制/字符串形式的 ``0``/``1``。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip()
        try:
            return float.fromhex(text) != 0.0
        except ValueError:
            pass
    try:
        return float(value) != 0.0
    except (TypeError, ValueError):
        return False


def state_from_metrics(metrics: Any, *, strict: bool = False) -> Dict[str, float]:
    """从 ``ProofMetrics``（对象）或 ``.kpt`` step 的 ``before``（dict）构造特征。

    集中在这里是为了让"指标 → 特征"的映射只有一份：``proof_orchestrator``
    的 ``_select_move``、``training_data.proof_steps_to_samples`` 与
    ``scripts/training/surrogate_data_from_traces.py`` 都走它。此前两处各自手写 4 个
    字段，扩维时必然漂移。

    Parameters
    ----------
    metrics : ProofMetrics | dict
        同名字段的对象，或 ``.kpt`` 里反序列化出的 ``before`` dict。
    strict : bool
        为 True 时，"字段一个都没读到"会抛 ``ValueError`` 而不是回退默认值。

        **为什么需要这个开关**：全部字段都有默认值意味着"传错了载体"
        （如把 dict 当成对象）会得到一份**看起来正常**的全默认特征，训练
        于是学不到任何东西却不报错。脚本侧（从 traces 生成训练集）必须用
        ``strict=True`` 把这种失误变成硬错误；而热路径上的
        ``_select_move`` 用宽松模式，因为指标缺失时"给中性值继续跑"是可
        接受的降级。

    Returns
    -------
    state : dict[str, float]
        含 ``FEATURE_NAMES`` 全部键。
    """
    if strict:
        _require_metric_fields(metrics)

    n_tubes = _to_float(_read_field(metrics, "n_tubes", 0), 0.0)
    n_grains = _to_float(_read_field(metrics, "n_grains", 0), 0.0)
    a = _to_float(_read_field(metrics, "a", 0.0), 0.0)
    c = _to_float(_read_field(metrics, "c", 0.0), 0.0)
    return {
        "rho": _to_float(_read_field(metrics, "rho", 0.5), 0.5),
        "delta": _to_float(_read_field(metrics, "delta", 0.01), 0.01),
        "local_error": _to_float(_read_field(metrics, "local_error", 1.0), 1.0),
        "volume_ratio": _to_float(_read_field(metrics, "volume_ratio", 1.0), 1.0),
        "log_n_tubes": float(np.log1p(n_tubes)),
        "n_grains_norm": (n_grains / n_tubes) if n_tubes > 0 else 0.0,
        "sticky": 1.0 if _to_bool(_read_field(metrics, "is_sticky", False)) else 0.0,
        "aspect": (c / a) if a > 0 else 1.0,
    }


#: ``state_from_metrics(strict=True)`` 要求必须存在的字段。
#: 挑这四个是因为它们**只有真实指标才有**、且非零即为真值：缺任一个都说明
#: 传入的不是指标（而是空 dict／写错的载体）。
_REQUIRED_METRIC_FIELDS: Tuple[str, ...] = ("delta", "rho", "volume_ratio", "n_tubes")


def _require_metric_fields(metrics: Any) -> None:
    """``strict`` 模式的守门：载体不对就抛错，而不是给一份默认特征。"""
    missing = [
        name for name in _REQUIRED_METRIC_FIELDS
        if _read_field(metrics, name, None) is None
    ]
    if missing:
        raise ContractViolationError(
            "strict 模式下缺少指标字段 "
            f"{missing}（载体为 {type(metrics).__name__}）。"
            "若传的是 .kpt 的 before/after dict，请确认字段名与 ProofMetrics "
            "一致；若传的是对象，请确认它真的是 ProofMetrics。",
            missing=tuple(missing),
        )


def heuristic_labels(
    states: List[Dict[str, float]],
    noise: float = 0.25,
    seed: Optional[int] = None,
) -> np.ndarray:
    """返回来自确定性启发式 + 噪声的走法分标签。

    Parameters
    ----------
    states : list of dict
    noise : float
        施加到每个分数上的高斯噪声标准差。
    seed : int or None

    Returns
    -------
    Y : np.ndarray, shape (n_states, 3)
    """
    rng = np.random.default_rng(seed)
    Y = np.zeros((len(states), 3), dtype=float)
    for i, s in enumerate(states):
        # 启发式只看得懂历史 4 个字段；扩展特征不参与启发式（见 heuristics.py）。
        scores = heuristic_move_scores(
            None,
            s["rho"],
            s["delta"],
            s["local_error"],
            s["volume_ratio"],
        )
        Y[i, 0] = scores[1]
        Y[i, 1] = scores[2]
        Y[i, 2] = scores[3]
        if noise > 0:
            Y[i] += rng.normal(0.0, noise, size=3)
            Y[i] = np.clip(Y[i], 0.0, 10.0)
    return Y


def llm_labels(
    states: List[Dict[str, float]],
    prompt_builder: Callable[[Dict[str, float]], str],
    scorer: Optional[Callable] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """从 LLM 打分器收集走法分标签。

    Parameters
    ----------
    states : list of dict
    prompt_builder : callable
        把 state dict 映射成 prompt 字符串。
    scorer : callable or None
        默认为 ``two_stage_score``。

    Returns
    -------
    Y : np.ndarray, shape (n_states, 3)
    stats : dict
        各来源标签的计数（heuristic / fast / deep / cache / ...）。
    """
    if scorer is None:
        scorer = two_stage_score
    Y = np.zeros((len(states), 3), dtype=float)
    stats: Dict[str, int] = {}
    for i, s in enumerate(states):
        prompt = prompt_builder(s)
        try:
            scores, source = scorer(prompt, state=s, use_heuristic_fallback=True)
        except Exception:
            # 任何失败都回落到启发式。
            scores = heuristic_move_scores(
                None, s["rho"], s["delta"], s["local_error"], s["volume_ratio"]
            )
            source = "heuristic"
        Y[i, 0] = scores.get(1, 5.0)
        Y[i, 1] = scores.get(2, 5.0)
        Y[i, 2] = scores.get(3, 5.0)
        stats[source] = stats.get(source, 0) + 1
    return Y, stats


def default_prompt(state: Dict[str, float]) -> str:
    """从单个 state 收集 LLM 标签的最小 prompt。"""
    lines = [
        "You are selecting the next move in a Kakeya-set proof search.",
        "State:",
    ]
    for name in FEATURE_NAMES:
        lines.append(f"  {name}={float(state.get(name, 0.0)):.4f}")
    lines.append("Return exactly this JSON and nothing else:")
    lines.append('{"move_1": <0-10>, "move_2": <0-10>, "move_3": <0-10>}')
    return "\n".join(lines)


def build_training_data(
    n_states: int = 2000,
    seed: Optional[int] = None,
    use_llm_if_available: bool = True,
    prompt_builder: Optional[Callable[[Dict[str, float]], str]] = None,
    scorer: Optional[Callable] = None,
    heuristic_noise: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """为 surrogate 打分器构建 (X, Y) 训练矩阵。

    Parameters
    ----------
    n_states : int
    seed : int or None
    use_llm_if_available : bool
        若为 True 且 Ollama 可达，则收集 LLM 标签。
        否则使用启发式标签。
    prompt_builder : callable or None
    scorer : callable or None
    heuristic_noise : float
        使用启发式标签时的噪声水平。

    Returns
    -------
    X : np.ndarray, shape (n_states, len(FEATURE_NAMES))
    Y : np.ndarray, shape (n_states, 3)
    meta : dict
        包含 'label_source'、'source_counts'（若有 LLM）、状态范围，
        以及 ``feature_names``（推理端据此校验模型与特征清单是否配套）。
    """
    states = sample_states(n_states, seed=seed)
    X = np.array(
        [state_to_features(s) for s in states],
        dtype=float,
    )

    llm_available = use_llm_if_available and is_server_ready()
    if llm_available:
        if prompt_builder is None:
            prompt_builder = default_prompt
        Y, stats = llm_labels(states, prompt_builder, scorer=scorer)
        meta: Dict[str, Any] = {
            "label_source": "llm",
            "source_counts": stats,
            "n_states": n_states,
        }
    else:
        Y = heuristic_labels(states, noise=heuristic_noise, seed=seed)
        meta = {
            "label_source": "heuristic",
            "heuristic_noise": heuristic_noise,
            "n_states": n_states,
        }

    # 把特征归一化到零均值／单位方差，以便更容易优化。
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_norm = (X - mean) / std

    meta["feature_mean"] = mean.tolist()
    meta["feature_std"] = std.tolist()
    meta["feature_names"] = list(FEATURE_NAMES)

    # 标签的类别分布：**"训练集里 Move #k 各有多少个"必须留痕**。
    #
    # 理由：argmax 一致率在类别单一/严重失衡时**无判别力**（"永远猜 Move #3"
    # 也能满分，本项目 `eval_surrogate.py` 已在 2 样本场景实证）。把分布写进
    # meta 后，事后能回答"这个权重文件当时是用什么训的"——此前 meta 只记
    # `label_source` 与 `n_states`，看不出类别是否单一。
    argmax = np.argmax(Y, axis=1)  # 从 0 开始计数
    meta["label_class_counts"] = {
        str(int(k) + 1): int((argmax == k).sum()) for k in range(Y.shape[1])
    }

    return X_norm, Y, meta


# 把 proof-orchestrator 的 action 映射到 surrogate 的 move id（1/2/3）。
_PROOF_ACTION_TO_MOVE_ID = {
    "MOVE1_LONGER_GRAINS": 1,
    "MOVE2_REPLACE_SQUARE_GRAINS": 2,
    "MOVE3_WIDER_GRAINS": 3,
}


def proof_steps_to_samples(
    steps: List[Any],
    *,
    strict: bool = False,
    include_noop: bool = False,
) -> List[Tuple[Dict[str, float], Dict[int, float]]]:
    """Convert step records into surrogate training samples.

    Each returned sample is a ``(state_dict, target_dict)`` pair where the
    state carries the geometric features observed *before* the step (built by
    ``state_from_metrics``, so it always carries every ``FEATURE_NAMES`` key),
    and the target assigns 10.0 to the move that was actually executed and 5.0
    (neutral) to the others.  Steps whose action is not one of Moves
    #1/#2/#3 are skipped.

    Parameters
    ----------
    steps : list
        **两种载体都接受**：

        - ``ProofResult.steps``（``ProofStep`` 对象，属性访问，``action``
          是 ``ProofAction`` 枚举）；
        - ``trace_format.read_trace(path).steps`` 或 ``iter_steps(path)``
          产出的 dict（``action`` 是字符串、``before`` 是含十六进制浮点的
          dict）。

        后者是"从入库的 ``docs/traces/*.kpt`` 回流训练"的入口——此前只认
        前者，dict 走进去会**静默返 0 个样本**（``{}.name`` 抛
        ``AttributeError`` 被吞掉）。
    strict : bool
        透传给 ``state_from_metrics``。从文件读样本时应为 True：让"字段名
        对不上"变成硬错误，而不是悄悄训出一条水平线。
    include_noop : bool
        **默认 False**：跳过"应用了 Move 但没带来任何可观测推进"的步
        （``ProofStep.achieved_gain is False``）。

        为什么默认跳过：本函数把"实际执行过的 Move"当**正样本**（打 10.0）。
        而实测（2026-10-01）一条 trace 的 3 个 Move 样本里有 **2 个是
        ``MOVE2`` 的 no-op**（``achieved_gain=False``，见局限 #14 ——
        ``Move #2`` 的前置条件在默认参数下结构性不可达）。把它们当正样本
        = **教模型偏好一个什么都不做的 Move**。这与"用错误标签教模型"同族，
        是训练闭环里最该避免的一类污染。

        三态语义：``True`` 收下；``False`` 跳过（除非 ``include_noop``）；
        ``None``（旧文件无此字段 / 非 Move 步）**照收**——因为"未记录"不等于
        "没推进"，不能凭空剔除（与 ``selection_source`` 处理同策略）。

    Returns
    -------
    samples : list of (state_dict, target_dict)
    """
    samples: List[Tuple[Dict[str, float], Dict[int, float]]] = []
    for step in steps:
        action = _read_field(step, "action", None)
        # 对象载体：action 是枚举，取 .name；dict 载体：action 已经是字符串。
        #
        # **显式收窄成 ``str | None``**：``getattr(...)`` 的返回类型是 ``Any``，
        # 直接传给 ``dict.get(key: str)`` 会被类型检查器拒绝（Any 可能不是 str）。
        # 而运行时语义本来就只可能是这两种之一——``getattr(..., None)`` 拿不到
        # 时是 None，拿到的是枚举的 ``.name``（str）。把这层事实写进类型标注，
        # 下游 ``.get(action_name)`` 才不必 ignore。
        action_name: str | None = None
        if isinstance(action, str):
            action_name = action
        else:
            name = getattr(action, "name", None)   # ``Any`` —— 无法静态证明是 str
            if isinstance(name, str):
                action_name = name
        # 取不到动作名 ⇒ 这一步不是 Move（或载体缺字段），跳过。
        # **显式提前退出**而不是把 None 交给 ``.get()``：后者行为相同
        # （``.get(None)`` 返回 None），但类型上 ``.get`` 要求 ``str``，
        # 且显式退出让"无动作名"这条路径在代码里可见。
        if action_name is None:
            continue
        move_id = _PROOF_ACTION_TO_MOVE_ID.get(action_name)
        if move_id is None:
            continue
        # 缺口二：**明确**判定为 no-op 的 Move 不入正样本（见参数说明）。
        # 只有显式 ``False`` 才跳过；``None``（未记录）照收——不把"不知道"
        # 当成"没推进"。
        if not include_noop and _read_field(step, "achieved_gain", None) is False:
            continue
        before = _read_field(step, "before", None)
        if before is None:
            continue
        state_dict = state_from_metrics(before, strict=strict)
        target_dict = {1: 5.0, 2: 5.0, 3: 5.0}
        target_dict[move_id] = 10.0
        samples.append((state_dict, target_dict))
    return samples


# ---------------------------------------------------------------------------
# 样本充足性判据
# ---------------------------------------------------------------------------

#: surrogate 的默认结构（与 ``surrogate_scorer.SurrogateScorer`` 的默认值一致）。
#:
#: **为什么在这里复制一份**：本模块（``ai/training_data.py``）**不导入**
#: ``surrogate_scorer``——后者带 numpy 权重与推理逻辑，而本模块要能被
#: "只算样本数"的轻量调用方（如 ``scripts/training/surrogate_data_from_traces.py``）
#: 使用。故这里只放**结构常量**，并加测试钉住它与真实模型默认值一致
#: （见 ``tests/test_sample_sufficiency.py::test_default_architecture_matches_scorer``）。
DEFAULT_INPUT_DIM = 4
DEFAULT_HIDDEN_DIMS = (16, 16)
DEFAULT_OUTPUT_DIM = 3


def count_mlp_parameters(
    input_dim: int = DEFAULT_INPUT_DIM,
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    output_dim: int = DEFAULT_OUTPUT_DIM,
) -> int:
    """算 MLP 的参数量（权重 + 偏置），用于推导"需要多少样本"。

    **为什么要能算而不是写死**：阈值原先的依据是注释里的一句
    "≈ 400 个参数"，而参数量**随 ``input_dim`` 变**——落盘权重是 4 维
    （403 参数），而 ``FEATURE_NAMES`` 是 8 维（467 参数）。写死一个数
    就必然与某一种配置不符（2026-09-30 实测发现：原标题写成"8 维输入 ×
    ... ≈ 400"，但 8 维其实是 467，403 对应的是 4 维）。

    Examples
    --------
    >>> count_mlp_parameters()          # 默认 4 -> 16 -> 16 -> 3
    403
    >>> count_mlp_parameters(8)         # 8 维特征（FEATURE_NAMES）
    467
    """
    dims = [int(input_dim), *[int(h) for h in hidden_dims], int(output_dim)]
    return sum(
        dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)
    )


#: 样本数与参数量的目标比值（经验法则：每 1 个参数配这么多样本）。
#:
#: 取 5（而非 10）是本项目的**保守下界**：5× 已远高于现状，若取 10× 则
#: 目标更加遥不可及（见下面的 ``MIN_SAMPLES_FOR_TRAINING`` 说明）。
SAMPLES_PER_PARAMETER = 5


#: 认为"可以开始训练一个可信的 surrogate"的最小样本数。
#:
#: **取值依据（2026-09-30 重写；原表述有误，见下）**
#:
#: 模型（默认 4→16→16→3）有 :func:`count_mlp_parameters` = **403** 个参数。
#:
#: 经验法则要求样本数 ≥ 5–10× 参数量，即 **2015–4030**。**当前取 200**，
#: 是**明显低于**该法则的**工程折中**，理由有两条：
#:
#: 1. **现状约束**：可信样本只有个位数（见 内部设计文档 §7.0.2 第 6 步），
#:    把阈值定在 2000+ 会让判定**长期停在 ``insufficient``**，失去"催人攒数据"
#:    的作用——一个永远报警的指标等于没有指标。
#: 2. **性质保守**：200 是**下限**而非目标。它回答的是"**能不能开始试训**"，
#:    不是"训出来就可信"。样本不足时训出的模型会**静默**给出看似正常的分数，
#:    比报错危险，故这里宁可让人看到"还差很多"。
#:
#: ⚠️ **原表述的错误（保留记录）**：旧注释写"模型有 8 维输入 × 2 隐层
#: (16,16) × 3 输出 ≈ 400 个参数"，并称"取 200 作为'至少是同量级'的下界"。
#: 两处不准：
#:
#: - **数字张冠李戴**：403 个参数对应的是 **4 维**输入（落盘权重的实际配置），
#:   不是 8 维（8 维是 **467**）。
#: - **"同量级"不成立**：200 = **0.50×** 参数量（4 维口径），是参数量的一半，
#:   不是"同量级"。真正的同量级下界应是 403 本身。
#:
#: 现状是"**明知低于经验法则而取之**"，这与旧注释暗示的"已满足同量级"
#: 是两回事——后者会让读者以为 200 已足够可信。
#:
#: **何时该改大**：当可信样本数接近 2000 时，应把它抬到
#: ``SAMPLES_PER_PARAMETER × count_mlp_parameters()``。
MIN_SAMPLES_FOR_TRAINING = 200

#: 按经验法则"够用"的样本数（``5 × 参数量``），供报告显示差距用。
#:
#: **为什么单列**：``MIN_SAMPLES_FOR_TRAINING`` 是"能不能开始试训"的**下限**，
#: 而这个是"训出来**可信**"的**目标**。两者差一个数量级（200 vs 2015），
#: 报告里应当**同时**给出，免得读者把"达标"误读为"够了"。
TARGET_SAMPLES_FOR_TRAINING = SAMPLES_PER_PARAMETER * count_mlp_parameters()

#: 认为"类别分布有判别力"的最小占比。单一类别占比超过它就报警。
#:
#: 理由：argmax 一致率在类别严重不平衡时无判别力——测试集只有 Move #3 时，
#: "永远猜 Move #3"也能拿满分。这是本项目已记录的实证（`eval_surrogate.py`
#: 在 2 样本场景下正确输出"⚠ 测试集只有一种 Move"）。
DOMINANT_CLASS_SHARE_WARN = 0.8

#: 充足性判定的三态取值（与 ``proof_monitor.HealthVerdict`` 同构）。
SUFFICIENCY_SUFFICIENT = "sufficient"
SUFFICIENCY_INSUFFICIENT = "insufficient"
SUFFICIENCY_UNKNOWN = "unknown"


def assess_sample_sufficiency(
    samples: Sequence[Tuple[Dict[str, float], Dict[int, float]]],
    *,
    min_samples: int = MIN_SAMPLES_FOR_TRAINING,
) -> Dict[str, Any]:
    """评估一批训练样本是否足以训练可信的 surrogate。

    **为什么需要它**：管道此前能算出"样本数 = 2"并如实打印，但**没有任何地方
    判断这个数够不够**。于是 2 个样本会一路走到训练、静默产出一个过拟合的模型，
    而下游看到的是"训练成功 + 损失下降"——**整条链路上没有任何信号**。

    三态判定（对齐 ``proof_monitor`` 的 ``unknown`` 语义）：

    - ``sufficient``：样本数达标**且**类别分布不单一；
    - ``insufficient``：样本数不足，或类别严重失衡（给出**还差多少**）；
    - ``unknown``：样本为空——不是"不足"而是"没有数据"，处置不同
      （前者要去攒数据，后者要先确认管道接对了）。

    Parameters
    ----------
    samples : sequence of (state_dict, target_dict)
        ``proof_steps_to_samples`` 的产出。
    min_samples : int
        充足所需的最小样本数，默认 :data:`MIN_SAMPLES_FOR_TRAINING`。

    Returns
    -------
    dict，键：

    - ``verdict``：上述三态之一；
    - ``n_samples``：样本数；
    - ``class_counts``：各 Move 被标为 10.0 的次数（``{1: n1, 2: n2, 3: n3}``）；
    - ``reasons``：人类可读的原因列表（空 = 无问题）；
    - ``shortfall``：距 ``min_samples``（**试训下限**）还差多少；
    - ``target_shortfall``：距 ``TARGET_SAMPLES_FOR_TRAINING``
      （**经验法则够用**，5× 参数量）还差多少；
    - ``target_samples``：上述目标值本身。

    ⚠️ **``shortfall`` 与 ``target_shortfall`` 是两个不同的量**，别混：
    前者判"能不能开始试训"（200），后者判"训出来可不可信"（2015）。
    两者差一个数量级，只报前者会让 ``sufficient`` 被误读为"够了"。
    """
    n = len(samples)
    class_counts = {1: 0, 2: 0, 3: 0}
    stray_keys: List[Any] = []
    for _state, target in samples:
        # target 里 10.0 的那个是"实际执行的动作"。
        for move_id, score in target.items():
            if score <= 5.0:
                continue
            # **只认整数键**：导出的 JSON 会把键写成字符串（``"1"``），
            # 不做还原就会混进一个新"类别"，使 1/2/3 的计数全为 0
            # ——静默给出全零分布。这里显式区分并记录，由调用方还原
            # （见 ``scripts/surrogate_data_from_traces._as_sample_pairs``）。
            if isinstance(move_id, int) and not isinstance(move_id, bool):
                class_counts[move_id] = class_counts.get(move_id, 0) + 1
            else:
                stray_keys.append(move_id)

    reasons: List[str] = []
    shortfall = max(0, min_samples - n)
    # 距"经验法则够用"（5× 参数量）还差多少。**与 ``shortfall`` 分开**：
    # 前者是"能不能开始试训"的下限，后者是"训出来可信"的目标，
    # 两者差一个数量级（200 vs 2015），混为一谈会让读者把"达标"误读为"够了"。
    target_shortfall = max(0, TARGET_SAMPLES_FOR_TRAINING - n)

    if stray_keys:
        reasons.append(
            f"target 的键不是整数：{sorted(set(map(str, stray_keys)))[:5]}。"
            "多为从 JSON 读回的样本——调用方须先 ``int(k)`` 还原键"
            "（否则类别统计会静默错位）。"
        )

    if n == 0:
        # 与 insufficient 分开：没有数据 ≠ 数据不够。
        return {
            "verdict": SUFFICIENCY_UNKNOWN,
            "n_samples": 0,
            "class_counts": class_counts,
            "reasons": reasons + [
                "没有任何训练样本——先确认数据管道接对了（轨迹目录、字段名），"
                "而不是直接去攒更多数据。"
            ],
            "shortfall": shortfall,
            "target_shortfall": target_shortfall,
            "target_samples": TARGET_SAMPLES_FOR_TRAINING,
        }

    if n < min_samples:
        reasons.append(
            f"样本数 {n} < 阈值 {min_samples}（还差 {shortfall}）——"
            f"模型有 {count_mlp_parameters()} 个参数，该阈值仅为"
            "'能否开始试训'的**下限**（低于经验法则的 5× 参数量）；"
            "样本不足时训出的模型会**静默**给出看似正常的分数。"
        )
    # ⚠️ **"已达下限但未达目标"不进 ``reasons``**（2026-09-30 修正）。
    #
    # 初版把它 append 进 ``reasons``，立刻打红了两条既有测试：
    # ``reasons == []`` 是**既有的对外契约**（"没有原因 = 一切正常"），
    # 而"低于经验法则目标"**不是一个问题**——它是**信息**：
    # ``sufficient`` 依然成立（可以试训），只是别把"可以试训"读成"够了"。
    #
    # 混进 ``reasons`` 会让 ``verdict`` 也跟着变成 insufficient，
    # 等于把"信息"降级成"故障"。故这里只填 ``target_shortfall`` 字段，
    # 由渲染层（``format_sufficiency_report``）决定怎么展示。

    # 类别分布：只在实际使用的类别上判断（全 0 的情况已在上面处理）。
    used = {k: v for k, v in class_counts.items() if v > 0}
    if len(used) == 1:
        only = next(iter(used))
        reasons.append(
            f"所有样本都是 Move #{only}——此时任何指标（argmax 一致率）都无判别力。"
        )
    elif used:
        total_used = sum(used.values())
        top_move, top_count = max(used.items(), key=lambda kv: kv[1])
        share = top_count / total_used
        if share > DOMINANT_CLASS_SHARE_WARN:
            reasons.append(
                f"类别严重失衡：Move #{top_move} 占 {share:.0%}"
                f"（>{DOMINANT_CLASS_SHARE_WARN:.0%}）——需补充其他 Move 的样本。"
            )

    verdict = SUFFICIENCY_SUFFICIENT if not reasons else SUFFICIENCY_INSUFFICIENT
    return {
        "verdict": verdict,
        "n_samples": n,
        "class_counts": class_counts,
        "reasons": reasons,
        "shortfall": shortfall,
        "target_shortfall": target_shortfall,
        "target_samples": TARGET_SAMPLES_FOR_TRAINING,
    }


def format_sufficiency_report(assessment: Dict[str, Any]) -> str:
    """把 :func:`assess_sample_sufficiency` 的结论渲染成多行文本。"""
    verdict = assessment["verdict"]
    n = assessment["n_samples"]
    counts = assessment["class_counts"]
    icon = {
        SUFFICIENCY_SUFFICIENT: "✓",
        SUFFICIENCY_INSUFFICIENT: "⚠",
        SUFFICIENCY_UNKNOWN: "?",
    }.get(verdict, "?")
    lines = [
        "",
        f"{icon} 样本充足性：{verdict}（{n} 个样本）",
        f"   类别分布：Move #1={counts.get(1, 0)}, "
        f"#2={counts.get(2, 0)}, #3={counts.get(3, 0)}",
        # 把两个门缝写在同一行：读者一眼能看出"试训下限 200"与
        # "经验法则够用 2015"不是一个量级（否则会把 sufficient 读成够了）。
        f"   门槛：试训下限 {MIN_SAMPLES_FOR_TRAINING} 个；"
        f"经验法则够用 {assessment.get('target_samples', TARGET_SAMPLES_FOR_TRAINING)} 个"
        f"（{SAMPLES_PER_PARAMETER}× {count_mlp_parameters()} 参数）",
    ]
    for r in assessment["reasons"]:
        lines.append(f"   · {r}")
    # **"可达下限但未达目标"在此处显示**（不在 ``reasons`` 里，见
    # ``assess_sample_sufficiency`` 的注释）：它是**信息**而非**问题**——
    # ``verdict`` 仍是 ``sufficient``，可开始试训；但不写这句，
    # 读者会把 ``sufficient`` 当成"够了"。
    if (
        verdict == SUFFICIENCY_SUFFICIENT
        and assessment.get("target_shortfall", 0) > 0
    ):
        lines.append(
            f"   ℹ 但距经验法则目标 {assessment['target_samples']} 个"
            f"还差 {assessment['target_shortfall']} 个——"
            "可开始试训，泛化能力预期有限。"
        )
    if verdict == SUFFICIENCY_SUFFICIENT:
        lines.append("   可以用于训练。")
    return "\n".join(lines)

