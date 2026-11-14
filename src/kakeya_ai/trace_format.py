r""".kpt —— 证明轨迹文件协议（Kakeya Proof Trace）。

**为什么需要它**：``ProofResult`` 只是内存里的对象；一旦只留下 ``trace_summary()``
的文本，第三方就无法复核——文本里没有几何输入，也没有内容指纹，改一个数字看不
出来。这个模块把轨迹落成**自包含、可独立复核**的文件：带上输入的管族、每一步的
状态指纹、以及哈希链，使"没验证的不入库"这条原则在文件层面也成立。

文件格式（JSON + 分隔符）
------------------------

总体是一个 UTF-8 文本文件，**不是**纯 JSON：文件头与数据体之间用一行分隔符分开，
读的人第一眼就能看到协议与参数，不必解析整个 JSON；追加步骤时也能流式写：

    KPT1
    {"protocol": "kpt/1", "created": "...", "params": {...}, "n_steps": 3,
     "failure_reason": "BUDGET_EXHAUSTED", "config": {"KAKEYA_MARCH": "x86-64-v2"}}
    ---
    {"kind": "geometry", "tubes": [...], "shading": {...}, "hash": "..."}
    {"kind": "step", "index": 0, "action": "BROADNESS_REDUCTION", ...}
    {"kind": "step", "index": 1, ...}
    ---
    {"n_steps": 3, "head": "<最后一步的哈希>", "file_hash": "..."}

- 第 1 行固定 ``KPT1``（协议魔数 + 版本），便于 ``head``/``file`` 命令识别。
- 第 2 行 header：参数指纹（epsilon/K/lambda_min/grid_n/…）与步数，
  以及**协议版本** ``protocol``（如 ``"kpt/1"``）。

**头部的两个"为什么跑成这样"字段**（均为可选新增，不抬协议版本）：

- ``failure_reason``：失败归因（``FailureReason`` 的枚举名）。此前只有
  ``success`` 一个布尔值，复核方分不清"预算用尽（加 max_iterations 可能就出
  结果）"与"判定性失败（这条路走不通）"——两者后续处置相反。
- ``config``：**被环境变量实际覆盖**的配置项。``params`` 是"显式指定的参数"，
  而环境变量（``KAKEYA_MARCH`` 等）同样影响结果；复核方拿同一份参数重跑却
  对不上时，第一个要排除的就是它。

两者缺省时读作空串 / 空字典，表示"生成时无此字段"，与"无失败 / 无覆盖"
是两回事（旧文件确实无从得知）。

**版本与演进**：``protocol`` 字段是格式版本的单一真源（不是单独再放一个
``schema_version`` —— 那是重复真源，两者一旦不同步就没有可信答案）。
读侧按 ``SUPPORTED_PROTOCOLS`` 校验：不认识的版本**直接拒绝并说明支持范围**，
绝不按当前规则去猜着读——误读一份旧文件的代价远大于拒绝它。允许的改动
（向后兼容的加字段 vs 必须抬版本号的不兼容改动）见 ``SUPPORTED_PROTOCOLS``
的注释。
- 第 3 行 ``---`` 分隔线。
- 之后每行一个 JSON 对象（JSONL）：首行 geometry，其余每步一行。
- 末尾再一条 ``---`` 分隔线 + footer，含哈希链的 head 与整文件哈希。

**哈希是精确的，不做取整**：浮点按 ``float.hex()`` 的十六进制形式参与哈希，
逐位精确（十进制 "0.1" 在不同机器上格式化可能不同，hex 不会）。
代价是**同一构型在不同机器/不同 numpy 版本下若产生不同的末位浮点，哈希就会不同**
——这是刻意的选择：宁可报"不一致"让人去看，也不要悄悄放过一个末位差异。

哈希链
------

    h_0 = H(geometry_hash ‖ step_0 的规范化内容)
    h_i = H(h_{i-1} ‖ step_i 的规范化内容)

篡改任意一步都会让它之后的所有 h_i 对不上，**能定位到具体哪一步**
（而不是只验一个整文件哈希、只告诉你"整个文件坏了"）。

**防护边界（如实说明）**：哈希链是**完整性**校验，不是**认证**。它防的是
意外损坏、传输错误、以及事后悄悄改一个数（没重算工具就会露馅）；它**防不住
有动机的攻击者**——没有密钥，任何人改动内容后都可以把哈希链与 footer 一起
重算，得到一个自洽的假文件。要抵抗这种场景必须上签名（HMAC/公钥），那是另一
层机制，本协议暂时不做，也不假装做到了。

**双重表示的一致性**：几何与指标各存两份——十进制（给人看，``round`` 到 17 位）
与 hex（供复核，逐位精确）。复核优先信 hex；但两份不一致本身就是异常
（人看到的与机器验的不是同一个数），所以 ``verify_trace`` 会单独报
``DECIMAL_HEX_MISMATCH``，而不是静默采用 hex 了事。

用法
----

    from kakeya_ai.trace_format import write_trace, read_trace, verify_trace

    write_trace(result, "docs/traces/run1.kpt", params={...})
    report = verify_trace("docs/traces/run1.kpt")   # 独立复核，不依赖 orchestrator
    print(report.summary())

``verify_trace`` 会**重新计算**每一步的体积下界指标并与文件比对。几何计算走
``accel.volume_voxel``（CPU），与生成时同一套内核——这是"解释器"的自我一致性
检查，不是形式化证明。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .errors import InputValidationError, TraceFormatError
from .primitives import Direction, Shading, Tube

if TYPE_CHECKING:
    from .proof_orchestrator import ProofMetrics, ProofResult

#: 协议魔数与版本。改格式时递增版本号，读到不认识的版本直接拒绝。
MAGIC = "KPT1"
PROTOCOL = "kpt/1"

#: 当前写入的**协议主版本号**（头部 ``protocol`` 字段的数值部分）。
#: 与包版本（``kakeya_ai.version.__version__``）**解耦**：包可以升到 0.5.0
#: 而协议仍是 1，只要文件格式没变。
PROTOCOL_VERSION = 1

#: **受支持的协议版本集合**。读侧据此判断能否处理一份文件。
#:
#: 演进规则（对外承诺，改动格式前必读）
#: ------------------------------------
#: 1. **向后兼容的改动不动版本号**：新增＊＊可选＊＊字段（如 ``params`` 里加一项、
#:    header 加一个可选键）只需在文档里注明，读侧对新字段做缺省处理即可。
#:    历史文件（``docs/traces/20260915_141647_demo.kpt``）没有 ``domain_radius``
#:    而仍能复核通过，就是这条规则在起作用。
#: 2. **不兼容的改动必须抬版本号**：改动已有字段的语义、删除字段、改变
#:    哈希链的计算口径，都要把 ``PROTOCOL`` 与 ``PROTOCOL_VERSION`` 一起递增，
#:    并把旧版本从本集合移除——**宁可直接拒绝，也不要按新规则误读旧文件**。
#: 3. **拒绝时要说清怎么办**：读到不认识的版本时抛出的错误里必须写明
#:    "本版本支持哪些协议"与"该文件的协议是什么"，让使用者能判断是自己
#:    的文件太旧、还是软件太旧。
#: 4. **不改哈希口径就不改版本**：纯文档/注释改动不影响文件字节语义。
SUPPORTED_PROTOCOLS: frozenset = frozenset({"kpt/1"})

#: 分隔线（header / body / footer 之间）。
SEP = "---"

#: 十进制表示的写法。**用 repr 而不是 round**：Python 的 ``repr(float)`` 保证
#: "最短且能往返"——同一个双精度值 write/read 后完全相等（round(1e-17, 17)
#: 仍是 1e-17，对小数毫无保留作用，照样与 hex 不一致）。
#: 所以十进制与 hex 两种表示对同一个值而言**应当逐位相等**，一致性检查才有意义。
_HUMAN_PRECISION = 17


def _dec(value: float) -> float:
    """十进制表示（可往返）：走一下 repr 再回 float，保证写出去的就是精确值。"""
    return float(repr(float(value)))


# ---------------------------------------------------------------------------
# 规范化与哈希
# ---------------------------------------------------------------------------


def _num(value: float) -> str:
    """把浮点规范成逐位精确的字符串（参与哈希）。

    用 ``float.hex()`` 而不是 ``repr()``：十进制表示（如 ``0.1``）在不同平台
    上可能有多种等价写法，hex 是 IEEE-754 的精确文本形式，跨平台一致。
    """
    f = float(value)
    # ``f != f`` 是 NaN 检测的**标准惯用法**（NaN 是唯一不等于自身的值），
    # 而 `_num` 的输入可能来自被破坏的文件，NaN 必须被识别并规范成 "nan"，
    # 否则 `f.hex()` 会产出 "nan" 之外的意外形态，破坏哈希稳定性。
    # 注：ruff 的 PLR0124（"与自身比较"）会盯上这行，但该规则不在本项目的
    # select 里，故无需 noqa。写清理由比加一条空转指令更能防止后来者
    # "顺手修掉"而引入真 bug。
    if f != f:
        return "nan"
    if f == float("inf"):
        return "inf"
    if f == float("-inf"):
        return "-inf"
    return f.hex()


def _hash_payload(payload: Any) -> str:
    """对任意 JSON 可序列化对象求 SHA-256（键按字典序，保证稳定）。"""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tube_hash(tube: Tube) -> str:
    """单根管的精确指纹。"""
    return _hash_payload({
        "center": [_num(c) for c in np.asarray(tube.center, dtype=float).ravel()],
        "direction": [_num(c) for c in np.asarray(tube.direction.v, dtype=float).ravel()],
        "length": _num(tube.length),
        "delta": _num(tube.delta),
    })


def geometry_hash(tubes: Sequence[Tube], shading: Optional[Shading] = None) -> str:
    """管族（+ 可选染色）的精确指纹。

    用整条哈希链而不是"逐管哈希再哈希"：后者需要额外的顺序约定（管序重排后
    几何相同却哈希不同）。这里按给定顺序逐管喂入，顺序因此成为指纹的一部分
    ——**管序变了就算不同构型**，这是刻意的（grains 的 ``tube_indices`` 依赖
    管序，重排会改变后续一切）。
    """
    h = hashlib.sha256()
    h.update(b"kakeya.geometry.v1\n")
    h.update(f"n_tubes={len(tubes)}\n".encode("utf-8"))
    for t in tubes:
        h.update(tube_hash(t).encode("ascii"))
        h.update(b"\n")
    if shading is not None:
        h.update(b"shading\n")
        for lam in np.asarray(shading.densities, dtype=float).ravel():
            h.update(_num(lam).encode("ascii"))
            h.update(b"\n")
    else:
        h.update(b"shading=None\n")
    return h.hexdigest()


def metrics_payload(metrics: "ProofMetrics") -> Dict[str, Any]:
    """``ProofMetrics`` → 参与哈希的规范化字典（浮点走 hex）。"""
    return {
        "delta": _num(metrics.delta),
        "rho": _num(metrics.rho),
        "a": _num(metrics.a),
        "b": _num(metrics.b),
        "c": _num(metrics.c),
        "local_error": _num(metrics.local_error),
        "volume_ratio": _num(metrics.volume_ratio),
        "union_volume": _num(metrics.union_volume),
        "lower_bound": _num(metrics.lower_bound),
        "n_tubes": int(metrics.n_tubes),
        "n_grains": int(metrics.n_grains),
        "is_sticky": bool(metrics.is_sticky),
        # 域半径：体积估计的盒 = [-r, r]³。必须在文件里，否则复核方
        # 会在另一个盒上重算（历史数值全按 r=1.0 算）。
        "domain_radius": _num(getattr(metrics, "domain_radius", 1.0)),
        # 该步用的网格数。与域同理：必须与记录值一致才能重算出同一个数。
        "grid_n": int(getattr(metrics, "grid_n", 0) or 0),
    }


def _failure_reason_name(result: Any) -> str:
    """取 ``result.failure_reason`` 的枚举名，供写入 trace 头部。

    兼容三种情形：属性缺失（旧调用点构造的鸭子类型对象）、属性为 ``None``
    （尚未归因）、属性是枚举（正常）。任一情况下都返回可写进 JSON 的字符串，
    绝不抛异常——写 trace 失败会让一次真实的证明搜索白跑。
    """
    reason = getattr(result, "failure_reason", None)
    if reason is None:
        return ""
    name = getattr(reason, "name", None)
    return str(name) if name is not None else str(reason)


def _proof_scope_name(result: Any) -> str:
    """取 ``result.scope`` 的枚举名，供写入 trace 头部。

    **为什么必须落盘**（2026-09-30，L1 数学闭合）：``success`` 是布尔，
    而它有两种完全不同的含义——"原输入成立"（``FULL``）与"只在 broad
    子集上成立"（``BROAD_ONLY``）。若 ``scope`` 不进文件，复核方打开 trace
    看到 ``success: true`` 就**只能**得出前一个结论，而那可能是假的。

    与 ``_failure_reason_name`` 同一兼容口径：属性缺失（旧鸭子类型对象）
    或为 ``None`` 时返回空串（"生成时无此记录"），绝不抛异常——写 trace
    失败会让一次真实的证明搜索白跑。
    """
    scope = getattr(result, "scope", None)
    if scope is None:
        return ""
    name = getattr(scope, "name", None)
    return str(name) if name is not None else str(scope)


def _uncovered_narrow_tubes(result: Any) -> int:
    """取被 broadness 剥离且未被覆盖的管数，供写入 trace 头部。

    **为什么必须落盘。** ``success`` 是 trace 头部的一等结论，而它只管 broad
    子集。若"管不到哪些管"这件事不写进文件，复核方打开 trace 看到
    ``success: true`` 就会当成对**原输入**成立——这正是缺口 ② 的存续方式。

    旧结果对象没有该属性时返回 ``0``（"生成时无此记录"）——与
    ``domain_radius``/``failure_reason`` 的兼容口径一致。注意 ``0`` 在此
    表示"未记录"，不等于"当时没有剥离"；旧 trace 需自行查 ``steps`` 里的
    ``BROADNESS_REDUCTION`` 步骤。
    """
    value = getattr(result, "uncovered_narrow_tubes", 0)
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _config_overrides() -> Dict[str, Any]:
    """返回**被环境变量实际覆盖**的配置项（只含覆盖项）。

    为什么只记覆盖项：默认值在所有机器上都一样，写进去没有信息量；而"某人
    把 ``KAKEYA_GRID_N`` 调成了 256"这类事实，恰恰是复核方在数值对不上时
    第一个要排除的变量。

    **任何失败都不应让导出失败**：写 trace 是证明搜索的唯一产出，配置读取
    出了问题（注册表损坏、环境变量类型非法）也不能连带把它丢掉。因此这里
    吞掉异常并返回带 ``_error`` 的字典——如实说明"配置没读到"，而不是
    假装"没有任何覆盖"。
    """
    try:
        from .config import snapshot

        snap = snapshot()
        overridden = set(snap.overridden)
        return {k: v for k, v in snap.as_dict().items() if k in overridden}
    except Exception as exc:
        # 裸 except 是**刻意**的（见 docstring：trace 导出不能被配置读取拖垮）。
        # ruff 的 BLE001 会盯上它，但该规则不在本项目的 select 里，
        # 故不加 noqa（无效指令会误导读者以为有豁免机制）。
        return {"_error": f"{type(exc).__name__}: {exc}"}


def metrics_human(metrics: "ProofMetrics") -> Dict[str, float]:
    """给人看的十进制表示（可往返，不参与哈希）。"""
    return {
        "delta": _dec(metrics.delta),
        "rho": _dec(metrics.rho),
        "a": _dec(metrics.a),
        "b": _dec(metrics.b),
        "c": _dec(metrics.c),
        "local_error": _dec(metrics.local_error),
        "volume_ratio": _dec(metrics.volume_ratio),
        "union_volume": _dec(metrics.union_volume),
        "lower_bound": _dec(metrics.lower_bound),
    }


def _tube_payload(tube: Tube) -> Dict[str, Any]:
    """管的 JSON 表达：十进制给人看，hex 供复核（两者逐位等价）。"""
    center = [float(c) for c in np.asarray(tube.center, dtype=float).ravel()]
    direction = [float(c) for c in np.asarray(tube.direction.v, dtype=float).ravel()]
    return {
        "center": [_dec(c) for c in center],
        "center_hex": [_num(c) for c in center],
        "direction": [_dec(c) for c in direction],
        "direction_hex": [_num(c) for c in direction],
        "length": _dec(tube.length),
        "length_hex": _num(tube.length),
        "delta": _dec(tube.delta),
        "delta_hex": _num(tube.delta),
    }


def _unit_direction(vec: np.ndarray) -> Direction:
    """从已归一化的向量构造 ``Direction``，**不做二次归一化**。

    ``Direction.__post_init__`` 会无条件 ``v / ||v||``。读文件时向量已经是单位
    向量（写入前归一化过），再除一遍自己那个"接近 1 但末位不是 1"的范数，
    会把末位翻转——实测 200 个随机方向里有 2 个中招（``||v||=1.0000000000000002``
    或 ``0.9999999999999999``）。后果是**写出去再读回来几何哈希就变了**，
    ``verify_trace`` 会误报"几何哈希不匹配"。

    这里绕过构造器直接写入 ``__dict__``（frozen dataclass 用 ``object.__setattr__``），
    保证读回的向量与写入时**逐位相同**。用 ``_validate_direction`` 做容错：
    文件里的向量若明显不是单位向量（被手改过），仍走正规构造器重新归一化。
    """
    arr = np.asarray(vec, dtype=float).copy()
    norm = float(np.linalg.norm(arr))
    if norm < 1e-12:
        raise InputValidationError("Direction vector must be non-zero.")
    # 与 1 的偏差在几个 ULP 之内 → 认为已是单位向量，原样保留（逐位精确）
    if abs(norm - 1.0) <= 1e-9:
        d = Direction.__new__(Direction)
        object.__setattr__(d, "v", arr)
        return d
    return Direction(arr)      # 明显不是单位向量：交给构造器归一化


def _tube_from_payload(payload: Dict[str, Any]) -> Tube:
    """从 JSON 还原管——**优先用 hex**，保证逐位精确。"""
    def vec(key: str) -> np.ndarray:
        if f"{key}_hex" in payload:
            return np.array([float.fromhex(h) for h in payload[f"{key}_hex"]])
        return np.asarray(payload[key], dtype=float)

    def scalar(key: str) -> float:
        if f"{key}_hex" in payload:
            return float.fromhex(payload[f"{key}_hex"])
        return float(payload[key])

    return Tube(
        center=vec("center"),
        direction=_unit_direction(vec("direction")),
        length=scalar("length"),
        delta=scalar("delta"),
    )


def _decimal_hex_mismatches(payload: Dict[str, Any], key: str) -> List[str]:
    """比对同一对象里 ``key``（十进制）与 ``key_hex``（十六进制）是否一致。

    两份表示是刻意存的：十进制给人读，hex 供复核。但**人看到的与机器验的
    必须是同一个数**——否则一个被改过的文件里，人读着 "0.5"、复核却按未改的
    hex 放行，就成了伪证。所以不一致要单独报出来，不能静默采用 hex。

    Returns
    -------
    list of str
        不一致项的说明（空列表 = 一致）。
    """
    # 入参可能来自被破坏的文件（payload 不是 dict、字段不是标量/数组）：
    # 统一返回"不一致"的说明，而不是抛 TypeError（那看起来像程序 bug）。
    if not isinstance(payload, dict):
        return [f"{key}：所在对象应为 dict，实际为 {type(payload).__name__}"]

    hex_key = f"{key}_hex"
    if hex_key not in payload or key not in payload:
        return []

    dec = payload[key]
    hx = payload[hex_key]
    out: List[str] = []

    def compare(dec_v: Any, hex_v: Any, where: str) -> None:
        if hex_v is None:
            return
        if isinstance(hex_v, list) and not isinstance(dec_v, list):
            out.append(f"{where}：hex 是数组但十进制不是（{type(dec_v).__name__}）")
            return
        if not isinstance(hex_v, list) and isinstance(dec_v, list):
            out.append(f"{where}：十进制是数组但 hex 不是（{type(hex_v).__name__}）")
            return
        if not isinstance(hex_v, list) and not isinstance(hex_v, str):
            out.append(f"{where}：hex 应为字符串或数组，实际为 {type(hex_v).__name__}")
            return
        try:
            # hex 字符串按 float.hex() 精确解析；十进制经 float() 再比较——
            # 两者都转成同一个 IEEE-754 双精度值比较，逐位精确。
            if isinstance(hex_v, list):
                for j, h in enumerate(hex_v):
                    d = dec_v[j] if isinstance(dec_v, list) and j < len(dec_v) else None
                    if d is None:
                        continue
                    # ``d`` 来自 ``Any`` 的下标访问，``float()`` 需要
                    # ``ConvertibleToFloat``。这里只接受数值与字符串（与
                    # 其余分支的宽容口径一致）：其余类型交给外层
                    # ``except TypeError`` 统一报"无法解析"。
                    if not isinstance(d, (int, float, str)):
                        out.append(
                            f"{where}[{j}]：十进制项应为数值或字符串，"
                            f"实际为 {type(d).__name__}"
                        )
                        continue
                    if float.fromhex(h) != float(d):
                        out.append(f"{where}[{j}]：十进制 {d!r} vs hex {h}")
            else:
                # 同上面的数组分支：``dec_v`` 是 ``Any``，``float()`` 需要
                # ``ConvertibleToFloat``。只接受数值与字符串，其余交给外层
                # ``except TypeError`` 统一报"无法解析"（与数组分支同一口径）。
                if not isinstance(dec_v, (int, float, str)):
                    out.append(
                        f"{where}：十进制应为数值或字符串，"
                        f"实际为 {type(dec_v).__name__}"
                    )
                    return
                if float.fromhex(hex_v) != float(dec_v):
                    out.append(f"{where}：十进制 {dec_v!r} vs hex {hex_v}")
        except (TypeError, ValueError):
            out.append(f"{where}：无法解析（dec={dec_v!r}, hex={hex_v!r}）")

    compare(dec, hx, key)
    return out


# ---------------------------------------------------------------------------
# 写入
# ---------------------------------------------------------------------------


def _step_payload(step: Any, prev_hash: str, index: int) -> Dict[str, Any]:
    """把一步转成 JSON 行，并给出它自己的 hash（哈希链的一环）。

    把**该步的域**也写进去（``domain_radius``）：域会在 NSP 抬升管族时变大，
    抬升前后的步骤各自算在不同大小的盒上。只记一个全局域的话，复核方会在
    错误的盒上重算早期步骤。域参与哈希，改它会被发现。
    """
    after_radius = float(getattr(step.after, "domain_radius", 1.0))
    before_radius = float(getattr(step.before, "domain_radius", after_radius))
    core = {
        "index": index,
        "iteration": int(step.iteration),
        "action": step.action.name,
        "before": metrics_payload(step.before),
        "after": metrics_payload(step.after),
        "notes": list(step.notes),
        # 状态变化参与哈希：复核方靠它重建中间管族，改它必须被发现。
        "state_change": getattr(step, "state_change", None),
        # Move 是谁选的（``SelectionSource`` 的枚举名）。参与哈希，因此
        # 事后把 "HEURISTIC" 改成 "LLM" 会被链校验发现——这一点重要：
        # 该字段的**唯一用途**就是统计"哪条路径覆盖了多少"，若可随意改写
        # 就没有任何结论价值。
        #
        # 用 ``getattr`` 取（旧 ``ProofStep`` 可能没有这个属性），缺省为空串，
        # 读侧据此区分"旧协议文件"与"明确的不适用"。
        "selection_source": getattr(
            getattr(step, "selection_source", None), "name", ""
        ),
        # 本步 Move 是否达成定量增益（缺口二）。三态：True / False / None
        # （None = 非 Move 步）。**参与哈希**：它是"这个 Move 到底动没动"的
        # 定量断言，事后把 False 改成 True 必须被链校验发现。
        # 用 ``getattr`` 取（旧 ``ProofStep`` 可能没有该属性），缺省 ``None``，
        # 读侧据"缺键"与"显式 null"区分——与 ``selection_source`` 同策略。
        "achieved_gain": getattr(step, "achieved_gain", None),
        # 本步 Move **实际执行的合并轮数**（缺口二补强）。三态：正整数 / 0 /
        # None（非 Move 步）。**参与哈希**：它是"这个 Move 动了几轮"的定量断言，
        # 事后把 1 改成 3 必须被链校验发现。旧 ``ProofStep`` 无该属性 ⇒ 缺省
        # ``None``，读侧据"缺键"与"显式 null"区分（与 ``achieved_gain`` 同策略）。
        "iters_used": getattr(step, "iters_used", None),
        # 逐步骤的域半径（也写进两侧 metrics_payload，这里显式一份便于人工检视）。
        "domain_radius_before": before_radius,
        "domain_radius_after": after_radius,
    }
    chained = {"prev": prev_hash, **core}
    core["prev_hash"] = prev_hash
    core["hash"] = _hash_payload(chained)
    # 人读的十进制近似另放，不参与哈希
    core["before_human"] = metrics_human(step.before)
    core["after_human"] = metrics_human(step.after)
    return core


def default_params(result: Any) -> Dict[str, Any]:
    """从 orchestrator 或 None 构造参数指纹。

    ``ProofResult`` 本身不记参数（它是运行**产物**），所以导出时若手边有
    orchestrator，把它传进来；否则用 ``params`` 显式给。缺失的参数不会
    伪造——如实留空并在 ``missing`` 里列出。
    """
    return {}


def write_trace(
    result: "ProofResult",
    path: str | Path,
    *,
    params: Optional[Dict[str, Any]] = None,
    initial_tubes: Optional[Sequence[Tube]] = None,
    shading: Optional[Shading] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """把一个 ``ProofResult`` 落成 ``.kpt`` 文件，返回写入路径。

    Parameters
    ----------
    result : ProofResult
        编排器跑出来的结果。
    path : str or Path
        目标路径（父目录会自动创建）。
    params : dict, optional
        参数指纹（``epsilon``/``K``/``lambda_min``/``grid_n``/``broadness_*``
        等）。**强烈建议给出**——复核方要靠它复现同样的体积估计。
    initial_tubes : sequence of Tube, optional
        初始管族。默认取 ``result.initial_tubes``。
    shading : Shading, optional
        初始染色。
    extra : dict, optional
        任意附加信息（会原样写进 header，但**不参与哈希**）。

    Returns
    -------
    Path
        实际写入的路径。
    """
    tubes = list(initial_tubes if initial_tubes is not None
                 else (result.initial_tubes or []))
    if not tubes:
        raise TraceFormatError(
            "无法导出可复核的 trace：缺少初始管族。"
            "请传 initial_tubes=...，或让编排器在 run() 时记录它。"
        )

    geo_hash = geometry_hash(tubes, shading)

    # 域半径写进 params：体积估计的盒 = [-r, r]³。基准 1.0（历史值）；
    # 管族被 NSP 副本带出基准域时会抬升为 2 的幂。复核方必须知道它，
    # 否则会在另一个盒上重算，数值对不上（见 domain.py 的三条不变量）。
    params = dict(params or {})
    params.setdefault("domain_radius", float(getattr(result, "domain_radius", 1.0)))

    # 随机种子：主循环里唯一的随机来源是 NSP 因子化，复核方必须用它重放副本，
    # 否则拿同样的参数也走不出同一条路径（见 ProofOrchestrator 的 seed 说明）。
    #
    # **键名用 ``run_seed`` 而不是 ``seed``**：两者含义不同，混用会让复核方
    # 误用种子。``seed`` 在既有 trace 里指**输入管族**的构造种子（由调用方
    # 传入，如 ``demo_proof_orchestrator.py`` 的 ``params={"seed": 7}``），
    # 它对编排器的 NSP 副本毫无作用；``run_seed`` 才是"这次证明搜索用的种子"，
    # 缺了它 NSP 步骤就重放不出来。
    #
    # 两种情况都**显式写出**，因为它们含义不同：
    #   - 整数：确定性运行，复核方可用它逐位重放 NSP 步骤。
    #   - ``null``：调用方显式选了非确定性探索模式（``seed=None``），
    #     此时 **NSP 步骤不可重放**，必须如实声明。
    # 为什么不干脆省略该字段：省略会被读成"协议早期的文件没有这个字段"，
    # 与非确定性是两回事。写 ``null`` 才是明确的可复现性声明。
    run_seed = getattr(result, "seed", None)
    params.setdefault("run_seed", None if run_seed is None else int(run_seed))

    header = {
        "protocol": PROTOCOL,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_steps": len(result.steps),
        "success": bool(result.success),
        # ``success`` 的**作用域**（枚举名，可选新增字段——不抬版本号）。
        #
        # 紧跟在 ``success`` 之后是**有意的**：这两个字段必须一起读。
        # 复核方打开 trace 第一眼看到的是 ``success``，若 ``scope`` 埋在
        # 十几行之后，等于没声明。``BROAD_ONLY`` 明确表示"这个 success 不是
        # 对原输入的断言"（论文中缺口由『小帽上的归纳』承接，本实现未实现）。
        "scope": _proof_scope_name(result),
        "exhausted": bool(result.exhausted),
        # 失败归因（枚举名）。**可选新增字段**（协议演进规则第 1 条）：
        # 旧读侧对它做缺省处理即可，故不抬版本号。用枚举名而不是数值，
        # 是为了让文件自解释——读一眼就知道是哪种失败。
        "failure_reason": _failure_reason_name(result),
        # ``success`` 的**作用域限制**：被 broadness 前置过滤剥离、且未被任何
        # 后续步骤覆盖的管数。同属"可选新增字段"（不抬版本号）。
        #
        # 必须与 ``success`` 一起读：``success=True`` 且本字段 > 0 时，
        # 正确读法是"下界在 broad 子集上成立"，而非"原输入成立"。
        "uncovered_narrow_tubes": _uncovered_narrow_tubes(result),
        "message": result.message,
        "geometry_hash": geo_hash,
        "params": params,
        # **生效的环境配置快照**。与 ``params`` 分开，因为两者性质不同：
        # ``params`` 是"这次运行被显式指定的参数"，``config`` 是"跑的时候
        # 环境里那些环境变量是什么值"。后者此前完全没记录——复核方拿同一份
        # 参数重跑，若环境变量不同（如 ``KAKEYA_GRID_N``、内存护栏阈值），
        # 结果可能对不上，却无从得知差异来自哪里。
        #
        # 只写**被环境覆盖过的**键（``overridden``）：默认值没有信息量，全写
        # 进去会让头部膨胀且掩盖真正需要关注的项。空字典 = 全部按默认跑。
        "config": _config_overrides(),
        # 结果摘要写进头部：**快速打开只需读前两行**就能看到结论，
        # 不必解析数据体（长 trace 可能几十 MB）。
        "final": {
            **metrics_human(result.final_metrics),
            "n_tubes": int(result.final_metrics.n_tubes),
            "n_grains": int(result.final_metrics.n_grains),
            "is_sticky": bool(result.final_metrics.is_sticky),
        },
        # 输入规模也放头部：看元数据就能知道"这份 trace 多大"。
        "n_tubes": len(tubes),
    }
    if extra:
        header["extra"] = dict(extra)

    lines: List[str] = [MAGIC, json.dumps(header, ensure_ascii=False), SEP]

    geometry_line = {
        "kind": "geometry",
        "n_tubes": len(tubes),
        "tubes": [_tube_payload(t) for t in tubes],
        "shading": (
            None if shading is None
            else [_dec(x) for x in shading.densities]
        ),
        "shading_hex": (
            None if shading is None
            else [_num(x) for x in shading.densities]
        ),
        "hash": geo_hash,
    }
    lines.append(json.dumps(geometry_line, ensure_ascii=False))

    prev = geo_hash
    for i, step in enumerate(result.steps):
        payload = _step_payload(step, prev, i)
        prev = payload["hash"]
        lines.append(json.dumps(payload, ensure_ascii=False))

    lines.append(SEP)
    body = "\n".join(lines)
    file_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()
    footer = {
        "n_steps": len(result.steps),
        "head": prev,
        "file_hash": file_hash,
    }
    # **footer 自身也要被保护**：整文件哈希只覆盖到它前面那条分隔线，
    # 所以 footer 里的 head / file_hash / n_steps 原本是"改了就没人发现"的
    # 裸字段（fuzz 800 轮抓到了 11 处：把 head 删掉、把 n_steps 改掉都能通过）。
    # 做法：给 footer 一个覆盖自身（不含本字段）的校验和。
    footer["footer_hash"] = _hash_payload(footer)
    lines.append(json.dumps(footer, ensure_ascii=False))
    lines.append("")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# 快速打开 / 快速导入
# ---------------------------------------------------------------------------


def _check_protocol(header: Mapping[str, Any]) -> None:
    """校验头部声明的协议版本是否受支持。

    两条判断，都是为了"宁可拒绝、不要误读"：

    - ``protocol`` 缺失或不是字符串 → 拒绝（元数据损坏，或根本不是本格式）。
    - 值不在 ``SUPPORTED_PROTOCOLS`` 中 → 拒绝，并同时报出"文件是什么版本"
      与"本程序支持哪些版本"，让使用者能判断是谁太旧。

    注意这里**不做版本号的数值大小比较**：只有"认识/不认识"两态。
    曾考虑按 ``kpt/2 < kpt/10`` 之类的数值顺序判断，但那样会鼓励"向前兼容地
    猜着读"——对一份定义了哈希链的文件，猜错的后果是静默产出错误的复核结论。
    """
    value = header.get("protocol")
    if not isinstance(value, str) or not value.strip():
        raise TraceFormatError(
            "头部缺少有效的 protocol 字段（应为字符串，如 "
            f"{PROTOCOL!r}）；实际为 {value!r}。"
            "该文件可能不是 .kpt，或头部已损坏。"
        )
    if value not in SUPPORTED_PROTOCOLS:
        supported = "、".join(sorted(SUPPORTED_PROTOCOLS))
        raise TraceFormatError(
            f"不支持的 .kpt 协议版本：{value!r}。"
            f"本程序支持：{supported}。"
            "若该文件由更新版本的工具生成，请升级本程序；"
            "若是旧文件，请用生成它的版本导出为受支持协议。"
        )


@dataclass
class TraceHeader:
    """**快速打开**的结果：只读文件头，不碰数据体。

    ``peek_trace`` 只读文件的前若干行，因此对几十 MB 的长 trace 也是常量开销
    （微秒级）。元数据（协议、参数、结论摘要、几何哈希、步数）全在头部，
    所以"看一眼这份 trace 是什么"不需要解析整个文件。
    """

    path: str
    protocol: str
    created: str
    n_steps: int
    n_tubes: int
    success: bool
    exhausted: bool
    message: str
    #: 失败归因（``FailureReason`` 的枚举名）。带默认值，且**必须**放在所有
    #: 无默认值字段之后——否则 dataclass 会因"非默认参数跟在默认参数后面"
    #: 直接报错。旧协议文件没有这个键，读出来是空串：那表示"生成时无此字段"，
    #: 与"这是一次无失败的运行"是两回事。
    failure_reason: str = ""
    #: ``success`` 的**作用域**（``ProofScope`` 的枚举名）。2026-09-30 新增。
    #:
    #: **必须与 ``success`` 一起读**：``FULL`` 才表示"对原输入成立"；
    #: ``BROAD_ONLY`` 表示"只在剥离后的 broad 子集上成立"（此时 ``success``
    #: 按不变量必为 ``False``）。旧协议文件没有这个键，读出来是空串：
    #: **表示"生成时无此字段"，不等于当时覆盖了全部输入**——与
    #: ``failure_reason``/``uncovered_narrow_tubes`` 同一兼容口径。
    scope: str = ""
    #: ``success`` 的**作用域限制**：被 broadness 剥离且未被覆盖的管数。
    #:
    #: ``> 0`` 表示"``success`` 只在 broad 子集上成立，不是对原输入"。
    #: 旧协议文件没有这个键，读出来是 ``0``——与 ``failure_reason``/``config``
    #: 同一兼容口径：**表示"生成时未记录"，不等于"当时没有剥离"**。
    uncovered_narrow_tubes: int = 0
    geometry_hash: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    #: **被环境变量覆盖**的配置项（空字典 = 全部按默认值跑）。
    #: 旧协议文件没有这个键，读出来是空字典——那表示"生成时未记录"，
    #: 而不等于"没有任何覆盖"（旧文件确实无从得知）。
    config: Dict[str, Any] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    #: 头部自身的哈希（与实际不符说明头部被改过）。需完整校验请用 ``verify_trace``。
    header_hash: str = ""

    def summary(self) -> str:
        lines = [
            f"KPT {self.protocol}  {Path(self.path).name}",
            f"  创建：{self.created}",
            f"  规模：{self.n_tubes} 根管 → {self.n_steps} 步",
            f"  结论：{'成功' if self.success else '未成功'}"
            + ("（预算用尽，非终止判定）" if self.exhausted else "")
            + f"  {self.message}",
        ]
        if self.failure_reason:
            lines.append(f"  失败归因：{self.failure_reason}")
        # 紧跟在结论之后：读者看到"成功"时就该知道它管不到哪些管。
        if self.uncovered_narrow_tubes > 0:
            lines.append(
                f"   ⚠ 作用域：本次下界仅覆盖 broad 子集——有 "
                f"{self.uncovered_narrow_tubes} 根窄簇管被剥离且未被覆盖，"
                "不是对原输入管族成立"
            )
        lines.append(f"  几何哈希：{self.geometry_hash[:16]}…")
        if self.final:
            vr = self.final.get("volume_ratio")
            if vr is not None:
                lines.append(f"  最终 volume_ratio：{vr:.6g}")
        if self.params:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(self.params.items()))
            lines.append(f"  参数：{shown}")
        if self.config:
            shown = ", ".join(f"{k}={v}" for k, v in sorted(self.config.items()))
            lines.append(f"  环境覆盖：{shown}")
        return "\n".join(lines)


def peek_trace(path: str | Path, max_header_lines: int = 8) -> TraceHeader:
    """**快速打开**：只读文件头，返回元数据而不解析数据体。

    对长 trace 至关重要——数据体可能有几十万行（每步一行），而
    "这是什么、结论如何、参数是什么"全在头两行。代价是 O(头部大小) 而不是
    O(文件大小)。

    Parameters
    ----------
    path : str or Path
         ``.kpt`` 文件。
    max_header_lines : int
        最多读多少行找 header（默认 8，实际只需要 2 行；留余量应对未来
        头部注释）。

    Raises
    ------
    ValueError
        不是 ``.kpt`` 文件，或头部无法解析。

    Notes
    -----
    这里对**字段值**坏了是宽容的（``n_steps`` 若是字符串就退回默认值，
    不抛异常——元数据局部损坏不该让"看一眼这是什么文件"彻底失效），
    但对**协议版本**是严格的：不认识的版本直接拒绝。

    两者不矛盾，判据是"能否安全地未知而行"：字段值坏了，退回默认值只是显示
    不准，调用方仍知道这是份 KPT 文件；而协议版本不认识意味着**后续字段的
    语义可能完全不同**，此时给出一份"看着像、其实按错误规则解释"的元数据，
    比拒绝危险得多。
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline()
        if first.startswith("\ufeff"):
            # **统一拒绝 BOM，不静默容忍**：BOM 会改变文件字节，整文件哈希
            # 因此判为"被改动"；如果只在某几个入口容忍，就会出现
            # "能快开、却读不了 / 验不过"的分裂行为。与其半容忍，不如明确拒绝
            # 并说清原因（提示如何修复）。
            raise TraceFormatError(
                "文件带 UTF-8 BOM：请去掉 BOM 后重试"
                "（如 `sed -i '' '1s/^\\xef\\xbb\\xbf//' 文件` 或另存为无 BOM）。"
                "BOM 会改变字节，整文件哈希会判它被改动过。"
            )
        if first.strip() != MAGIC:
            raise TraceFormatError(
                f"不是有效的 .kpt 文件：首行应为 {MAGIC!r}，实际为 {first[:40]!r}"
            )
        header_line = fh.readline()

    # 格式规定第 2 行就是 header。**不做"向后扫描找 JSON"的容错**：曾如此写过，
    # 结果 header 被破坏成数组（``[1,2,3]``）时，扫描会越过分隔线把 geometry
    # 行当作 header 读进来，静默产出一个半空的元数据对象（``protocol=''``、
    # ``n_tubes`` 却是 geometry 的值）。宁可报错，也不要给一份看着像、其实错的元数据。
    if header_line.strip() == SEP or not header_line.strip():
        raise TraceFormatError(
            "文件结构损坏：第 2 行应是 header，实际是分隔线或空行"
        )
    try:
        header = json.loads(header_line)
    except json.JSONDecodeError as exc:
        raise TraceFormatError(f"头部不是合法 JSON：{exc}") from exc
    if not isinstance(header, dict):
        raise TraceFormatError(
            f"头部应为 JSON 对象，实际为 {type(header).__name__}"
        )

    _check_protocol(header)

    # 头部哈希：覆盖 magic + header 两行（与 verify_trace 的整文件哈希不同口径，
    # 这里只保证"元数据没被单独改过"这个较弱但廉价的保证）。
    header_hash = hashlib.sha256(
        (first + header_line).encode("utf-8")
    ).hexdigest()

    def _as_int(key: str, default: int) -> int:
        value = header.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default      # 元数据坏了不该让"快速打开"崩掉

    def _as_str(key: str, default: str = "") -> str:
        value = header.get(key, default)
        return value if isinstance(value, str) else default

    def _as_dict(key: str) -> Dict[str, Any]:
        value = header.get(key)
        return dict(value) if isinstance(value, dict) else {}

    return TraceHeader(
        path=str(path),
        protocol=_as_str("protocol"),
        created=_as_str("created"),
        n_steps=_as_int("n_steps", 0),
        n_tubes=_as_int("n_tubes", -1),
        success=bool(header.get("success", False)),
        exhausted=bool(header.get("exhausted", False)),
        message=_as_str("message"),
        # 旧文件没有这个键 → 空串（表示"生成时无此字段"，不等于"无失败"）。
        failure_reason=str(header.get("failure_reason", "") or ""),
        # scope 是 2026-09-30 新增的可选字段：缺键读成空串（"生成时无此字段"）。
        scope=str(header.get("scope", "") or ""),
        # 旧文件没有这个键 → 0（同上："未记录"不等于"无剥离"）。
        uncovered_narrow_tubes=_as_int("uncovered_narrow_tubes", 0),
        geometry_hash=_as_str("geometry_hash"),
        params=_as_dict("params"),
        config=_as_dict("config"),
        final=_as_dict("final"),
        extra=_as_dict("extra"),
        header_hash=header_hash,
    )


def iter_steps(path: str | Path, *, limit: Optional[int] = None,
               offset: int = 0):
    """**快速导入（流式）**：逐步产出数据体里的 step，不把整个文件读进内存。

    适合"只要某几步"或"长搜索边写边看"的场景：读一行、解析一行、产出一行。
    geometry 行被跳过（它可能很大；需要管族请用 ``read_trace``）。

    Parameters
    ----------
    path : str or Path
        ``.kpt`` 文件。
    limit : int, optional
        最多产出多少步。
    offset : int
        跳过前多少步。

    Yields
    ------
    dict
        每步的 JSON 对象（含 ``action``/``before``/``after``/``hash`` 等）。

    Notes
    -----
    流式读**不做哈希校验**——校验需要前后文（哈希链），与"流式"的取舍相反。
    要校验请用 ``verify_trace``。

    **但协议版本校验照做**：流式读同样是"读这份文件"，若不校验版本，
    就会出现"``read_trace`` 拒绝、``iter_steps`` 照读"的分裂行为——同一份
    不认识的版本，一个入口说不能读、另一个入口给出数据。这里只解析 header
    这一行（O(1) 开销），不破坏流式特性。
    """
    path = Path(path)
    yielded = 0
    skipped = 0
    with path.open("r", encoding="utf-8") as fh:
        if fh.readline().strip() != MAGIC:
            raise TraceFormatError(f"不是有效的 .kpt 文件：{path}")

        header_line = fh.readline()
        try:
            header = json.loads(header_line)
        except json.JSONDecodeError:
            raise TraceFormatError(
                f"header 不是合法 JSON（文件可能被截断）：{path}"
            ) from None
        if not isinstance(header, dict):
            raise TraceFormatError(
                f"header 应为 JSON 对象，实际为 {type(header).__name__}"
            )
        _check_protocol(header)

        # 跳到第一条分隔线
        for line in fh:
            if line.strip() == SEP:
                break
        else:
            raise TraceFormatError("文件结构损坏：找不到数据体起始的分隔线")

        fh.readline()                       # geometry 行（跳过）

        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s == SEP:                    # 数据体结束
                break
            if not s.startswith("{"):
                continue
            if skipped < offset:
                skipped += 1
                continue
            if limit is not None and yielded >= limit:
                return
            try:
                yield json.loads(s)
            except json.JSONDecodeError:
                # 坏行不静默跳过——它是"文件损坏"的证据，必须让调用方知道。
                raise TraceFormatError(
                    f"第 {skipped + yielded} 步的 JSON 无法解析（文件可能损坏）"
                ) from None
            yielded += 1


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


@dataclass
class TraceFile:
    """解析后的 ``.kpt`` 内容（**未验证**，只是结构）。"""

    header: Dict[str, Any]
    tubes: List[Tube]
    shading: Optional[Shading]
    geometry_hash: str
    steps: List[Dict[str, Any]]
    footer: Dict[str, Any]
    #: body 的**原始字节**（魔数 → 末尾分隔线，含）。整文件哈希基于它，
    #: 因此 CRLF、行尾空格这类字节级改动也会被检出。
    raw_body_bytes: bytes = b""
    #: 几何行的原始 payload（含十进制与 hex 两套表示）。复核时要拿它
    #: 比对"人读到的数"与"机器验的数"是否一致，所以不能只留还原后的 Tube。
    geometry_payload: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def params(self) -> Dict[str, Any]:
        """返回参数字典；字段类型不对时返回空字典（**不抛异常**）。

        fuzz 发现 ``params`` 被改成 ``null``/数字时，``dict(None)`` 会抛
        ``TypeError: 'NoneType' object is not iterable``——元数据坏了不该让
        "读取"本身失败。
        """
        value = self.header.get("params")
        return dict(value) if isinstance(value, dict) else {}


def read_trace(path: str | Path) -> TraceFile:
    """读取 ``.kpt`` 文件（只解析结构，不做一致性校验——那是 ``verify_trace``）。"""
    path = Path(path)          # 后续要用 .read_bytes()，先规范化
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != MAGIC:
        raise TraceFormatError(
            f"不是有效的 .kpt 文件：首行应为 {MAGIC!r}，实际为 "
            f"{lines[0][:40]!r}" if lines else "空文件"
        )

    sep_idx = [i for i, ln in enumerate(lines) if ln.strip() == SEP]
    if len(sep_idx) < 2:
        raise TraceFormatError("缺少分隔线：格式应为 header --- body --- footer")

    head_end, body_end = sep_idx[0], sep_idx[-1]

    def _parse_json_line(text: str, what: str) -> dict:
        """解析一行 JSON，**并校验是对象**。

        不校验的话，``header`` 是数组时会在后面取 ``.get`` 时抛
        ``AttributeError``（应给 ``ValueError``）；行被截断时会抛
        ``JSONDecodeError`` 而未附上下文——对使用者来说等于"诡异失败"。
        """
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise TraceFormatError(
                f"{what} 不是合法 JSON（文件可能被截断或损坏）：{exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise TraceFormatError(
                f"{what} 应为 JSON 对象，实际为 {type(obj).__name__}"
            )
        return obj

    if head_end != 2:
        # 格式固定为 [MAGIC, header, SEP]，中间插任何行都是结构错误。
        raise TraceFormatError(
            f"结构损坏：第 3 行应为分隔线 {SEP!r}，实际在其之前还有 "
            f"{max(0, head_end - 2)} 行内容"
        )
    header = _parse_json_line(lines[1], "header")
    _check_protocol(header)
    body_lines = lines[head_end + 1: body_end]
    footer = (_parse_json_line(lines[body_end + 1], "footer")
              if body_end + 1 < len(lines) else {})

    # footer 之后必须是空行（允许文件以换行结尾）。多出的内容说明文件被追加过
    # ——fuzz 发现追加一行"假 footer"或一个 "{" 都会被完全忽略、verify 照样通过。
    # 那些字节不在整文件哈希覆盖范围内，所以必须在这里显式拒绝。
    trailing = [ln for ln in lines[body_end + 2:] if ln.strip()]
    if trailing:
        raise TraceFormatError(
            f"footer 之后还有 {len(trailing)} 行多余内容（文件被追加过）："
            f"{trailing[0][:60]!r}"
        )

    if not body_lines:
        raise TraceFormatError("数据体为空：至少应有一行 geometry")

    geo = _parse_json_line(body_lines[0], "geometry 行")
    if geo.get("kind") != "geometry":
        raise TraceFormatError(f"数据体首行应为 geometry，实际为 {geo.get('kind')!r}")

    # 字段类型必须显式校验：fuzz 发现不做校验时会抛 KeyError / TypeError
    # （``geo["tubes"]`` 缺失、或被改成 float/None），看起来像程序 bug，
    # 而实际是"文件内容不对"——应统一表达为 ValueError。
    raw_tubes = geo.get("tubes")
    if not isinstance(raw_tubes, list):
        raise TraceFormatError(
            f"geometry 行的 tubes 应为数组，实际为 {type(raw_tubes).__name__}"
        )
    bad = [i for i, tp in enumerate(raw_tubes) if not isinstance(tp, dict)]
    if bad:
        raise TraceFormatError(
            f"geometry 行的 tubes[{bad[0]}] 应为对象，实际为 "
            f"{type(raw_tubes[bad[0]]).__name__}"
        )
    tubes = []
    for i, tp in enumerate(raw_tubes):
        try:
            tubes.append(_tube_from_payload(tp))
        except (KeyError, TypeError, ValueError) as exc:
            raise TraceFormatError(
                f"geometry 行的 tubes[{i}] 无法解析为管：{exc}"
            ) from exc

    # shading 两个字段（十进制/hex）也必须是数组：fuzz 发现被换成 int 时
    # 会在迭代上抛 TypeError。
    for key in ("shading", "shading_hex"):
        value = geo.get(key)
        if value is not None and not isinstance(value, list):
            raise TraceFormatError(
                f"geometry 行的 {key} 应为数组或 null，实际为 "
                f"{type(value).__name__}"
            )
    shading = None
    if geo.get("shading_hex") is not None:
        raw = geo["shading_hex"]
        if not all(isinstance(h, str) for h in raw):
            bad = next(h for h in raw if not isinstance(h, str))
            raise TraceFormatError(
                f"geometry 行的 shading_hex 元素应为字符串，实际为 "
                f"{type(bad).__name__}"
            )
        try:
            shading = Shading(np.array([float.fromhex(h) for h in raw]))
        except (ValueError, OverflowError) as exc:
            raise TraceFormatError(f"geometry 行的 shading_hex 无法解析：{exc}") from exc
    elif geo.get("shading") is not None:
        raw = geo["shading"]
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                   for x in raw):
            bad = next(x for x in raw
                       if not (isinstance(x, (int, float))
                               and not isinstance(x, bool)))
            raise TraceFormatError(
                f"geometry 行的 shading 元素应为数字，实际为 {type(bad).__name__}"
            )
        try:
            shading = Shading(np.asarray(raw, dtype=float))
        except (TypeError, ValueError) as exc:
            raise TraceFormatError(f"geometry 行的 shading 无法解析：{exc}") from exc

    steps: List[Dict[str, Any]] = []
    for k, ln in enumerate(body_lines[1:]):
        payload = _parse_json_line(ln, f"第 {k} 步")
        notes = payload.get("notes")
        if notes is not None and not isinstance(notes, list):
            raise TraceFormatError(
                f"第 {k} 步的 notes 应为数组，实际为 {type(notes).__name__}"
            )
        steps.append(payload)
    # 整文件哈希覆盖"魔数 → body 末尾的那条分隔线（含）"，与写入端一致：
    # 写入时 body 是 [MAGIC, header, SEP, geometry, steps..., SEP]，
    # 所以这里必须**含** lines[body_end] 那条 SEP，否则会少一个 "---"
    # （曾因此把干净文件的哈希算错，误报"文件被改动"）。
    #
    # **按实际字节算，不按 splitlines() 规范化后的文本**：文本规范化会把
    # CRLF、行首 \r、不换行空格等差异吃掉——那意味着"文件被改过"却验不出来。
    # 早先版本先 splitlines() 再 join，CRLF 版的同一份 trace 哈希不变（漏检）。
    # 做法：在 **bytes** 上按 b"\n" 切，取到 body 末尾那条分隔线为止。
    raw_bytes = path.read_bytes()
    raw_lines = raw_bytes.split(b"\n")
    if len(raw_lines) <= body_end:
        raise TraceFormatError("文件结构损坏：字节行数与文本行数不一致")
    raw_body_bytes = b"\n".join(raw_lines[: body_end + 1])

    return TraceFile(
        header=header,
        tubes=tubes,
        shading=shading,
        geometry_hash=geo.get("hash", ""),
        steps=steps,
        footer=footer,
        raw_body_bytes=raw_body_bytes,
        geometry_payload=geo,
    )


def _apply_state_change(
    current: List[Tube],
    change: Optional[Dict[str, Any]],
    step_index: int,
    report: "VerificationReport",
) -> Optional[List[Tube]]:
    """按 ``state_change`` 把管族推进到该步之后的样子。

    返回 None 表示推不动（缺记录或记录非法）——调用方会如实标 UNVERIFIED。
    支持四种操作，与 ``ProofOrchestrator._describe_state_change`` 一一对应：

    - ``keep``：不变。
    - ``subset``：取索引子集。
    - ``rescale``：逐根重建为新的 delta（center/direction/length 不变）。
    - ``replace``：直接用记录里的整份管族。
    """
    if change is None:
        return None
    op = change.get("op")

    if op == "keep":
        return list(current)

    if op == "subset":
        indices = change.get("indices")
        if not isinstance(indices, list):
            return None
        if any(not (0 <= int(j) < len(current)) for j in indices):
            report.notes.append(f"step {step_index}: subset 索引越界，无法重建")
            return None
        return [current[int(j)] for j in indices]

    if op == "rescale":
        # 优先用记录里的精确 delta（hex）；没有才退回倍率相乘。
        delta_hex = change.get("delta_hex")
        delta = change.get("delta")
        if isinstance(delta_hex, str):
            new_delta = float.fromhex(delta_hex)
        elif isinstance(delta, (int, float)) and delta > 0:
            new_delta = float(delta)
        else:
            factor = change.get("factor")
            if not isinstance(factor, (int, float)) or factor <= 0:
                return None
            return [
                Tube(center=t.center, direction=t.direction, length=t.length,
                     delta=t.delta * float(factor))
                for t in current
            ]
        return [
            Tube(center=t.center, direction=t.direction, length=t.length,
                 delta=new_delta)
            for t in current
        ]

    if op == "replace":
        tubes = change.get("tubes")
        if not isinstance(tubes, list):
            return None
        try:
            return [_tube_from_payload(tp) for tp in tubes]
        except (KeyError, ValueError, TypeError):
            report.notes.append(f"step {step_index}: replace 记录无法解析")
            return None

    return None


# ---------------------------------------------------------------------------
# 复核
# ---------------------------------------------------------------------------


@dataclass
class StepVerdict:
    """单步的复核结论。"""

    index: int
    action: str
    status: str                      #: 校验状态字段，其取值只能是 OK / MISMATCH / HASH_BROKEN / UNVERIFIED 中的一种
    detail: str = ""
    recomputed: Optional[Dict[str, float]] = None


@dataclass
class VerificationReport:
    """整体复核报告。"""

    path: str
    ok: bool
    n_steps: int
    verdicts: List[StepVerdict] = field(default_factory=list)
    file_hash_ok: bool = True
    geometry_hash_ok: bool = True
    chain_ok: bool = True
    decimal_hex_ok: bool = True
    #: 元数据自洽（header/footer 声明的步数与实际一致）。
    metadata_ok: bool = True
    #: footer 的校验和是否通过（footer 不在整文件哈希覆盖范围内）。
    footer_ok: bool = True
    #: 这份 trace 的 **NSP 步骤能否重放**：
    #:
    #: - ``"seeded"``：``params.run_seed`` 是整数，副本可逐位重放。
    #: - ``"non-deterministic"``：``run_seed=null``，调用方选了探索模式。
    #: - ``"unknown"``：协议早期文件没有该字段，实际能力未知。
    #: - ``""``：尚未判定（手工构造 report 时）。
    #:
    #: **它不影响 ``ok``**：非确定性不是错误，只是能力受限。但复核方必须
    #: 能一眼看到限制，而不是以为拿到参数就能复现。
    reproducibility: str = ""
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        bad = [v for v in self.verdicts if v.status != "OK"]
        head = (f"KPT 复核：{'通过' if self.ok else '未通过'}  "
                f"（{self.n_steps} 步，{len(bad)} 处异常）")
        lines = [head]
        if not self.file_hash_ok:
            lines.append("  · 整文件哈希不匹配：文件被改动过")
        if not self.geometry_hash_ok:
            lines.append("  · 几何哈希不匹配：管族与 header/几何行不一致")
        if not self.decimal_hex_ok:
            lines.append("  · 十进制与 hex 表示不一致：人读到的数不是复核的数")
        if not self.footer_ok:
            lines.append("  · footer 校验失败：footer 字段被改动或缺失校验和")
        if not self.metadata_ok:
            lines.append("  · 元数据与内容不符：声明的步数不是实际的步数")
        if not self.chain_ok:
            lines.append("  · 哈希链断裂：某一步的内容与其 hash 对不上")
        # 可复现性单独一行：它不是"错误"，而是复核方必须知道的**能力限制**。
        if self.reproducibility == "non-deterministic":
            lines.append("  · 可复现性：run_seed=null（探索模式），NSP 步骤不可重放")
        elif self.reproducibility == "unknown":
            lines.append("  · 可复现性：未知（旧协议文件，无 run_seed 字段）")
        for v in bad:
            lines.append(f"  · step {v.index} [{v.action}] {v.status}: {v.detail}")
        for n in self.notes:
            lines.append(f"  · {n}")
        if self.ok:
            lines.append(
                "  注：通过只说明文件自洽（未被事后改动、重算一致），"
                "不是数学证明，也不防有动机的伪造（那需要签名）。"
            )
        return "\n".join(lines)


def verify_trace(
    path: str | Path,
    *,
    epsilon: float = 0.05,
    K: float = 2.0,
    lambda_min: float = 0.85,
    grid_n: int = 64,
    backend: str = "numpy",
    recompute: bool = True,
) -> VerificationReport:
    """独立复核一个 ``.kpt`` 文件。

    Parameters
    ----------
    path : str or Path
        ``.kpt`` 文件。
    epsilon, K, lambda_min, grid_n : float
        复核用的参数。默认取文件 header 里的 ``params``（若给出），
        否则用 Theorem 1.2 的默认值。
    backend : str
        体积估计后端，默认 ``"numpy"``（复核不需要加速，用最保守的实现）。
    recompute : bool
        是否重算每步的 ``volume_ratio``/``lower_bound``。关掉时只验哈希链
        （快，但只能证明"没被改动"，不能证明"算得对"）。

    Returns
    -------
    VerificationReport
    """
    from . import accel
    from .domain import bounds_for_radius
    from .volume import theorem_1_2_lower_bound

    path = Path(path)
    tf = read_trace(path)

    # header/footer 已是 dict（read_trace 保证）；但**字段值**可能是错的类型，
    # fuzz 发现 ``params`` 被换成 int 时会在 .get 上抛 AttributeError，
    # ``footer`` 被换成 bool 时会在迭代上抛 TypeError。逐处校验类型。
    if not isinstance(tf.header, dict):
        raise TraceFormatError("header 应为 JSON 对象")
    if not isinstance(tf.footer, dict):
        raise TraceFormatError("footer 应为 JSON 对象")

    # 先把 field-level 的类型问题挡在入口，后面的逻辑就不必到处防御。
    for key in ("n_steps", "n_tubes"):
        v = tf.header.get(key)
        if v is not None and not isinstance(v, int):
            raise TraceFormatError(
                f"header 的 {key} 应为整数，实际为 {type(v).__name__}"
            )
    for key in ("params",):
        v = tf.header.get(key)
        if v is not None and not isinstance(v, dict):
            raise TraceFormatError(
                f"header 的 {key} 应为对象，实际为 {type(v).__name__}"
            )

    params = tf.params()

    def _coerce(key: str, default, cast, *, positive: bool = False):
        """从 header 取参数并做类型/取值校验。

        非法值（如 ``grid_n`` 是字符串或负数）应给出**可读的错误**，而不是
        让它在深处抛 ``ValueError: Number of samples, -5, must be non-negative``
        ——使用者看到的是 numpy 的内部报错，不知道问题出在文件参数上。
        """
        if key not in params:
            return default
        raw = params[key]
        try:
            value = cast(raw)
        except (TypeError, ValueError) as exc:
            raise TraceFormatError(
                f"参数 {key}={raw!r} 无法转换为 {cast.__name__}"
            ) from exc
        if positive and value <= 0:
            raise InputValidationError(
                f"必须为正数，实际为 {raw!r}", field=key, value=raw,
            )
        return value

    epsilon = _coerce("epsilon", epsilon, float)
    K = _coerce("K", K, float)
    lambda_min = _coerce("lambda_min", lambda_min, float)
    grid_n = _coerce("grid_n", grid_n, int, positive=True)
    # 域半径：老文件（协议早期，没写这个字段）按历史基准 1.0 复核，
    # 这样已入库的 trace 仍能逐位对上；新文件则按其声明的域重算。
    domain_radius = _coerce("domain_radius", 1.0, float, positive=True)
    # 注意：不用在此处预先算 bounds——各步的域与 grid_n 可能不同（NSP 抬升后
    # 变大），重算时按**每一步自己**记录的 domain_radius / grid_n 取盒，
    # 见下方按步重算处的 bounds_for_radius(step_radius)。

    report = VerificationReport(path=str(path), ok=True, n_steps=tf.n_steps)

    # --- 0) 元数据自洽：声明步数 vs 实际步数 ---
    # ``peek_trace`` 报的是 header 里的**声明值**（那是它能读到的全部），
    # 而 ``read_trace`` 数的是**实际行数**。两者不一致说明文件被改过或写坏了，
    # 必须报出来——否则"快速打开"给出的规模是假的，而使用者无从知道。
    declared = tf.header.get("n_steps")
    if isinstance(declared, int) and declared != tf.n_steps:
        report.metadata_ok = False
        report.notes.append(
            f"header 声明 {declared} 步，实际 {tf.n_steps} 步（元数据与内容不符）"
        )
    footer_steps = tf.footer.get("n_steps")
    if footer_steps is not None and not isinstance(footer_steps, int):
        report.metadata_ok = False
        report.notes.append(
            f"footer 的 n_steps 应为整数，实际为 {type(footer_steps).__name__}"
        )
    if isinstance(footer_steps, int) and footer_steps != tf.n_steps:
        report.metadata_ok = False
        report.notes.append(
            f"footer 声明 {footer_steps} 步，实际 {tf.n_steps} 步"
        )

    # --- 0b) 可复现性声明：NSP 步骤能否重放 ---
    # 主循环唯一的随机来源是 NSP 因子化。``run_seed`` 是整数时，复核方能用它
    # 逐位重放副本；为 ``null`` 时**明确不可重放**（调用方选了探索模式）。
    # 这里**只报告、不判定失败**：非确定性运行本身不是错误，只是能力受限。
    # 关键是让复核方一眼看到限制，而不是以为"我拿到参数就能复现"。
    if "run_seed" not in params:
        report.reproducibility = "unknown"
        report.notes.append(
            "params 缺 run_seed 字段（协议早期文件）：NSP 步骤的可复现性未知，"
            "若该 trace 含 NSP_FACTORIZATION 步骤则无法重放副本"
        )
    elif params.get("run_seed") is None:
        report.reproducibility = "non-deterministic"
        report.notes.append(
            "run_seed=null：本次运行为非确定性探索模式，NSP 步骤不可重放"
        )
    else:
        report.reproducibility = "seeded"

    # --- 1) 整文件哈希（字节级）---
    actual_file_hash = hashlib.sha256(tf.raw_body_bytes).hexdigest()
    report.file_hash_ok = (actual_file_hash == tf.footer.get("file_hash"))

    # --- 1b) footer 自校验（footer 不在整文件哈希覆盖范围内）---
    recorded_footer_hash = tf.footer.get("footer_hash")
    if isinstance(recorded_footer_hash, str):
        bare_footer = {k: v for k, v in tf.footer.items() if k != "footer_hash"}
        if _hash_payload(bare_footer) != recorded_footer_hash:
            report.footer_ok = False
            report.notes.append(
                "footer 校验和不符：footer 的字段（head/file_hash/n_steps）被改动过"
            )
    else:
        # 老文件没有 footer_hash：如实标注（不假装它没问题）
        report.footer_ok = False
        report.notes.append(
            "footer 缺 footer_hash 字段（旧协议文件）：footer 内容无法校验"
        )

    # --- 2) 几何哈希（重算）---
    recomputed_geo = geometry_hash(tf.tubes, tf.shading)
    recorded_geo = tf.geometry_hash if isinstance(tf.geometry_hash, str) else ""
    report.geometry_hash_ok = (recorded_geo == recomputed_geo)
    if not isinstance(tf.geometry_hash, str):
        # header 的 geometry_hash 被改成非字符串：切片会抛 KeyError/TypeError
        report.notes.append(
            f"header 的 geometry_hash 应为字符串，实际为 "
            f"{type(tf.geometry_hash).__name__}"
        )
    elif not report.geometry_hash_ok:
        report.notes.append(
            f"几何哈希：文件 {recorded_geo[:16]}… vs 重算 {recomputed_geo[:16]}…"
        )

    # --- 2b) 十进制与 hex 表示一致性 ---
    # 人看的十进制与机器验的 hex 必须是同一个数；不一致就单独报，不能静默信 hex。
    mismatch_notes: List[str] = []
    raw_tubes_payload = tf.geometry_payload.get("tubes", [])
    if not isinstance(raw_tubes_payload, list):
        mismatch_notes.append(
            f"geometry 行的 tubes 应为数组，实际为 {type(raw_tubes_payload).__name__}"
        )
        raw_tubes_payload = []
    for i, tp in enumerate(raw_tubes_payload):
        for key in ("center", "direction", "length", "delta"):
            for msg in _decimal_hex_mismatches(tp, key):
                mismatch_notes.append(f"geometry tube[{i}] {msg}")
    for i, payload in enumerate(tf.steps):
        if not isinstance(payload, dict):
            continue
        for side in ("before_human", "after_human"):
            human = payload.get(side)
            hexside = payload.get(side.replace("_human", ""))
            # 类型必须两边都是对象：fuzz 发现被换成 int/bool 时
            # 会在 .items()/.get() 上抛 AttributeError（看似程序 bug，
            # 实为文件内容不对）。
            if not isinstance(human, dict) or not isinstance(hexside, dict):
                if human is not None and not isinstance(human, dict):
                    mismatch_notes.append(
                        f"step {i} {side}：应为对象，实际为 {type(human).__name__}"
                    )
                continue
            for key, dec_v in human.items():
                hx = hexside.get(key)
                if hx is None or not isinstance(hx, str):
                    continue
                try:
                    if float.fromhex(hx) != float(dec_v):
                        mismatch_notes.append(
                            f"step {i} {side}.{key}：十进制 {dec_v!r} vs hex {hx}"
                        )
                except (TypeError, ValueError):
                    mismatch_notes.append(f"step {i} {side}.{key}：无法解析")
    if mismatch_notes:
        report.decimal_hex_ok = False
        report.notes.extend(mismatch_notes[:5])
        if len(mismatch_notes) > 5:
            report.notes.append(f"… 另有 {len(mismatch_notes) - 5} 处十进制/hex 不一致")

    # --- 3) 哈希链 ---
    prev = tf.geometry_hash
    broken: set[int] = set()
    for i, payload in enumerate(tf.steps):
        if not isinstance(payload, dict):
            report.chain_ok = False
            broken.add(i)
            report.verdicts.append(StepVerdict(
                index=i, action="?",
                status="HASH_BROKEN",
                detail=f"该步应为 JSON 对象，实际为 {type(payload).__name__}",
            ))
            prev = ""
            continue
        if payload.get("prev_hash") != prev:
            report.chain_ok = False
            broken.add(i)
            # prev 可能不是字符串（前一步的 hash 字段被改坏）：切片会抛
            # TypeError/KeyError。做成可读的提示而不是崩。
            prev_show = prev[:16] if isinstance(prev, str) else repr(prev)[:16]
            report.verdicts.append(StepVerdict(
                index=i, action=payload.get("action", "?"),
                status="HASH_BROKEN",
                detail=f"prev_hash 不匹配（应为 {prev_show}…）",
            ))
            prev = payload.get("hash", "")
            continue
        # 用 .get 而不是下标：文件被剪掉字段时要给"哈希链断裂"这类明确结论，
        # 而不是 KeyError（曾因缺 after 直接抛 KeyError，看起来像程序崩了）。
        missing = [k for k in ("prev_hash", "index", "iteration", "action",
                               "before", "after", "notes")
                   if k not in payload]
        if missing:
            report.chain_ok = False
            broken.add(i)
            report.verdicts.append(StepVerdict(
                index=i, action=payload.get("action", "?"),
                status="HASH_BROKEN",
                detail=f"该步缺少字段 {missing}（文件被裁剪或损坏）",
            ))
            prev = payload.get("hash", "")
            continue
        chained = {
            "prev": payload["prev_hash"],
            "index": payload["index"],
            "iteration": payload["iteration"],
            "action": payload["action"],
            "before": payload["before"],
            "after": payload["after"],
            "notes": payload["notes"],
            "state_change": payload.get("state_change"),
        }
        # 逐步骤的域半径与网格数：**新文件**由 _step_payload 写入并参与哈希；
        # **老文件**没有这些字段（协议早期）。写入端与复核端必须用同一套字段，
        # 否则干净文件会被误判"哈希链断裂"（我曾只改了一侧，7 个测试因此失败）。
        # 判据取"文件里有没有"：有就必须匹配，没有就两边都不算。
        for extra_key in ("domain_radius_before", "domain_radius_after",
                          "selection_source", "achieved_gain", "iters_used"):
            if extra_key in payload:
                chained[extra_key] = payload[extra_key]
        expected = _hash_payload(chained)
        if expected != payload.get("hash"):
            report.chain_ok = False
            broken.add(i)
            report.verdicts.append(StepVerdict(
                index=i, action=payload.get("action", "?"),
                status="HASH_BROKEN",
                detail="内容与自身 hash 不一致（该步被改动）",
            ))
        prev = payload.get("hash", "")

    if tf.footer.get("head") and tf.footer["head"] != prev:
        report.chain_ok = False
        report.notes.append("footer 的 head 与最后一步的 hash 不一致")

    # --- 4) 重算每步指标（靠 state_change 重建中间管族）---
    if recompute:
        current = list(tf.tubes)
        for i, payload in enumerate(tf.steps):
            action = payload.get("action", "?")
            if i in broken:
                continue    # 该步哈希已坏，重算无意义（已有 HASH_BROKEN 结论）

            after = payload.get("after", {})
            lb_hex = after.get("lower_bound")
            uv_hex = after.get("union_volume")
            if lb_hex is None or not uv_hex:
                report.verdicts.append(StepVerdict(
                    index=i, action=action, status="UNVERIFIED",
                    detail="该步缺少 union_volume/lower_bound，无法重算",
                ))
                continue

            # 先按 state_change 把管族推进到"该步之后"的样子；推不动就如实报。
            change = payload.get("state_change")
            advanced = _apply_state_change(current, change, i, report)
            if advanced is None:
                report.verdicts.append(StepVerdict(
                    index=i, action=action, status="UNVERIFIED",
                    detail=("该步缺少可重建的状态变化记录（state_change），"
                            "无法重建中间管族"),
                ))
                continue
            current = advanced

            # 用**该步自己的域与网格**重算：域会在 NSP 抬升管族时变大，
            # grid_n 随之抬到 2 的幂，每步的参数可能不同。老文件没写这些字段
            # → 回退到全局值（历史基准：r=1.0、文件参数里的 grid_n）。
            step_radius = domain_radius
            step_grid_n = grid_n
            after_payload = payload.get("after")
            if isinstance(after_payload, dict):
                raw_radius = after_payload.get("domain_radius")
                if isinstance(raw_radius, str):
                    try:
                        step_radius = float.fromhex(raw_radius)
                    except ValueError:
                        step_radius = domain_radius
                raw_grid = after_payload.get("grid_n")
                if isinstance(raw_grid, int) and raw_grid > 0:
                    step_grid_n = raw_grid
            step_bounds = bounds_for_radius(step_radius)

            recomputed_uv = float(accel.volume_voxel(
                current, grid_n=step_grid_n, backend=backend, bounds=step_bounds))
            recomputed_lb = float(theorem_1_2_lower_bound(
                current, epsilon=epsilon, K=K, lambda_min=lambda_min))
            recomputed_ratio = recomputed_uv / recomputed_lb if recomputed_lb > 0 else 0.0

            recorded_uv = float.fromhex(uv_hex)
            tol = 1e-12 * max(1.0, abs(recomputed_uv))
            if abs(recomputed_uv - recorded_uv) > tol:
                report.verdicts.append(StepVerdict(
                    index=i, action=action, status="MISMATCH",
                    detail=(f"union_volume 记录 {recorded_uv:.10g} "
                            f"vs 重算 {recomputed_uv:.10g}"),
                    recomputed={"union_volume": recomputed_uv,
                                "lower_bound": recomputed_lb,
                                "volume_ratio": recomputed_ratio},
                ))
            else:
                report.verdicts.append(StepVerdict(
                    index=i, action=action, status="OK",
                    recomputed={"union_volume": recomputed_uv,
                                "lower_bound": recomputed_lb,
                                "volume_ratio": recomputed_ratio},
                ))
    else:
        report.verdicts = [
            StepVerdict(index=i, action=p.get("action", "?"), status="OK")
            for i, p in enumerate(tf.steps)
        ]

    report.ok = (
        report.file_hash_ok
        and report.geometry_hash_ok
        and report.decimal_hex_ok
        and report.metadata_ok
        and report.footer_ok
        and report.chain_ok
        and all(v.status == "OK" for v in report.verdicts)
    )
    return report
