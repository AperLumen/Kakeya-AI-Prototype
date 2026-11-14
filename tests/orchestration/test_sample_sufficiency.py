"""样本充足性判据的守门测试。

**为什么需要它。** 缺口的形态不是"管道坏了"——实测管道完全可用（`docs/traces`
的 7 个 step 能正确转出样本、特征 8 维、目标标注正确）。真正的问题是：
**"样本数 = 2" 这个数字被算出来、被打印，但没有任何地方判断它够不够**
——2 个样本会一路走到训练、静默产出一个过拟合的模型，而下游看到的是
"训练成功 + 损失下降"。**整条链路上没有任何信号**。

本文件钉住三件事：

1. **三态语义**（``sufficient`` / ``insufficient`` / ``unknown``）——尤其是
   "没有数据"与"数据不够"必须分开（前者要先查管道，后者要去攒数据）。
2. **类别分布统计正确**——包含一个真实踩过的坑：导出的 JSON 把 target 键转成
   字符串，不做还原会把 ``'1'`` 当成新类别，统计出全零分布（静默错结论）。
3. **判据可复现**——阈值显式、不读环境、不看墙钟，同一输入任何机器同一结论。

变异验证（开发时手工做过）：把 ``n == 0`` 分支合并进 ``n < min_samples`` → 红；
把类别分布改成只数第一类 → 红。
"""

from __future__ import annotations

import pytest

from kakeya_ai.ai.training_data import (
    DOMINANT_CLASS_SHARE_WARN,
    MIN_SAMPLES_FOR_TRAINING,
    SUFFICIENCY_INSUFFICIENT,
    SUFFICIENCY_SUFFICIENT,
    SUFFICIENCY_UNKNOWN,
    assess_sample_sufficiency,
    build_training_data,
    format_sufficiency_report,
)


def _sample(move_id: int):
    """造一个 (state, target) 样本：``move_id`` 得 10 分，其余 5 分。"""
    target = {1: 5.0, 2: 5.0, 3: 5.0}
    target[move_id] = 10.0
    return ({"rho": 0.5}, target)


# ---------------------------------------------------------------------------
# 1. 三态语义
# ---------------------------------------------------------------------------


def test_empty_samples_report_unknown_not_insufficient():
    """**空样本必须报 ``unknown``，不是 ``insufficient``。**

    两者处置相反：``insufficient`` 意味着"管道通了、去攒更多数据"；
    ``unknown`` 意味着"根本没有数据——先查管道接对没有"。混为一谈会让人
    去漫无目的地攒数据，而实际问题可能是路径写错了。
    """
    a = assess_sample_sufficiency([])
    assert a["verdict"] == SUFFICIENCY_UNKNOWN
    assert a["n_samples"] == 0
    assert a["reasons"], "unknown 也必须给出原因（否则没人知道去查什么）"


def test_few_samples_report_insufficient_with_shortfall():
    """样本数不足时报 ``insufficient``，并给出**还差多少**。

    实测场景：``docs/traces/`` 只有 2 个可用样本，阈值 200 → 还差 198。
    """
    a = assess_sample_sufficiency([_sample(1), _sample(3)])
    assert a["verdict"] == SUFFICIENCY_INSUFFICIENT
    assert a["n_samples"] == 2
    assert a["shortfall"] == MIN_SAMPLES_FOR_TRAINING - 2
    assert any("198" in r or "样本数" in r for r in a["reasons"])


def test_single_class_reported_even_when_count_is_high():
    """**类别单一即使数量达标也要报问题**（数量够 ≠ 有判别力）。

    这是本项目已实证的一条：测试集只有一种 Move 时，argmax 一致率可以轻松
    是 1.0（"永远猜它"也满分）。故充足性不能只看样本数。
    """
    a = assess_sample_sufficiency([_sample(3)] * (MIN_SAMPLES_FOR_TRAINING + 50))
    assert a["verdict"] == SUFFICIENCY_INSUFFICIENT
    assert a["class_counts"][3] == MIN_SAMPLES_FOR_TRAINING + 50
    assert any("Move #3" in r for r in a["reasons"])


def test_dominant_class_reported():
    """某类占比超过阈值要报失衡。"""
    n = 300
    n_major = int(n * (DOMINANT_CLASS_SHARE_WARN + 0.1))
    samples = [_sample(1)] * n_major + [_sample(2)] * (n - n_major)
    a = assess_sample_sufficiency(samples)
    assert a["verdict"] == SUFFICIENCY_INSUFFICIENT
    assert any("失衡" in r for r in a["reasons"])


def test_sufficient_when_counts_and_distribution_ok():
    """数量达标且分布均衡 → ``sufficient``。"""
    samples = [_sample(1), _sample(2), _sample(3)] * 100
    a = assess_sample_sufficiency(samples)
    assert a["verdict"] == SUFFICIENCY_SUFFICIENT
    assert a["reasons"] == []
    assert a["shortfall"] == 0


# ---------------------------------------------------------------------------
# 2. 类别分布统计的正确性（含真实踩过的坑）
# ---------------------------------------------------------------------------


def test_class_counts_identify_the_executed_move():
    """类别计数必须数"实际执行的那个 Move"（target 里 10.0 的那个）。"""
    a = assess_sample_sufficiency([_sample(2), _sample(2), _sample(3)])
    assert a["class_counts"] == {1: 0, 2: 2, 3: 1}


def test_string_keys_are_not_counted_as_new_classes():
    """**target 键为字符串时不得统计错**（导出的 JSON 键就是字符串）。

    这是真实踩过的坑：数据导出脚本写 JSON 时键被转成字符串，若判定函数
    直接 ``target.items()`` 会把 ``'1'``/``'2'``/``'3'``
    当成 3 个新类别，于是既有整数键的计数全为 0——**静默给出全零分布**。

    本测试钉住"判定函数只认整数键"这一契约：键还原是调用方的责任，
    判定函数不做隐式转换（否则静默给出全零分布，比报错更糟）。
    """
    # 传字符串键：当前契约是判定函数**不**做键还原（那是调用方的责任），
    # 故这里断言"它不会把字符串键误当成合法类别"——即不计入 1/2/3。
    #
    # ``# type: ignore[arg-type]`` 是**负向用例的固有摩擦**：这条用例的全部
    # 意义就是"喂进不合规的输入，看它是否被正确拒绝"，故类型上必然不兼容。
    # 与 ``test_proof_monitor.test_invalid_on_step_is_rejected`` 同一处置
    # ——**不是**在掩盖缺陷（真缺陷不该被 ignore）。
    a = assess_sample_sufficiency(
        [({"x": 1.0}, {"1": 10.0, "2": 5.0, "3": 5.0})]  # type: ignore[arg-type]
    )
    assert a["class_counts"] == {1: 0, 2: 0, 3: 0}, (
        "字符串键被误当成了 Move 类别——调用方必须先 int(k) 还原键"
    )


# ---------------------------------------------------------------------------
# 3. 可复现与报告
# ---------------------------------------------------------------------------


def test_assessment_is_deterministic():
    """同一输入必须得出同一结论（不读环境、不看时间）。"""
    samples = [_sample(1), _sample(2), _sample(3)] * 100
    a1 = assess_sample_sufficiency(samples)
    a2 = assess_sample_sufficiency(samples)
    assert a1 == a2


def test_threshold_is_overridable_and_explicit():
    """阈值必须可显式覆盖（便于测试与不同规模的部署）。"""
    a = assess_sample_sufficiency([_sample(1), _sample(2)], min_samples=2)
    assert a["verdict"] == SUFFICIENCY_SUFFICIENT, (
        "min_samples 覆盖未生效——阈值写死了"
    )
    assert a["shortfall"] == 0


def test_report_mentions_verdict_and_counts():
    """报告文本要包含判定词与类别分布（便于日志直接读）。"""
    a = assess_sample_sufficiency([_sample(1)])
    text = format_sufficiency_report(a)
    assert SUFFICIENCY_INSUFFICIENT in text
    assert "Move #1=1" in text


# ---------------------------------------------------------------------------
# 4. 训练数据构建留痕
# ---------------------------------------------------------------------------


def test_build_training_data_records_class_distribution():
    """``build_training_data`` 的 meta 必须记录标签类别分布。

    理由：权重文件的 meta.json 是事后回答"这个模型当时用什么训的"的唯一依据。
    此前只记 ``label_source`` 与 ``n_states``，看不出类别是否单一。
    """
    _X, Y, meta = build_training_data(n_states=40, seed=0, use_llm_if_available=False)
    assert "label_class_counts" in meta, "meta 未记录标签类别分布"
    counts = meta["label_class_counts"]
    assert set(counts) <= {"1", "2", "3"}
    # 分布之和必须等于样本数（每行都有一个 argmax）
    assert sum(counts.values()) == Y.shape[0]


# ---------------------------------------------------------------------------
# 3. 阈值依据的正确性（2026-09-30 新增）
# ---------------------------------------------------------------------------
#
# **为什么单列一组**：原注释写"模型有 8 维输入 × 2 隐层 (16,16) × 3 输出
# ≈ 400 个参数"，并称"取 200 作为'至少是同量级'的下界"。实测发现两处不准：
#
# - **数字张冠李戴**：403 个参数对应 **4 维**输入（落盘权重的实际配置），
#   8 维是 **467**；
# - **"同量级"不成立**：200 = **0.50×** 参数量，是参数量的一半。
#
# 本组把"结构常量 ↔ 真实模型"和"阈值比例"都钉住——否则那位注释里的
# "≈400"会随模型改动继续漂移，而没人会注意到。


def test_default_architecture_matches_scorer():
    """结构常量必须与 ``SurrogateScorer`` 的**真实默认值**一致。

    ``training_data`` 刻意**不导入** ``surrogate_scorer``（后者带 numpy
    权重与推理逻辑，而本模块要能被"只算样本数"的轻量调用方使用），
    所以结构常量是**复制**过来的。复制就会漂移 ⇒ 用这条测试钉住。

    **不做的事**：不改成"导入真实默认值"——那会把 numpy 依赖拉进
    轻量路径，代价大于收益。重复 + 守门是本项目已验证的模式。
    """
    from kakeya_ai.ai.surrogate_scorer import SurrogateScorer

    from kakeya_ai.ai.training_data import (
        DEFAULT_HIDDEN_DIMS,
        DEFAULT_INPUT_DIM,
        DEFAULT_OUTPUT_DIM,
    )

    model = SurrogateScorer()
    assert model.input_dim == DEFAULT_INPUT_DIM, (
        f"SurrogateScorer 默认 input_dim={model.input_dim}，"
        f"而 training_data 里写的是 {DEFAULT_INPUT_DIM} —— 两处已漂移，"
        "请同步（并检查 MIN_SAMPLES_FOR_TRAINING 的依据是否需要跟着更新）"
    )
    assert tuple(model.hidden_dims) == tuple(DEFAULT_HIDDEN_DIMS)
    assert model.output_dim == DEFAULT_OUTPUT_DIM


def test_count_mlp_parameters_formula():
    """参数量公式：每层 ``in*out + out``（权重 + 偏置）。"""
    from kakeya_ai.ai.training_data import count_mlp_parameters

    # 4->16->16->3: (4*16+16) + (16*16+16) + (16*3+3) = 80+272+51 = 403
    assert count_mlp_parameters(4, (16, 16), 3) == 403
    # 8->16->16->3 = (8*16+16) + 272 + 51 = 144+272+51 = 467
    assert count_mlp_parameters(8, (16, 16), 3) == 467
    # 单层平凡情形
    assert count_mlp_parameters(2, (), 1) == 2 * 1 + 1


def test_target_samples_is_derived_not_hardcoded():
    """``TARGET_SAMPLES_FOR_TRAINING`` 必须是**算出来的**（5× 参数量）。

    **钉住的是"派生关系"而非"具体数值"**：数值会随模型结构变，派生关系
    不该变。若有人把它改成字面量，模型一改它就静默过期。
    """
    from kakeya_ai.ai.training_data import (
        SAMPLES_PER_PARAMETER,
        TARGET_SAMPLES_FOR_TRAINING,
        count_mlp_parameters,
    )

    assert TARGET_SAMPLES_FOR_TRAINING == (
        SAMPLES_PER_PARAMETER * count_mlp_parameters()
    )
    assert TARGET_SAMPLES_FOR_TRAINING == 2015  # 5 × 403


def test_min_samples_is_documented_as_below_rule_of_thumb():
    """``MIN_SAMPLES_FOR_TRAINING`` 必须**小于**经验法则目标。

    这条是**有意的方向性断言**：若哪天有人把它抬到 ≥ 目标值，
    说明"下限"与"目标"的区分已经消失——那时要么更新本测试的理由，
    要么承认阈值已被当成目标使用（那是另一种设计，需明说）。
    """
    from kakeya_ai.ai.training_data import (
        MIN_SAMPLES_FOR_TRAINING,
        SAMPLES_PER_PARAMETER,
        TARGET_SAMPLES_FOR_TRAINING,
        count_mlp_parameters,
    )

    assert MIN_SAMPLES_FOR_TRAINING < TARGET_SAMPLES_FOR_TRAINING
    ratio = MIN_SAMPLES_FOR_TRAINING / count_mlp_parameters()
    assert ratio < 1.0, (
        f"阈值 {MIN_SAMPLES_FOR_TRAINING} 是参数量的 {ratio:.2f}× —— "
        "旧注释曾称这是'同量级'，但 <1× 显然不是。若确实改到了 ≥1×，"
        "请同步更新 MIN_SAMPLES_FOR_TRAINING 的说明文字。"
    )
    # 记录现状：200 / 403 = 0.496（约为参数量的一半）。
    assert ratio == pytest.approx(0.496, abs=0.01)
    assert SAMPLES_PER_PARAMETER == 5


def test_sufficient_does_not_require_reaching_target():
    """**过了下限即为 ``sufficient``**，不要求达到经验法则目标。

    这条钉住"下限"与"目标"的**分工**：``verdict`` 只回答"能不能开始试训"
    （下限 200）；"够不够可信"是 ``target_shortfall`` 的事。
    若哪天改成"必须达标才 sufficient"，本条会红并提示这是有意变更。
    """
    from kakeya_ai.ai.training_data import (
        MIN_SAMPLES_FOR_TRAINING,
        TARGET_SAMPLES_FOR_TRAINING,
    )

    n = MIN_SAMPLES_FOR_TRAINING + 10
    assert n < TARGET_SAMPLES_FOR_TRAINING
    samples = [_sample(1), _sample(2), _sample(3)] * ((n // 3) + 1)
    a = assess_sample_sufficiency(samples[:n])
    assert a["verdict"] == SUFFICIENCY_SUFFICIENT
    assert a["target_shortfall"] > 0, "应仍提示距目标有差距"
    assert a["target_samples"] == TARGET_SAMPLES_FOR_TRAINING


def test_target_shortfall_is_not_a_reason():
    """**"低于目标"是信息、不是问题** —— 不得进 ``reasons``。

    ``reasons == []`` 是**既有的对外契约**（"没有原因 = 一切正常"）。
    初版把这条信息 append 进 ``reasons``，立刻打红了两条既有测试
    （``test_sufficient_when_counts_and_distribution_ok`` 与
    ``test_threshold_is_overridable_and_explicit``）——因为 ``verdict``
    会跟着变成 ``insufficient``，等于把"信息"降级成"故障"。

    本测试**替代不了**那两条既有测试（它们测的是各自的原意），
    而是把这条新边界单独钉住。
    """
    from kakeya_ai.ai.training_data import MIN_SAMPLES_FOR_TRAINING

    n = MIN_SAMPLES_FOR_TRAINING + 10
    samples = [_sample(1), _sample(2), _sample(3)] * ((n // 3) + 1)
    a = assess_sample_sufficiency(samples[:n])

    assert a["reasons"] == [], (
        "'已达下限但未达目标'不应作为原因 —— 它是信息，"
        "进了 reasons 会让 verdict 变成 insufficient"
    )
    # 但报告里必须显示出来（否则 sufficient 会被读成"够了"）。
    report = format_sufficiency_report(a)
    assert "经验法则目标" in report
    assert "泛化能力预期有限" in report


def test_report_always_shows_both_thresholds():
    """报告必须**同时**给出两个门槛——否则读者无法判断差距量级。"""
    a = assess_sample_sufficiency([_sample(1), _sample(2), _sample(3)])
    report = format_sufficiency_report(a)
    assert str(MIN_SAMPLES_FOR_TRAINING) in report
    assert "经验法则够用" in report
