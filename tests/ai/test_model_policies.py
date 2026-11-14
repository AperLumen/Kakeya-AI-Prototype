"""模型行为策略（src/kakeya_ai/ai/model_policies.py）的守门。

钉住 2026-11-12 实测的两条相反策略（详见
内部评测报告）：

- xian-reason：``think:false`` 会剥训练预填、输出退化为常量 → 必须改写为
  省略 think 字段；预算低于 1200 会被思维链截断 → 抬到下限；
- 其它模型（含基座 qwen3.5:4b）：原样透传，**不得**被策略误改。
"""

import pytest

from kakeya_ai.ai.model_policies import (
    apply_generate_policy,
    policy_for,
)


def test_policy_for_matches_with_and_without_tag():
    assert policy_for("xian-reason") is not None
    assert policy_for("xian-reason:latest") is not None
    assert policy_for("qwen3.5:4b") is None


def test_xian_reason_think_false_rewritten_to_omit():
    think, options = apply_generate_policy(
        "xian-reason", False, {"num_predict": 512}
    )
    assert think is None  # think:false → 省略字段（不是改成 True！）
    assert options["num_predict"] == 1200  # 抬到下限


def test_xian_reason_think_none_untouched():
    think, options = apply_generate_policy(
        "xian-reason:latest", None, {"num_predict": 2048}
    )
    assert think is None
    assert options["num_predict"] == 2048  # 已高于下限，不动


def test_xian_reason_think_true_kept():
    think, _ = apply_generate_policy("xian-reason", True, {})
    assert think is True


def test_other_models_untouched():
    think, options = apply_generate_policy(
        "qwen3.5:4b", False, {"num_predict": 512}
    )
    assert think is False
    assert options["num_predict"] == 512


def test_no_options_dict_handled():
    think, options = apply_generate_policy("xian-reason", False, None)
    assert think is None
    assert options == {}  # 调用方没给预算时不编造数字


def test_options_not_mutated_in_place():
    src = {"num_predict": 512}
    apply_generate_policy("xian-reason", False, src)
    assert src["num_predict"] == 512  # 不改调用方的 dict


@pytest.mark.parametrize("model", ["xian-reason", "xian-reason:latest"])
def test_warning_logged_on_override(model, caplog):
    with caplog.at_level("WARNING"):
        apply_generate_policy(model, False, {"num_predict": 512})
    assert "think:false" in caplog.text


def test_wrap_prompt_for_generate_only_for_finetuned():
    from kakeya_ai.ai.model_policies import wrap_prompt_for_generate
    # 微调模型（含 kakeya-fast 别名）：包训练用 ChatML（含 <think> 预填）
    wrapped = wrap_prompt_for_generate("xian-reason", "P")
    assert wrapped == "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n<think>\n"
    assert wrap_prompt_for_generate("kakeya-fast", "P") == wrapped
    # 基座：原样透传（包模板会改变所有存量调用方行为）
    assert wrap_prompt_for_generate("qwen3.5:4b", "P") == "P"


def test_prompt_prefix_direct_mode():
    from kakeya_ai.ai.model_policies import prepare_generate_prompt
    # kakeya-fast（v4 定稿）：直答模式走 think:false 通道（force_think），
    # **不再需要**【直答】前缀——但训练格式包装仍要（微调模型见裸 prompt
    # 会出分布）。
    out = prepare_generate_prompt("kakeya-fast", "P")
    assert out == "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # kakeya-deep：无前缀（deep 位 = 完整推理）
    out2 = prepare_generate_prompt("kakeya-deep", "P")
    assert out2 == "<|im_start|>user\nP<|im_end|>\n<|im_start|>assistant\n<think>\n"
    # 基座：不加前缀不包装
    assert prepare_generate_prompt("qwen3.5:4b", "P") == "P"


def test_fast_direct_mode_reverted():
    from kakeya_ai.ai.model_policies import apply_generate_policy
    # v4 直答模式实测 top1 仅 45%（快但判断塌）——kakeya-fast 回退为
    # 推理模式：think=False 仍被禁止（省略字段），预算下限 1200。
    think, options = apply_generate_policy("kakeya-fast", False, {"num_predict": 512})
    assert think is None
    assert options["num_predict"] == 1200
