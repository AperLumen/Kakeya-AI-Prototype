"""模型行为策略注册表：think 标志与预算下限的**统一拦截点**（2026-11-12）。

**为什么需要它**：不同模型对 ``think`` 标志的依赖**方向相反**——

- ``xian-reason``（项目自训 LoRA）：训练模板末尾带打开的 ``<think>`` 预填，
  Ollama 的 renderer 在 ``think:false`` 时会把它**从 prompt 里剥掉**——剥掉
  即出训练分布，输出退化为与输入无关的常量 JSON（实测 top-1 一致率 3.3%，
  见评测记录）。
- 基座 ``qwen3.5:4b`` 恰恰相反：不传 ``think:false`` 时思维链吃光
  ``num_predict``，JSON 还没产出就被截断（parse 全灭）。

靠每个调用点自己记"这个模型该怎么传 think"必踩（本项目实测踩了两次才
定位）。统一在这里拦：调用方照原样传，本模块按模型改写并留 warning。

**覆盖范围**：``ollama_client.generate`` 与 ``scripts/interfaces/ollama_chat``
（都是 /api/generate）。/api/chat 的直接调用方（如评测脚本）请自行参考
本表的策略（chat 的 thinking 字段截断问题见评测报告）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPolicy:
    """单个模型的调用策略。"""

    #: ``think:false`` 会破坏该模型（剥训练预填）；遇之改为省略 think 字段。
    forbid_think_false: bool = False
    #: 强制 think 取值（None=不动）。v4 起 kakeya-fast 用 think:false
    #: 走直答模式（2.4s vs 10s；直答样本治好了 v1 的无预填塌缩）——
    #: 这是"think 可关"的生产开关，与 prompt_prefix 配套。
    force_think: "bool | None" = None
    #: ``num_predict`` 下限——think 型模型的思维链要占掉几百 token，
    #: 预算太低时正文会被截断甚至全空。
    min_num_predict: int = 0
    #: /api/generate 的裸 prompt 要不要先包一层训练用 ChatML（含 ``<think>``
    #: 预填）。generate 不走 Modelfile 的 TEMPLATE（那是 chat 的事），
    #: 微调模型见不到训练格式会给出近乎任意的输出（2026-11-12 实测：
    #: 同一模型 /api/chat 下 MAE 0.137，/api/generate 裸发下答非所问）。
    wrap_chatml_for_generate: bool = False
    #: prompt 前缀（v4 直答模式）：kakeya-fast 加"【直答】"让微调模型
    #: 跳过思维链直接出分——这是"think 可关"在生产侧的开关
    #: （训练时学的条件行为，见 build_xian_reason_data 的直答变体）。
    #: deep 位（kakeya-deep）不加——它要的就是完整推理。
    prompt_prefix: str = ""
    #: 策略原因（进 warning，排查时可读）。
    note: str = ""


POLICIES: Dict[str, ModelPolicy] = {
    "xian-reason": ModelPolicy(
        forbid_think_false=True,
        min_num_predict=1200,
        wrap_chatml_for_generate=True,
        note="训练模板带 <think> 预填，think:false 会被 renderer 剥掉预填，"
             "输出退化为常量（2026-11-12 实测）",
    ),
    # kakeya-fast 当前是 xian-reason 的别名（2026-11-12 起，用户定调
    # "fast/deep 按实测延迟分档：≈10s 内算 fast，≈50s 内算 deep"）。
    # ⚠️ 别名是活的：将来 re-point 到别的模型时，本条必须同步改。
    "kakeya-fast": ModelPolicy(
        forbid_think_false=True,
        min_num_predict=1200,
        wrap_chatml_for_generate=True,
        note="kakeya-fast 别名当前指向 xian-reason，策略同上；v4 实测"
             "think:false 直答 top1 仅 45%（快但判断塌），已回退——"
             "推理模式仍是唯一可用形态",
    ),
    # kakeya-deep 自 v4 起也指向 xian-reason（同一 int8 权重、独立别名）：
    # fast 位直答（快）、deep 位推理（准），一份权重两个角色。
    "kakeya-deep": ModelPolicy(
        forbid_think_false=True,
        min_num_predict=1200,
        wrap_chatml_for_generate=True,
        note="kakeya-deep 别名当前指向 xian-reason（v4 起），策略同上",
    ),
}


#: 训练模板（``data/xian_reason/base_mlx/chat_template.jinja`` 对纯文本
#: 消息的渲染结果）的等价包装——改训练模板时这里必须同步。
_CHATML_WRAP = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def prepare_generate_prompt(model: str, prompt: str) -> str:
    """按策略处理裸 prompt（/api/generate 用）：先加前缀，再包训练格式。"""
    policy = policy_for(model)
    if policy is None:
        return prompt
    if policy.prompt_prefix:
        prompt = policy.prompt_prefix + prompt
    if policy.wrap_chatml_for_generate:
        prompt = _CHATML_WRAP.format(prompt=prompt)
    return prompt


#: 旧名保留（向后兼容）；新代码请用 ``prepare_generate_prompt``。
def wrap_prompt_for_generate(model: str, prompt: str) -> str:
    return prepare_generate_prompt(model, prompt)


def policy_for(model: str) -> Optional[ModelPolicy]:
    """按模型名取策略。``xian-reason:latest`` 与 ``xian-reason`` 命中同一条。"""
    name = model.split(":", 1)[0]
    return POLICIES.get(name)


def apply_generate_policy(
    model: str,
    think: Optional[bool],
    options: Optional[Dict[str, Any]],
) -> Tuple[Optional[bool], Dict[str, Any]]:
    """按模型策略改写 (think, options)，返回改写后的值。

    规则：
    - ``forbid_think_false`` 且调用方传了 ``think=False`` → 改为 ``None``
      （省略字段），并记 warning；
    - ``min_num_predict`` 大于调用方给的 ``num_predict`` → 抬到下限并记
      warning（调用方没给则不补——"没给"由服务端默认处理，不在这里编数）。
    """
    policy = policy_for(model)
    if policy is None:
        return think, dict(options) if options else {}
    out_options = dict(options) if options else {}
    out_think = think
    if policy.force_think is not None and think != policy.force_think:
        _logger.warning(
            "模型 %s 强制 think=%s（%s）", model, policy.force_think, policy.note,
        )
        out_think = policy.force_think
    if policy.forbid_think_false and think is False:
        _logger.warning(
            "模型 %s 禁止 think:false（%s）——已改为省略 think 字段",
            model, policy.note,
        )
        out_think = None
    if policy.min_num_predict and "num_predict" in out_options:
        cur = out_options["num_predict"]
        if cur < policy.min_num_predict:
            _logger.warning(
                "模型 %s 的 num_predict=%s 低于下限 %s（思维链会吃预算）"
                "——已抬到下限", model, cur, policy.min_num_predict,
            )
            out_options["num_predict"] = policy.min_num_predict
    return out_think, out_options
