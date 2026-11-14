"""纯 NumPy 实现的 Nesterov 加速梯度下降优化器。

这是一个小型的、无依赖的优化器，用于训练 AI 推理层的 surrogate 打分模型。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class NesterovOptimizer:
    """带可选权重衰减的 Nesterov 加速梯度下降。

    更新规则（对每个参数向量 ``w``）：

        v_{t+1} = mu * v_t - lr * grad(L)(w_t + mu * v_t)
        w_{t+1} = w_t + v_{t+1}

    Parameters
    ----------
    lr : float
        学习率（默认 1e-3）。
    mu : float
        动量系数（默认 0.9）。
    weight_decay : float
        L2 正则化系数（默认 0.0）。
    lr_decay : float
        每个 epoch 施加的学习率乘性衰减（默认 1.0）。
    """

    def __init__(
        self,
        lr: float = 1e-3,
        mu: float = 0.9,
        weight_decay: float = 0.0,
        lr_decay: float = 1.0,
        max_grad_norm: Optional[float] = None,
    ):
        self.lr = float(lr)
        self.mu = float(mu)
        self.weight_decay = float(weight_decay)
        self.lr_decay = float(lr_decay)
        self.max_grad_norm = max_grad_norm
        self._velocity: Dict[int, np.ndarray] = {}
        self._step = 0

    def step(self, params: Dict[int, np.ndarray], grads: Dict[int, np.ndarray]) -> None:
        """对所有参数数组施加一次 Nesterov 更新。

        Parameters
        ----------
        params : dict[int, np.ndarray]
            参数 id 到参数数组的映射。**原地**更新。
        grads : dict[int, np.ndarray]
            参数 id 到梯度数组的映射。假定梯度是在前瞻点 ``w + mu * v``
            处求得的。
        """
        if self.max_grad_norm is not None:
            total_sq = sum(float(np.sum(g * g)) for g in grads.values())
            total_norm = total_sq**0.5
            if total_norm > self.max_grad_norm:
                scale = self.max_grad_norm / (total_norm + 1e-12)
                grads = {pid: g * scale for pid, g in grads.items()}

        lr = self.lr * (self.lr_decay**self._step)
        for pid, w in params.items():
            g = grads[pid]
            if self.weight_decay > 0.0:
                g = g + self.weight_decay * w
            v = self._velocity.setdefault(pid, np.zeros_like(w))
            v_new = self.mu * v - lr * g
            self._velocity[pid] = v_new
            # ``+=`` 对 ndarray 是**原地**更新，正是优化器要的效果：调用方传入
            # 的 ``params`` 字典里的数组被直接改写（``w = w + v_new`` 只会让
            # 局部名指向新数组，参数不会被更新——那才是真 bug）。
            #
            # 注：这条写法会被 ruff 的 PLW2901（"循环变量被赋值覆盖"）盯上，
            # 但该规则**不在本项目的 select 里**（只开 F + E9），故无须豁免。
            # 措辞刻意不写出那个指令的字面量：ruff 会把注释里出现的它当成
            # 真指令来解析，从而报"畸形指令"警告（已在 test_lint_config 踩过）。
            w += v_new
        self._step += 1

    def state_dict(self) -> Dict[str, Any]:
        """返回可序列化的状态字典。"""
        return {
            "lr": self.lr,
            "mu": self.mu,
            "weight_decay": self.weight_decay,
            "lr_decay": self.lr_decay,
            "step": self._step,
            "velocity": {pid: v.tolist() for pid, v in self._velocity.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """恢复优化器状态。"""
        self.lr = float(state["lr"])
        self.mu = float(state["mu"])
        self.weight_decay = float(state["weight_decay"])
        self.lr_decay = float(state["lr_decay"])
        self._step = int(state["step"])
        self._velocity = {
            int(pid): np.array(v, dtype=float) for pid, v in state["velocity"].items()
        }

    def __repr__(self) -> str:
        return (
            f"NesterovOptimizer(lr={self.lr}, mu={self.mu}, "
            f"weight_decay={self.weight_decay}, lr_decay={self.lr_decay}, step={self._step})"
        )
