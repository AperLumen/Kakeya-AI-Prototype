"""AI 推理层的 surrogate 打分模型。

一个用纯 NumPy 实现的小型全连接神经网络。它从几何状态向量
（rho, delta, local_error, volume_ratio）学习预测三个走法分数
（move_1, move_2, move_3）。训练使用 Nesterov 加速梯度下降。

推理时，若模型的 top-two 分数差超过阈值，则直接返回 surrogate 答案；否则调用方
可以回落到基于 LLM 的打分器。
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .nesterov_optimizer import NesterovOptimizer
from ..errors import InputValidationError


class SurrogateScorer:
    """用于模仿 LLM 走法打分的小型 MLP surrogate。

    Parameters
    ----------
    input_dim : int
        状态向量的维数（默认 4）。
    hidden_dims : list[int]
        隐藏层宽度（默认 [16, 16]）。
    output_dim : int
        要打分的走法数（默认 3）。
    seed : int or None
        权重初始化的随机种子。
    feature_mean, feature_std : np.ndarray or None
        可选的归一化统计量。两者都设置时，原始状态特征会在推理前被标准化
        （std 为 0 的项按 1 处理）。
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: Optional[List[int]] = None,
        output_dim: int = 3,
        seed: Optional[int] = None,
        output_bias_init: Optional[np.ndarray] = None,
        feature_mean: Optional[np.ndarray] = None,
        feature_std: Optional[np.ndarray] = None,
        feature_names: Optional[Sequence[str]] = None,
    ):
        self.input_dim = int(input_dim)
        self.hidden_dims = list(hidden_dims) if hidden_dims is not None else [16, 16]
        self.output_dim = int(output_dim)
        self.rng = np.random.default_rng(seed)
        # 特征归一化统计量。两者都设置时，原始状态特征会在前向传播前被标准化到
        # 零均值／单位方差，与 ``training_data.build_training_data`` 一致。
        self.feature_mean = (
            None if feature_mean is None else np.asarray(feature_mean, dtype=float)
        )
        self.feature_std = (
            None if feature_std is None else np.asarray(feature_std, dtype=float)
        )
        # 特征名清单：决定 ``_state_to_vector`` 取哪些字段、按什么顺序。
        # 默认为历史 4 特征（推理端不传时保持旧行为）；训练路径会把
        # ``FEATURE_NAMES`` 传进来，从而支持扩展特征。
        if feature_names is None:
            from .training_data import FEATURE_NAMES as _ALL

            feature_names = _ALL[: self.input_dim]
        self.feature_names: Tuple[str, ...] = tuple(feature_names)

        # 构建各层形状：input -> hidden -> ... -> output。
        layer_dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        self.params: Dict[int, np.ndarray] = {}
        self.param_shapes: Dict[int, Tuple[int, ...]] = {}
        pid = 0
        n_layers = len(layer_dims) - 1
        for layer_idx, (in_dim, out_dim) in enumerate(
            # 同一列表的错位切片：长度必然差 1，strict=True 恒成立。
            # 写上它不是为了防错，而是让"这两条必然等长"变成代码里可见的断言。
            zip(layer_dims[:-1], layer_dims[1:], strict=True)
        ):
            # Xavier/Glorot 初始化，使归一化输入下的训练更稳定。
            scale = np.sqrt(1.0 / max(in_dim, 1))
            W = self.rng.normal(0.0, scale, size=(in_dim, out_dim)).astype(float)
            if layer_idx < n_layers - 1:
                b = np.zeros(out_dim, dtype=float)
            else:
                # 若提供了目标均值，则用它初始化输出偏置，
                # 否则用中性值 5.0（[0, 10] 分数区间的中点）。
                if output_bias_init is not None:
                    b = np.asarray(output_bias_init, dtype=float).reshape(out_dim)
                else:
                    b = np.full(out_dim, 5.0, dtype=float)
            self.params[pid] = W
            self.param_shapes[pid] = W.shape
            pid += 1
            self.params[pid] = b
            self.param_shapes[pid] = b.shape
            pid += 1

    @property
    def n_parameters(self) -> int:
        return sum(p.size for p in self.params.values())

    def forward(
        self,
        X: np.ndarray,
        params: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """返回每个走法的 logits 的前向传播。

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
        params : dict[int, np.ndarray] or None
            若为 None，则使用 ``self.params``。

        Returns
        -------
        logits : np.ndarray, shape (n_samples, output_dim)
        """
        if params is None:
            params = self.params
        X = np.atleast_2d(X).astype(float)
        a = X
        n_layers = len(params) // 2
        for layer_idx in range(n_layers):
            W = params[2 * layer_idx]
            b = params[2 * layer_idx + 1]
            # 用 einsum 以避开某些平台上特定矩阵形状下的 BLAS 除零警告。
            z = np.einsum("ni,ij->nj", a, W) + b
            # 隐藏层用 ReLU；输出层用线性。
            if layer_idx < n_layers - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z
        return a

    def predict_scores(self, state: Dict[str, float]) -> Dict[int, float]:
        """为单个 state dict 返回走法分数 {1, 2, 3}。

        分数被裁剪到 [0, 10]，以匹配 LLM 打分器的约定。
        """
        x = self._state_to_vector(state)
        logits = self.forward(x[None, :])
        scores = np.clip(logits[0], 0.0, 10.0)
        return {i + 1: float(scores[i]) for i in range(self.output_dim)}

    def confidence(self, scores: Dict[int, float]) -> float:
        """返回 top-two 分数差；越大表示越有把握。"""
        values = sorted(scores.values(), reverse=True)
        if len(values) < 2:
            return 0.0
        return float(values[0] - values[1])

    def backward(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        params: Optional[Dict[int, np.ndarray]] = None,
    ) -> Dict[int, np.ndarray]:
        """MSE 损失的反向传播。返回每个参数 id 的梯度。"""
        if params is None:
            params = self.params
        X = np.atleast_2d(X).astype(float)
        n_samples = y_true.shape[0]
        # 输出层关于 z 的梯度 dL/dz = (y_pred - y_true) / n_samples
        delta = (y_pred - y_true) / max(n_samples, 1)

        # 重建各层激活值。
        activations = [X]
        a = X
        n_layers = len(params) // 2
        for layer_idx in range(n_layers):
            W = params[2 * layer_idx]
            b = params[2 * layer_idx + 1]
            z = np.einsum("ni,ij->nj", a, W) + b
            if layer_idx < n_layers - 1:
                a = np.maximum(z, 0.0)
            else:
                a = z
            activations.append(a)

        grads: Dict[int, np.ndarray] = {}
        for layer_idx in range(n_layers - 1, -1, -1):
            a_prev = activations[layer_idx]
            W_pid = 2 * layer_idx
            b_pid = W_pid + 1
            grads[W_pid] = np.einsum("ni,nj->ij", a_prev, delta)
            grads[b_pid] = delta.sum(axis=0)
            if layer_idx > 0:
                W = params[W_pid]
                da_prev = np.einsum("nj,kj->nk", delta, W)
                z_prev = activations[layer_idx]
                delta = da_prev * (z_prev > 0).astype(float)
        return grads

    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方误差。"""
        return float(np.mean((y_pred - y_true) ** 2))

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        optimizer: Optional[NesterovOptimizer] = None,
        epochs: int = 1000,
        val_split: float = 0.1,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        print_every: int = 100,
        seed: Optional[int] = None,
    ) -> Dict[str, List[float]]:
        """用 Nesterov 加速梯度下降训练 surrogate。

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, input_dim)
        Y : np.ndarray, shape (n_samples, output_dim)
        optimizer : NesterovOptimizer or None
            默认为 ``NesterovOptimizer(lr=1e-3, mu=0.9)``。
        epochs : int
        val_split : float
            留出用于验证的数据比例。
        batch_size : int or None
            若为 None，则做全批量训练。
        verbose : bool
        print_every : int
        seed : int or None
            在训练/验证划分前打乱数据。

        Returns
        -------
        history : dict
            每个 epoch 的训练/验证损失列表。
        """
        X = np.atleast_2d(X).astype(float)
        Y = np.atleast_2d(Y).astype(float)
        n_samples = X.shape[0]
        if n_samples == 0:
            raise InputValidationError("Empty training set.")

        # 用经验目标均值初始化输出偏置，以加快收敛。
        output_pid = 2 * ((len(self.params) // 2) - 1) + 1
        if np.allclose(self.params[output_pid], 5.0):
            self.params[output_pid] = Y.mean(axis=0).copy()

        rng = np.random.default_rng(seed)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        n_val = int(n_samples * val_split)
        n_train = n_samples - n_val
        X_train, Y_train = X[:n_train], Y[:n_train]
        X_val, Y_val = X[n_train:], Y[n_train:]

        if optimizer is None:
            optimizer = NesterovOptimizer(lr=1e-3, mu=0.9)

        # 收窄到**局部变量**：``batch_size`` 参数是 ``Optional[int]``，上面
        # 赋予了 ``n_train``；但赋值语句本身不会把参数的类型改成 ``int``
        # （检查器只认声明），于是 ``range(..., batch_size)`` 与切片都用不了。
        # 取局部变量这一步把"到这里必非 None"变成代码里可见的事实。
        eff_batch_size = n_train if batch_size is None else batch_size

        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # 小批量循环。
            perm = rng.permutation(n_train)
            for start in range(0, n_train, eff_batch_size):
                batch_idx = perm[start : start + eff_batch_size]
                Xb = X_train[batch_idx]
                Yb = Y_train[batch_idx]
                # Nesterov 前瞻：在 w + mu * v 处计算梯度。
                lookahead_params = self._lookahead_params(optimizer)
                y_pred = self.forward(Xb, lookahead_params)
                grads = self.backward(Xb, Yb, y_pred, lookahead_params)
                optimizer.step(self.params, grads)

            train_pred = self.forward(X_train)
            train_loss = self.loss(Y_train, train_pred)
            history["train_loss"].append(train_loss)

            if n_val > 0:
                val_pred = self.forward(X_val)
                val_loss = self.loss(Y_val, val_pred)
                history["val_loss"].append(val_loss)
            else:
                history["val_loss"].append(train_loss)

            if verbose and (epoch + 1) % print_every == 0:
                msg = f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.6f}"
                if n_val > 0:
                    msg += f"  val_loss={history['val_loss'][-1]:.6f}"
                print(msg)

        return history

    def _state_to_vector(self, state: Dict[str, float]) -> np.ndarray:
        """把 state dict 转成模型输入向量。

        特征**按 ``self.feature_names`` 取**（而不是硬编码字段名）：模型是用
        哪组特征训练的，推理就取哪组。旧权重文件没有 ``feature_names``，
        ``load`` 会给它补上历史 4 特征，行为与修复前逐位一致。

        当 ``feature_mean`` 与 ``feature_std`` 都设置时（模型是在归一化特征上训练的），
        原始特征会先被标准化；std 为 0 的项按 1 处理，与
        ``training_data.build_training_data`` 一致。
        """
        from .training_data import _DEFAULT_FEATURE

        x = np.array(
            [float(state.get(name, _DEFAULT_FEATURE.get(name, 0.0)))
             for name in self.feature_names],
            dtype=float,
        )
        if x.shape[0] != self.input_dim:
            raise InputValidationError(
                f"特征向量长度 {x.shape[0]} 与模型 input_dim {self.input_dim} 不符"
                f"（feature_names={list(self.feature_names)}）。"
                "这通常意味着特征清单被改过而权重文件没重新训练。"
            )
        if self.feature_mean is not None and self.feature_std is not None:
            std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
            x = (x - self.feature_mean) / std
        return x

    def _lookahead_params(self, optimizer: NesterovOptimizer) -> Dict[int, np.ndarray]:
        """为每个参数返回 w + mu * v（Nesterov 前瞻点）。"""
        lookahead: Dict[int, np.ndarray] = {}
        for pid, w in self.params.items():
            v = optimizer._velocity.get(pid)
            if v is None:
                lookahead[pid] = w.copy()
            else:
                lookahead[pid] = w + optimizer.mu * v
        return lookahead

    def save(self, path: str) -> None:
        """把模型权重与结构保存到 JSON 文件。"""
        payload = {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "output_dim": self.output_dim,
            "params": {str(pid): p.tolist() for pid, p in self.params.items()},
        }
        # 旧文件缺少这些字段；加载它们会走原始特征行为，故只写出实际设置了的统计量。
        if self.feature_mean is not None:
            payload["feature_mean"] = np.asarray(self.feature_mean, dtype=float).tolist()
        if self.feature_std is not None:
            payload["feature_std"] = np.asarray(self.feature_std, dtype=float).tolist()
        payload["feature_names"] = list(self.feature_names)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SurrogateScorer":
        """从 JSON 文件加载一个 surrogate 打分器。

        在引入特征归一化之前写出的文件不含 ``feature_mean`` / ``feature_std``；
        这些文件以原始特征（未归一化）推理的方式加载，从而保留历史行为。同样，
        在 ``feature_names`` 存在之前写出的文件会得到历史的前 ``input_dim`` 个
        名字，从而与旧的硬编码行为完全一致。
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        input_dim = payload["input_dim"]
        feature_names = payload.get("feature_names")
        if feature_names is None:
            from .training_data import FEATURE_NAMES

            feature_names = list(FEATURE_NAMES[:input_dim])
        elif len(feature_names) != input_dim:
            raise InputValidationError(
                f"权重文件自相矛盾：input_dim={input_dim} 但 feature_names "
                f"有 {len(feature_names)} 项。拒绝加载——静默按行数截断会"
                "让推理读到错位的特征。"
            )
        model = cls(
            input_dim=input_dim,
            hidden_dims=payload["hidden_dims"],
            output_dim=payload["output_dim"],
            feature_names=feature_names,
        )
        for pid_str, vals in payload["params"].items():
            pid = int(pid_str)
            model.params[pid] = np.array(vals, dtype=float)
        if "feature_mean" in payload:
            model.feature_mean = np.array(payload["feature_mean"], dtype=float)
        if "feature_std" in payload:
            model.feature_std = np.array(payload["feature_std"], dtype=float)
        return model

    def __repr__(self) -> str:
        return (
            f"SurrogateScorer(input_dim={self.input_dim}, hidden_dims={self.hidden_dims}, "
            f"output_dim={self.output_dim}, n_parameters={self.n_parameters})"
        )
