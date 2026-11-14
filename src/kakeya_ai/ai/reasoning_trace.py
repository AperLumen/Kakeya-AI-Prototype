"""推理轨迹的摘要、轻量向量化与注意力锚点。

本模块保存快思考者的推理轨迹，用一个小的领域词表把它们变成稀疏 numpy 向量，
并检索出最相关的历史轨迹，作为注意力锚点前置到深层模型的 prompt 中。

不需要新的外部依赖：分词与向量运算只用 Python 标准库与 numpy。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Kakeya grains/moves 推理轨迹的领域词表。
# 词频向量建立在这个固定词表之上，这样轨迹之间的相似度计算既快又确定，
# 不需要 embedding 模型。
DOMAIN_VOCABULARY = [
    # 状态参数
    "rho",
    "delta",
    "local_error",
    "volume_ratio",
    "volume",
    "error",
    # 走法
    "move_1",
    "move_2",
    "move_3",
    "lengthen",
    "longer",
    "square",
    "widen",
    "reduce",
    "increase",
    "decrease",
    # 几何
    "grain",
    "grains",
    "tube",
    "tubes",
    "prism",
    "direction",
    # 证明概念
    "bound",
    "lower",
    "estimate",
    "exceed",
    "close",
    "large",
    "small",
    "high",
    "low",
    "moderate",
    "candidate",
    "reasoning",
    "promising",
    "uncertain",
]


@dataclass
class ReasoningTrace:
    """一条已存储的推理轨迹及其派生表示。"""

    state_key: str
    reasoning: str
    summary: str
    vector: np.ndarray
    scores: Dict[int, float]
    timestamp: float = field(default_factory=time.time)


class TraceVectorizer:
    """基于固定领域词表的轻量词袋向量化器。"""

    def __init__(self, vocabulary: Sequence[str] = DOMAIN_VOCABULARY):
        self.vocabulary = tuple(vocabulary)
        self._index = {term: idx for idx, term in enumerate(self.vocabulary)}

    def _tokenize(self, text: str) -> List[str]:
        """转小写，并按非字母数字字符切分。"""
        return re.findall(r"[a-z0-9_]+", text.lower())

    def transform(self, text: str) -> np.ndarray:
        """返回 ``text`` 的稀疏词频向量。"""
        vec = np.zeros(len(self.vocabulary), dtype=float)
        for token in self._tokenize(text):
            idx = self._index.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        """把一串文本变换成二维数组（每行一条文本）。"""
        return np.vstack([self.transform(t) for t in texts])

    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """两个向量之间的余弦相似度，裁剪到 [0, 1]。"""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        cos = float(np.dot(a, b) / (norm_a * norm_b))
        # 作为注意力锚点，我们只关心正相似度。
        return max(0.0, min(1.0, cos))


def _extract_first_sentence(text: str, max_chars: int = 200) -> str:
    """返回第一句；若找不到句子结尾，则返回前缀。"""
    match = re.search(r"[^.!?]+[.!?]", text.strip())
    if match:
        sentence = match.group(0).strip()
    else:
        sentence = text.strip()
    if len(sentence) > max_chars:
        sentence = sentence[:max_chars].rsplit(" ", 1)[0] + "..."
    return sentence


class ReasoningTraceStore:
    """带基于向量的检索能力的内存推理轨迹存储。"""

    def __init__(self, vectorizer: Optional[TraceVectorizer] = None):
        self.vectorizer = vectorizer if vectorizer is not None else TraceVectorizer()
        self._traces: List[ReasoningTrace] = []

    @property
    def size(self) -> int:
        return len(self._traces)

    def add(
        self,
        state_key: str,
        reasoning: str,
        scores: Dict[int, float],
    ) -> ReasoningTrace:
        """新增一条轨迹，自动计算摘要与向量。"""
        trace = ReasoningTrace(
            state_key=state_key,
            reasoning=reasoning,
            summary=_extract_first_sentence(reasoning),
            vector=self.vectorizer.transform(reasoning),
            scores=dict(scores),
        )
        self._traces.append(trace)
        return trace

    def get_anchors(
        self,
        query_text: str,
        exclude_state_key: Optional[str] = None,
        top_k: int = 2,
    ) -> List[Tuple[str, float]]:
        """返回最相似的 ``top_k`` 条历史轨迹摘要。

        每项结果是一个元组 ``(summary, similarity)``。若存储为空或
        ``top_k <= 0``，则返回空列表。
        """
        if top_k <= 0 or not self._traces:
            return []

        query_vec = self.vectorizer.transform(query_text)
        candidates: List[Tuple[float, ReasoningTrace]] = []
        for trace in self._traces:
            if exclude_state_key is not None and trace.state_key == exclude_state_key:
                continue
            sim = TraceVectorizer.similarity(query_vec, trace.vector)
            candidates.append((sim, trace))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [(trace.summary, sim) for sim, trace in candidates[:top_k]]
