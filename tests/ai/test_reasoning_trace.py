"""推理轨迹摘要、向量化与锚点检索的测试。"""

import numpy as np
import pytest

from kakeya_ai.ai.reasoning_trace import (
    DOMAIN_VOCABULARY,
    ReasoningTraceStore,
    TraceVectorizer,
    _extract_first_sentence,
)


def test_vectorizer_transform_uses_domain_vocabulary():
    vectorizer = TraceVectorizer()
    vec = vectorizer.transform("Rho is high, so move_2 looks promising for square grains.")
    assert vec.shape == (len(DOMAIN_VOCABULARY),)
    assert vec[DOMAIN_VOCABULARY.index("rho")] > 0
    assert vec[DOMAIN_VOCABULARY.index("move_2")] > 0
    assert vec[DOMAIN_VOCABULARY.index("grains")] > 0


def test_vectorizer_similarity_is_between_zero_and_one():
    vectorizer = TraceVectorizer()
    a = vectorizer.transform("move_1 lengthen grains rho")
    b = vectorizer.transform("move_1 lengthen grains delta")
    sim = TraceVectorizer.similarity(a, b)
    assert 0.0 <= sim <= 1.0
    assert sim > 0.0


def test_vectorizer_similarity_of_identical_texts_is_one():
    vectorizer = TraceVectorizer()
    text = "move_3 widen grains to reduce local_error"
    vec = vectorizer.transform(text)
    assert TraceVectorizer.similarity(vec, vec) == pytest.approx(1.0)


def test_extract_first_sentence_truncates_long_text():
    text = "First sentence here. Second sentence there. " + "word " * 200
    sentence = _extract_first_sentence(text, max_chars=15)
    assert sentence.endswith("...")
    assert "Second" not in sentence


def test_trace_store_retrieves_relevant_anchors():
    store = ReasoningTraceStore()
    store.add("state_a", "move_2 is best because rho is close to one", {1: 3.0, 2: 9.0, 3: 4.0})
    store.add("state_b", "move_3 is best because local_error is huge", {1: 4.0, 2: 3.0, 3: 9.0})

    anchors = store.get_anchors("rho square grain move_2", top_k=1)
    assert len(anchors) == 1
    assert "move_2" in anchors[0][0]


def test_trace_store_excludes_same_state_key():
    store = ReasoningTraceStore()
    store.add("same", "move_1 reasoning", {1: 10.0, 2: 0.0, 3: 0.0})
    anchors = store.get_anchors("move_1 reasoning", exclude_state_key="same", top_k=1)
    assert anchors == []
