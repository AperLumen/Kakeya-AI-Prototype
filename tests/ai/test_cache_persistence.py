"""DecisionCache / PromptCache 可选 SQLite 后端的测试。

默认（不传 ``path``）的内存行为由 ``test_ai_optimizations.py`` 与
``test_prompt_cache.py`` 覆盖；本文件覆盖持久化模式：往返、跨实例持久化、
清空与 TTL。
"""

from kakeya_ai.ai.cache import DecisionCache
from kakeya_ai.ai.prompt_cache import PromptCache

_STATE = {"delta": 0.01, "rho": 0.5, "local_error": 1.0, "volume_ratio": 1.2}
_RESULT = {"move_1": 3, "move_2": 5, "move_3": 7}


def test_decision_cache_sqlite_roundtrip(tmp_path):
    cache = DecisionCache(path=str(tmp_path / "decisions.db"))
    assert cache.get(_STATE) is None
    cache.set(_STATE, _RESULT)
    assert cache.get(_STATE) == _RESULT
    assert cache.size == 1


def test_decision_cache_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "decisions.db")
    DecisionCache(path=path).set(_STATE, _RESULT)
    # 同一文件上的全新实例必须能看到该条目。
    reloaded = DecisionCache(path=path)
    assert reloaded.get(_STATE) == _RESULT
    assert reloaded.size == 1


def test_decision_cache_sqlite_clear(tmp_path):
    path = str(tmp_path / "decisions.db")
    cache = DecisionCache(path=path)
    cache.set(_STATE, _RESULT)
    cache.clear()
    assert cache.size == 0
    assert cache.get(_STATE) is None
    assert DecisionCache(path=path).get(_STATE) is None


def test_decision_cache_sqlite_different_keys_independent(tmp_path):
    cache = DecisionCache(path=str(tmp_path / "decisions.db"))
    other_state = dict(_STATE, rho=0.7)
    cache.set(_STATE, _RESULT)
    cache.set(other_state, {"move_1": 1})
    assert cache.get(_STATE) == _RESULT
    assert cache.get(other_state) == {"move_1": 1}
    assert cache.size == 2


def test_prompt_cache_sqlite_roundtrip_and_stats(tmp_path):
    cache = PromptCache(path=str(tmp_path / "prefixes.db"))
    prefix = cache.get_or_build("branch", lambda: "static prefix")
    assert prefix == "static prefix"
    assert cache.misses == 1 and cache.hits == 0
    assert cache.get_or_build("branch", lambda: "other") == "static prefix"
    assert cache.hits == 1 and cache.size == 1


def test_prompt_cache_sqlite_persists_across_instances(tmp_path):
    path = str(tmp_path / "prefixes.db")
    PromptCache(path=path).get_or_build("branch", lambda: "static prefix")
    reloaded = PromptCache(path=path)
    assert reloaded.get("branch") == "static prefix"
    assert reloaded.size == 1


def test_prompt_cache_sqlite_clear(tmp_path):
    path = str(tmp_path / "prefixes.db")
    cache = PromptCache(path=path)
    cache.get_or_build("branch", lambda: "static prefix")
    cache.clear()
    assert cache.size == 0
    assert cache.hits == 0 and cache.misses == 0
    assert PromptCache(path=path).get("branch") is None


def test_decision_cache_memory_ttl():
    import time as _time

    cache = DecisionCache(ttl_seconds=60)
    cache.set(_STATE, _RESULT)
    assert cache.get(_STATE) == _RESULT
    # 把内存中的条目回拨到 TTL 之外。
    k = cache.key(_STATE, (1, 2, 3))
    value, _ = cache._store[k]
    cache._store[k] = (value, _time.time() - 120)
    assert cache.get(_STATE) is None
    assert cache.size == 0


def test_decision_cache_sqlite_ttl(tmp_path):
    import sqlite3

    path = str(tmp_path / "decisions.db")
    cache = DecisionCache(path=path, ttl_seconds=60)
    cache.set(_STATE, _RESULT)
    assert cache.get(_STATE) == _RESULT
    # 把该行回拨到 TTL 之外，然后再次读取。
    conn = sqlite3.connect(path)
    conn.execute("UPDATE decisions SET inserted_at = inserted_at - 120")
    conn.commit()
    conn.close()
    assert cache.get(_STATE) is None
    assert cache.size == 0


def test_prompt_cache_sqlite_ttl(tmp_path):
    import sqlite3

    path = str(tmp_path / "prefixes.db")
    cache = PromptCache(path=path, ttl_seconds=60)
    assert cache.get_or_build("branch", lambda: "static prefix") == "static prefix"
    conn = sqlite3.connect(path)
    conn.execute("UPDATE prompt_prefixes SET inserted_at = inserted_at - 120")
    conn.commit()
    conn.close()
    # 过期的条目算未命中并被重建。
    assert cache.get_or_build("branch", lambda: "rebuilt") == "rebuilt"
    assert cache.misses == 2
