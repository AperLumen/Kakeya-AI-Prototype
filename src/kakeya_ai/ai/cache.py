"""AI 打分决策的简单缓存。

避免对几何上相似的证明状态重复调用 LLM。

两种后端：

- 内存（默认）：不带 ``path`` 的 ``DecisionCache()``；进程退出后条目即丢失。
- SQLite（``DecisionCache(path="...")``）：条目在重启后仍存活。key 以
  SHA-256 哈希存储（主键），值以 JSON 序列化存储，另带插入时间戳。线程安全
  依赖单个连接（``check_same_thread=False``）加一把锁。

可选 TTL：``ttl_seconds``（两种后端）在读取时使条目过期——过期条目被视为
未命中并被清除。``None``（默认）表示条目永久保留，直到 ``clear()``。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    key_hash TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    inserted_at REAL NOT NULL
);
"""


@dataclass
class DecisionCache:
    """以可哈希的状态摘要为键的内容寻址缓存。

    Parameters
    ----------
    path : str, optional
        持久化后端的 SQLite 文件路径。若为 None（默认），则缓存是普通的内存 dict。
    ttl_seconds : float, optional
        若设置，则读取时早于该秒数的条目被视为不存在并被清除（两种后端皆然）。
        None（默认）表示不过期。
    """

    path: Optional[str] = None
    ttl_seconds: Optional[float] = None
    _store: Dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _conn: Optional[sqlite3.Connection] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()

    def key(self, state: dict, move_indices: tuple) -> str:
        """从状态字段的一个子集创建稳定 key。"""
        # 对浮点数取整，避免产生近乎重复的 key。
        parts = [
            round(state.get("delta", 0.0), 6),
            round(state.get("rho", 0.0), 3),
            round(state.get("local_error", 0.0), 3),
            round(state.get("volume_ratio", 0.0), 3),
            move_indices,
        ]
        return str(tuple(parts))

    @staticmethod
    def _key_hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _expired(self, inserted_at: float) -> bool:
        return self.ttl_seconds is not None and (time.time() - inserted_at) > self.ttl_seconds

    def get(self, state: dict, move_indices: tuple = (1, 2, 3)) -> Optional[dict]:
        key = self.key(state, move_indices)
        if self._conn is None:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, inserted_at = entry
            if self._expired(inserted_at):
                del self._store[key]
                return None
            return value
        key_hash = self._key_hash(key)
        with self._lock:
            row = self._conn.execute(
                "SELECT value, inserted_at FROM decisions WHERE key_hash=?", (key_hash,)
            ).fetchone()
        if row is None:
            return None
        if self._expired(row[1]):
            with self._lock:
                self._conn.execute("DELETE FROM decisions WHERE key_hash=?", (key_hash,))
                self._conn.commit()
            return None
        return json.loads(row[0])

    def set(self, state: dict, result: dict, move_indices: tuple = (1, 2, 3)) -> None:
        key = self.key(state, move_indices)
        if self._conn is None:
            self._store[key] = (result, time.time())
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions (key_hash, key, value, inserted_at) "
                "VALUES (?, ?, ?, ?)",
                (self._key_hash(key), key, json.dumps(result), time.time()),
            )
            self._conn.commit()

    def clear(self) -> None:
        if self._conn is None:
            self._store.clear()
            return
        with self._lock:
            self._conn.execute("DELETE FROM decisions")
            self._conn.commit()

    @property
    def size(self) -> int:
        if self._conn is None:
            return len(self._store)
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        return int(row[0])
