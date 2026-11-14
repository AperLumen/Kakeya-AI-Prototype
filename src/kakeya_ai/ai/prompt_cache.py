"""面向 KV-cache 友好的轻量 prompt 前缀缓存。

Ollama（经由 llama.cpp）可以跨请求复用共享 prompt 前缀的 KV 状态。本模块让
这件事显式化：它保存 prompt 模板的静态部分，并在末尾追加动态状态块，从而让
重复调用共享尽量长的前缀。

缓存本身只是应用层的簿记；真正的 KV 复用由 Ollama 服务端执行。

两种后端：

- 内存（默认）：不带 ``path`` 的 ``PromptCache()``；进程退出后前缀即丢失。
- SQLite（``PromptCache(path="...")``）：前缀在重启后仍存活。key 以 SHA-256
  哈希存储（主键），另存前缀文本与插入时间戳。线程安全依赖单个连接
  （``check_same_thread=False``）加一把锁。

可选 TTL：``ttl_seconds``（两种后端）在读取时使条目过期——过期条目被视为
未命中、会重建并重新缓存。``None``（默认）表示条目永久保留，直到 ``clear()``。
命中／未命中统计始终是进程本地的，即使在 SQLite 模式下也是如此。
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prompt_prefixes (
    key_hash TEXT PRIMARY KEY,
    key TEXT NOT NULL,
    prefix TEXT NOT NULL,
    inserted_at REAL NOT NULL
);
"""


@dataclass
class PromptCache:
    """缓存静态 prompt 前缀并记录每个前缀被复用的次数。

    Parameters
    ----------
    path : str, optional
        持久化后端的 SQLite 文件路径。若为 None（默认），前缀存放在普通的内存 dict 中。
    ttl_seconds : float, optional
        若设置，则读取时早于该秒数的条目被视为不存在并被重建（两种后端皆然）。
        None（默认）表示不过期。
    """

    path: Optional[str] = None
    ttl_seconds: Optional[float] = None
    _prefixes: Dict[str, tuple] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _conn: Optional[sqlite3.Connection] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            with self._lock:
                self._conn.executescript(_SCHEMA)
                self._conn.commit()

    @staticmethod
    def _key_hash(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _expired(self, inserted_at: float) -> bool:
        return self.ttl_seconds is not None and (time.time() - inserted_at) > self.ttl_seconds

    def _load(self, key: str) -> Optional[str]:
        if self._conn is None:
            entry = self._prefixes.get(key)
            if entry is None:
                return None
            prefix, inserted_at = entry
            if self._expired(inserted_at):
                del self._prefixes[key]
                return None
            return prefix
        key_hash = self._key_hash(key)
        with self._lock:
            row = self._conn.execute(
                "SELECT prefix, inserted_at FROM prompt_prefixes WHERE key_hash=?",
                (key_hash,),
            ).fetchone()
        if row is None:
            return None
        if self._expired(row[1]):
            with self._lock:
                self._conn.execute(
                    "DELETE FROM prompt_prefixes WHERE key_hash=?", (key_hash,)
                )
                self._conn.commit()
            return None
        return row[0]

    def _save(self, key: str, prefix: str) -> None:
        if self._conn is None:
            self._prefixes[key] = (prefix, time.time())
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO prompt_prefixes (key_hash, key, prefix, inserted_at) "
                "VALUES (?, ?, ?, ?)",
                (self._key_hash(key), key, prefix, time.time()),
            )
            self._conn.commit()

    def get_or_build(self, key: str, builder: Callable[[], str]) -> str:
        """返回缓存的前缀，或构建并缓存之。"""
        prefix = self._load(key)
        if prefix is not None:
            self.hits += 1
            return prefix
        prefix = builder()
        self._save(key, prefix)
        self.misses += 1
        return prefix

    def get(self, key: str) -> Optional[str]:
        """返回缓存的前缀而不构建它；没有则返回 None。"""
        prefix = self._load(key)
        if prefix is not None:
            self.hits += 1
        return prefix

    def clear(self) -> None:
        """清空所有缓存前缀并重置统计。"""
        if self._conn is None:
            self._prefixes.clear()
        else:
            with self._lock:
                self._conn.execute("DELETE FROM prompt_prefixes")
                self._conn.commit()
        self.hits = 0
        self.misses = 0

    @property
    def size(self) -> int:
        if self._conn is None:
            return len(self._prefixes)
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM prompt_prefixes").fetchone()
        return int(row[0])

    @staticmethod
    def build_full_prompt(prefix: str, dynamic_block: str) -> str:
        """拼接静态前缀与动态块。

        动态块放在末尾，这样前缀能跨请求被最大程度地共享。
        """
        return f"{prefix}\n\n{dynamic_block}"
