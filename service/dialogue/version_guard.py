"""版本号控制

客户打断后会产生新 question，旧的 LLM/TTS 推送链路应当立刻停止。
每个 session 维护一个单调递增 version：
- 新 question 进来 → bump(session) 得到新 version
- 旧链路的每个 slice 消费前 → is_expired(session, my_version) 检查
  若 latest > my_version → 该链路已过期，跳过后续输出

单进程内存实现（原分布式系统用 Redis，这里单机够用）。
"""

from __future__ import annotations

import threading


class VersionGuard:
    def __init__(self):
        self._versions: dict[str, int] = {}
        self._lock = threading.Lock()

    def bump(self, session_id: str) -> int:
        """开启新一轮，返回新 version。"""
        with self._lock:
            v = self._versions.get(session_id, 0) + 1
            self._versions[session_id] = v
            return v

    def current(self, session_id: str) -> int:
        with self._lock:
            return self._versions.get(session_id, 0)

    def is_expired(self, session_id: str, version: int) -> bool:
        """my version 是否已被更新的 question 覆盖。"""
        with self._lock:
            return self._versions.get(session_id, 0) > version

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._versions.pop(session_id, None)
