"""ASR final 条件合并窗

解决 VAD 把一个完整问题切碎、导致 AI 抢话的问题。

逻辑：
- 收到 final 时，看距离上一次 partial 的时间间隔：
  - >= quick_fire_ms (默认 500ms)：客户已停止说话 → 立即 fire
  - <  quick_fire_ms：客户可能还在说 → 排一个 merge_window_ms (默认 1.2s) 的窗
    - 窗内若来新 partial → 取消重排（说明还在说）
    - 窗到期且未被新 final 覆盖 → fire 累加内容
- 用单调递增的 pending_seq 判断窗是否被新 final 覆盖

依赖 asyncio。fire 通过回调 on_fire(full_content, new_content) 触发。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass
class MergeConfig:
    quick_fire_ms: int = 500       # 距上次 partial 超过此值 → 立即 fire
    merge_window_ms: int = 1200    # 合并窗时长


FireCallback = Callable[[str, str], Awaitable[None]]


class SentenceMergeWindow:
    def __init__(self, on_fire: FireCallback, config: MergeConfig | None = None):
        self.cfg = config or MergeConfig()
        self._on_fire = on_fire
        self._last_partial_ts = 0.0
        self._accumulated = ""
        self._pending_seq = 0
        self._pending_task: asyncio.Task | None = None

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0

    def on_partial(self, text: str) -> None:
        """ASR 中间识别。刷新时间戳；若有待发窗则取消（客户还在说）。"""
        self._last_partial_ts = self._now_ms()
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

    async def on_final(self, text: str) -> None:
        """ASR 最终识别。决定立即 fire 还是排合并窗。"""
        self._accumulated = (self._accumulated + text).strip()
        since_partial = self._now_ms() - self._last_partial_ts

        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
            self._pending_task = None

        if since_partial >= self.cfg.quick_fire_ms:
            await self._fire(text)
        else:
            self._pending_seq += 1
            seq = self._pending_seq
            self._pending_task = asyncio.ensure_future(self._delayed_fire(seq, text))

    async def _delayed_fire(self, seq: int, new_content: str) -> None:
        try:
            await asyncio.sleep(self.cfg.merge_window_ms / 1000.0)
        except asyncio.CancelledError:
            return
        if seq == self._pending_seq:
            await self._fire(new_content)

    async def _fire(self, new_content: str) -> None:
        full = self._accumulated
        self._accumulated = ""
        self._pending_seq += 1  # 使任何在途窗失效
        await self._on_fire(full, new_content)

    def reset(self) -> None:
        if self._pending_task and not self._pending_task.done():
            self._pending_task.cancel()
        self._pending_task = None
        self._accumulated = ""
        self._last_partial_ts = 0.0
