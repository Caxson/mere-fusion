"""三层打断保护 + 打断意图判定

当用户说话时，AI 是否应该停下来（打断）取决于：
- AI 没在播放 → 直接走 LLM（不存在"打断"问题）
- AI 在播放 → 检查三层保护：
    1. 开场白保护期：开场白播放期间不打断 (TTL = 文本长度 × per_char_ms)
    2. AI 句首保护期：每句 TTS 首帧后 sentence_guard_ms 内不打断
    3. 意图判定：判断用户这句到底是「真打断」还是「附和/噪声」
        - 优先用 LLM 二分类（硬超时，超时默认不打断 = 保守）
        - 无 LLM 时退化为编辑距离匹配语气词库

关键设计：意图判定喂的是 full_content（累加的全部 ASR final），
而非单次 new_content，避免碎片化 final 被逐个判成「不打断」吃掉。

时间相关状态用单调时钟 + TTL，单进程内存实现。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher


@dataclass
class InterruptConfig:
    opening_guard_per_char_ms: int = 200   # 开场白保护 TTL = 文本长度 × 此值
    sentence_guard_ms: int = 2000          # AI 句首保护期
    classifier_timeout_ms: int = 300       # LLM 分类硬超时
    edit_distance_long_threshold: int = 8  # 超过此长度直接判为问题（打断）
    edit_distance_similarity: float = 0.8  # 与语气词相似度 ≥ 此值 → 语气词（不打断）
    modal_particles_path: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(__file__), "modal_particles.txt"
        )
    )


# LLM 二分类器：输入累加内容，返回 True=打断 / False=不打断。
# 超时或异常由 InterruptDecider 兜底为 False。
InterruptClassifier = Callable[[str], bool]


class InterruptDecider:
    def __init__(
        self,
        config: InterruptConfig | None = None,
        classifier: InterruptClassifier | None = None,
    ):
        self.cfg = config or InterruptConfig()
        self._classifier = classifier
        self._opening_until: dict[str, float] = {}
        self._sentence_until: dict[str, float] = {}
        self._modal_particles = self._load_modal_particles()

    def _load_modal_particles(self) -> list[str]:
        try:
            with open(self.cfg.modal_particles_path, encoding="utf-8") as f:
                return [ln.strip() for ln in f if ln.strip()]
        except OSError:
            return ["嗯", "啊", "哦", "对", "好", "好的", "知道了"]

    @staticmethod
    def _now_ms() -> float:
        return time.monotonic() * 1000.0

    # ── 保护期标记 ──────────────────────────────────────────
    def mark_opening(self, session_id: str, opening_text: str) -> None:
        ttl = max(1, len(opening_text)) * self.cfg.opening_guard_per_char_ms
        self._opening_until[session_id] = self._now_ms() + ttl

    def mark_sentence_start(self, session_id: str) -> None:
        """每句 TTS 首帧到达时调用。"""
        self._sentence_until[session_id] = self._now_ms() + self.cfg.sentence_guard_ms

    def _in_opening_guard(self, session_id: str) -> bool:
        return self._now_ms() < self._opening_until.get(session_id, 0.0)

    def _in_sentence_guard(self, session_id: str) -> bool:
        return self._now_ms() < self._sentence_until.get(session_id, 0.0)

    # ── 主决策 ──────────────────────────────────────────────
    def should_interrupt(
        self, session_id: str, full_content: str, is_playing: bool
    ) -> bool:
        """返回是否应打断当前 AI 播放。"""
        if not is_playing:
            return True  # 没在播放，等同于直接回应
        if self._in_opening_guard(session_id):
            return False
        if self._in_sentence_guard(session_id):
            return False
        return self._classify(full_content)

    def _classify(self, content: str) -> bool:
        content = (content or "").strip()
        if not content:
            return False
        if self._classifier is not None:
            try:
                return bool(self._run_with_timeout(content))
            except Exception:
                return False  # 超时/异常 → 保守不打断
        return self._edit_distance_fallback(content)

    def _run_with_timeout(self, content: str) -> bool:
        """在硬超时内运行 LLM 分类器，超时抛异常由上层兜底。"""
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._classifier, content)
            return fut.result(timeout=self.cfg.classifier_timeout_ms / 1000.0)

    def _edit_distance_fallback(self, content: str) -> bool:
        # 长内容直接判为实质问题
        if len(content) > self.cfg.edit_distance_long_threshold:
            return True
        # 短内容与语气词库比相似度
        for particle in self._modal_particles:
            sim = SequenceMatcher(None, content, particle).ratio()
            if sim >= self.cfg.edit_distance_similarity:
                return False  # 是语气词，不打断
        return True  # 不像语气词，判为打断

    def clear(self, session_id: str) -> None:
        self._opening_until.pop(session_id, None)
        self._sentence_until.pop(session_id, None)
