"""对话质量链路单元测试。

运行: pytest tests/test_dialogue.py -v
"""

import asyncio
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from service.dialogue import (
    AdaptiveEnergyGate,
    GateConfig,
    GateState,
    InterruptConfig,
    InterruptDecider,
    MergeConfig,
    SentenceMergeWindow,
    VersionGuard,
)

SR = 16000
FRAME = 320  # 20ms @ 16kHz


def _speech_frame(amp=8000.0, freq=300.0, n=FRAME, sr=SR):
    """模拟人声帧：低频基频 + 高频谐波（HFER 较高）。int16 标度。"""
    t = np.arange(n) / sr
    sig = amp * (np.sin(2 * np.pi * freq * t) + 0.5 * np.sin(2 * np.pi * 2500 * t))
    return (sig / 32768.0).astype(np.float32)  # 归一化到 [-1,1]


def _noise_frame(amp=120.0, n=FRAME):
    """模拟低能量环境噪声帧。"""
    rng = np.random.default_rng(42)
    return ((rng.standard_normal(n) * amp) / 32768.0).astype(np.float32)


# ── AdaptiveEnergyGate ───────────────────────────────────────

class TestEnergyGate:
    def test_idle_gates_noise(self):
        gate = AdaptiveEnergyGate()
        passed_any = False
        for _ in range(20):
            r = gate.process(_noise_frame())
            passed_any = passed_any or r.passed
        assert not passed_any
        assert gate.state == GateState.IDLE

    def test_speech_opens_gate(self):
        gate = AdaptiveEnergyGate(GateConfig(floor_init=100.0))
        opened = False
        for _ in range(10):
            r = gate.process(_speech_frame())
            if r.state == GateState.SPEAKING:
                opened = True
                break
        assert opened

    def test_lookback_emitted_on_open(self):
        """进入 SPEAKING 时应补发 lookback 帧保护句首。"""
        cfg = GateConfig(lookback_frames=4)
        gate = AdaptiveEnergyGate(cfg)
        # 先灌几帧噪声填充 lookback 缓冲
        for _ in range(4):
            gate.process(_noise_frame())
        r = gate.process(_speech_frame())
        # 第一次开门会补发 lookback（>1 帧）
        if r.state == GateState.SPEAKING:
            assert len(r.emit_frames) >= 1

    def test_hysteresis_exit_debounce(self):
        """退出需要连续多帧低能量（去抖），单帧低能量不退出。"""
        cfg = GateConfig(exit_debounce_frames=5)
        gate = AdaptiveEnergyGate(cfg)
        for _ in range(10):
            gate.process(_speech_frame())
        assert gate.state == GateState.SPEAKING
        # 单帧静音不应退出
        gate.process(_noise_frame())
        assert gate.state == GateState.SPEAKING
        # 连续 5 帧静音后退出
        for _ in range(5):
            gate.process(_noise_frame())
        assert gate.state == GateState.IDLE

    def test_playing_freezes_floor(self):
        gate = AdaptiveEnergyGate()
        gate.set_playing(True)
        floor_before = gate.noise_floor
        for _ in range(10):
            gate.process(_noise_frame(amp=300.0))
        assert gate.noise_floor == floor_before  # 播放期基线冻结


# ── SentenceMergeWindow ──────────────────────────────────────

class TestMergeWindow:
    def test_immediate_fire_when_paused(self):
        fired = []

        async def on_fire(full, new):
            fired.append((full, new))

        async def run():
            mw = SentenceMergeWindow(on_fire, MergeConfig(quick_fire_ms=500))
            # 不调 on_partial → last_partial_ts=0 → 间隔很大 → 立即 fire
            await mw.on_final("你好我想咨询一下")
            await asyncio.sleep(0.05)

        asyncio.run(run())
        assert len(fired) == 1
        assert fired[0][0] == "你好我想咨询一下"

    def test_merge_window_waits(self):
        fired = []

        async def on_fire(full, new):
            fired.append(full)

        async def run():
            mw = SentenceMergeWindow(on_fire, MergeConfig(quick_fire_ms=500, merge_window_ms=300))
            mw.on_partial("我想")          # 刚说过 partial
            await mw.on_final("我想问")     # 间隔 < 500ms → 排窗
            assert len(fired) == 0          # 窗内还没 fire
            await asyncio.sleep(0.4)        # 等过窗
            assert len(fired) == 1

        asyncio.run(run())

    def test_new_partial_cancels_window(self):
        fired = []

        async def on_fire(full, new):
            fired.append(full)

        async def run():
            mw = SentenceMergeWindow(on_fire, MergeConfig(quick_fire_ms=500, merge_window_ms=300))
            mw.on_partial("我想")
            await mw.on_final("我想问")     # 排窗
            await asyncio.sleep(0.1)
            mw.on_partial("我想问的是")     # 新 partial → 取消窗
            await asyncio.sleep(0.3)
            assert len(fired) == 0          # 被取消，没 fire

        asyncio.run(run())


# ── InterruptDecider ─────────────────────────────────────────

class TestInterrupt:
    def test_not_playing_always_interrupt(self):
        d = InterruptDecider()
        assert d.should_interrupt("s1", "随便什么", is_playing=False) is True

    def test_opening_guard_blocks(self):
        d = InterruptDecider()
        d.mark_opening("s1", "您好这里是测试")  # TTL = 7×200=1400ms
        assert d.should_interrupt("s1", "这是一个实质问题对吧", is_playing=True) is False

    def test_sentence_guard_blocks(self):
        d = InterruptDecider()
        d.mark_sentence_start("s1")
        assert d.should_interrupt("s1", "怎么操作", is_playing=True) is False

    def test_modal_particle_not_interrupt(self):
        d = InterruptDecider()
        assert d.should_interrupt("s1", "嗯", is_playing=True) is False
        assert d.should_interrupt("s1", "好的", is_playing=True) is False

    def test_real_question_interrupts(self):
        d = InterruptDecider()
        assert d.should_interrupt("s1", "那你刚才说的费用是什么意思", is_playing=True) is True

    def test_classifier_timeout_defaults_no_interrupt(self):
        import time

        def slow_classifier(content):
            time.sleep(1.0)  # 超过 300ms 硬超时
            return True

        d = InterruptDecider(InterruptConfig(classifier_timeout_ms=300), classifier=slow_classifier)
        assert d.should_interrupt("s1", "一段很长的实质性问题内容", is_playing=True) is False

    def test_classifier_used_when_fast(self):
        d = InterruptDecider(classifier=lambda c: True)
        assert d.should_interrupt("s1", "嗯", is_playing=True) is True  # 分类器说打断就打断


# ── VersionGuard ─────────────────────────────────────────────

class TestVersionGuard:
    def test_bump_and_expire(self):
        vg = VersionGuard()
        v1 = vg.bump("s1")
        assert v1 == 1
        assert vg.is_expired("s1", v1) is False
        v2 = vg.bump("s1")
        assert v2 == 2
        assert vg.is_expired("s1", v1) is True   # 旧版本过期
        assert vg.is_expired("s1", v2) is False

    def test_independent_sessions(self):
        vg = VersionGuard()
        vg.bump("s1")
        assert vg.current("s2") == 0
        assert vg.is_expired("s2", 0) is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
