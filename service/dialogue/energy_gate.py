"""自适应能量门 (AdaptiveEnergyGate)

两层降噪的核心：在送入 ASR 之前，用「双门限滞回 + HFER 高频能量比二次确认」
把环境噪声 / 远场弱混响 / 电流噪声挡在门外，只放行真正的人声段。

设计要点：
- 状态机 IDLE ↔ SPEAKING，双门限滞回避免临界抖动
- HFER (High-Frequency Energy Ratio) 二次确认：噪声频谱单一，人声高频占比更高
- AI 播放期冻结噪声基线，防止 TTS 回声污染基线导致后续误判
- 句首 lookback：进入 SPEAKING 时补发前几帧，避免切掉句首

输入帧约定（与 baseasr.py 一致）：16kHz 单声道 float32，每帧 320 样本（20ms）。
文档里的能量阈值是按 int16 标度调的，内部统一换算到 int16 标度计算 RMS。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GateState(Enum):
    IDLE = "IDLE"
    SPEAKING = "SPEAKING"


@dataclass
class GateConfig:
    sample_rate: int = 16000
    high_multiplier: float = 3.0          # IDLE → SPEAKING 触发倍数
    low_multiplier: float = 1.5           # SPEAKING → IDLE 触发倍数
    exit_debounce_frames: int = 5         # 退出去抖帧数 (~100ms @ 20ms/帧)
    lookback_frames: int = 4              # 句首保护缓冲 (~80ms)
    floor_init: float = 100.0             # 噪声基线初值 (int16 标度)
    floor_min: float = 50.0               # 基线下限，防止过低导致误触发
    ema_alpha: float = 0.05               # 基线 EMA 学习率
    hfer_enabled: bool = True
    hfer_threshold: float = 0.05          # 高频/低频能量比阈值
    hfer_ema_alpha: float = 0.2           # HFER EMA 平滑
    fft_size: int = 256
    low_band: tuple[int, int] = (200, 2000)    # Hz
    high_band: tuple[int, int] = (2000, 4000)  # Hz


@dataclass
class GateResult:
    """单帧处理结果。

    passed: 该帧是否应送入 ASR
    emit_frames: 实际要送入 ASR 的帧序列（进入 SPEAKING 时含 lookback 补发帧）
    state: 处理后状态
    rms: 当前帧 RMS（int16 标度）
    hfer: 当前帧 HFER EMA 值
    """

    passed: bool
    emit_frames: list[np.ndarray]
    state: GateState
    rms: float
    hfer: float


class AdaptiveEnergyGate:
    def __init__(self, config: GateConfig | None = None):
        self.cfg = config or GateConfig()
        self.state = GateState.IDLE
        self.noise_floor = self.cfg.floor_init
        self.hfer_ema = 0.0
        self._below_count = 0
        self._lookback: deque[np.ndarray] = deque(maxlen=self.cfg.lookback_frames)
        self._is_playing = False
        # 统计
        self.switches = 0
        self.gated = 0
        self.passed = 0
        self.hfer_rejected = 0
        # 预计算 FFT 频段 bin 范围
        freqs = np.fft.rfftfreq(self.cfg.fft_size, d=1.0 / self.cfg.sample_rate)
        self._low_bins = np.where(
            (freqs >= self.cfg.low_band[0]) & (freqs < self.cfg.low_band[1])
        )[0]
        self._high_bins = np.where(
            (freqs >= self.cfg.high_band[0]) & (freqs < self.cfg.high_band[1])
        )[0]

    def set_playing(self, playing: bool) -> None:
        """AI 是否正在播放音频。播放期冻结噪声基线更新。"""
        self._is_playing = playing

    def reset(self) -> None:
        self.state = GateState.IDLE
        self.noise_floor = self.cfg.floor_init
        self.hfer_ema = 0.0
        self._below_count = 0
        self._lookback.clear()

    @staticmethod
    def _to_int16_scale(frame: np.ndarray) -> np.ndarray:
        """把 float32 [-1,1] 帧换算到 int16 标度用于能量计算。"""
        if frame.dtype == np.int16:
            return frame.astype(np.float32)
        # 兼容已经是 int16 标度的 float
        peak = np.max(np.abs(frame)) if frame.size else 0.0
        if peak <= 1.5:  # 认为是 [-1,1] 归一化
            return frame.astype(np.float32) * 32768.0
        return frame.astype(np.float32)

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

    def _compute_hfer(self, samples: np.ndarray) -> float:
        """高频能量比：high_band 能量 / low_band 能量。"""
        n = self.cfg.fft_size
        buf = samples[:n]
        if buf.size < n:
            buf = np.pad(buf, (0, n - buf.size))
        window = np.hanning(n)
        spectrum = np.abs(np.fft.rfft(buf * window)) ** 2
        low_e = float(np.sum(spectrum[self._low_bins])) + 1e-9
        high_e = float(np.sum(spectrum[self._high_bins]))
        return high_e / low_e

    def process(self, frame: np.ndarray) -> GateResult:
        samples = self._to_int16_scale(np.asarray(frame).flatten())
        rms = self._rms(samples)

        if self.cfg.hfer_enabled:
            hfer = self._compute_hfer(samples)
            self.hfer_ema = (
                self.cfg.hfer_ema_alpha * hfer
                + (1 - self.cfg.hfer_ema_alpha) * self.hfer_ema
            )

        emit: list[np.ndarray] = []

        if self.state == GateState.IDLE:
            self._lookback.append(frame)
            high_thresh = self.noise_floor * self.cfg.high_multiplier
            hfer_ok = (not self.cfg.hfer_enabled) or (
                self.hfer_ema >= self.cfg.hfer_threshold
            )
            if rms > high_thresh and hfer_ok:
                # IDLE → SPEAKING：补发 lookback 帧保护句首
                self.state = GateState.SPEAKING
                self.switches += 1
                self._below_count = 0
                emit = list(self._lookback)
                self._lookback.clear()
                self.passed += len(emit)
                return GateResult(True, emit, self.state, rms, self.hfer_ema)
            else:
                if rms > high_thresh and not hfer_ok:
                    self.hfer_rejected += 1
                # 播放期冻结基线，否则用 EMA 缓慢跟踪环境噪声
                if not self._is_playing:
                    self.noise_floor = max(
                        self.cfg.floor_min,
                        (1 - self.cfg.ema_alpha) * self.noise_floor
                        + self.cfg.ema_alpha * rms,
                    )
                self.gated += 1
                return GateResult(False, [], self.state, rms, self.hfer_ema)

        # SPEAKING 状态
        low_thresh = self.noise_floor * self.cfg.low_multiplier
        if rms < low_thresh:
            self._below_count += 1
            if self._below_count >= self.cfg.exit_debounce_frames:
                # SPEAKING → IDLE
                self.state = GateState.IDLE
                self.switches += 1
                self._below_count = 0
                self._lookback.clear()
        else:
            self._below_count = 0

        # SPEAKING 期间所有帧都透传
        self.passed += 1
        return GateResult(True, [frame], self.state, rms, self.hfer_ema)

    def stats(self) -> dict:
        return {
            "state": self.state.value,
            "noise_floor": round(self.noise_floor, 1),
            "hfer_ema": round(self.hfer_ema, 4),
            "switches": self.switches,
            "gated": self.gated,
            "passed": self.passed,
            "hfer_rejected": self.hfer_rejected,
        }
