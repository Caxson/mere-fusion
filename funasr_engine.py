"""FunASR 流式 ASR 引擎

中文场景下 Paraformer-streaming 比 Whisper 延迟更低、CER 更优。
基于 FunASR 官方流式 demo 的 chunk + cache 范式：cache 必须跨 chunk 复用，
否则退化成非流式。

依赖：pip install funasr
模型：paraformer-zh-streaming（首次运行自动从 ModelScope 下载）

约定（与 baseasr.py 一致）：16kHz 单声道 float32。
"""

from __future__ import annotations

import numpy as np


class FunASRStreaming:
    def __init__(
        self,
        model: str = "paraformer-zh-streaming",
        device: str = "cpu",
        chunk_size: tuple[int, int, int] = (0, 10, 5),  # [0,10,5]=600ms; [0,8,4]=480ms
        encoder_chunk_look_back: int = 4,
        decoder_chunk_look_back: int = 1,
        sample_rate: int = 16000,
    ):
        from funasr import AutoModel  # 延迟导入，未装 funasr 时不影响其他引擎

        self.model = AutoModel(model=model, device=device, disable_update=True)
        self.chunk_size = list(chunk_size)
        self.encoder_chunk_look_back = encoder_chunk_look_back
        self.decoder_chunk_look_back = decoder_chunk_look_back
        self.sample_rate = sample_rate
        # 每 chunk 采样点数：chunk_size[1] * 960 (60ms × 16k)
        self.chunk_stride = self.chunk_size[1] * 960
        self._cache: dict = {}

    def reset(self) -> None:
        """一段语音结束后清空流式状态。"""
        self._cache = {}

    def feed_chunk(self, audio_chunk: np.ndarray, is_final: bool = False) -> str:
        """喂一个音频 chunk，返回该 chunk 的增量识别文本（可能为空）。"""
        chunk = np.asarray(audio_chunk).flatten().astype(np.float32)
        res = self.model.generate(
            input=chunk,
            cache=self._cache,
            is_final=is_final,
            chunk_size=self.chunk_size,
            encoder_chunk_look_back=self.encoder_chunk_look_back,
            decoder_chunk_look_back=self.decoder_chunk_look_back,
        )
        if res and isinstance(res, list) and "text" in res[0]:
            return res[0]["text"]
        return ""

    def transcribe(self, audio: np.ndarray, is_final: bool = True) -> str:
        """离线整段转写：按 chunk_stride 切块逐块喂，拼接结果。"""
        audio = np.asarray(audio).flatten().astype(np.float32)
        self.reset()
        total = int((len(audio) - 1) / self.chunk_stride + 1) if len(audio) else 0
        out = []
        for i in range(total):
            seg = audio[i * self.chunk_stride : (i + 1) * self.chunk_stride]
            last = i == total - 1
            text = self.feed_chunk(seg, is_final=last or is_final and last)
            if text:
                out.append(text)
        self.reset()
        return "".join(out)
