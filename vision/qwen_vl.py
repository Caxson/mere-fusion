"""Qwen 多模态视觉理解

用一个多模态大模型（Qwen3.x-VL）替代原来的 YOLO + DeepFace + EasyOCR 三件套：
物体/人脸/情绪/文字理解统一交给 VL 模型，代码量大幅下降，理解能力更强。

工作方式（实时场景）：
- 每秒截 1 帧画面缓存
- 用户说完一句话时，把「最近一帧图 + 这句话文本」一起发给 VL 模型，
  得到结合画面的理解/回答

接口：DashScope OpenAI 兼容模式，环境变量 DASHSCOPE_API_KEY。
图像以 base64 data URL 放进 OpenAI 多模态 messages 格式。
"""

from __future__ import annotations

import base64
import io
import os
import time
from threading import Lock

import numpy as np

DASHSCOPE_OPENAI_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _encode_image(frame) -> str:
    """np.ndarray(BGR/RGB) 或 PIL.Image → base64 data URL (JPEG)。"""
    from PIL import Image

    if isinstance(frame, np.ndarray):
        arr = frame
        # OpenCV 常见 BGR → 转 RGB
        if arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1]
        img = Image.fromarray(arr.astype(np.uint8))
    else:
        img = frame
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class FrameBuffer:
    """1Hz 截帧缓存：只保留最近一帧（线程安全）。"""

    def __init__(self, interval_sec: float = 1.0):
        self.interval = interval_sec
        self._last_ts = 0.0
        self._frame = None
        self._lock = Lock()

    def offer(self, frame) -> bool:
        """送入一帧，按间隔节流。返回是否被采纳。"""
        now = time.monotonic()
        if now - self._last_ts < self.interval:
            return False
        with self._lock:
            self._frame = frame
            self._last_ts = now
        return True

    def latest(self):
        with self._lock:
            return self._frame


class QwenVision:
    def __init__(
        self,
        model: str = "qwen3.6-plus",
        api_key: str | None = None,
        base_url: str = DASHSCOPE_OPENAI_BASE,
        system_prompt: str = "你是数字人的视觉助手，结合画面内容自然地回答用户。",
    ):
        from openai import OpenAI

        self.model = model
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=base_url,
        )

    def understand(self, text: str, frame=None) -> str:
        """把文本（+可选画面）一起发给 VL 模型，返回理解结果。"""
        content: list[dict] = [{"type": "text", "text": text}]
        if frame is not None:
            content.insert(
                0, {"type": "image_url", "image_url": {"url": _encode_image(frame)}}
            )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
        )
        return resp.choices[0].message.content
