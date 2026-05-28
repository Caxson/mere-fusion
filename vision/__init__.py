"""视觉理解（2026 升级）

用 Qwen 多模态大模型统一替代 YOLO + DeepFace + EasyOCR。
- FrameBuffer: 1Hz 截帧节流缓存
- QwenVision: 图文一起发给 VL 模型做理解
"""

from .qwen_vl import FrameBuffer, QwenVision

__all__ = ["FrameBuffer", "QwenVision"]
