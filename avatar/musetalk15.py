"""MuseTalk 1.5 版本/权重解析

MuseTalk 1.5 相对 1.0 是「权重 + 配置」升级（训练加了 perceptual / GAN / sync loss，
视觉保真和唇形精度更好），推理代码结构基本一致。这里提供一个版本解析器，
让上层按 `--musetalk_version` 切换到 1.5 的权重目录与配置，而不改动已跑通的渲染逻辑。

权重下载（上游 TMElyralab/MuseTalk）：
    huggingface-cli download TMElyralab/MuseTalk --local-dir models/
    1.5 权重位于 models/musetalkV15/unet.pth + musetalk.json

作为低端 GPU 兜底方案：MuseTalk 4GB VRAM 即可跑 30+ FPS。
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class MuseTalkWeights:
    version: str
    unet_path: str
    unet_config: str


def resolve_weights(models_root: str = "models", version: str = "1.5") -> MuseTalkWeights:
    """返回指定版本的 UNet 权重 + 配置路径。"""
    if version in ("1.5", "v15", "musetalkV15"):
        base = os.path.join(models_root, "musetalkV15")
        return MuseTalkWeights(
            version="1.5",
            unet_path=os.path.join(base, "unet.pth"),
            unet_config=os.path.join(base, "musetalk.json"),
        )
    # 1.0 默认
    base = os.path.join(models_root, "musetalk")
    return MuseTalkWeights(
        version="1.0",
        unet_path=os.path.join(base, "pytorch_model.bin"),
        unet_config=os.path.join(base, "musetalk.json"),
    )


def weights_available(weights: MuseTalkWeights) -> bool:
    return os.path.isfile(weights.unet_path) and os.path.isfile(weights.unet_config)
