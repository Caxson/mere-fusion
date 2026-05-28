"""数字人渲染后端封装（2026 升级）

- SoulXFlashHead: 实时流式数字人（最佳实时方案，~96fps@4090）
- EchoMimicV3:    离线半身+手势（录视频最佳，AAAI 2026）
- musetalk15:     MuseTalk 1.5 权重解析（低端 GPU 兜底）

重依赖均为延迟导入：未安装对应上游环境时，import avatar 不报错，
只有在实际构造对应类时才需要依赖就位。
"""

from . import musetalk15

__all__ = ["musetalk15", "get_soulx", "get_echomimic_v3"]


def get_soulx(*args, **kwargs):
    """惰性构造 SoulXFlashHead，避免无 flash_head 环境时 import 失败。"""
    from .soulx_flashhead import SoulXFlashHead
    return SoulXFlashHead(*args, **kwargs)


def get_echomimic_v3(*args, **kwargs):
    from .echomimic_v3 import EchoMimicV3
    return EchoMimicV3(*args, **kwargs)
