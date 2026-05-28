"""对话质量链路

把「单向播放 → 等用户说完 → 处理 → 回复」的玩具式对话，升级到生产级：
- AdaptiveEnergyGate: 双层降噪（双门限滞回 + HFER 二次确认 + 播放期冻结基线）
- SentenceMergeWindow: ASR final 条件合并窗，避免 VAD 切碎完整问题导致抢话
- InterruptDecider: 三层打断保护（开场白/句首/意图判定）
- VersionGuard: 版本号控制，打断后旧 LLM/TTS 链路自动失效
"""

from .energy_gate import AdaptiveEnergyGate, GateConfig, GateResult, GateState
from .interrupt import InterruptConfig, InterruptDecider
from .merge_window import MergeConfig, SentenceMergeWindow
from .version_guard import VersionGuard

__all__ = [
    "AdaptiveEnergyGate",
    "GateConfig",
    "GateResult",
    "GateState",
    "SentenceMergeWindow",
    "MergeConfig",
    "InterruptDecider",
    "InterruptConfig",
    "VersionGuard",
]
