"""火山引擎(豆包)语音能力（可选高端档）

- DoubaoASR:      流式语音识别 (Seed-ASR)
- DoubaoTTS:      大模型 TTS V3 双向流式
- DoubaoRealtime: 端到端实时语音对话 (S2S)
- protocol:       三者共用的二进制帧协议

依赖 websockets。鉴权用 appid + access_token（环境变量 DOUBAO_APPID /
DOUBAO_ACCESS_TOKEN）。Resource-Id 因套餐而异，构造时可覆盖。
"""

from . import protocol
from .asr import DoubaoASR
from .realtime import DoubaoRealtime
from .tts import DoubaoTTS

__all__ = ["protocol", "DoubaoASR", "DoubaoTTS", "DoubaoRealtime"]
