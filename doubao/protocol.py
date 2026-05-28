"""火山引擎(豆包)语音 WebSocket 二进制帧协议

ASR / TTS V3 / 实时对话三个产品共用同一套帧协议。
帧 = 4字节 header + 可选扩展字段(event / session_id / sequence / error_code)
     + 4字节 payload 长度 + payload。

header 位定义：
  byte0: protocol_version(高4) | header_size(低4, =1 表示 4 字节)
  byte1: message_type(高4)     | flags(低4)
  byte2: serialization(高4)    | compression(低4)
  byte3: reserved 0x00

协议细节经 BytePlus 官方文档 + 多个生产实现交叉核实。
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

PROTOCOL_VERSION = 0b0001
HEADER_SIZE = 0b0001  # 单位 4 字节

# message_type (高4位)
MSG_FULL_CLIENT = 0b0001       # 客户端完整请求(JSON)
MSG_AUDIO_CLIENT = 0b0010      # 客户端纯音频
MSG_FULL_SERVER = 0b1001       # 服务端完整响应
MSG_AUDIO_SERVER = 0b1011      # 服务端纯音频 / ACK
MSG_ERROR = 0b1111             # 服务端错误(扩展字段含 error_code)

# flags (低4位)
FLAG_NO_SEQ = 0b0000
FLAG_POS_SEQ = 0b0001
FLAG_LAST_NO_SEQ = 0b0010      # 尾包(无序号)
FLAG_NEG_SEQ = 0b0011          # 负序号 = 最后一包
FLAG_WITH_EVENT = 0b0100       # 带 event

# serialization (高4位 of byte2)
SERIAL_RAW = 0b0000            # 裸音频
SERIAL_JSON = 0b0001
SERIAL_THRIFT = 0b0011

# compression (低4位 of byte2)
COMPRESS_NONE = 0b0000
COMPRESS_GZIP = 0b0001

# event 枚举
EV_START_CONNECTION = 1
EV_FINISH_CONNECTION = 2
EV_CONNECTION_STARTED = 50
EV_CONNECTION_FAILED = 51
EV_CONNECTION_FINISHED = 52
EV_START_SESSION = 100
EV_CANCEL_SESSION = 101
EV_FINISH_SESSION = 102
EV_SESSION_STARTED = 150
EV_SESSION_FINISHED = 152
EV_SESSION_FAILED = 153
EV_TASK_REQUEST = 200
EV_SAY_HELLO = 300
EV_TTS_SENTENCE_START = 350
EV_TTS_SENTENCE_END = 351
EV_TTS_RESPONSE = 352          # payload 是裸音频
EV_TTS_ENDED = 359
EV_ASR_INFO = 450
EV_ASR_RESPONSE = 451
EV_ASR_ENDED = 459
EV_CHAT_TTS_TEXT = 500
EV_CHAT_RESPONSE = 550
EV_CHAT_ENDED = 559

# 连接级事件：不携带 session_id
CONNECTION_EVENTS = frozenset({
    EV_START_CONNECTION, EV_FINISH_CONNECTION,
    EV_CONNECTION_STARTED, EV_CONNECTION_FAILED, EV_CONNECTION_FINISHED,
})


def generate_header(
    message_type: int,
    flags: int = FLAG_NO_SEQ,
    serialization: int = SERIAL_JSON,
    compression: int = COMPRESS_GZIP,
) -> bytes:
    return bytes([
        (PROTOCOL_VERSION << 4) | HEADER_SIZE,
        (message_type << 4) | flags,
        (serialization << 4) | compression,
        0x00,
    ])


def build_frame(
    message_type: int,
    *,
    flags: int = FLAG_NO_SEQ,
    serialization: int = SERIAL_JSON,
    compression: int = COMPRESS_GZIP,
    event: int | None = None,
    session_id: str | None = None,
    sequence: int | None = None,
    error_code: int | None = None,
    payload: bytes = b"",
) -> bytes:
    out = bytearray(generate_header(message_type, flags, serialization, compression))
    if (flags & FLAG_WITH_EVENT) and event is not None:
        out += struct.pack(">i", event)
        if event not in CONNECTION_EVENTS and session_id is not None:
            sid = session_id.encode()
            out += struct.pack(">I", len(sid)) + sid
    if sequence is not None and (flags & (FLAG_POS_SEQ | FLAG_NEG_SEQ)):
        out += struct.pack(">i", sequence)
    if error_code is not None and message_type == MSG_ERROR:
        out += struct.pack(">I", error_code)
    out += struct.pack(">I", len(payload)) + payload
    return bytes(out)


@dataclass
class ParsedFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    event: int | None = None
    session_id: str | None = None
    sequence: int | None = None
    error_code: int | None = None
    payload: bytes = b""


def parse_frame(raw: bytes) -> ParsedFrame:
    message_type = raw[1] >> 4
    flags = raw[1] & 0x0F
    serialization = raw[2] >> 4
    compression = raw[2] & 0x0F
    header_size = (raw[0] & 0x0F) * 4
    off = header_size

    event = session_id = sequence = error_code = None

    if flags & FLAG_WITH_EVENT:
        event = struct.unpack(">i", raw[off:off + 4])[0]
        off += 4
        if event not in CONNECTION_EVENTS:
            sl = struct.unpack(">I", raw[off:off + 4])[0]
            off += 4
            if sl > 0:
                session_id = raw[off:off + sl].decode(errors="replace")
                off += sl
    if flags & (FLAG_POS_SEQ | FLAG_NEG_SEQ):
        sequence = struct.unpack(">i", raw[off:off + 4])[0]
        off += 4
    if message_type == MSG_ERROR:
        error_code = struct.unpack(">I", raw[off:off + 4])[0]
        off += 4

    size = struct.unpack(">I", raw[off:off + 4])[0]
    off += 4
    payload = raw[off:off + size]
    return ParsedFrame(
        message_type, flags, serialization, compression,
        event, session_id, sequence, error_code, payload,
    )
