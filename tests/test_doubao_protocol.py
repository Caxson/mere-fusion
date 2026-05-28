"""豆包二进制帧协议 round-trip 单元测试。

只测 protocol.py 的编解码（离线，无网络）。这是最易出 bug 的部分。
运行: pytest tests/test_doubao_protocol.py -v
"""

import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from doubao import protocol as p


class TestHeader:
    def test_header_bits(self):
        h = p.generate_header(p.MSG_FULL_CLIENT, p.FLAG_POS_SEQ, p.SERIAL_JSON, p.COMPRESS_GZIP)
        assert len(h) == 4
        assert h[0] == (p.PROTOCOL_VERSION << 4) | p.HEADER_SIZE
        assert h[1] == (p.MSG_FULL_CLIENT << 4) | p.FLAG_POS_SEQ
        assert h[2] == (p.SERIAL_JSON << 4) | p.COMPRESS_GZIP
        assert h[3] == 0x00


class TestRoundTrip:
    def test_sequence_frame(self):
        payload = b"hello-audio"
        raw = p.build_frame(
            p.MSG_AUDIO_CLIENT, flags=p.FLAG_POS_SEQ,
            serialization=p.SERIAL_RAW, compression=p.COMPRESS_NONE,
            sequence=7, payload=payload,
        )
        f = p.parse_frame(raw)
        assert f.message_type == p.MSG_AUDIO_CLIENT
        assert f.flags == p.FLAG_POS_SEQ
        assert f.sequence == 7
        assert f.payload == payload
        assert f.event is None

    def test_negative_sequence_tail(self):
        raw = p.build_frame(
            p.MSG_AUDIO_CLIENT, flags=p.FLAG_NEG_SEQ,
            serialization=p.SERIAL_RAW, compression=p.COMPRESS_NONE,
            sequence=-12, payload=b"",
        )
        f = p.parse_frame(raw)
        assert f.sequence == -12

    def test_event_frame_with_session(self):
        payload = b'{"text":"hi"}'
        raw = p.build_frame(
            p.MSG_FULL_CLIENT, flags=p.FLAG_WITH_EVENT,
            serialization=p.SERIAL_JSON, compression=p.COMPRESS_NONE,
            event=p.EV_TASK_REQUEST, session_id="sess-abc", payload=payload,
        )
        f = p.parse_frame(raw)
        assert f.event == p.EV_TASK_REQUEST
        assert f.session_id == "sess-abc"
        assert f.payload == payload

    def test_connection_event_has_no_session(self):
        raw = p.build_frame(
            p.MSG_FULL_CLIENT, flags=p.FLAG_WITH_EVENT,
            serialization=p.SERIAL_JSON, compression=p.COMPRESS_NONE,
            event=p.EV_START_CONNECTION, session_id="should-be-ignored", payload=b"{}",
        )
        f = p.parse_frame(raw)
        assert f.event == p.EV_START_CONNECTION
        assert f.session_id is None  # 连接级事件不带 session_id
        assert f.payload == b"{}"

    def test_audio_server_response(self):
        audio = bytes(range(256))
        raw = p.build_frame(
            p.MSG_AUDIO_SERVER, flags=p.FLAG_WITH_EVENT,
            serialization=p.SERIAL_RAW, compression=p.COMPRESS_NONE,
            event=p.EV_TTS_RESPONSE, session_id="s1", payload=audio,
        )
        f = p.parse_frame(raw)
        assert f.message_type == p.MSG_AUDIO_SERVER
        assert f.event == p.EV_TTS_RESPONSE
        assert f.payload == audio

    def test_error_frame(self):
        raw = p.build_frame(
            p.MSG_ERROR, flags=p.FLAG_NO_SEQ,
            serialization=p.SERIAL_JSON, compression=p.COMPRESS_NONE,
            error_code=45000001, payload=b'{"error":"bad"}',
        )
        f = p.parse_frame(raw)
        assert f.message_type == p.MSG_ERROR
        assert f.error_code == 45000001
        assert f.payload == b'{"error":"bad"}'

    def test_payload_length_prefix(self):
        """payload 前 4 字节大端长度应与实际一致。"""
        payload = b"x" * 1000
        raw = p.build_frame(
            p.MSG_FULL_CLIENT, flags=p.FLAG_NO_SEQ,
            serialization=p.SERIAL_JSON, compression=p.COMPRESS_NONE,
            payload=payload,
        )
        # 末尾 4+len 字节
        size = struct.unpack(">I", raw[4:8])[0]
        assert size == 1000
        assert raw[8:] == payload


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
