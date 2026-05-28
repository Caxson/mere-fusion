"""豆包流式语音识别 (Seed-ASR / BigASR)

端点: wss://openspeech.bytedance.com/api/v3/sauc/bigmodel
鉴权: X-Api-App-Key / X-Api-Access-Key / X-Api-Resource-Id (header)
音频: PCM 16k/16bit/单声道, gzip, sequence 驱动 (尾包用负序号)
"""

from __future__ import annotations

import gzip
import json
import uuid
from collections.abc import AsyncIterator

from . import protocol as p

ASR_URL = "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel"


class DoubaoASR:
    def __init__(
        self,
        app_id: str,
        access_token: str,
        resource_id: str = "volc.bigasr.sauc.duration",  # 或 .concurrent
        sample_rate: int = 16000,
        url: str = ASR_URL,
    ):
        self.app_id = app_id
        self.access_token = access_token
        self.resource_id = resource_id
        self.sample_rate = sample_rate
        self.url = url

    def _headers(self) -> dict:
        return {
            "X-Api-App-Key": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": self.resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    async def stream(self, pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """喂 PCM chunk 异步迭代器，yield 识别结果 dict（含 text / utterances）。"""
        import websockets

        init = {
            "user": {"uid": "mere-fusion"},
            "audio": {"format": "pcm", "codec": "raw",
                      "rate": self.sample_rate, "bits": 16, "channel": 1},
            "request": {"model_name": "bigmodel", "enable_itn": True,
                        "enable_punc": True, "show_utterances": True},
        }
        async with websockets.connect(
            self.url, additional_headers=self._headers(), max_size=16 * 1024 * 1024
        ) as ws:
            await ws.send(p.build_frame(
                p.MSG_FULL_CLIENT, flags=p.FLAG_POS_SEQ,
                serialization=p.SERIAL_JSON, compression=p.COMPRESS_GZIP,
                sequence=1, payload=gzip.compress(json.dumps(init).encode()),
            ))

            seq = 1
            async for chunk in pcm_chunks:
                seq += 1
                await ws.send(p.build_frame(
                    p.MSG_AUDIO_CLIENT, flags=p.FLAG_POS_SEQ,
                    serialization=p.SERIAL_RAW, compression=p.COMPRESS_GZIP,
                    sequence=seq, payload=gzip.compress(chunk),
                ))
                # 非阻塞收一次结果
                async for res in self._drain(ws):
                    yield res

            # 尾包（负序号）
            await ws.send(p.build_frame(
                p.MSG_AUDIO_CLIENT, flags=p.FLAG_NEG_SEQ,
                serialization=p.SERIAL_RAW, compression=p.COMPRESS_GZIP,
                sequence=-seq, payload=gzip.compress(b""),
            ))
            async for res in self._iter_until_final(ws):
                yield res

    async def _drain(self, ws):
        import asyncio
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.001)
        except (asyncio.TimeoutError, Exception):
            return
        yield self._decode(raw)

    async def _iter_until_final(self, ws):
        async for raw in ws:
            frame = p.parse_frame(raw)
            yield self._decode_parsed(frame)
            if frame.flags & p.FLAG_LAST_NO_SEQ:
                break

    def _decode(self, raw: bytes) -> dict:
        return self._decode_parsed(p.parse_frame(raw))

    def _decode_parsed(self, frame: p.ParsedFrame) -> dict:
        data = frame.payload
        if frame.compression == p.COMPRESS_GZIP and data:
            try:
                data = gzip.decompress(data)
            except OSError:
                pass
        try:
            obj = json.loads(data)
        except (ValueError, TypeError):
            return {"raw": data}
        return obj.get("result", obj)
