"""豆包端到端实时语音大模型 (Realtime / S2S 语音对话)

类 GPT-4o Realtime：听(ASR) + 想(LLM) + 说(TTS) 一体。
端点: wss://openspeech.bytedance.com/api/v3/realtime/dialogue
鉴权: X-Api-App-ID / X-Api-Access-Key / X-Api-Resource-Id: volc.speech.dialog
      X-Api-App-Key: PlgvMymc7f3tQnJ6  (该产品固定公共 App-Key, 非用户密钥)

流程: StartConnection→StartSession→持续推音频(event=200)→
      服务端流式回 ASRResponse(451)/ChatResponse(550)/TTSResponse(352 裸音频)
需自带保活：静默时每 5s 推 100ms 静音 PCM。
"""

from __future__ import annotations

import gzip
import json
import uuid
from collections.abc import AsyncIterator

from . import protocol as p

REALTIME_URL = "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
PUBLIC_APP_KEY = "PlgvMymc7f3tQnJ6"


class DoubaoRealtime:
    def __init__(
        self,
        app_id: str,
        access_token: str,
        bot_name: str = "小塔",
        sample_rate: int = 24000,
        url: str = REALTIME_URL,
    ):
        self.app_id = app_id
        self.access_token = access_token
        self.bot_name = bot_name
        self.sample_rate = sample_rate
        self.url = url

    def _headers(self) -> dict:
        return {
            "X-Api-App-ID": self.app_id,
            "X-Api-Access-Key": self.access_token,
            "X-Api-Resource-Id": "volc.speech.dialog",
            "X-Api-App-Key": PUBLIC_APP_KEY,
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }

    async def dialog(self, mic_pcm_chunks: AsyncIterator[bytes]) -> AsyncIterator[dict]:
        """推麦克风 PCM(16k)，yield 事件 dict：
        {"type":"asr"|"chat"|"tts_audio", "data":...}
        """
        import asyncio

        import websockets

        sid = str(uuid.uuid4())

        def frame(event, payload_obj_or_bytes, audio=False):
            if audio:
                return p.build_frame(
                    p.MSG_AUDIO_CLIENT, flags=p.FLAG_WITH_EVENT,
                    serialization=p.SERIAL_RAW, compression=p.COMPRESS_GZIP,
                    event=event, session_id=sid,
                    payload=gzip.compress(payload_obj_or_bytes),
                )
            return p.build_frame(
                p.MSG_FULL_CLIENT, flags=p.FLAG_WITH_EVENT,
                serialization=p.SERIAL_JSON, compression=p.COMPRESS_GZIP,
                event=event,
                session_id=None if event in p.CONNECTION_EVENTS else sid,
                payload=gzip.compress(json.dumps(payload_obj_or_bytes).encode()),
            )

        async with websockets.connect(
            self.url, additional_headers=self._headers(), ping_interval=5
        ) as ws:
            await ws.send(frame(p.EV_START_CONNECTION, {}))
            start = {"dialog": {"bot_name": self.bot_name},
                     "tts": {"audio_config": {"channel": 1, "format": "pcm",
                                              "sample_rate": self.sample_rate}}}
            await ws.send(frame(p.EV_START_SESSION, start))

            async def push():
                async for chunk in mic_pcm_chunks:
                    await ws.send(frame(p.EV_TASK_REQUEST, chunk, audio=True))

            asyncio.create_task(push())

            async for raw in ws:
                f = p.parse_frame(raw)
                if f.event == p.EV_TTS_RESPONSE:
                    yield {"type": "tts_audio", "data": f.payload}
                elif f.event == p.EV_ASR_RESPONSE:
                    yield {"type": "asr", "data": self._json(f.payload)}
                elif f.event == p.EV_CHAT_RESPONSE:
                    yield {"type": "chat", "data": self._json(f.payload)}
                elif f.event in (p.EV_SESSION_FINISHED, p.EV_CHAT_ENDED):
                    break

    @staticmethod
    def _json(data: bytes):
        try:
            return json.loads(gzip.decompress(data))
        except OSError:
            try:
                return json.loads(data)
            except ValueError:
                return {"raw": data}
