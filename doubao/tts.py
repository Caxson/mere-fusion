"""豆包大模型 TTS V3 (双向流式 / Seed-TTS 2.0)

端点: wss://openspeech.bytedance.com/api/v3/tts/bidirection
鉴权: X-Api-App-Key / X-Api-Access-Key / X-Api-Resource-Id
事件驱动: StartConnection→StartSession→TaskRequest→(audio frames)→FinishSession
音色: req_params.speaker (如 zh_female_vv_uranus_bigtts)

注意: Resource-Id 各套餐不同（volc.service_type.10029 / seed-tts-2.0 ...），
做成可配置，以你火山控制台开通的为准。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator

from . import protocol as p

TTS_URL = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"


class DoubaoTTS:
    def __init__(
        self,
        app_id: str,
        access_token: str,
        speaker: str = "zh_female_vv_uranus_bigtts",
        resource_id: str = "volc.service_type.10029",
        audio_format: str = "mp3",
        sample_rate: int = 24000,
        url: str = TTS_URL,
    ):
        self.app_id = app_id
        self.access_token = access_token
        self.speaker = speaker
        self.resource_id = resource_id
        self.audio_format = audio_format
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

    def _req_params(self) -> dict:
        return {
            "speaker": self.speaker,
            "audio_params": {
                "format": self.audio_format,
                "sample_rate": self.sample_rate,
                "speech_rate": 0,
            },
        }

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """流式合成，yield 音频字节块（格式由 audio_format 决定）。"""
        import websockets

        sid = str(uuid.uuid4())
        base = {"user": {"uid": "mere-fusion"}, "namespace": "BidirectionalTTS",
                "req_params": self._req_params()}

        def ev(event, payload_obj, with_session=True):
            return p.build_frame(
                p.MSG_FULL_CLIENT, flags=p.FLAG_WITH_EVENT,
                serialization=p.SERIAL_JSON, compression=p.COMPRESS_NONE,
                event=event, session_id=sid if with_session else None,
                payload=json.dumps(payload_obj, ensure_ascii=False).encode(),
            )

        async with websockets.connect(
            self.url, additional_headers=self._headers(), max_size=16 * 1024 * 1024
        ) as ws:
            await ws.send(ev(p.EV_START_CONNECTION, {}, with_session=False))
            await ws.recv()  # ConnectionStarted
            await ws.send(ev(p.EV_START_SESSION, {**base, "event": p.EV_START_SESSION}))
            await ws.recv()  # SessionStarted

            task = {**base, "event": p.EV_TASK_REQUEST}
            task["req_params"] = {**self._req_params(), "text": text}
            await ws.send(ev(p.EV_TASK_REQUEST, task))
            await ws.send(ev(p.EV_FINISH_SESSION, {}))

            async for raw in ws:
                frame = p.parse_frame(raw)
                if frame.message_type == p.MSG_AUDIO_SERVER:
                    yield frame.payload  # 裸音频
                if frame.event in (p.EV_TTS_ENDED, p.EV_SESSION_FINISHED):
                    break
