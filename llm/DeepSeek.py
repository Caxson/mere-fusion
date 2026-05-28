"""DeepSeek-V4 适配器

OpenAI 兼容接口。实时数字人推荐 `deepseek-v4-flash`（284B 总参/13B 激活，
快且省），非思考模式，流式输出可直接逐句喂给 TTS。

环境变量：DEEPSEEK_API_KEY
注意：旧 model id `deepseek-chat` / `deepseek-reasoner` 将于 2026-07-24 下线。
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeek:
    def __init__(
        self,
        model_path: str = "deepseek-v4-flash",
        api_key: str | None = None,
        base_url: str = DEEPSEEK_BASE_URL,
        system_prompt: str = "你是一个友好的 AI 数字人助手，回答简洁自然，适合语音播报。",
    ):
        self.model_path = model_path
        self.system_prompt = system_prompt
        self.client = OpenAI(
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
            base_url=base_url,
        )

    def chat(self, message: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
        )
        return resp.choices[0].message.content

    def chat_stream(self, message: str) -> Iterator[str]:
        """流式输出 token 增量。配合断句逐句喂 TTS，降低首响延迟。"""
        stream = self.client.chat.completions.create(
            model=self.model_path,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": message},
            ],
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
