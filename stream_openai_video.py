import argparse
import json
import multiprocessing
import queue
import threading
from functools import reduce
from typing import Callable, Generator
import logging

import cv2
import openai
import pyaudio
from dotenv import load_dotenv
from flask import Flask
from flask_sockets import Sockets

from app import active_sessions

# ernerf
from ernerf.nerf_triplane.utils import *

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
sockets = Sockets(app)
nerfreals = []
statreals = []

# 常量定义
DELIMITERS = [f"{d} " for d in (".", "?", "!")]  # 确定短语结束位置的标点符号
MINIMUM_PHRASE_LENGTH = 200  # 确保音频流畅的最短短语长度
TTS_CHUNK_SIZE = 1024  # TTS 音频块大小

# 默认值
DEFAULT_RESPONSE_MODEL = "gpt-3.5-turbo"  # 默认的 OpenAI 模型
DEFAULT_TTS_MODEL = "tts-1"  # 默认的 TTS 模型
DEFAULT_VOICE = "alloy"  # 默认的语音选择

load_dotenv()


class OpenAISessionManager:
    def __init__(self, session_id):
        self.session_id = session_id
        self.stop_event = threading.Event()
        self.text_in_queue = queue.Queue()
        self.video_in_queue = queue.Queue()
        self.phrase_queue = queue.Queue()
        self.video_queue = queue.Queue()
        self.OPENAI_CLIENT = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # 初始化线程
        self.phrase_generation_thread = threading.Thread(
            target=self.phrase_generator, daemon=True
        )
        self.ttv_thread = threading.Thread(
            target=self.text_to_speech_processor_with_video, daemon=True
        )
        self.video_playback_thread = threading.Thread(
            target=self.video_player, daemon=True
        )

        # 启动线程
        self.start_threads()

    def start_threads(self):
        """
        启动短语生成、TTS处理和视频播放的线程。
        """
        self.phrase_generation_thread.start()
        self.ttv_thread.start()
        self.video_playback_thread.start()
        logging.info(f"Session {self.session_id} started with threads running.")

    def text_produce(self, text):
        logging.info(f"Text Producer is producing: {text}")
        self.text_in_queue.put(text)

    def video_produce(self, text):
        logging.info(f"Video Producer is producing: {text}")
        self.video_in_queue.put(text)

    def stream_delimited_completion(
            self,
            messages: list[dict],
            model: str = DEFAULT_RESPONSE_MODEL,
            content_transformers: list[Callable[[str], str]] = [],
            phrase_transformers: list[Callable[[str], str]] = [],
            delimiters: list[str] = DELIMITERS,
    ) -> Generator[str, None, None]:
        """
        从 OpenAI 的聊天完成流中生成分隔短语。
        """
        working_string = ""
        for chunk in self.OPENAI_CLIENT.chat.completions.create(
                messages=messages, model=model, stream=True
        ):
            if self.stop_event.is_set():
                yield None
                return

            content = chunk.choices[0].delta.content or ""
            if content:
                working_string += reduce(lambda c, t: t(c), content_transformers, content)
                while len(working_string) >= MINIMUM_PHRASE_LENGTH:
                    delimiter_index = -1
                    for delimiter in delimiters:
                        index = working_string.find(delimiter, MINIMUM_PHRASE_LENGTH)
                        if index != -1 and (delimiter_index == -1 or index < delimiter_index):
                            delimiter_index = index

                    if delimiter_index == -1:
                        break

                    phrase, working_string = (
                        working_string[: delimiter_index + len(delimiter)],
                        working_string[delimiter_index + len(delimiter):],
                    )
                    yield reduce(lambda p, t: t(p), phrase_transformers, phrase)

        if working_string.strip():
            yield working_string.strip()
        yield None

    def phrase_generator(self):
        """
        从 text_in_queue 获取消息并生成短语放入 phrase_queue。
        """
        while not self.stop_event.is_set():
            try:
                text_message = self.text_in_queue.get(timeout=1)
                if text_message is None:
                    self.phrase_queue.put(None)
                    return
                video_message = "目前你没有看到画面"
                if not self.video_in_queue.empty():
                    video_message = self.video_in_queue.get()
                logging.info(
                    f"Consuming text: {text_message}, video: {video_message}"
                )
                input_messages = [
                    {"role": "system",
                     "content": "你需要扮演一个人类，现在需要不断根据我发给你的画面数据和聊天数据生成对应的回答和响应，你需要把我发给你的文本当作我给你说的话，我发给你的画面当作你看到的，你只需要按照文本内容给出响应，画面内容只作为参考,用中文回复并把自己当作人来应答，确保您的输出格式可以直接从您的响应中大声读出(它将被转换成口语)。"},
                    {"role": "user", "content": f"文本：{text_message}\n -画面：{video_message}"},
                ]
                for phrase in self.stream_delimited_completion(
                        input_messages,
                        content_transformers=[lambda c: c.replace("\n", " ")],
                        phrase_transformers=[lambda p: p.strip()],
                ):
                    if phrase is None:
                        self.phrase_queue.put(None)
                        return

                    logging.info(f"> {phrase}")
                    self.phrase_queue.put(phrase)

            except queue.Empty:
                continue
            finally:
                self.close()

    def video_player(self):
        """
        播放 video_queue 中的虚拟人脸帧。
        """
        while not self.stop_event.is_set():
            frame = self.video_queue.get()
            if frame is None:
                break
            cv2.imshow('Virtual Avatar', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def text_to_speech_processor_with_video(self):
        """
        处理短语并将其转换为语音和虚拟人脸帧。
        """
        while not self.stop_event.is_set():
            phrase = self.phrase_queue.get()
            if phrase is None:
                self.video_queue.put(None)
                return

            try:
                with self.OPENAI_CLIENT.audio.speech.with_streaming_response.create(
                        model=DEFAULT_TTS_MODEL, voice=DEFAULT_VOICE, response_format="pcm", input=phrase
                ) as response:
                    for chunk in response.iter_bytes(chunk_size=1024):
                        # 发给数字人model处理
                        session_obj = active_sessions.get(self.session_id)
                        session_obj.model.put_msg_txt(chunk)
                        if self.stop_event.is_set():
                            return
            except Exception as e:
                logging.error(f"Error in TTS with video: {e}")
                self.video_queue.put(None)
                return
            finally:
                self.close()

    async def close(self):
        """
        停止所有队列消费线程和清理资源。
        """
        self.stop_event.set()
        self.phrase_queue.put(None)
        self.video_queue.put(None)
        self.phrase_generation_thread.join()
        self.ttv_thread.join()
        self.video_playback_thread.join()
        logging.info(f"Session {self.session_id} has been stopped.")
