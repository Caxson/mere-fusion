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
from flask import Flask
from flask_sockets import Sockets

from threading import Thread, Event
import multiprocessing

from aiohttp import web
import aiohttp_cors

import argparse
import asyncio

# ernerf
from ernerf.nerf_triplane.provider import NeRFDataset_Test
from ernerf.nerf_triplane.utils import *
from ernerf.nerf_triplane.network import NeRFNetwork
from nerfreal import NeRFReal

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

# 提示常量
AUDIO_FRIENDLY_INSTRUCTION = "Make sure your output is formatted in such a way that it can be read out loud (it will be turned into spoken words) from your response directly."
PROMPT_OPTIONS = {
    "getty": "explain the gettysburg address to a ten year old. then say the speech in a way they'd understand",
    "toast": "write a six sentence story about toast",
    "counter": "Count to 15, with a comma between each number, unless it's a multiple of 3 (including 3), then use only a period (ex. '4, 5, 6. 7,'), and no newlines. E.g., 1, 2, 3, ...",
    "punc": "say five sentences. each one ending with different punctuation. at least one question. each sentence should be at least 15 words long.",
}

PROMPT_TO_USE = f"{PROMPT_OPTIONS['getty']}. {AUDIO_FRIENDLY_INSTRUCTION}"

# 初始化 OpenAI 客户端，使用环境变量中的 OPENAI_API_KEY
OPENAI_CLIENT = openai.OpenAI()

# 全局停止事件，用于管理线程的结束
stop_event = threading.Event()

# 定义一个全局队列
text_in_queue = queue.Queue()
video_in_queue = queue.Queue()


# 生产者函数，用于随时添加数据到队列
def text_produce(text):
    logging.info(f"Text Producer is producing: {text}")
    text_in_queue.put(text)


def video_produce(text):
    logging.info(f"Video producer is producing: {text}")
    video_in_queue.put(text)


def stream_delimited_completion(
        messages: list[dict],
        client: openai.OpenAI = OPENAI_CLIENT,
        model: str = DEFAULT_RESPONSE_MODEL,
        content_transformers: list[Callable[[str], str]] = [],
        phrase_transformers: list[Callable[[str], str]] = [],
        delimiters: list[str] = DELIMITERS,
) -> Generator[str, None, None]:
    """
    从 OpenAI 的聊天完成流中生成分隔短语。

    :param messages: 消息列表，作为 OpenAI 的输入
    :param client: OpenAI 客户端实例
    :param model: 使用的模型
    :param content_transformers: 应用于内容的转换函数列表
    :param phrase_transformers: 应用于短语的转换函数列表
    :param delimiters: 短语分隔符列表
    :return: 生成短语的生成器
    """

    def apply_transformers(s: str, transformers: list[Callable[[str], str]]) -> str:
        return reduce(lambda c, transformer: transformer(c), transformers, s)

    working_string = ""
    for chunk in client.chat.completions.create(
            messages=messages, model=model, stream=True
    ):
        # 如果全局停止事件被设置，则发出信号停止
        if stop_event.is_set():
            yield None
            return

        content = chunk.choices[0].delta.content or ""
        if content:
            # 在添加到工作字符串之前应用所有转换器
            working_string += apply_transformers(content, content_transformers)
            while len(working_string) >= MINIMUM_PHRASE_LENGTH:
                delimiter_index = -1
                for delimiter in delimiters:
                    index = working_string.find(delimiter, MINIMUM_PHRASE_LENGTH)
                    if index != -1 and (
                            delimiter_index == -1 or index < delimiter_index
                    ):
                        delimiter_index = index

                if delimiter_index == -1:
                    break

                phrase, working_string = (
                    working_string[: delimiter_index + len(delimiter)],
                    working_string[delimiter_index + len(delimiter):],
                )
                yield apply_transformers(phrase, phrase_transformers)

    # 处理剩余的内容
    if working_string.strip():
        yield working_string.strip()

    yield None  # 发送终止信号，表示没有更多内容


def phrase_generator(phrase_queue: queue.Queue):
    """
    从 message_queue 获取消息，调用 stream_delimited_completion 生成短语，并将其放入 phrase_queue 中。

    :param phrase_queue: 用于存放短语的队列
    """
    while not stop_event.is_set():
        try:
            text_message = text_in_queue.get(timeout=1)  # 获取新的消息
            if text_message is None:
                phrase_queue.put(None)
                return
            video_message = "目前你没有看到画面"
            if not video_in_queue.empty():
                video_message = video_in_queue.get()
            logging.info(
                f"Consumer_Log ---------------------------------- consuming text: {text_message}, video: {video_message}")
            input_messages = [
                {"role": "system",
                 "content": "你需要扮演一个可爱的女孩子，现在需要不断根据我发给你的画面数据和聊天数据生成对应的回答和响应，你需要把我发给你的文本当作我给你说的话，我发给你的画面当作你看到的，你只需要按照文本内容给出响应，画面内容只作为参考,用中文回复并把自己当作人来应答，确保您的输出格式可以直接从您的响应中大声读出(它将被转换成口语)。"},
                {"role": "user",
                 "content": f"文本（前两个数字是文本开始和结束长度）：{text_message}\n -画面：{video_message}"},
            ]
            for phrase in stream_delimited_completion(
                    input_messages,
                    content_transformers=[lambda c: c.replace("\n", " ")],
                    phrase_transformers=[lambda p: p.strip()],
            ):
                if phrase is None:
                    phrase_queue.put(None)
                    return

                logger.info(f"> {phrase}")
                phrase_queue.put(phrase)

        except queue.Empty:
            continue


def generate_face_frame(audio_chunk, model):
    """
    基于音频数据生成虚拟人脸动画帧（假设使用 NeRF 或 MuseTalk）。

    :param audio_chunk: 音频数据块
    :param model: 虚拟人脸生成模型
    :return: 生成的虚拟人脸图像帧
    """
    # 假设模型能够接收音频特征并生成虚拟人脸图像帧
    frame = model.render_frame(audio_chunk)
    return frame


def video_player(video_queue: queue.Queue):
    """
    播放视频队列中的虚拟人脸帧。

    :param video_queue: 包含视频帧的队列
    """
    while not stop_event.is_set():
        frame = video_queue.get()
        if frame is None:
            break
        # 显示图像帧
        cv2.imshow('Virtual Avatar', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


# 文本转视频
def text_to_speech_processor_with_video(
        phrase_queue: queue.Queue,
        video_queue: queue.Queue,
        client: openai.OpenAI = OPENAI_CLIENT,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_VOICE
):
    """
    处理短语并将其转换为语音和虚拟人脸帧，放入音频和视频队列。

    :param phrase_queue: 短语队列
    :param video_queue: 视频帧队列
    :param client: OpenAI 客户端实例
    :param model: TTS 模型
    :param voice: 使用的语音
    """
    while not stop_event.is_set():
        phrase = phrase_queue.get()
        if phrase is None:
            # audio_queue.put(None)
            video_queue.put(None)
            return

        try:
            with client.audio.speech.with_streaming_response.create(
                    model=model, voice=voice, response_format="pcm", input=phrase
            ) as response:
                for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    # 放入音频队列
                    # audio_queue.put(chunk)

                    # 生成虚拟人脸帧!!!
                    nerfreal.put_msg_txt(chunk)

                    if stop_event.is_set():
                        return
        except Exception as e:
            logger.error(f"Error in text_to_speech_processor_with_video: {e}")
            # audio_queue.put(None)
            video_queue.put(None)
            return


# 文本转语音
def text_to_speech_processor(
        phrase_queue: queue.Queue,
        audio_queue: queue.Queue,
        client: openai.OpenAI = OPENAI_CLIENT,
        model: str = DEFAULT_TTS_MODEL,
        voice: str = DEFAULT_VOICE,
):
    """
    处理短语并将其转换为语音，放入音频队列。

    :param phrase_queue: 短语队列
    :param audio_queue: 音频队列
    :param client: OpenAI 客户端实例
    :param model: TTS 模型
    :param voice: 使用的语音
    """
    while not stop_event.is_set():
        phrase = phrase_queue.get()
        if phrase is None:
            audio_queue.put(None)
            return

        try:
            with client.audio.speech.with_streaming_response.create(
                    model=model, voice=voice, response_format="pcm", input=phrase
            ) as response:
                for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    audio_queue.put(chunk)
                    if stop_event.is_set():
                        return
        except Exception as e:
            logger.error(f"Error in text_to_speech_processor: {e}")
            audio_queue.put(None)
            return


def audio_player(audio_queue: queue.Queue):
    """
    播放音频队列中的音频数据。

    :param audio_queue: 包含音频数据的队列
    """
    p = pyaudio.PyAudio()
    player_stream = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

    try:
        while not stop_event.is_set():
            audio_data = audio_queue.get()
            if audio_data is None:
                break
            player_stream.write(audio_data)
    except Exception as e:
        logger.error(f"Error in audio_player: {e}")
    finally:
        player_stream.stop_stream()
        player_stream.close()
        p.terminate()


def wait_for_enter():
    """
    等待用户按下回车键以停止操作。
    """
    input("Press Enter to stop...\n\n")
    stop_event.set()
    logger.info("STOP instruction received. Working to quit...")


def stream_out_video_main(opt):
    """
    主函数 数字人参数准备和模型导入
    """
    global nerfreal
    # app.config.from_object(opt)
    # print(app.config)
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)

    if opt.model == 'ernerf':
        # from ernerf.nerf_triplane.provider import NeRFDataset_Test
        # from ernerf.nerf_triplane.utils import *
        # from ernerf.nerf_triplane.network import NeRFNetwork
        # from nerfreal import NeRFReal
        # assert test mode
        opt.test = True
        opt.test_train = False
        # opt.train_camera =True
        # explicit smoothing
        opt.smooth_path = True
        opt.smooth_lips = True

        assert opt.pose != '', 'Must provide a pose source'

        # if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.exp_eye = True
        opt.smooth_eye = True

        if opt.torso_imgs == '':  # no img,use model output
            opt.torso = True

        # assert opt.cuda_ray, "Only support CUDA ray mode."
        opt.asr = True

        if opt.patch_size > 1:
            # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
            assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
        seed_everything(opt.seed)
        print(opt)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = NeRFNetwork(opt)

        criterion = torch.nn.MSELoss(reduction='none')
        metrics = []  # use no metric in GUI for faster initialization...
        print(model)
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16,
                          metrics=metrics, use_checkpoint=opt.ckpt)

        test_loader = NeRFDataset_Test(opt, device=device).dataloader()
        model.aud_features = test_loader._data.auds
        model.eye_areas = test_loader._data.eye_area

        # we still need test_loader to provide audio features for testing.
        for _ in range(opt.max_session):
            nerfreal = NeRFReal(opt, trainer, test_loader)
            nerfreals.append(nerfreal)
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        print(opt)
        for _ in range(opt.max_session):
            nerfreal = MuseReal(opt)
            nerfreals.append(nerfreal)
    elif opt.model == 'wav2lip':
        from lipreal import LipReal
        print(opt)
        for _ in range(opt.max_session):
            nerfreal = LipReal(opt)
            nerfreals.append(nerfreal)

    for _ in range(opt.max_session):
        statreals.append(0)

    # if opt.transport == 'rtmp':
    #     thread_quit = Event()
    #     rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
    #     rendthrd.start()

    #######################################################################################

    """
    初始化队列和线程，并启动整个 TTS 过程。
    """
    phrase_queue = queue.Queue()
    # audio_queue = queue.Queue()
    video_queue = queue.Queue()

    phrase_generation_thread = threading.Thread(
        target=phrase_generator, args=(phrase_queue,)
    )
    # tts_thread = threading.Thread(
    #     target=text_to_speech_processor, args=(phrase_queue, audio_queue)
    # )
    ttv_thread = threading.Thread(
        target=text_to_speech_processor_with_video, args=(phrase_queue, video_queue)
    )
    # audio_player_thread = threading.Thread(target=audio_player, args=(audio_queue,))
    # video_player_thread = threading.Thread(target=video_player, args=(video_queue,))

    # 启动线程
    phrase_generation_thread.start()
    ttv_thread.start()
    # tts_thread.start()
    # audio_player_thread.start()
    # video_player_thread.start()

    # 创建并启动 "按下回车停止" 线程。守护线程不会阻止脚本退出
    threading.Thread(target=wait_for_enter, daemon=True).start()

    phrase_generation_thread.join()
    logger.info("## all phrases enqueued. phrase generation thread terminated.")
    # tts_thread.join()
    ttv_thread.join()
    logger.info("## all tts complete and enqueued. tts thread terminated.")
    # audio_player_thread.join()
    # video_player_thread.join()
    logger.info("## audio output complete. video player thread terminated.")
    return nerfreal
