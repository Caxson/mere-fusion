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


def stream_out_video_main():
    """
    主函数 数字人参数准备和模型导入
    """
    global nerfreal
    multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")

    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")

    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')

    parser.add_argument('--num_rays', type=int, default=4096 * 16,
                        help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16,
                        help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0,
                        help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                        help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### loss set
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")

    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1,
                        help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")

    parser.add_argument('--torso_shrink', type=float, default=0.8,
                        help="shrink bg coords to allow more flexibility in deform")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0,
                        help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=1,
                        help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1 / 256,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10,
                        help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01,
                        help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1,
                        help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    ### else
    parser.add_argument('--att', type=int, default=2,
                        help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='',
                        help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")

    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000,
                        help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true',
                        help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # asr
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")

    # parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')  #
    # parser.add_argument('--asr_model', type=str, default='facebook/wav2vec2-large-960h-lv60-self')
    # parser.add_argument('--asr_model', type=str, default='facebook/hubert-large-ls960-ft')

    parser.add_argument('--asr_save_feats', action='store_true')
    # audio FPS
    parser.add_argument('--fps', type=int, default=50)
    # sliding window left-middle-right length (unit: 20ms)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)

    # musetalk opt
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # parser.add_argument('--customvideo', action='store_true', help="custom video")
    # parser.add_argument('--customvideo_img', type=str, default='data/customvideo/img')
    # parser.add_argument('--customvideo_imgnum', type=int, default=1)

    parser.add_argument('--customvideo_config', type=str, default='')

    parser.add_argument('--tts', type=str, default='edgetts')  # xtts gpt-sovits cosyvoice
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')  # http://localhost:9000
    # parser.add_argument('--CHARACTER', type=str, default='test')
    # parser.add_argument('--EMOTION', type=str, default='default')

    parser.add_argument('--model', type=str, default='musetalk')  # musetalk wav2lip

    parser.add_argument('--transport', type=str, default='rtcpush')  # rtmp webrtc rtcpush
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')  # rtmp://localhost/live/livestream

    parser.add_argument('--max_session', type=int, default=1)  # multi session count
    parser.add_argument('--listenport', type=int, default=8010)

    opt = parser.parse_args()
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

    if opt.transport == 'rtmp':
        thread_quit = Event()
        rendthrd = Thread(target=nerfreals[0].render, args=(thread_quit,))
        rendthrd.start()

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
    return opt, nerfreal
