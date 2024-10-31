# 主函数
import asyncio
import io
import json
import logging
import subprocess
import uuid

import librosa
from av.video.stream import VideoStream
from dotenv import load_dotenv
from flask import Flask
from flask_sockets import Sockets

from ernerf.nerf_triplane.provider import NeRFDataset_Test
from ernerf.nerf_triplane.utils import *
from ernerf.nerf_triplane.network import NeRFNetwork
from nerfreal import NeRFReal
from whisper_online_server import WhisperRTCServerProcessor
from yolo_opencv import yolo_opencv_main, YoloOpencvProcessor
from aiohttp import web
import aiohttp
import aiohttp_cors
import soundfile as sf
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
import multiprocessing
import argparse

import random
import asyncio

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiohttp import ClientSession

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# 定义最大会话数
MAX_SESSIONS = 5

# 存储当前活跃的会话
active_sessions = {}
session_lock = asyncio.Lock()

# 当前会话计数
current_sessions = 0

pull_url = ''
push_url = ''


def llm_response(message):
    from llm.LLM import LLM
    llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    response = llm.chat(message)
    print(response)
    return response


##############################################################
pcs = set()


async def start_session(request):
    """
    启动会话的 API 端点
    请求 JSON 格式：
    {}
    """
    # 自动分配一个 session_id
    session_id = str(uuid.uuid4())

    async with session_lock:
        global current_sessions
        # 检查是否达到最大会话数
        if current_sessions >= MAX_SESSIONS:
            return web.json_response({'code': 1, 'message': 'Maximum number of sessions reached'}, status=429)

        # 定义拉流和推流的流 URL，包含 session_id
        consume_stream_url = f'webrtc://<server_ip>/live/stream_{session_id}'
        produce_stream_url = f'webrtc://<server_ip>/live/processed_stream_{session_id}'

        # 创建会话实例
        session_obj = ConnectSession(session_id, consume_stream_url, produce_stream_url)

        # 启动会话
        await session_obj.start()

        # 添加到活跃会话字典
        active_sessions[session_id] = session_obj
        current_sessions += 1
        logger.info(f"Started session {session_id}. Current sessions: {current_sessions}")

    return web.json_response({'code': 0, 'message': 'Session started', 'session_id': session_id})


async def stop_session(request):
    """
    停止会话的 API 端点
    请求 JSON 格式：
    {
        "session_id": "unique_session_id"
    }
    """
    data = await request.json()
    session_id = data.get('session_id')
    if not session_id:
        return web.json_response({'code': 1, 'message': 'session_id is required'}, status=400)

    async with session_lock:
        global current_sessions
        session_obj = active_sessions.get(session_id)
        if not session_obj:
            return web.json_response({'code': 1, 'message': 'Session not found'}, status=404)

        # 关闭会话
        await session_obj.close()

        # 从活跃会话字典中移除
        del active_sessions[session_id]
        current_sessions -= 1
        logger.info(f"Stopped session {session_id}. Current sessions: {current_sessions}")

    return web.json_response({'code': 0, 'message': 'Session stopped'})


async def interrupt(request):
    params = await request.json()
    session_id = params.get('session_id', 0)
    session_obj = active_sessions.get(session_id)
    if not session_obj:
        return web.json_response({'code': 1, 'message': 'Session not found'}, status=404)
    session_obj.model.pause_talk()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def talk(request):
    params = await request.json()

    session_id = params.get('session_id', 0)
    session_obj = active_sessions.get(session_id)
    if not session_obj:
        return web.json_response({'code': 1, 'message': 'Session not found'}, status=404)
    if params.get('interrupt'):
        session_obj.model.pause_talk()
    if params['type'] == 'echo':
        session_obj.model.put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response(params['text']))
        session_obj.model.put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def set_audio_type(request):
    params = await request.json()

    session_id = params.get('session_id', 0)
    session_obj = active_sessions.get(session_id)
    if not session_obj:
        return web.json_response({'code': 1, 'message': 'Session not found'}, status=404)
    session_obj.model.set_curr_state(params['audio_type'], params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def record(request):
    params = await request.json()

    session_id = params.get('session_id', 0)
    session_obj = active_sessions.get(session_id)
    if not session_obj:
        return web.json_response({'code': 1, 'message': 'Session not found'}, status=404)
    if params['type'] == 'start_record':
        session_obj.model.start_recording("data/record_lasted.mp4")
    elif params['type'] == 'end_record':
        session_obj.model.stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


class UserSession:
    """每个用户会话的封装类"""

    def __init__(self, session_id, pc):
        self.session_id = session_id
        self.pc = pc
        self.video_buffer = None
        self.audio_buffer = None

    def add_track(self, track):
        """根据轨道类型添加处理逻辑"""
        if track.kind == "audio":
            logger.info(f"会话 {self.session_id}: 接收到音频轨道")
            self.audio_buffer = AudioStreamTrack(track, self.session_id)
            # TODO DEL
            self.pc.addTrack(self.audio_buffer)
        elif track.kind == "video":
            logger.info(f"会话 {self.session_id}: 接收到视频轨道")
            self.video_buffer = VideoStreamTrack(track, self.session_id)
            # TODO DEL
            self.pc.addTrack(self.video_buffer)

    async def close(self):
        if self.video_buffer:
            await self.video_buffer.close()
        if self.audio_buffer:
            await self.audio_buffer.close()


class AudioStreamTrack(MediaStreamTrack):
    """处理音频流数据"""

    kind = "audio"

    def __init__(self, track, session_id):
        super().__init__()
        self.track = track
        self.session_id = session_id
        self.buffer = []
        self.processor = WhisperRTCServerProcessor(session_id, min_chunk_r=1)

    async def recv(self):
        frame = await self.track.recv()
        raw_bytes = frame.to_bytes()
        self.processor.process(raw_bytes)
        return frame

    async def close(self):
        await self.processor.close()


class VideoStreamTrack(MediaStreamTrack):
    """处理视频流数据"""

    kind = "video"

    def __init__(self, track, session_id):
        super().__init__()
        self.track = track
        # 为每个会话创建独立的处理器
        self.processor = YoloOpencvProcessor(session_id)

    async def recv(self):
        frame = await self.track.recv()
        # 在每一帧调用时
        self.processor.process_frame(frame)
        return frame

    async def close(self):
        await self.processor.close()


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def send_request(session, url, params, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=params) as resp:
                return await resp.json()
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
            await asyncio.sleep(2 ** attempt)  # 指数退避
    logger.error(f"All {max_retries} attempts failed for {url}")
    return None


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


async def post_json(url, json):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=json) as response:
                return await response.json()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


class ConnectSession:
    def __init__(self, session_id, consume_stream_url, produce_stream_url):
        self.session_id = session_id
        self.opt = opt
        self.model = None
        self.push_url = push_url
        self.pull_url = pull_url
        self.consume_stream_url = consume_stream_url
        self.produce_stream_url = produce_stream_url
        self.consume_pc = RTCPeerConnection()
        self.produce_pc = RTCPeerConnection()
        self.task = None  # 用于跟踪后台任务
        self.active = True  # 会话状态
        # self.processor = stream_out_video_main(opt)
        self.consume_connected = asyncio.Event()  # 拉流连接建立事件
        self.use_session = None


    async def initialize_model(self):
        """根据配置初始化模型实例"""
        if self.opt.model == 'ernerf':
            return await self.initialize_ernerf()
        elif self.opt.model == 'musetalk':
            from musereal import MuseReal
            logger.info(f"Initializing MuseReal for session {self.session_id}")
            return MuseReal(self.opt)
        elif self.opt.model == 'wav2lip':
            from lipreal import LipReal
            logger.info(f"Initializing LipReal for session {self.session_id}")
            return LipReal(self.opt)
        else:
            raise ValueError(f"Unknown model type: {self.opt.model}")

    async def initialize_ernerf(self):
        """初始化 Ernerf 模型"""
        logger.info(f"Initializing Ernerf for session {self.session_id}")
        # 载入配置
        opt = self.opt
        opt.customopt = []
        if opt.customvideo_config != '':
            with open(opt.customvideo_config, 'r') as file:
                opt.customopt = json.load(file)

        if opt.model == 'ernerf':
            opt.test = True
            opt.test_train = False
            opt.smooth_path = True
            opt.smooth_lips = True

            assert opt.pose != '', 'Must provide a pose source'

            opt.fp16 = True
            opt.cuda_ray = True
            opt.exp_eye = True
            opt.smooth_eye = True

            if opt.torso_imgs == '':  # no img,use model output
                opt.torso = True

            opt.asr = True

            if opt.patch_size > 1:
                assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."
            seed_everything(opt.seed)
            print(opt)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = NeRFNetwork(opt)

            criterion = torch.nn.MSELoss(reduction='none')
            metrics = []  # use no metric in GUI for faster initialization...
            print(model)
            trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion,
                              fp16=opt.fp16,
                              metrics=metrics, use_checkpoint=opt.ckpt)

            test_loader = NeRFDataset_Test(opt, device=device).dataloader()
            model.aud_features = test_loader._data.auds
            model.eye_areas = test_loader._data.eye_area

            nerfreal = NeRFReal(opt, trainer, test_loader)
            return nerfreal

    async def start(self):
        logger.info(f"Starting session {self.session_id}")
        # 添加接收器，表示我们希望接收音视频流
        self.model = await self.initialize_model()
        self.consume_pc.addTransceiver('audio', direction='recvonly')
        self.consume_pc.addTransceiver('video', direction='recvonly')
        self.task = asyncio.create_task(self.run())

    async def run(self):
        async with ClientSession() as session:
            # 设置拉流连接状态变化事件处理器
            @self.consume_pc.on('connectionstatechange')
            async def on_consume_connectionstatechange():
                logger.info(
                    f'Consume PC connection state for session {self.session_id}: {self.consume_pc.connectionState}')
                if self.consume_pc.connectionState == 'connected':
                    self.consume_connected.set()
                elif self.consume_pc.connectionState in ('failed', 'closed', 'disconnected'):
                    # TODO 重试机制
                    await self.close()

            # 设置推流连接状态变化事件处理器
            @self.produce_pc.on('connectionstatechange')
            async def on_produce_connectionstatechange():
                logger.info(
                    f'Produce PC connection state for session {self.session_id}: {self.produce_pc.connectionState}')
                if self.produce_pc.connectionState in ('failed', 'closed', 'disconnected'):
                    # TODO 重试机制
                    await self.close()

            self.use_session = UserSession(self.session_id, self.consume_pc)

            @self.consume_pc.on('track')
            def on_track(track):
                logger.info(f'Track {track.kind} received, id: {track.id} for session {self.session_id}')
                self.use_session.add_track(track)

            # 创建并设置拉流的本地描述
            try:
                consume_offer = await self.consume_pc.createOffer()
                await self.consume_pc.setLocalDescription(consume_offer)
                logger.info(f"Consume Offer SDP for session {self.session_id}: {consume_offer.sdp}")
                play_params = {
                    'api': self.pull_url,
                    'streamurl': self.consume_stream_url,
                    'clientip': None,
                    'sdp': self.consume_pc.localDescription.sdp,
                    'tid': str(random.randint(10000, 99999)),
                    'action': 'play'
                }
            except Exception as e:
                logger.error(f"Error during offer creation for consume_pc in session {self.session_id}: {e}")
                await self.close()
                return

            # 发送拉流请求到 SRS
            try:
                async with session.post(play_params['api'], json=play_params) as resp:
                    res = await resp.json()
                    logger.info(f"SRS Response for play session {self.session_id}: {res}")
                    if res.get('code') and res['code'] != 0:
                        logger.error(f"Failed to play stream for session {self.session_id}: {res}")
                        await self.close()
                        return
                    answer = RTCSessionDescription(sdp=res['sdp'], type='answer')
                    await self.consume_pc.setRemoteDescription(answer)
            except Exception as e:
                logger.error(f"Error during SRS play request for session {self.session_id}: {e}")
                await self.close()
                return

            # 数字人启动
            player = HumanPlayer(self.model)
            self.produce_pc.addTrack(player.audio)
            self.produce_pc.addTrack(player.video)

            # 等待拉流连接完全建立
            try:
                await asyncio.wait_for(self.consume_connected.wait(), timeout=15)  # 设置一个超时时间
                logger.info(f"Consume connection established for session {self.session_id}")
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for consume connection in session {self.session_id}")
                await self.close()
                return

            # 推流部分：等待拉流连接后再启动推流
            try:
                produce_offer = await self.produce_pc.createOffer()
                await self.produce_pc.setLocalDescription(produce_offer)
                logger.info(f"Produce Offer SDP for session {self.session_id}: {produce_offer.sdp}")
                publish_params = {
                    'api': self.push_url,
                    'streamurl': self.produce_stream_url,
                    'clientip': None,
                    'sdp': self.produce_pc.localDescription.sdp,
                    'tid': str(random.randint(10000, 99999)),
                    'action': 'publish'
                }
            except Exception as e:
                logger.error(f"Error during offer creation for produce_pc in session {self.session_id}: {e}")
                await self.close()
                return

            # 发送推流请求到 SRS
            try:
                async with session.post(publish_params['api'], json=publish_params) as resp:
                    res = await resp.json()
                    logger.info(f"SRS Publish Response for session {self.session_id}: {res}")
                    if res.get('code') and res['code'] != 0:
                        logger.error(f"Failed to publish stream for session {self.session_id}: {res}")
                        await self.close()
                        return
                    answer = RTCSessionDescription(sdp=res['sdp'], type='answer')
                    await self.produce_pc.setRemoteDescription(answer)
            except Exception as e:
                logger.error(f"Error during SRS publish request for session {self.session_id}: {e}")
                await self.close()
                return

            # 保持连接，直到会话被关闭
            logger.info(f"Stream relaying for session {self.session_id} established. Running until stopped.")
            try:
                while self.active:
                    await asyncio.sleep(1)
            except asyncio.CancelledError:
                logger.info(f"Stream relaying for session {self.session_id} cancelled.")
            finally:
                await self.close()

    async def close(self):
        if not self.active:
            return
        self.active = False
        await self.consume_pc.close()
        await self.produce_pc.close()
        await self.use_session.close()
        logger.info(f"Stream relaying for session {self.session_id} closed.")


def run_server(runner):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, '0.0.0.0', opt.listen_port)
    loop.run_until_complete(site.start())
    # if opt.transport == 'rtc':
    #     loop.run_until_complete(run(opt.push_url, opt.pull_url))
    loop.run_forever()


##########################################

if __name__ == "__main__":
    # 加载 .env 文件中的环境变量，例如 API 密钥
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

    parser.add_argument('--model', type=str, default='ernerf')  # ernerf musetalk wav2lip

    parser.add_argument('--transport', type=str, default='rtc')  # rtmp webrtc rtc
    parser.add_argument('--push_url', type=str,
                        default='http://<server_ip>:1985/rtc/v1/publish/')  # rtmp://localhost/live/livestream    http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream
    parser.add_argument('--pull_url', type=str,
                        default='http://<server_ip>:1985/rtc/v1/play/')

    parser.add_argument('--max_session', type=int, default=10)  # multi session count
    parser.add_argument('--listen_port', type=int, default=8010)

    opt = parser.parse_args()

    # 音频输入识别主函数
    # whisper_main()
    # 视频输入识别主函数
    # yolo_opencv_main()
    # 流式处理和输出主函数
    # nerfreal = stream_out_video_main(opt)

    # 参数初始化
    push_url = opt.push_url
    pull_url = opt.pull_url
    MAX_SESSIONS = opt.max_session

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post('/start_session', start_session)
    appasync.router.add_post('/stop_session', stop_session)
    appasync.router.add_post("/interrupt", interrupt)
    appasync.router.add_post("/talk", talk)
    appasync.router.add_post("/set_audio_type", set_audio_type)
    appasync.router.add_post("/record", record)
    appasync.router.add_static('/', path='web')

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    print('start http server; http://<serverip>:' + str(opt.listenport) + '/' + pagename)

    run_server(web.AppRunner(appasync))
