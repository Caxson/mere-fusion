# 主函数
import asyncio
import io
import json
import logging
import subprocess

import librosa
from dotenv import load_dotenv
from flask import Flask
from flask_sockets import Sockets

from stream_openai_video import stream_out_video_main
from whisper_online_server import whisper_main, WhisperRTCServerProcessor
from yolo_opencv import yolo_opencv_main, YoloOpencvProcessor
from aiohttp import web
import aiohttp
import aiohttp_cors
import soundfile as sf
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer
import multiprocessing
import argparse

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
sockets = Sockets(app)
session_ids = []
session_resources = []


@sockets.route('/msg_text')
def echo_socket(ws):
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    else:
        print('建立连接！')
        while True:
            message = ws.receive()
            if not message or len(message) == 0:
                return '输入信息为空'
            else:
                nerfreal.put_msg_txt(message)


@sockets.route('/msg_text_ai')
def chat_socket(ws):
    if not ws:
        print('未建立连接！')
        return 'Please use WebSocket'
    else:
        print('建立连接！')
        while True:
            message = ws.receive()
            if len(message) == 0:
                return '输入信息为空'
            else:
                res = llm_response(message)
                nerfreal.put_msg_txt(res)


def llm_response(message):
    from llm.LLM import LLM
    llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    response = llm.chat(message)
    print(response)
    return response


##############################################################
pcs = set()


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    session_id = len(session_ids)
    for index, value in enumerate(session_resources):
        if value == 0:
            session_id = index
            break
    if session_id >= len(session_ids):
        print('reach max session')
        return -1
    session_resources[session_id] = 1

    pc = RTCPeerConnection()
    pcs.add(pc)

    # Monitor the input of video and audio streams

    session = UserSession(session_id, pc)

    @pc.on("track")
    async def on_track(track):
        """根据轨道类型，将轨道添加到会话中"""
        session.add_track(track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            session_resources[session_id] = 0
        if pc.connectionState == "closed":
            pcs.discard(pc)
            session_resources[session_id] = 0

    player = HumanPlayer(session_ids[session_id])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)

    await pc.setRemoteDescription(offer)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": session_id}
        ),
    )


async def human(request):
    params = await request.json()

    sessionid = params.get('sessionid', 0)
    if params.get('interrupt'):
        session_ids[sessionid].pause_talk()

    if params['type'] == 'echo':
        session_ids[sessionid].put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response(params['text']))
        session_ids[sessionid].put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid', 0)
    session_ids[sessionid].set_curr_state(params['audiotype'], params['reinit'])

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def record(request):
    params = await request.json()

    sessionid = params.get('sessionid', 0)
    if params['type'] == 'start_record':
        session_ids[sessionid].start_recording("data/record_lasted.mp4")
    elif params['type'] == 'end_record':
        session_ids[sessionid].stop_recording()
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
        self.video_buffer = []  # 用于存储视频帧的缓冲区

    def add_track(self, track):
        """根据轨道类型添加处理逻辑"""
        if track.kind == "audio":
            logging.info(f"会话 {self.session_id}: 接收到音频轨道")
            self.pc.addTrack(AudioStreamTrack(track))
        elif track.kind == "video":
            logging.info(f"会话 {self.session_id}: 接收到视频轨道")
            self.pc.addTrack(VideoStreamTrack(track))


class AudioStreamTrack(MediaStreamTrack):
    """处理音频流数据"""

    kind = "audio"

    def __init__(self, track):
        super().__init__()
        self.track = track
        self.buffer = []
        self.processor = WhisperRTCServerProcessor()

    async def recv(self):
        frame = await self.track.recv()
        raw_bytes = frame.to_bytes()
        self.processor.process(raw_bytes)
        return frame


class VideoStreamTrack(MediaStreamTrack):
    """处理视频流数据"""

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track
        # 为每个会话创建独立的处理器
        self.processor = YoloOpencvProcessor()

    async def recv(self):
        frame = await self.track.recv()
        # 在每一帧调用时
        self.processor.process_frame(frame)
        return frame


async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()


async def post(url, data):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()
    except aiohttp.ClientError as e:
        print(f'Error: {e}')


async def run(push_url):
    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    player = HumanPlayer(session_ids[0])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)

    await pc.setLocalDescription(await pc.createOffer())
    answer = await post(push_url, pc.localDescription.sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))


def run_server(runner):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
    loop.run_until_complete(site.start())
    if opt.transport == 'rtcpush':
        loop.run_until_complete(run(opt.push_url))
    loop.run_forever()


##########################################

if __name__ == "__main__":
    # 加载 .env 文件中的环境变量，例如 API 密钥
    load_dotenv()
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

    # 音频输入识别主函数
    # whisper_main()
    # 视频输入识别主函数
    yolo_opencv_main()
    # 流式处理和输出主函数
    nerfreal = stream_out_video_main(opt)

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/set_audiotype", set_audiotype)
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
