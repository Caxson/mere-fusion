# 主函数
import asyncio
import json
import logging
import subprocess
from dotenv import load_dotenv
from flask import Flask
from flask_sockets import Sockets

from stream_openai_video import stream_out_video_main
from whisper_online_server import whisper_main
from yolo_opencv import yolo_opencv_main
from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
sockets = Sockets(app)
nerfreals = []
statreals = []


@sockets.route('/humanecho')
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


def llm_response(message):
    from llm.LLM import LLM
    llm = LLM().init_model('VllmGPT', model_path='THUDM/chatglm3-6b')
    response = llm.chat(message)
    print(response)
    return response


@sockets.route('/humanchat')
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


##############################################################
pcs = set()

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    sessionid = len(nerfreals)
    for index, value in enumerate(statreals):
        if value == 0:
            sessionid = index
            break
    if sessionid >= len(nerfreals):
        print('reach max session')
        return -1
    statreals[sessionid] = 1

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)
            statreals[sessionid] = 0
        if pc.connectionState == "closed":
            pcs.discard(pc)
            statreals[sessionid] = 0

    player = HumanPlayer(nerfreals[sessionid])
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
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid": sessionid}
        ),
    )


async def human(request):
    params = await request.json()

    sessionid = params.get('sessionid', 0)
    if params.get('interrupt'):
        nerfreals[sessionid].pause_talk()

    if params['type'] == 'echo':
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        res = await asyncio.get_event_loop().run_in_executor(None, llm_response(params['text']))
        nerfreals[sessionid].put_msg_txt(res)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


async def set_audiotype(request):
    params = await request.json()

    sessionid = params.get('sessionid', 0)
    nerfreals[sessionid].set_curr_state(params['audiotype'], params['reinit'])

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
        nerfreals[sessionid].start_recording("data/record_lasted.mp4")
    elif params['type'] == 'end_record':
        nerfreals[sessionid].stop_recording()
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )


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

    player = HumanPlayer(nerfreals[0])
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

    # 音频输入识别主函数
    whisper_main()
    # 视频输入识别主函数
    # yolo_opencv_main()
    # 流式处理和输出主函数
    opt, nerfreal = stream_out_video_main()

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
