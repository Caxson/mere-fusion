import asyncio
import json
import random

from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiohttp import ClientSession
import av
import cv2


class RelayVideoStreamTrack(MediaStreamTrack):
    """
    用于对接收到的视频帧进行处理，然后发送出去。
    """

    kind = "video"

    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        print('Processing frame in VideoTransformTrack')
        # 对视频帧进行处理，例如将图像转换为灰度
        # img = frame.to_ndarray(format="bgr24")
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        #
        # new_frame = av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        # new_frame.pts = frame.pts
        # new_frame.time_base = frame.time_base
        # return new_frame
        return frame

class RelayAudioStreamTrack(MediaStreamTrack):
    """
    音频轨道的中继类，直接转发接收到的音频帧。
    """

    kind = "audio"

    def __init__(self, source):
        super().__init__()  # 初始化父类
        self.source = source  # 源音频轨道

    async def recv(self):
        frame = await self.source.recv()
        return frame  # 直接返回接收到的音频帧


async def consume_and_produce_stream():
    # 拉流：从 SRS 拉取前端推送的流
    consume_pc = RTCPeerConnection()

    # 添加接收器，表示我们希望接收音视频流
    consume_pc.addTransceiver('audio', direction='recvonly')
    consume_pc.addTransceiver('video', direction='recvonly')

    # 推流：将处理后的流推回 SRS
    produce_pc = RTCPeerConnection()

    # 保存接收到的媒体流
    media_tracks = []

    async with ClientSession() as session:
        # 拉流部分
        play_url = 'http://123.56.254.166:1985/rtc/v1/play/'
        play_params = {
            'api': play_url,
            'streamurl': 'webrtc://123.56.254.166/live/stream',
            'clientip': None,
            'sdp': '',
            'tid': str(random.randint(10000, 99999)),
            'action': 'play'
        }

        # 当接收到媒体流时，保存轨道并准备处理
        @consume_pc.on('track')
        def on_track(track):
            print(f'Track {track.kind} received, id: {track.id}')
            if track.kind == 'video':
                relay_video = RelayVideoStreamTrack(track)
                produce_pc.addTrack(relay_video)
                print('Video track relayed to produce_pc')
            elif track.kind == 'audio':
                relay_audio = RelayAudioStreamTrack(track)
                produce_pc.addTrack(relay_audio)
                print('Audio track relayed to produce_pc')

                # @track.on('frame')
                # async def on_frame(frame):
                #     print(f'Received frame: {frame}')

        # Add connection state change handler
        @consume_pc.on('connectionstatechange')
        async def on_connectionstatechange():
            print(f'Consume PC connection state: {consume_pc.connectionState}')
            if consume_pc.connectionState == 'failed':
                await consume_pc.close()


        # 创建 offer
        consume_offer = await consume_pc.createOffer()
        await consume_pc.setLocalDescription(consume_offer)
        play_params['sdp'] = consume_offer.sdp

        # 发送请求到 SRS
        async with session.post(play_url, json=play_params) as resp:
            res = await resp.json()
            if res.get('code') and res['code'] != 0:
                print(f"Failed to play stream: {res}")
                return
            # 设置远端描述
            answer = RTCSessionDescription(sdp=res['sdp'], type='answer')
            await consume_pc.setRemoteDescription(answer)

        # 等待媒体流接收完成
        # while not media_tracks:
        #     await asyncio.sleep(0.1)

        # await asyncio.sleep(1)

        # 推流部分
        publish_url = 'http://123.56.254.166:1985/rtc/v1/publish/'
        publish_params = {
            'api': publish_url,
            'streamurl': 'webrtc://123.56.254.166/live/processed_stream',
            'clientip': None,
            'sdp': '',
            'tid': str(random.randint(10000, 99999)),
            'action': 'publish'
        }

        # 创建并设置本地描述
        produce_offer = await produce_pc.createOffer()
        await produce_pc.setLocalDescription(produce_offer)
        print("Produce Offer SDP:", produce_offer.sdp)
        publish_params['sdp'] = produce_offer.sdp

        # 发送推流请求到 SRS
        async with session.post(publish_url, json=publish_params) as resp:
            res = await resp.json()
            print("SRS Publish Response:", res)
            if res.get('code') and res['code'] != 0:
                print(f"Failed to publish stream: {res}")
                return
            answer = RTCSessionDescription(sdp=res['sdp'], type='answer')
            print("Produce Answer SDP:", answer.sdp)
            await produce_pc.setRemoteDescription(answer)

        # 保持连接
        print("Stream relaying established. Running for 1 hour.")
        await asyncio.sleep(3600)


def main():
    asyncio.run(consume_and_produce_stream())


if __name__ == '__main__':
    main()
