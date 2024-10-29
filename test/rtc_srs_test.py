import asyncio
import json
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
import websockets

# 配置日志
logging.basicConfig(level=logging.DEBUG)


# 示例：处理流（这里只是简单的转发）
async def process_stream(track: MediaStreamTrack):
    logging.debug(f"处理轨道: {track.kind}")
    # 实现您的流处理逻辑
    pass


async def handle_publish_signaling(pc_publish: RTCPeerConnection):
    logging.debug("Connecting to SRS WebSocket for publish signaling")
    try:
        async with websockets.connect('ws://123.56.254.166:1985/rtc/v1/signaling') as websocket:
            logging.debug("Connected to SRS WebSocket for publish signaling")

            # 等待接收 SDP Offer
            offer_data = json.loads(await websocket.recv())
            logging.debug(f"Received SDP Offer: {offer_data}")
            offer = RTCSessionDescription(sdp=offer_data['sdp'], type=offer_data['type'])
            await pc_publish.setRemoteDescription(offer)

            # 创建 SDP Answer
            answer = await pc_publish.createAnswer()
            await pc_publish.setLocalDescription(answer)

            # 发送 SDP Answer
            await websocket.send(json.dumps({
                'sdp': pc_publish.localDescription.sdp,
                'type': pc_publish.localDescription.type
            }))
            logging.debug("Sent SDP Answer")

            # 处理 ICE 候选
            async for message in websocket:
                data = json.loads(message)
                logging.debug(f"Received message: {data}")
                if 'candidate' in data:
                    candidate = data['candidate']
                    await pc_publish.addIceCandidate(candidate)
                elif 'sdp' in data:
                    await pc_publish.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
    except Exception as e:
        logging.error(f"Error in handle_publish_signaling: {e}")


async def handle_play_signaling(pc_play: RTCPeerConnection):
    logging.debug("Connecting to SRS WebSocket for play signaling")
    try:
        async with websockets.connect('ws://123.56.254.166:1985/rtc/v1/signaling') as websocket:
            logging.debug("Connected to SRS WebSocket for play signaling")

            # 创建 SDP Offer
            offer = await pc_play.createOffer()
            await pc_play.setLocalDescription(offer)
            logging.debug(f"Created SDP Offer: {offer.sdp}")

            # 发送 SDP Offer
            await websocket.send(json.dumps({
                'sdp': pc_play.localDescription.sdp,
                'type': pc_play.localDescription.type
            }))
            logging.debug("Sent SDP Offer")

            # 等待接收 SDP Answer
            answer_data = json.loads(await websocket.recv())
            logging.debug(f"Received SDP Answer: {answer_data}")
            answer = RTCSessionDescription(sdp=answer_data['sdp'], type=answer_data['type'])
            await pc_play.setRemoteDescription(answer)

            # 处理 ICE 候选
            async for message in websocket:
                data = json.loads(message)
                logging.debug(f"Received message: {data}")
                if 'candidate' in data:
                    candidate = data['candidate']
                    await pc_play.addIceCandidate(candidate)
                elif 'sdp' in data:
                    await pc_play.setRemoteDescription(RTCSessionDescription(sdp=data['sdp'], type=data['type']))
    except Exception as e:
        logging.error(f"Error in handle_play_signaling: {e}")


async def main():
    # 发布端 RTCPeerConnection（接收前端推流）
    pc_publish = RTCPeerConnection()

    # 当接收到轨道时，进行处理并推回
    @pc_publish.on("track")
    async def on_track(track):
        logging.debug(f"收到轨道: {track.kind}")
        # 处理流
        await process_stream(track)
        # 将处理后的轨道添加到播放连接
        pc_play.addTrack(track)

    # 推流端 RTCPeerConnection（推送处理后的流）
    pc_play = RTCPeerConnection()

    # 连接发布和播放信令
    await asyncio.gather(
        handle_publish_signaling(pc_publish),
        handle_play_signaling(pc_play),
    )


if __name__ == '__main__':
    asyncio.run(main())
