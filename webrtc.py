import asyncio
import logging
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
from av.frame import Frame
from av.packet import Packet
import fractions

AUDIO_PTIME = 0.020  # 20ms audio
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()
        self.kind = kind
        self._player = player
        self._queue = asyncio.Queue()
        self.timelist = []
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0

    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            if hasattr(self, "_timestamp"):
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('video start:', self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else:  #audio
            if hasattr(self, "_timestamp"):
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                wait = self._start + (self._timestamp / SAMPLE_RATE) - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                print('audio start:', self._start)
            return self._timestamp, AUDIO_TIME_BASE

    # 生成的视频/音频帧接收和处理并返回给下一步处理
    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        frame = await self._queue.get()
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                print(f"------actual avg final fps:{self.framecount / self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
        return frame

    def stop(self):
        super().stop()
        if self._player is not None:
            self._player._stop(self)
            self._player = None


def player_worker_thread(
        quit_event,
        loop,
        container,
        audio_track,
        video_track
):
    container.render(quit_event, loop, audio_track, video_track)


class HumanPlayer:

    def __init__(
            self, nerfreal, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        self.__container = nerfreal

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    asyncio.get_event_loop(),
                    self.__container,
                    self.__audio,
                    self.__video
                ),
            )
            self.__thread.start()

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            self.__thread.join()
            self.__thread = None

        if not self.__started and self.__container is not None:
            self.__container = None

    def __log_debug(self, msg: str, *args) -> None:
        logger.debug(f"HumanPlayer {msg}", *args)
