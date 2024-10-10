import math
import torch
import numpy as np

# from .utils import *
import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob

from nerfasr import NerfASR
from ttsreal import EdgeTTS, VoitsTTS, XTTS

import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

# from imgcache import ImgCache

from tqdm import tqdm


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


class NeRFReal(BaseReal):
    def __init__(self, opt, trainer, data_loader, debug=True):
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H

        self.trainer = trainer
        self.data_loader = data_loader
        self.loader = iter(data_loader)
        frame_total_num = data_loader._data.end_index
        if opt.fullbody:
            input_img_list = glob.glob(os.path.join(self.opt.fullbody_img, '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.fullbody_list_cycle = read_imgs(input_img_list[:frame_total_num])

        # build asr
        self.asr = NerfASR(opt, self)
        self.asr.warm_up()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.opt.asr:
            self.asr.stop()

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()

    def test_step(self, loop=None, audio_track=None, video_track=None):

        try:
            data = next(self.loader)
        except StopIteration:
            self.loader = iter(self.data_loader)
            data = next(self.loader)

        if self.opt.asr:
            data['auds'] = self.asr.get_next_feat()

        audiotype1 = 0
        audiotype2 = 0
        for i in range(2):
            frame, type = self.asr.get_audio_out()
            if i == 0:
                audiotype1 = type
            else:
                audiotype2 = type
            if self.opt.transport == 'rtmp':
                self.streamer.stream_frame_audio(frame)
            else:
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)

        if audiotype1 != 0 and audiotype2 != 0 and self.custom_index.get(audiotype1) is not None:
            mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype1]), self.custom_index[audiotype1])
            image = self.custom_img_cycle[audiotype1][mirindex]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.custom_index[audiotype1] += 1
            if self.opt.transport == 'rtmp':
                self.streamer.stream_frame(image)
            else:
                new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
        else:
            outputs = self.trainer.test_gui_with_data(data, self.W, self.H)
            image = (outputs['image'] * 255).astype(np.uint8)
            if not self.opt.fullbody:
                if self.opt.transport == 'rtmp':
                    self.streamer.stream_frame(image)
                else:
                    new_frame = VideoFrame.from_ndarray(image, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            else:
                image_fullbody = self.fullbody_list_cycle[data['index'][0]]
                image_fullbody = cv2.cvtColor(image_fullbody, cv2.COLOR_BGR2RGB)
                start_x = self.opt.fullbody_offset_x  # 合并后小图片的起始x坐标
                start_y = self.opt.fullbody_offset_y  # 合并后小图片的起始y坐标
                image_fullbody[start_y:start_y + image.shape[0], start_x:start_x + image.shape[1]] = image
                if self.opt.transport == 'rtmp':
                    self.streamer.stream_frame(image_fullbody)
                else:
                    new_frame = VideoFrame.from_ndarray(image_fullbody, format="rgb24")
                    asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.init_customindex()
        count = 0
        totaltime = 0
        _starttime = time.perf_counter()
        _totalframe = 0

        self.tts.render(quit_event)
        while not quit_event.is_set():
            t = time.perf_counter()
            for _ in range(2):
                self.asr.run_step()
            self.test_step(loop, audio_track, video_track)
            totaltime += (time.perf_counter() - t)
            count += 1
            _totalframe += 1
            if count == 100:
                print(f"------actual avg infer fps:{count / totaltime:.4f}")
                count = 0
                totaltime = 0
            if self.opt.transport == 'rtmp':
                delay = _starttime + _totalframe * 0.04 - time.perf_counter()  # 40ms
                if delay > 0:
                    time.sleep(delay)
            else:
                if video_track._queue.qsize() >= 5:
                    time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        print('nerfreal thread stop')
