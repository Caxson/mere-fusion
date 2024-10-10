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
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import multiprocessing as mp

from musetalk.utils.utils import get_file_type, get_video_fps, datagen
# from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
from musetalk.utils.utils import load_all_model, load_diffusion_model, load_audio_model
from ttsreal import EdgeTTS, VoitsTTS, XTTS

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm


def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames


def __mirror_index(size, index):
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1


@torch.no_grad()
def inference(render_event, batch_size, latents_out_path, audio_feat_queue, audio_out_queue, res_frame_queue,
              ):  # vae, unet, pe,timesteps

    vae, unet, pe = load_diffusion_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    unet.model = unet.model.half()

    input_latent_list_cycle = torch.load(latents_out_path)
    length = len(input_latent_list_cycle)
    index = 0
    count = 0
    counttime = 0
    print('start inference')
    while True:
        if render_event.is_set():
            starttime = time.perf_counter()
            try:
                whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            is_all_silence = True
            audio_frames = []
            for _ in range(batch_size * 2):
                frame, type = audio_out_queue.get()
                audio_frames.append((frame, type))
                if type == 0:
                    is_all_silence = False
            if is_all_silence:
                for i in range(batch_size):
                    res_frame_queue.put((None, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index = index + 1
            else:
                # print('infer=======')
                t = time.perf_counter()
                whisper_batch = np.stack(whisper_chunks)
                latent_batch = []
                for i in range(batch_size):
                    idx = __mirror_index(length, index + i)
                    latent = input_latent_list_cycle[idx]
                    latent_batch.append(latent)
                latent_batch = torch.cat(latent_batch, dim=0)

                audio_feature_batch = torch.from_numpy(whisper_batch)
                audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                             dtype=unet.model.dtype)
                audio_feature_batch = pe(audio_feature_batch)
                latent_batch = latent_batch.to(dtype=unet.model.dtype)

                pred_latents = unet.model(latent_batch,
                                          timesteps,
                                          encoder_hidden_states=audio_feature_batch).sample
                recon = vae.decode_latents(pred_latents)
                counttime += (time.perf_counter() - t)
                count += batch_size
                # _totalframe += 1
                if count >= 100:
                    print(f"------actual avg infer fps:{count / counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(recon):
                    # self.__pushmedia(res_frame,loop,audio_track,video_track)
                    res_frame_queue.put((res_frame, __mirror_index(length, index), audio_frames[i * 2:i * 2 + 2]))
                    index = index + 1
        else:
            time.sleep(1)
    print('musereal inference processor stop')


@torch.no_grad()
class MuseReal(BaseReal):
    def __init__(self, opt):
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H

        self.fps = opt.fps

        #### musetalk
        self.avatar_id = opt.avatar_id
        self.video_path = ''  # video_path
        self.bbox_shift = opt.bbox_shift
        self.avatar_path = f"./data/avatars/{self.avatar_id}"
        self.full_imgs_path = f"{self.avatar_path}/full_imgs"
        self.coords_path = f"{self.avatar_path}/coords.pkl"
        self.latents_out_path = f"{self.avatar_path}/latents.pt"
        self.video_out_path = f"{self.avatar_path}/vid_output/"
        self.mask_out_path = f"{self.avatar_path}/mask"
        self.mask_coords_path = f"{self.avatar_path}/mask_coords.pkl"
        self.avatar_info_path = f"{self.avatar_path}/avator_info.json"
        self.avatar_info = {
            "avatar_id": self.avatar_id,
            "video_path": self.video_path,
            "bbox_shift": self.bbox_shift
        }
        self.batch_size = opt.batch_size
        self.idx = 0
        self.res_frame_queue = mp.Queue(self.batch_size * 2)
        self.__loadmodels()
        self.__loadavatar()

        self.asr = MuseASR(opt, self, self.audio_processor)
        self.asr.warm_up()
        # self.__warm_up()

        self.render_event = mp.Event()
        mp.Process(target=inference, args=(self.render_event, self.batch_size, self.latents_out_path,
                                           self.asr.feat_queue, self.asr.output_queue, self.res_frame_queue,
                                           )).start()  # self.vae, self.unet, self.pe,self.timesteps

    def __loadmodels(self):
        self.audio_processor = load_audio_model()

    def __loadavatar(self):
        with open(self.coords_path, 'rb') as f:
            self.coord_list_cycle = pickle.load(f)
        input_img_list = glob.glob(os.path.join(self.full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.frame_list_cycle = read_imgs(input_img_list)
        with open(self.mask_coords_path, 'rb') as f:
            self.mask_coords_list_cycle = pickle.load(f)
        input_mask_list = glob.glob(os.path.join(self.mask_out_path, '*.[jpJP][pnPN]*[gG]'))
        input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        self.mask_list_cycle = read_imgs(input_mask_list)

    def put_msg_txt(self, msg):
        self.tts.put_msg_txt(msg)

    def put_audio_frame(self, audio_chunk):  # 16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk)

    def pause_talk(self):
        self.tts.pause_talk()
        self.asr.pause_talk()

    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1

    def __warm_up(self):
        self.asr.run_step()
        whisper_chunks = self.asr.get_next_feat()
        whisper_batch = np.stack(whisper_chunks)
        latent_batch = []
        for i in range(self.batch_size):
            idx = self.__mirror_index(self.idx + i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        print('infer=======')
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                     dtype=self.unet.model.dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(latent_batch,
                                       self.timesteps,
                                       encoder_hidden_states=audio_feature_batch).sample
        recon = self.vae.decode_latents(pred_latents)

    def process_frames(self, quit_event, loop=None, audio_track=None, video_track=None):

        while not quit_event.is_set():
            try:
                res_frame, idx, audio_frames = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            if audio_frames[0][1] != 0 and audio_frames[1][1] != 0:  # 全为静音数据，只需要取fullimg
                audiotype = audio_frames[0][1]
                if self.custom_index.get(audiotype) is not None:  # 有自定义视频
                    mirindex = self.mirror_index(len(self.custom_img_cycle[audiotype]), self.custom_index[audiotype])
                    combine_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    combine_frame = self.frame_list_cycle[idx]
            else:
                bbox = self.coord_list_cycle[idx]
                ori_frame = copy.deepcopy(self.frame_list_cycle[idx])
                x1, y1, x2, y2 = bbox
                try:
                    res_frame = cv2.resize(res_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                except:
                    continue
                mask = self.mask_list_cycle[idx]
                mask_crop_box = self.mask_coords_list_cycle[idx]
                combine_frame = get_image_blending(ori_frame, res_frame, bbox, mask, mask_crop_box)
                # print('blending time:',time.perf_counter()-t)

            image = combine_frame
            new_frame = VideoFrame.from_ndarray(image, format="bgr24")
            asyncio.run_coroutine_threadsafe(video_track._queue.put(new_frame), loop)
            if self.recording:
                self.recordq_video.put(new_frame)

            for audio_frame in audio_frames:
                frame, type = audio_frame
                frame = (frame * 32767).astype(np.int16)
                new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                new_frame.planes[0].update(frame.tobytes())
                new_frame.sample_rate = 16000
                asyncio.run_coroutine_threadsafe(audio_track._queue.put(new_frame), loop)
                if self.recording:
                    self.recordq_audio.put(new_frame)
        print('musereal process_frames thread stop')

    def render(self, quit_event, loop=None, audio_track=None, video_track=None):
        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event, loop, audio_track, video_track))
        process_thread.start()

        # start infer process render
        self.render_event.set()
        _starttime = time.perf_counter()
        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()
            if video_track._queue.qsize() >= 1.5 * self.opt.batch_size:
                print('sleep qsize=', video_track._queue.qsize())
                time.sleep(0.04 * video_track._queue.qsize() * 0.8)
        self.render_event.clear()
        print('musereal thread stop')
