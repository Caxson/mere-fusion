#!/usr/bin/env python3
from whisper_online import *
from stream_openai_video import text_produce

import argparse
import os
import logging
import numpy as np
import subprocess
import ffmpeg
import sys

logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

# server options
parser.add_argument("--host", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, default=18282)
parser.add_argument("--warmup-file", type=str, dest="data/whisper/whisper.wav",
                    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")

# options from whisper_online
add_shared_args(parser)
args = parser.parse_args()

set_logging(args, logger, other="")

# setting whisper object by args
SAMPLING_RATE = 16000
BUFFER_SIZE = 65536

size = args.model
language = args.lan
asr, online = asr_factory(args)
min_chunk = 1
args.warmup_file = "data/whisper/whisper.wav"

logger.info("Starting Whisper server on port")

######### Server objects

import io


class Connection:
    '''it wraps conn object'''

    def __init__(self, sock):
        self.sock = sock

    def non_blocking_receive_audio(self):
        r, addr = self.sock.recvfrom(BUFFER_SIZE)
        return r


class WhisperRTCServerProcessor:
    def __init__(self, online_asr_proc, min_chunk):
        # self.connection = Connection(sock)
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None
        self.chunk_count = 0

    def receive_audio_chunk(self, raw_bytes):
        out = []
        while sum(len(x) for x in out) < self.min_chunk * SAMPLING_RATE:
            if not raw_bytes:
                break
            with io.BytesIO(raw_bytes) as raw_io:
                with sf.SoundFile(raw_io, channels=1, samplerate=SAMPLING_RATE,
                                  subtype='PCM_16', format='RAW') as sound_file:
                    audio, _ = librosa.load(sound_file, sr=SAMPLING_RATE, dtype=np.float32)
                    out.append(audio)
        if not out:
            return None
        return np.concatenate(out)

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            # 将分析结果发送给消费者
            text_produce(msg)
            logger.info(f"res_msg: {msg}")
        else:
            logger.warning(f"No text in this segment")

    def process(self, raw_bytes):
        self.online_asr_proc.init()
        logger.info("##### 音频接收成功 #####")
        a = self.receive_audio_chunk(raw_bytes)
        if a is None:
            return None
        self.online_asr_proc.insert_audio_chunk(a)
        o = self.online_asr_proc.process_iter()
        self.send_result(o)


class WhisperRTPServerProcessor:
    def __init__(self, online_asr_proc, min_chunk):
        # self.connection = Connection(sock)
        self.online_asr_proc = online_asr_proc
        self.min_chunk = min_chunk
        self.last_end = None

        self.output_dir = '//'
        self.chunk_count = 0
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ffmpeg_process = self.create_ffmpeg_process()

    def receive_audio_chunk(self):
        out = []
        while sum(len(x) for x in out) < self.min_chunk * SAMPLING_RATE:
            raw_bytes = self.ffmpeg_process.stdout.read(BUFFER_SIZE)
            if not raw_bytes:
                break

            with io.BytesIO(raw_bytes) as raw_io:
                with sf.SoundFile(raw_io, channels=1, samplerate=SAMPLING_RATE,
                                  subtype='PCM_16', format='RAW') as sound_file:
                    audio, _ = librosa.load(sound_file, sr=SAMPLING_RATE, dtype=np.float32)
                    out.append(audio)
        if not out:
            return None
        # DEBUG 本地存储，调试用
        # concatenated_audio = np.concatenate(out)
        # self.save_audio_to_file(concatenated_audio)
        # return concatenated_audio
        return np.concatenate(out)

    def save_audio_to_file(self, audio_data):
        """将解码后的浮点格式音频数据保存为 .wav 文件"""
        self.chunk_count += 1
        file_path = os.path.join(self.output_dir, f"audio_chunk_{self.chunk_count}.wav")

        # 使用 soundfile 保存为 .wav 文件
        sf.write(file_path, audio_data, SAMPLING_RATE, format='WAV', subtype='PCM_16')
        logger.info(f"Saved audio chunk to {file_path}")

    def format_output_transcript(self, o):
        if o[0] is not None:
            beg, end = o[0] * 1000, o[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)
            self.last_end = end
            print("%1.0f %1.0f %s" % (beg, end, o[2]), flush=True, file=sys.stderr)
            return "%1.0f %1.0f %s" % (beg, end, o[2])
        else:
            logger.debug("No text in this segment")
            return None

    def send_result(self, o):
        msg = self.format_output_transcript(o)
        if msg is not None:
            # 将分析结果发送给消费者
            text_produce(msg)
            logger.info(f"res_msg: {msg}")
        else:
            logger.warning(f"No text in this segment")

    def process(self):
        self.online_asr_proc.init()
        logger.info("##### 音频处理服务启动成功 #####")
        while True:
            a = self.receive_audio_chunk()
            if a is None:
                continue
            self.online_asr_proc.insert_audio_chunk(a)
            o = self.online_asr_proc.process_iter()
            self.send_result(o)

    def create_ffmpeg_process(self):
        input_stream = ffmpeg.input(f'rtp://0.0.0.0:18282')

        # 设置输出到标准输出
        output_stream = ffmpeg.output(input_stream, 'pipe:1', format='wav', ar=16000)

        # 使用subprocess运行ffmpeg命令并捕获输出
        ffmpeg_process = subprocess.Popen(
            ffmpeg.compile(output_stream),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return ffmpeg_process


# Server loop for UDP
# with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
#     s.bind((args.host, args.port))
#     logger.info('Listening on'+str((args.host, args.port)))

# start_consumer_thread()
# 音频服务启动
def whisper_main():
    # warm up the ASR because the very first transcribe takes more time than the others.
    msg = "Whisper is not warmed up. The first chunk processing may take longer."
    if args.warmup_file:
        if os.path.isfile(args.warmup_file):
            a = load_audio_chunk(args.warmup_file, 0, 1)
            asr.transcribe(a)
            logger.info("Whisper is warmed up.")
        else:
            logger.critical("The warm up file is not available. " + msg)
            sys.exit(1)
    else:
        logger.warning(msg)

    # 音频处理服务启动
    while True:
        logger.info("whisper server waiting for a connection")
        proc = WhisperRTPServerProcessor(online, min_chunk)
        proc.process()

    logger.info('Whisper Server Connection closed, terminating.')
