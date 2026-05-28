import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum

class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def pause_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg): 
        self.msgqueue.put(msg)

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        print('ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = "zh-CN-YunxiaNeural"
        text = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        print(f'-------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0:
            print('edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            self.parent.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        stream, sample_rate = sf.read(byte_stream)
        print(f'[INFO]tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        communicate = edge_tts.Communicate(text, voicename)

        first = True
        async for chunk in communicate.stream():
            if first:
                first = False
            if chunk["type"] == "audio" and self.state==State.RUNNING:
                self.input_stream.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                pass

###########################################################################################
class VoitsTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.gpt_sovits(
                msg,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh",
                self.opt.TTS_SERVER,
            )
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'text_lang':language,
            'ref_audio_path':reffile,
            'prompt_text':reftext,
            'prompt_lang':language,
            'media_type':'raw',
            'streaming_mode':True
        }
        res = requests.post(
            f"{server_url}/tts",
            json=req,
            stream=True,
        )
        end = time.perf_counter()
        print(f"gpt_sovits Time to make POST: {end-start}s")

        if res.status_code != 200:
            print("Error:", res.text)
            return
            
        first = True
        for chunk in res.iter_content(chunk_size=16000): # 1280 32K*20ms*2
            if first:
                end = time.perf_counter()
                print(f"gpt_sovits Time to first chunk: {end-start}s")
                first = False
            if chunk and self.state==State.RUNNING:
                yield chunk

        print("gpt_sovits response.elapsed:", res.elapsed)

    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.cosy_voice(
                msg,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh",
                self.opt.TTS_SERVER,
            )
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
        res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
        
        end = time.perf_counter()
        print(f"cosy_voice Time to make POST: {end-start}s")

        if res.status_code != 200:
            print("Error:", res.text)
            return
            
        first = True
        for chunk in res.iter_content(chunk_size=16000): # 1280 32K*20ms*2
            if first:
                end = time.perf_counter()
                print(f"cosy_voice Time to first chunk: {end-start}s")
                first = False
            if chunk and self.state==State.RUNNING:
                yield chunk

        print("cosy_voice response.elapsed:", res.elapsed)

    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg): 
        self.stream_tts(
            self.xtts(
                msg,
                self.speaker,
                "zh-cn",
                self.opt.TTS_SERVER,
                "20"
            )
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size
        res = requests.post(
            f"{server_url}/tts_stream",
            json=speaker,
            stream=True,
        )
        end = time.perf_counter()
        print(f"xtts Time to make POST: {end-start}s")

        if res.status_code != 200:
            print("Error:", res.text)
            return

        first = True
        for chunk in res.iter_content(chunk_size=960):
            if first:
                end = time.perf_counter()
                print(f"xtts Time to first chunk: {end-start}s")
                first = False
            if chunk:
                yield chunk

        print("xtts response.elapsed:", res.elapsed)
    
    def stream_tts(self,audio_stream):
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk])
                    streamlen -= self.chunk
                    idx += self.chunk 
###########################################################################################
class CosyVoice2TTS(BaseTTS):
    """CosyVoice 2 / 3 进程内流式零样本克隆。

    不走 HTTP server，直接在进程内加载模型并流式合成（stream=True），
    每次 yield 一个 torch 音频块，重采样到 16kHz 后逐帧推给数字人。

    依赖：clone FunAudioLLM/CosyVoice 并把仓库根目录 + third_party/Matcha-TTS
    加入 PYTHONPATH。模型目录通过 opt.cosyvoice_model_dir 指定
    （CosyVoice2-0.5B 或 CosyVoice3-* ）。
    需要 opt.REF_FILE（参考音频）+ opt.REF_TEXT（参考音频对应文字）。
    """

    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self._model = None
        self._model_sr = 24000  # CosyVoice2 默认 24k

    def _ensure_model(self):
        if self._model is not None:
            return
        model_dir = getattr(self.opt, "cosyvoice_model_dir", "pretrained_models/CosyVoice2-0.5B")
        # CosyVoice3 目录含 cosyvoice3.yaml；否则用 CosyVoice2。
        use_v3 = "cosyvoice3" in str(model_dir).lower()
        if use_v3:
            from cosyvoice.cli.cosyvoice import CosyVoice3 as _CV
            self._model = _CV(model_dir, load_trt=False, load_vllm=False, fp16=False)
        else:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as _CV
            self._model = _CV(model_dir, load_jit=False, load_trt=False, load_vllm=False, fp16=False)
        self._model_sr = getattr(self._model, "sample_rate", 24000)

    def txt_to_audio(self, msg):
        try:
            self._ensure_model()
        except Exception as e:
            print(f"[CosyVoice2TTS] 模型加载失败 ({type(e).__name__}: {e})，跳过本句")
            return
        self.stream_tts(self.cosyvoice2(msg))

    def cosyvoice2(self, text) -> Iterator[np.ndarray]:
        start = time.perf_counter()
        first = True
        for out in self._model.inference_zero_shot(
            text,
            self.opt.REF_TEXT,
            self.opt.REF_FILE,
            stream=True,
        ):
            if first:
                print(f"CosyVoice2 time to first chunk: {time.perf_counter()-start:.3f}s")
                first = False
            if self.state != State.RUNNING:
                break
            speech = out["tts_speech"]  # torch.Tensor [1, T]
            yield speech.cpu().numpy().flatten().astype(np.float32)

    def stream_tts(self, audio_stream):
        for stream in audio_stream:
            if stream is None or len(stream) == 0:
                continue
            if self._model_sr != self.sample_rate:
                stream = resampy.resample(x=stream, sr_orig=self._model_sr, sr_new=self.sample_rate)
            streamlen = stream.shape[0]
            idx = 0
            while streamlen >= self.chunk and self.state == State.RUNNING:
                self.parent.put_audio_frame(stream[idx:idx + self.chunk])
                streamlen -= self.chunk
                idx += self.chunk
