import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCTC, AutoProcessor, Wav2Vec2Processor, HubertModel


import queue
from queue import Queue
#from collections import deque
from threading import Thread, Event

from baseasr import BaseASR

class NerfASR(BaseASR):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if 'esperanto' in self.opt.asr_model:
            self.audio_dim = 44
        elif 'deepspeech' in self.opt.asr_model:
            self.audio_dim = 29
        elif 'hubert' in self.opt.asr_model:
            self.audio_dim = 1024
        else:
            self.audio_dim = 32

        # each segment is (stride_left + ctx + stride_right) * 20ms, latency should be (ctx + stride_right) * 20ms
        self.context_size = opt.m
        self.stride_left_size = opt.l
        self.stride_right_size = opt.r

        # pad left frames
        if self.stride_left_size > 0:
            self.frames.extend([np.zeros(self.chunk, dtype=np.float32)] * self.stride_left_size)

        # create wav2vec model
        print(f'[INFO] loading ASR model {self.opt.asr_model}...')
        if 'hubert' in self.opt.asr_model:
            self.processor = Wav2Vec2Processor.from_pretrained(opt.asr_model)
            self.model = HubertModel.from_pretrained(opt.asr_model).to(self.device) 
        else:   
            self.processor = AutoProcessor.from_pretrained(opt.asr_model)
            self.model = AutoModelForCTC.from_pretrained(opt.asr_model).to(self.device)

        # the extracted features
        self.feat_buffer_size = 4
        self.feat_buffer_idx = 0
        self.feat_queue = torch.zeros(self.feat_buffer_size * self.context_size, self.audio_dim, dtype=torch.float32, device=self.device)

        self.front = self.feat_buffer_size * self.context_size - 8
        self.tail = 8
        # attention window...
        self.att_feats = [torch.zeros(self.audio_dim, 16, dtype=torch.float32, device=self.device)] * 4

        # warm up steps needed: mid + right + window_size + attention_size
        self.warm_up_steps = self.context_size + self.stride_left_size + self.stride_right_size

    def get_audio_frame(self):         
        try:
            frame = self.queue.get(block=False)
            type = 0
            #print(f'[INFO] get frame {frame.shape}')
        except queue.Empty:
            if self.parent and self.parent.curr_state>1:
                frame = self.parent.get_audio_stream(self.parent.curr_state)
                type = self.parent.curr_state
            else:
                frame = np.zeros(self.chunk, dtype=np.float32)
                type = 1

        return frame,type

    def get_next_feat(self):
        if self.opt.att>0:
            while len(self.att_feats) < 8:
                if self.front < self.tail:
                    feat = self.feat_queue[self.front:self.tail]
                else:
                    feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

                self.front = (self.front + 2) % self.feat_queue.shape[0]
                self.tail = (self.tail + 2) % self.feat_queue.shape[0]

                self.att_feats.append(feat.permute(1, 0))
            
            att_feat = torch.stack(self.att_feats, dim=0) # [8, 44, 16]

            self.att_feats = self.att_feats[1:]
        else:
            if self.front < self.tail:
                feat = self.feat_queue[self.front:self.tail]
            else:
                feat = torch.cat([self.feat_queue[self.front:], self.feat_queue[:self.tail]], dim=0)

            self.front = (self.front + 2) % self.feat_queue.shape[0]
            self.tail = (self.tail + 2) % self.feat_queue.shape[0]

            att_feat = feat.permute(1, 0).unsqueeze(0)


        return att_feat

    def run_step(self):

        # get a frame of audio
        frame,type = self.get_audio_frame()
        self.frames.append(frame)
        self.output_queue.put((frame,type))
        if len(self.frames) < self.stride_left_size + self.context_size + self.stride_right_size:
            return
        
        inputs = np.concatenate(self.frames) # [N * chunk]

        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

        logits, labels, text = self.__frame_to_text(inputs)
        feats = logits
        start = self.feat_buffer_idx * self.context_size
        end = start + feats.shape[0]
        self.feat_queue[start:end] = feats
        self.feat_buffer_idx = (self.feat_buffer_idx + 1) % self.feat_buffer_size
    

        
    def __frame_to_text(self, frame):
        # frame: [N * 320], N = (context_size + 2 * stride_size)
        
        inputs = self.processor(frame, sampling_rate=self.sample_rate, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            result = self.model(inputs.input_values.to(self.device))
            if 'hubert' in self.opt.asr_model:
                logits = result.last_hidden_state
            else:
                logits = result.logits

        left = max(0, self.stride_left_size)
        right = min(logits.shape[1], logits.shape[1] - self.stride_right_size + 1)
        logits = logits[:, left:right]
        return logits[0], None,None
    

    def warm_up(self):       
        print(f'[INFO] warm up ASR live model, expected latency = {self.warm_up_steps / self.fps:.6f}s')
        t = time.time()
        for _ in range(self.warm_up_steps):
            self.run_step()
        t = time.time() - t
        print(f'[INFO] warm-up done, actual latency = {t:.6f}s')