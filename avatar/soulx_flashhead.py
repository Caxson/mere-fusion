"""SoulX-FlashHead 实时数字人封装

Soul AI Lab 2026-02 开源的流式音频驱动数字人。Lite 档单张 RTX 4090 可达 ~96 FPS、
VRAM ~6.4GB，原生流式、零身份漂移，是当前实时通话场景的最佳开源方案。

依赖（不发 PyPI，需 clone + 装环境 + 下权重）：
    git clone https://github.com/Soul-AILab/SoulX-FlashHead
    huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B
    huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h
    并把 SoulX-FlashHead 仓库根目录加入 PYTHONPATH。

接口对应上游 flash_head/inference.py 的：
    get_pipeline / get_base_data / get_infer_params / get_audio_embedding / run_pipeline
"""

from __future__ import annotations

import numpy as np


class SoulXFlashHead:
    def __init__(
        self,
        ckpt_dir: str = "models/SoulX-FlashHead-1_3B",
        wav2vec_dir: str = "models/wav2vec2-base-960h",
        model_type: str = "lite",   # "lite"=实时 96fps / "pro"=高质量
        world_size: int = 1,
        use_face_crop: bool = False,
        base_seed: int = 42,
    ):
        from flash_head.inference import get_pipeline, get_infer_params

        self._fh = __import__("flash_head.inference", fromlist=["*"])
        self.pipeline = get_pipeline(
            world_size=world_size,
            ckpt_dir=ckpt_dir,
            wav2vec_dir=wav2vec_dir,
            model_type=model_type,
        )
        self.infer_params = get_infer_params()
        self.sample_rate = self.infer_params["sample_rate"]
        self.tgt_fps = self.infer_params["tgt_fps"]
        self.frame_num = self.infer_params["frame_num"]
        self._use_face_crop = use_face_crop
        self._base_seed = base_seed
        self._cond_ready = False

    def set_reference(self, cond_image_path_or_dir: str) -> None:
        """用参考图/目录初始化形象（每次换形象调用一次）。"""
        self._fh.get_base_data(
            self.pipeline,
            cond_image_path_or_dir,
            base_seed=self._base_seed,
            use_face_crop=self._use_face_crop,
        )
        self._cond_ready = True

    def stream_generate(self, audio: np.ndarray):
        """流式：按 frame_num 切音频 embedding，逐块 yield 视频帧 tensor。

        适合实时通话：把 ASR/TTS 产出的音频边喂边出画面。
        """
        if not self._cond_ready:
            raise RuntimeError("call set_reference() before generating")
        audio = np.asarray(audio).flatten().astype(np.float32)
        embedding = self._fh.get_audio_embedding(self.pipeline, audio)
        total = embedding.shape[1]
        slice_len = self.frame_num
        i = 0
        while i + self.frame_num <= total:
            chunk = embedding[:, i : i + self.frame_num].contiguous()
            yield self._fh.run_pipeline(self.pipeline, chunk)
            i += slice_len

    def generate(self, audio: np.ndarray) -> list:
        """离线：一次性返回所有视频帧块。"""
        return [v.cpu() for v in self.stream_generate(audio)]
