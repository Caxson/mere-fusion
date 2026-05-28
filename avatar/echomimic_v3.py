"""EchoMimicV3 离线半身数字人封装

蚂蚁集团 AAAI 2026，1.3B 参数统一多模态：音频驱动 + 半身 + 手势 + 表情。
适合离线录视频（非流式，整段 diffusion 采样生成 mp4）。录技术分享视频的最佳选择。

上游是脚本式（infer_flash.py + argparse），没有干净的 Python 函数 API，
因此这里通过子进程调用，参数对齐 run_flash.sh。

依赖：
    git clone https://github.com/antgroup/echomimic_v3
    并按上游 README 下载权重（transformer / chinese-wav2vec2-base）。
"""

from __future__ import annotations

import os
import subprocess


class EchoMimicV3:
    def __init__(
        self,
        repo_dir: str,                       # echomimic_v3 仓库根目录
        python_bin: str = "python",
        config_path: str = "config/config.yaml",
        model_name: str = "Wan2.1-Fun-V1.1-1.3B-InP",
        transformer_path: str = "models/transformer/diffusion_pytorch_model.safetensors",
        wav2vec_model_dir: str = "models/chinese-wav2vec2-base",
    ):
        self.repo_dir = repo_dir
        self.python_bin = python_bin
        self.config_path = config_path
        self.model_name = model_name
        self.transformer_path = transformer_path
        self.wav2vec_model_dir = wav2vec_model_dir

    def generate(
        self,
        image_path: str,
        audio_path: str,
        save_path: str,
        prompt: str = "A person is speaking.",
        num_inference_steps: int = 8,
        video_length: int = 81,
        guidance_scale: float = 6.0,
        audio_guidance_scale: float = 3.0,
        sample_size: tuple[int, int] = (768, 768),
        fps: int = 25,
        seed: int = 43,
        weight_dtype: str = "bfloat16",
        extra_args: list[str] | None = None,
    ) -> str:
        """跑 EchoMimicV3 infer_flash.py，返回输出目录。"""
        cmd = [
            self.python_bin, "infer_flash.py",
            "--image_path", os.path.abspath(image_path),
            "--audio_path", os.path.abspath(audio_path),
            "--prompt", prompt,
            "--config_path", self.config_path,
            "--model_name", self.model_name,
            "--transformer_path", self.transformer_path,
            "--wav2vec_model_dir", self.wav2vec_model_dir,
            "--save_path", os.path.abspath(save_path),
            "--num_inference_steps", str(num_inference_steps),
            "--video_length", str(video_length),
            "--guidance_scale", str(guidance_scale),
            "--audio_guidance_scale", str(audio_guidance_scale),
            "--sample_size", str(sample_size[0]), str(sample_size[1]),
            "--fps", str(fps),
            "--seed", str(seed),
            "--weight_dtype", weight_dtype,
        ]
        if extra_args:
            cmd.extend(extra_args)

        subprocess.run(cmd, cwd=self.repo_dir, check=True)
        return os.path.abspath(save_path)
