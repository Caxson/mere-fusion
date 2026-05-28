"""
mere-fusion: 从 Markdown 讲稿生成数字人技术分享视频

流程: Markdown → 拆段 → TTS → 合并音频 → 数字人渲染 → ffmpeg 字幕合成

用法:
  python examples/record_tech_video/generate_video.py \
    --script examples/record_tech_video/tech_script.md \
    --tts edgetts --avatar musetalk \
    --output examples/record_tech_video/output.mp4
"""

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()


def parse_markdown(script_path: str) -> list[dict]:
    """把 Markdown 拆成段落列表，每段有 title + text"""
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()

    paragraphs = []
    current_title = ""

    for line in content.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            current_title = re.sub(r"^#+\s*", "", line)
            continue
        paragraphs.append({
            "title": current_title,
            "text": line,
        })

    return paragraphs


async def tts_segment(
    text: str,
    index: int,
    tts_type: str,
    output_dir: str,
    voice: str = "zh-CN-YunxiNeural",
    voice_prompt: str | None = None,
) -> str:
    """对单段文字做 TTS，返回 wav 路径"""
    wav_path = os.path.join(output_dir, f"segment_{index:03d}.wav")

    if tts_type == "edgetts":
        import edge_tts
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(wav_path)

    elif tts_type == "cosyvoice":
        import requests
        payload = {"text": text, "speaker": "default"}
        if voice_prompt:
            payload["reference_audio"] = voice_prompt
        resp = requests.post(
            "http://127.0.0.1:9880/tts",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        with open(wav_path, "wb") as f:
            f.write(resp.content)

    else:
        raise ValueError(f"不支持的 TTS: {tts_type}")

    return wav_path


def concat_audio(wav_files: list[str], output_path: str) -> None:
    """用 ffmpeg 拼接多段 wav"""
    list_file = output_path + ".list.txt"
    with open(list_file, "w") as f:
        for wav in wav_files:
            f.write(f"file '{os.path.abspath(wav)}'\n")

    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", output_path],
        check=True,
        capture_output=True,
    )
    os.remove(list_file)


def generate_srt(paragraphs: list[dict], wav_files: list[str], srt_path: str) -> None:
    """根据每段音频时长生成 SRT 字幕"""
    import soundfile as sf

    entries = []
    current_time = 0.0

    for i, (para, wav) in enumerate(zip(paragraphs, wav_files)):
        data, sr = sf.read(wav)
        duration = len(data) / sr

        start_ts = format_srt_time(current_time)
        end_ts = format_srt_time(current_time + duration)

        entries.append(f"{i + 1}\n{start_ts} --> {end_ts}\n{para['text']}\n")
        current_time += duration

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(entries))


def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def burn_subtitles(video_path: str, srt_path: str, output_path: str) -> None:
    """用 ffmpeg 把字幕烧录到视频"""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style='FontSize=22,PrimaryColour=&HFFFFFF&'",
            "-c:a", "copy",
            output_path,
        ],
        check=True,
        capture_output=True,
    )


async def run_avatar(audio_path: str, avatar_type: str, avatar_image: str, output_video: str) -> str:
    """音频驱动数字人生成视频"""
    if avatar_type == "musetalk":
        print(f"[Avatar] MuseTalk 推理: {audio_path} → {output_video}")
        # 任何原因（缺依赖/缺权重/无 GPU/版本不兼容）失败都优雅降级，
        # 不让整条录视频管道崩在最后一步。
        try:
            from musetalk.mere_musetalk import MuseTalkInference
            inferencer = MuseTalkInference(avatar_path=avatar_image)
            inferencer.run(audio_path=audio_path, output_path=output_video)
        except Exception as e:
            print(f"[Avatar] MuseTalk 推理不可用 ({type(e).__name__}: {e})")
            print("[Avatar] 回退: 用 ffmpeg 生成静态图片+音频视频")
            fallback_video(audio_path, avatar_image, output_video)
    elif avatar_type == "none":
        return audio_path
    else:
        raise ValueError(f"不支持的 Avatar: {avatar_type}")

    return output_video


def fallback_video(audio_path: str, avatar_image: str, output_path: str) -> None:
    """当 MuseTalk 不可用时，用静态图片 + 音频生成视频"""
    img_candidates = []
    if os.path.isdir(avatar_image):
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            import glob
            img_candidates.extend(glob.glob(os.path.join(avatar_image, ext)))
    elif os.path.isfile(avatar_image):
        img_candidates = [avatar_image]

    if not img_candidates:
        print("[Avatar] 没有找到头像图片，跳过视频生成")
        shutil.copy(audio_path, output_path.replace(".mp4", ".wav"))
        return

    img = img_candidates[0]
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-loop", "1", "-i", img,
            "-i", audio_path,
            "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-pix_fmt", "yuv420p",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    print(f"[Avatar] Fallback 视频: {output_path}")


async def main():
    parser = argparse.ArgumentParser(description="从 Markdown 讲稿生成数字人视频")
    parser.add_argument("--script", required=True, help="Markdown 讲稿路径")
    parser.add_argument("--tts", default="edgetts", choices=["edgetts", "cosyvoice"])
    parser.add_argument("--voice", default="zh-CN-YunxiNeural", help="EdgeTTS 音色")
    parser.add_argument("--voice_prompt", default=None, help="CosyVoice 参考音频 (.wav)")
    parser.add_argument("--avatar", default="none", choices=["musetalk", "none"])
    parser.add_argument("--avatar_image", default="data/avatars/avator_1", help="头像图片/目录")
    parser.add_argument("--output", default="examples/record_tech_video/output.mp4")
    parser.add_argument("--subtitle", default=True, type=lambda x: x.lower() != "false")
    args = parser.parse_args()

    print("=" * 60)
    print("mere-fusion: Markdown → 数字人技术视频")
    print("=" * 60)

    paragraphs = parse_markdown(args.script)
    print(f"[Parse] 讲稿段落数: {len(paragraphs)}")
    for i, p in enumerate(paragraphs):
        print(f"  [{i}] {p['text'][:60]}...")

    with tempfile.TemporaryDirectory(prefix="mere_video_") as tmpdir:
        # TTS: 每段生成语音
        print(f"\n[TTS] 开始合成 ({args.tts})...")
        t0 = time.time()
        wav_files = []
        for i, para in enumerate(paragraphs):
            wav = await tts_segment(
                para["text"], i, args.tts, tmpdir,
                voice=args.voice, voice_prompt=args.voice_prompt,
            )
            wav_files.append(wav)
            print(f"  [{i}] {os.path.basename(wav)} done")
        print(f"[TTS] 完成, 耗时 {time.time() - t0:.1f}s")

        # 拼接音频
        full_audio = os.path.join(tmpdir, "full_audio.wav")
        concat_audio(wav_files, full_audio)
        print(f"[Audio] 合并完成: {full_audio}")

        # Avatar
        if args.avatar != "none":
            raw_video = os.path.join(tmpdir, "raw_video.mp4")
            print(f"\n[Avatar] 生成数字人视频 ({args.avatar})...")
            t0 = time.time()
            await run_avatar(full_audio, args.avatar, args.avatar_image, raw_video)
            print(f"[Avatar] 完成, 耗时 {time.time() - t0:.1f}s")
        else:
            raw_video = None

        # 字幕
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        if raw_video and args.subtitle:
            srt_path = os.path.join(tmpdir, "subtitles.srt")
            generate_srt(paragraphs, wav_files, srt_path)
            print(f"[Subtitle] SRT 生成完成")
            burn_subtitles(raw_video, srt_path, args.output)
            print(f"[Subtitle] 字幕烧录完成")
        elif raw_video:
            shutil.copy(raw_video, args.output)
        else:
            shutil.copy(full_audio, args.output.replace(".mp4", ".wav"))

    print(f"\n{'=' * 60}")
    print(f"输出: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
