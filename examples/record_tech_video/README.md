# 录技术视频 Pipeline

从 Markdown 讲稿生成数字人技术分享视频。

## 流程

```
tech_script.md  (你的讲稿)
      │
      ▼ 按段落/标题拆分
paragraphs[]
      │
      ▼ 每段 → TTS
EdgeTTS / CosyVoice  →  segment_*.wav
      │
      ▼ 合并音频
full_audio.wav
      │
      ▼ 音频驱动数字人 (可选)
MuseTalk  ──失败/未配置──►  静态图片 + 音频 (ffmpeg 降级)
      │
      ▼ 生成 SRT 字幕并烧录
ffmpeg  →  output.mp4
```

> **关于数字人渲染**：`--avatar musetalk` 需要本地配置好 MuseTalk（模型权重 + mmlab 依赖 + GPU）。
> 若推理不可用（缺依赖/缺权重/无 GPU），管道会**自动降级**为「静态头像图 + 音频」的视频，
> 不会中断。先用 `--avatar none` 跑通 TTS，再逐步接入真正的数字人渲染。

## 使用

### 1. 准备讲稿

编辑 `tech_script.md`，示例已提供。每个段落会被拆分为一段连续语音。

### 2. 生成视频

```bash
cd mere-fusion

# 方式 A: EdgeTTS (免费，不需要本地 TTS 服务)
python examples/record_tech_video/generate_video.py \
  --script examples/record_tech_video/tech_script.md \
  --tts edgetts \
  --avatar musetalk \
  --output examples/record_tech_video/output.mp4

# 方式 B: CosyVoice 3 零样本克隆 (需要先起 CosyVoice server)
python examples/record_tech_video/generate_video.py \
  --script examples/record_tech_video/tech_script.md \
  --tts cosyvoice \
  --voice_prompt examples/record_tech_video/my_voice.wav \
  --avatar musetalk \
  --output examples/record_tech_video/output.mp4
```

### 3. 参数说明

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--script` | Markdown 讲稿路径 | (必填) |
| `--tts` | TTS 引擎: edgetts / cosyvoice | edgetts |
| `--voice_prompt` | 零样本克隆的参考音频 (CosyVoice) | None |
| `--avatar` | 数字人: musetalk / none | none (仅生成音频) |
| `--avatar_image` | 数字人参考图片 | data/avatars/avator_1 |
| `--output` | 输出视频路径 | output.mp4 |
| `--subtitle` | 是否加字幕 | true |
| `--voice` | EdgeTTS 音色 | zh-CN-YunxiNeural |

### 4. 自定义你的形象

1. 拍一张正面免冠照 (最好 512×512 以上)
2. 放到 `data/avatars/my_avatar/`
3. 使用 `--avatar_image data/avatars/my_avatar`

### 5. 发布

生成的 `output.mp4` 可以直接上传到:
- B 站
- 小红书
- 抖音
- YouTube

建议用剪映/CapCut 做最后的 BGM + 片头片尾。
