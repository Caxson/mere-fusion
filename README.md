# mere-fusion

> Real-time AI digital human that holds sub-second voice conversations.
> Full pipeline: microphone → ASR → LLM → TTS → lip-synced avatar → WebRTC video stream.
> Put your own face on AI.

[English](#what-you-can-do-with-it) | [中文](#你能用它做什么)

---

## What You Can Do With It

| Use Case | Mode | Pipeline |
|---|---|---|
| **Live digital customer service** | Real-time | WebRTC → ASR → LLM → TTS → avatar → stream |
| **Virtual anchor / streamer** | Real-time | Same as above, relay to OBS / Bilibili |
| **AI voice assistant with a face** | Real-time | Browser → mere-fusion → browser |
| **Language learning partner** | Real-time | Speak → ASR → LLM coach → TTS → avatar |
| **Record tutorial / demo videos** | Offline | Text → TTS → avatar → mp4 |

---

## Architecture

```
Browser (mic + camera)
    │  WebRTC
    ▼
SRS WebRTC Relay  (docker, one container)
    │
    ▼
┌─ mere-fusion backend (Python) ──────────────────────┐
│                                                       │
│  Audio in ──► ASR    (Whisper / FasterWhisper)        │
│                 │                                     │
│                 ▼                                     │
│              LLM     (ChatGPT / Qwen / Gemini)        │
│                 │                                     │
│                 ▼                                     │
│              TTS     (EdgeTTS / CosyVoice / XTTS)     │
│                 │                                     │
│                 ▼                                     │
│            Avatar    (MuseTalk / ErNeRF / Wav2Lip)    │
│                 │                                     │
└─────────────────┼─────────────────────────────────────┘
                  ▼
            SRS ──► Browser (video + audio)
```

The backend pulls the client's audio/video from SRS, runs the ASR → LLM → TTS → avatar
pipeline, and pushes the generated talking-head stream back to SRS for the browser to play.

---

## Quick Start

### Prerequisites

- **Python 3.10**
- **GPU**: NVIDIA + CUDA recommended for avatar rendering
- **Docker**: for the SRS relay server
- macOS: `brew install portaudio ffmpeg`
- Linux: `sudo apt install libportaudio2 portaudio19-dev ffmpeg`

### 1. Setup

```bash
git clone https://github.com/Caxson/mere-fusion.git
cd mere-fusion

python3.10 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env   # then add your OPENAI_API_KEY
```

MuseTalk also needs mmlab packages:

```bash
pip install --no-cache-dir -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```

### 2. Start SRS (one command)

```bash
docker-compose up -d
# Verify: curl http://localhost:1985/api/v1/streams/
```

Or set a public candidate IP for remote access:

```bash
SRS_CANDIDATE=your.server.ip docker-compose up -d
```

### 3. Run the backend

```bash
python app.py --model musetalk --tts edgetts
```

### 4. Frontend

Vue.js client: [Caxson/mere_web_client](https://github.com/Caxson/mere_web_client)

---

## Modules

### ASR (Speech-to-Text)

| Engine | Notes |
|---|---|
| Whisper / WhisperTimestamped | OpenAI Whisper with timestamps |
| FasterWhisper | Optimized Whisper, faster transcription |
| OpenAI API ASR | Cloud speech-to-text |
| InsanelyFastWhisper | HuggingFace `transformers` pipeline |

Modes: offline / chunked / online (real-time), with optional VAD.

### LLM

| Engine | Notes |
|---|---|
| ChatGPT | OpenAI API |
| Qwen | Open-source, local or cloud |
| Gemini | Google multimodal |

### TTS (Text-to-Speech)

| Engine | Notes |
|---|---|
| EdgeTTS | Microsoft Azure voices, free, no GPU |
| CosyVoice | Zero-shot voice cloning from reference audio |
| GPT-SoVITS | Reference-timbre synthesis |
| XTTS | Voice cloning from a reference clip |

### Avatar (Digital Human)

| Engine | Notes |
|---|---|
| MuseTalk | Real-time lip-sync via latent inpainting |
| ErNeRF | NeRF-based 3D talking head |
| Wav2Lip | Classic lip-sync |

### Vision (optional)

YOLO object detection + DeepFace (age/gender/emotion) + EasyOCR text recognition
on the incoming video frames.

### Dialogue Quality (real-time, implemented)

Production-grade conversation pipeline in `service/dialogue/` that turns the basic
"wait until the user stops → reply" loop into an interruptible, noise-robust one.
Adapted from a real production telephony stack; **17 unit tests passing**.

| Mechanism | Module |
|---|---|
| Adaptive energy gate — dual-threshold hysteresis + HFER spectral confirmation + floor-freeze during playback | `energy_gate.py` |
| Conditional 1.2s sentence merge window — avoids cutting in on VAD-fragmented speech | `merge_window.py` |
| Three-layer barge-in guard — opening / sentence-head / LLM intent (+ edit-distance fallback) | `interrupt.py` |
| Version guard — a barge-in retires stale LLM/TTS streams | `version_guard.py` |

Wiring guide: [`docs/INTEGRATION.md`](docs/INTEGRATION.md).

---

## Project Structure

```
mere-fusion/
├── app.py                    # Main aiohttp + WebRTC server
├── docker-compose.yml        # One-line SRS relay
├── .env.example              # Environment variable template
│
├── whisper_online.py         # ASR engines
├── stream_openai_video.py    # LLM streaming + session management
├── ttsreal.py                # TTS engines
│
├── musereal.py / museasr.py  # MuseTalk avatar
├── nerfreal.py / nerfasr.py  # ErNeRF avatar
├── lipreal.py  / lipasr.py   # Wav2Lip avatar
├── webrtc.py                 # WebRTC HumanPlayer track
├── yolo_opencv.py            # Vision processing
│
├── llm/                      # LLM adapters (ChatGPT/Qwen/Gemini/DeepSeek)
├── service/dialogue/         # Noise gate, merge window, barge-in, version guard
├── avatar/ vision/ doubao/   # 2026 engines (opt-in): SoulX/EchoMimicV3, Qwen-VL, Doubao
├── musetalk/ ernerf/ wav2lip/# Avatar model code + weights
├── examples/record_tech_video/ # Markdown → mp4 pipeline
├── tests/                    # Unit tests (dialogue 17, doubao protocol 8)
├── data/                     # Avatar assets, pretrained weights
└── docs/                     # Architecture, roadmap, upgrade, integration
```

---

## Roadmap

The 2026 modernization plan (streaming avatars, faster ASR, lower end-to-end latency)
is tracked in [`docs/UPGRADE_PLAN_2026.md`](docs/UPGRADE_PLAN_2026.md) and
[`docs/ROADMAP.md`](docs/ROADMAP.md). New engines under evaluation (opt-in, see
[`docs/INTEGRATION.md`](docs/INTEGRATION.md)):

- **Avatar**: SoulX-FlashHead (streaming, 96 fps) for real-time, EchoMimicV3 (half-body + gestures) for offline video
- **ASR**: FunASR for low-latency Chinese
- **LLM**: DeepSeek-V4 Flash for fast TTFT
- **TTS**: CosyVoice 3 streaming + zero-shot clone

> The real-time dialogue quality pipeline (noise gate / merge window / barge-in /
> version guard) is **already implemented** — see Modules above.

---

## License

[MIT](LICENSE)

---

## 你能用它做什么

| 场景 | 模式 | 链路 |
|---|---|---|
| **数字人客服** | 实时 | WebRTC → ASR → LLM → TTS → 数字人 → 推流 |
| **虚拟主播** | 实时 | 同上，转推 OBS / B 站 |
| **带脸的 AI 语音助手** | 实时 | 浏览器 → mere-fusion → 浏览器 |
| **语言学习陪练** | 实时 | 说话 → ASR → LLM 教练 → TTS → 数字人 |
| **录制教程 / 演示视频** | 离线 | 文本 → TTS → 数字人 → mp4 |

---

## 架构

```
浏览器 (麦克风 + 摄像头)
    │  WebRTC
    ▼
SRS WebRTC 中转  (docker, 单容器)
    │
    ▼
┌─ mere-fusion 后端 (Python) ─────────────────────────┐
│                                                       │
│  音频 ──► ASR    (Whisper / FasterWhisper)            │
│             │                                         │
│             ▼                                         │
│          LLM     (ChatGPT / Qwen / Gemini)            │
│             │                                         │
│             ▼                                         │
│          TTS     (EdgeTTS / CosyVoice / XTTS)         │
│             │                                         │
│             ▼                                         │
│        数字人    (MuseTalk / ErNeRF / Wav2Lip)        │
│             │                                         │
└─────────────┼─────────────────────────────────────────┘
              ▼
        SRS ──► 浏览器 (视频 + 音频)
```

后端从 SRS 拉取客户端音视频流，跑完 ASR → LLM → TTS → 数字人 链路，
再把生成的数字人画面推回 SRS，由浏览器播放。

---

## 快速开始

### 系统要求

- Python 3.10
- GPU: 推荐 NVIDIA + CUDA（数字人渲染）
- Docker（SRS 中转）
- macOS: `brew install portaudio ffmpeg`
- Linux: `sudo apt install libportaudio2 portaudio19-dev ffmpeg`

### 1. 安装

```bash
git clone https://github.com/Caxson/mere-fusion.git
cd mere-fusion

python3.10 -m venv myenv
source myenv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env   # 然后填入你的 OPENAI_API_KEY
```

MuseTalk 还需要 mmlab 系列包：

```bash
pip install --no-cache-dir -U openmim
mim install mmengine "mmcv>=2.0.1" "mmdet>=3.1.0" "mmpose>=1.1.0"
```

### 2. 起 SRS（一行）

```bash
docker-compose up -d
# 验证: curl http://localhost:1985/api/v1/streams/
```

跨机器访问时指定公网 candidate IP：

```bash
SRS_CANDIDATE=你的服务器IP docker-compose up -d
```

### 3. 跑后端

```bash
python app.py --model musetalk --tts edgetts
```

### 4. 前端

Vue.js 客户端: [Caxson/mere_web_client](https://github.com/Caxson/mere_web_client)

---

## 模块

### ASR（语音识别）

| 引擎 | 说明 |
|---|---|
| Whisper / WhisperTimestamped | 带时间戳的 OpenAI Whisper |
| FasterWhisper | 优化版 Whisper，转录更快 |
| OpenAI API ASR | 云端语音转文字 |
| InsanelyFastWhisper | HuggingFace `transformers` 管道 |

模式：offline / 分块 / online（实时），可选 VAD。

### LLM

| 引擎 | 说明 |
|---|---|
| ChatGPT | OpenAI API |
| Qwen | 开源，本地或云端 |
| Gemini | Google 多模态 |

### TTS（语音合成）

| 引擎 | 说明 |
|---|---|
| EdgeTTS | 微软 Azure 音色，免费，不需要 GPU |
| CosyVoice | 零样本音色克隆 |
| GPT-SoVITS | 参考音色合成 |
| XTTS | 参考片段音色克隆 |

### 数字人

| 引擎 | 说明 |
|---|---|
| MuseTalk | 潜空间修复实时唇形同步 |
| ErNeRF | 基于 NeRF 的 3D 数字人 |
| Wav2Lip | 经典唇形同步 |

### 视觉（可选）

对输入视频帧做 YOLO 物体检测 + DeepFace（年龄/性别/情绪）+ EasyOCR 文字识别。

### 对话质量链路（实时，已实现）

`service/dialogue/` 把「等用户说完 → 回复」的玩具式对话，升级成可打断、抗噪的
生产级链路。移植自真实生产电话语音系统，**17 个单元测试全过**。

| 机制 | 模块 |
|---|---|
| 自适应能量门 — 双门限滞回 + HFER 高频能量比二次确认 + 播放期冻结基线 | `energy_gate.py` |
| 1.2s 条件合并窗 — 避免 VAD 切碎完整问题导致 AI 抢话 | `merge_window.py` |
| 三层打断保护 — 开场白 / 句首 / LLM 意图判定（+ 编辑距离 fallback） | `interrupt.py` |
| 版本号控制 — 打断后旧 LLM/TTS 流自动失效 | `version_guard.py` |

接入方法见 [`docs/INTEGRATION.md`](docs/INTEGRATION.md)。

---

## 路线图

2026 现代化计划（流式数字人、更快 ASR、更低端到端延迟）见
[`docs/UPGRADE_PLAN_2026.md`](docs/UPGRADE_PLAN_2026.md) 和
[`docs/ROADMAP.md`](docs/ROADMAP.md)。评估中的新引擎（可选接入，见
[`docs/INTEGRATION.md`](docs/INTEGRATION.md)）：

- **数字人**：实时用 SoulX-FlashHead（流式 96fps），离线录视频用 EchoMimicV3（半身+手势）
- **ASR**：FunASR 低延迟中文
- **LLM**：DeepSeek-V4 Flash 快速首响
- **TTS**：CosyVoice 3 流式 + 零样本克隆

> 实时对话质量链路（降噪门 / 合并窗 / 打断 / 版本号）**已实现** —— 见上方「模块」。

---

## 许可证

[MIT](LICENSE)
