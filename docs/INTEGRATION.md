# 2026 升级模块接入指南

本次升级的新引擎都是**独立可选模块**，默认不改变现有 Whisper/ChatGPT/EdgeTTS/MuseTalk
行为。下面说明如何把它们接进主流程（`app.py` 重构定稿后照此接线即可）。

重依赖（funasr / cosyvoice / flash_head / echomimic / dashscope）均为**延迟导入**：
不安装对应环境时，`import` 这些包不会报错，只有真正构造对应类时才需要依赖就位。

---

## 1. 对话质量链路 `service/dialogue/`（纯 Python，已单测）

把「等用户说完→处理→回复」升级为可打断的生产级对话。

```python
from service.dialogue import (
    AdaptiveEnergyGate, GateConfig,
    SentenceMergeWindow, MergeConfig,
    InterruptDecider, InterruptConfig,
    VersionGuard,
)

# (a) 降噪门：在送 ASR 前过一道
gate = AdaptiveEnergyGate(GateConfig())
gate.set_playing(is_ai_playing)          # AI 播放期冻结噪声基线
for frame in audio_frames:               # 16k float32, 320 样本/帧
    r = gate.process(frame)
    if r.passed:
        for f in r.emit_frames:          # 含句首 lookback 补发
            asr.feed(f)

# (b) 合并窗：ASR 回调里用
async def on_fire(full_content, new_content):
    version = guard.bump(session_id)     # 新一轮，旧 LLM/TTS 失效
    await run_llm(full_content, version)

mw = SentenceMergeWindow(on_fire, MergeConfig())
mw.on_partial(partial_text)              # 中间识别
await mw.on_final(final_text)            # 最终识别（自动决定立即/排窗）

# (c) 打断决策：用户说话时
decider = InterruptDecider(classifier=my_llm_yes_no)  # 可选 LLM 二分类
decider.mark_opening(session_id, opening_text)         # 开场白保护
decider.mark_sentence_start(session_id)                # 每句 TTS 首帧调用
if decider.should_interrupt(session_id, full_content, is_playing):
    stop_current_tts()

# (d) 版本号：LLM/TTS 每个 slice 消费前检查
if guard.is_expired(session_id, my_version):
    break   # 已被新一轮打断，停止推送
```

---

## 2. ASR：FunASR 流式 `funasr_engine.py`

```python
from funasr_engine import FunASRStreaming

asr = FunASRStreaming(model="paraformer-zh-streaming", device="cuda")
# 实时：逐 chunk 喂（cache 自动跨 chunk 复用）
text = asr.feed_chunk(audio_chunk, is_final=False)
# 一段话结束
asr.reset()
```

接 `app.py`：加 `--asr_engine funasr` 分支，构造 `FunASRStreaming` 替代 Whisper 处理器。

---

## 3. LLM：DeepSeek-V4 `llm/DeepSeek.py`

```python
from llm.DeepSeek import DeepSeek

llm = DeepSeek(model_path="deepseek-v4-flash")   # 环境变量 DEEPSEEK_API_KEY
for delta in llm.chat_stream(user_text):         # 流式，逐句喂 TTS
    sentence_splitter.feed(delta)
```

接 `llm/LLM.py`：在 `init_model` 里加 `'DeepSeek'` 分支。

---

## 4. TTS：CosyVoice 2/3 `ttsreal.py::CosyVoice2TTS`

进程内流式零样本克隆。需 clone FunAudioLLM/CosyVoice 并设 PYTHONPATH
（含 `third_party/Matcha-TTS`），下载 CosyVoice2-0.5B / CosyVoice3 权重。

```python
# app.py 选择 TTS 时：
elif opt.tts == "cosyvoice2":
    from ttsreal import CosyVoice2TTS
    tts = CosyVoice2TTS(opt, nerfreal)
# 需要 opt.cosyvoice_model_dir / opt.REF_FILE / opt.REF_TEXT
```

---

## 5. 数字人 `avatar/`

```python
# 实时（最佳）：SoulX-FlashHead lite
from avatar import get_soulx
fh = get_soulx(ckpt_dir="models/SoulX-FlashHead-1_3B",
               wav2vec_dir="models/wav2vec2-base-960h", model_type="lite")
fh.set_reference("my_avatar.png")
for frames in fh.stream_generate(audio):   # 流式逐块出帧
    push_to_webrtc(frames)

# 离线录视频（最佳）：EchoMimicV3 半身+手势
from avatar import get_echomimic_v3
eng = get_echomimic_v3(repo_dir="/path/to/echomimic_v3")
eng.generate(image_path="me.jpg", audio_path="speech.wav", save_path="out/")

# 低端 GPU 兜底：MuseTalk 1.5 权重
from avatar import musetalk15
w = musetalk15.resolve_weights("models", version="1.5")  # 指向 1.5 权重
```

录视频 pipeline 已接 `--avatar echomimic`（设环境变量 `ECHOMIMIC_V3_DIR`）。

---

## 6. 视觉：Qwen-VL `vision/`

```python
from vision import FrameBuffer, QwenVision

fb = FrameBuffer(interval_sec=1.0)         # 1Hz 截帧
fb.offer(video_frame)                      # 视频回调里持续喂，自动节流

vl = QwenVision(model="qwen3.6-plus")      # 环境变量 DASHSCOPE_API_KEY
# 用户说完话时，图文一起发
answer = vl.understand(user_text, frame=fb.latest())
```

替代 `yolo_opencv.py` 的 YOLO+DeepFace+OCR 三件套。

---

## 7. 豆包语音（可选高端档）`doubao/`

火山引擎流式 ASR / TTS V3 / 端到端实时对话，三者共用二进制帧协议
（`doubao/protocol.py`，已单测）。鉴权用 appid + access_token。

```python
from doubao import DoubaoASR, DoubaoTTS, DoubaoRealtime

# ASR：喂 PCM 16k chunk，收 partial/final
asr = DoubaoASR(app_id, access_token)        # resource_id 可按套餐覆盖
async for res in asr.stream(pcm_chunk_aiter):
    print(res.get("text"))

# TTS V3：流式合成
tts = DoubaoTTS(app_id, access_token, speaker="zh_female_vv_uranus_bigtts")
async for audio in tts.synthesize("你好，我是数字人"):
    play(audio)

# 端到端实时对话（听+想+说一体，类 GPT-4o Realtime）
rt = DoubaoRealtime(app_id, access_token)
async for ev in rt.dialog(mic_pcm_aiter):
    if ev["type"] == "tts_audio": play(ev["data"])
    elif ev["type"] == "asr":     print("用户:", ev["data"])
    elif ev["type"] == "chat":    print("豆包:", ev["data"])
```

> Resource-Id 因套餐而异（ASR `volc.bigasr.sauc.duration`/`.concurrent`、
> TTS `volc.service_type.10029`/`seed-tts-2.0`），以火山控制台开通的为准，
> 构造时传参覆盖即可。

---

## 环境变量一览（见 `.env.example`）

| 变量 | 用途 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek-V4 LLM |
| `DASHSCOPE_API_KEY` | Qwen-VL 视觉 |
| `ECHOMIMIC_V3_DIR` | EchoMimicV3 仓库路径（录视频半身） |
| `DOUBAO_APPID` / `DOUBAO_ACCESS_TOKEN` | 豆包语音（ASR/TTS/Realtime） |
| `OPENAI_API_KEY` | 既有 ChatGPT 路径 |
