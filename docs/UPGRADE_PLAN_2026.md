# 2026 技术栈升级计划 (UPGRADE_PLAN_2026)

> 本文是 mere-fusion 从 2024 技术栈升级到 2026 实时数字人 SOTA 的设计文档。
> 调研日期：2026-05。先调研不动代码，等评审通过后分阶段执行。

---

## 一、设计目标

把 mere-fusion 从「能跑通的玩具」升级到「面试拿得出手 + 自己能录技术视频」的双轨系统：

1. **实时通话轨**：保留 WebRTC + SRS 架构，端到端 P95 < 800ms，30+ FPS 流式
2. **离线录视频轨**：从 markdown 一键生成「半身+手势+对口型」的技术分享视频 mp4

两条轨道延迟需求差 100 倍，模型选型必须分开。

---

## 二、最终选型（先看结论）

| 模块 | 当前实现 | 升级目标 | 备选 |
|---|---|---|---|
| ASR (中文开源) | Whisper / FasterWhisper | **FunASR Paraformer-large 流式** | SenseVoice |
| ASR (闭源高端) | OpenAI Whisper API | **豆包 Seed-ASR (WebSocket)** | 阿里 NLS |
| LLM (主) | ChatGPT / Qwen 1.8B / Gemini | **DeepSeek-V4 Flash 非思考模式** | Qwen3-Plus / GLM-4-Plus |
| LLM (低延迟备选) | - | **Qwen3-Flash** (TTFT 竞速档) | - |
| TTS (开源) | EdgeTTS / GPT-SoVITS / XTTS | **CosyVoice 3 开源版（流式+零样本）** | FishSpeech / SparkTTS |
| TTS (闭源高端) | - | **豆包 SendTTS V3 (WebSocket)** | MiniMax speech-2.8 |
| 整体实时方案 | 拼接式 ASR+LLM+TTS | **(可选) 豆包 Realtime 端到端语音** | - |
| 视觉理解 | YOLO + DeepFace + EasyOCR | **Qwen3.6-Plus 多模态** (1Hz 截帧, 说完话图文一起发) | 通义 VL / GPT-5 Vision |
| **数字人 (实时流)** | ErNeRF / MuseTalk / Wav2Lip | **SoulX-FlashHead Lite** (Soul AI Lab 2026-02, 96fps) | MuseTalk 1.5 / Ditto |
| **数字人 (录视频)** | 无 | **EchoMimicV3** (Ant AAAI 2026, 半身+手势, 1.3B) | Hallo3 (画质最高) / OmniAvatar (全身) |
| 打断/降噪链路 | 单层 noisereduce + webrtcvad | **AdaptiveEnergyGate + 1.2s 合并窗 + 三层打断保护** | - |
| 部署 | 手写 SRS docker run + pip | **docker-compose 一行起 + .env 模板** | k8s helm chart (未来) |

**淘汰清单**：ErNeRF、Wav2Lip、Rasa、intent/、clean/、other/、XTTS、GPT-SoVITS。

---

## 三、数字人方案选型详细对比

### 候选与关键参数（2026-05 更新）

| 模型 | 类型 | 输出 | 速度 | VRAM | 适合场景 |
|---|---|---|---|---|---|
| **SoulX-FlashHead Lite** | 流式扩散 | 头部全表情 | **96 FPS** (RTX 4090) | 6.4 GB | ⭐ **实时通话最佳** |
| **SoulX-FlashHead Pro** | 流式扩散 | 头部高画质 | 10.8 FPS (4090) / 25+ (2x5090) | 12-20 GB | 实时高质量挡 |
| **EchoMimicV3 (Flash)** | 统一多模态扩散 | 半身+手势+表情 | 比 v2 大幅提升 | 16 GB | ⭐ **录视频最佳** |
| **EchoMimic v2 (加速版)** | Diffusion + Audio-Pose | 半身+手势 | ~2.4 FPS (A100) | 16 GB | 录视频备选 |
| **Ditto** | 运动空间扩散 | 头部+表情 | 实时 | - | 实时备选 |
| **Hallo3** | DiT | 头部高保真 | <1 FPS / 8xH100 | 很高 | 画质最高（离线） |
| **MuseTalk 1.5** | 潜空间 inpainting | 唇形 (256×256) | 30+ FPS (V100) | 4-8 GB | 低端 GPU 兜底 |
| **LatentSync 1.5** | SD + 时序层 | 唇形 | ~4 FPS | 7.8 GB | 唇形同步最高 (LSE-C 7.90) |
| **OmniAvatar** | 全身扩散 | 全身+物体交互 | 离线 | - | 创意型全身 |
| ~~ErNeRF~~ | NeRF | 头部 3D | - | - | **淘汰** |
| ~~Wav2Lip~~ | GAN (2020) | 唇形 | - | - | **淘汰** |
| **Sonic** | Motion bucket | 头部 + 头动 | ~1 FPS | ~12 GB | 表情备选 |
| ~~ErNeRF~~ | NeRF | 头部 3D | 实时但效果差 | - | **淘汰** |
| ~~Wav2Lip~~ | GAN (2020) | 唇形 (低质量) | 实时但低保真 | - | **淘汰** |

### 最终决策

**实时通话 → SoulX-FlashHead Lite（最佳）**

- **96 FPS** 在单张 RTX 4090 上，VRAM 仅 6.4 GB — 性能碾压所有竞品
- 原生流式架构，无限长度生成，零身份漂移（不是逐帧独立推理，有时序建模）
- HDTF 上 FID 9.97（流式）/ Sync-C 5.53，综合指标 2026 实时类 SOTA
- Soul AI Lab（国内团队，2026-02 开源），中文/亚洲面孔支持好
- 比 MuseTalk 快 3x，比 Ditto 快 2x，比 Hallo3 快 600x
- GitHub: <https://github.com/Soul-AILab/SoulX-FlashHead>

备选：MuseTalk 1.5（低端 GPU 兜底，4GB VRAM 即可跑，社区最成熟 5.4k stars）

**录技术视频 → EchoMimicV3（最佳）**

- **AAAI 2026** 收录，仅 1.3B 参数，蚂蚁集团最新一代
- 统一多模态多任务架构：音频驱动 + 半身 + 手势 + 表情 — 一个模型搞定
- Flash 加速版（2026-01）大幅提升推理速度
- 在 EchoMimicV2（CVPR 2025）基础上全面升级
- 支持 A100 / RTX 4090 / V100
- GitHub: <https://github.com/antgroup/echomimic_v3>

备选：Hallo3（画质指标最高 FID 20.36，但只做头部 + 需要 8xH100）

**淘汰理由**

- **ErNeRF / Wav2Lip**：2024 前的模型，已被 Diffusion 全面超越
- **MuseTalk 1.5 不再是实时首选**：SoulX-FlashHead 在速度和表情丰富度上全面碾压
- **EchoMimicV2 不再是录视频首选**：V3 是 V2 的全面升级，参数量更小效果更好

---

## 四、打断/降噪链路改造（参考实时语音对话生产实践）

mere-fusion 当前的对话流是「单向播放 → 等用户说完 → 处理 → 回复」，**没有打断 + 没有合并窗 + 没有版本控制**。生产可用建议实现以下 5 个机制：

### 4.1 双层降噪（AdaptiveEnergyGate）

```
PCM 20ms 帧 (16kHz)
    ↓
┌─ AdaptiveEnergyGate ──────────────────────────┐
│   状态机 IDLE ↔ SPEAKING                       │
│   双门限滞回 (high_mul=3.0, low_mul=1.5)       │
│   HFER 高频/低频能量比二次确认 (阈值 0.05)     │
│   AI 播放期冻结 noise floor                    │
│   句首 lookback 80ms (4 帧)                    │
└──────────────────────────────────────────────┘
    ↓
ASR 引擎 (FunASR / 豆包)
```

**核心参数**（沿用生产值）：

| 参数 | 默认 | 作用 |
|---|---|---|
| `high_multiplier` | 3.0 | IDLE→SPEAKING 触发倍数 |
| `low_multiplier` | 1.5 | SPEAKING→IDLE 触发倍数 |
| `exit_debounce_frames` | 5 (~100ms) | 退出去抖 |
| `lookback_frames` | 4 (~80ms) | 句首保护缓冲 |
| `floor_init` | 100.0 | 噪声基线初值 |
| `ema_alpha` | 0.05 | 基线 EMA 学习率 |
| `hfer_threshold` | 0.05 | HFER 高频能量比阈值 |

**HFER 原理**：每帧 256 点 FFT @ 16kHz，计算 2000-4000Hz / 200-2000Hz 能量比，EMA 平滑后低于阈值判为远场弱混响 / 电流噪声，拒绝进入 SPEAKING。这是 2026 真实生产环境（远场拾音、网络抖动）必须的二次过滤。

**移植成本**：Python 实现约 200 行，无外部依赖（只用 numpy + scipy.fft）。

### 4.2 ASR final 条件合并窗

```
final 到达 → 判断 (now - lastPartialTs)
  ≥ 500ms → "客户已停说话" → 立即 fire question
  < 500ms → "客户仍在说话" → 排 1.2s 合并窗
              ↓ 期间若来新 partial → cancel 重排
              ↓ 1.2s 到期未被新 final 覆盖 → fire
```

**意图**：避免 VAD 切碎完整问题导致 AI 抢话。这是从 mere-fusion 现状（直接拿到 final 就 fire）到生产可用的关键改造。

### 4.3 三层打断保护

```
用户说话 → AI 是否在播放?
  否 → 直接走 LLM
  是 → 检查 3 层保护：
    1. 开场白保护期 (Redis interrupt:opening:{uuid}, TTL = textLen × 200ms)
    2. AI 句首 2s 保护期 (Redis interrupt:sentence:{uuid}, TTL 2s)
    3. LLM 二分类 (DeepSeek-V4 Flash 非思考模式, 300ms 硬超时)
       → 输入 fullContent (累加 ASR 全部 final)
       → 1 = 真打断（明确叫停/实质问题/明确否定）
       → 0 = 不打断（附和词/噪声/半句话）
       → 超时默认 0 (保守策略)

    fallback: 编辑距离匹配语气词库 (modal_particles.txt)
```

**关键设计**：判 fullContent 而非 newContent，避免碎片化 final 被逐个判 0 吃掉。

### 4.4 版本号控制

```
每个 question 进来 → Redis uu_question_v:{uuid}++
每个 LLM/TTS slice 消费前 → 读最新 version
  if (latest > 当前 req.v) → versionExpired = true → 跳过

效果：客户打断后新 question 进来，旧 LLM/TTS 推送链路自动停止
```

### 4.5 端到端关键耗时指标（SLO）

| 指标 | 含义 | 目标 |
|---|---|---|
| `agent_ms` | 端到端首响 (ASR final → TTS 首包到耳朵) | < 1500ms |
| `tts_first_frame_ms` | TTS API 首帧延迟 | < 800ms |
| `firstQuestionTime` | ASR final → LLM 首句完成 | < 2000ms |
| `llm_ttft_ms` | LLM 首 token 延迟 | < 600ms |
| `racing_threshold` | 竞速备选模型切换阈值 | 600ms |

**Python 实现路径**：`service/dialogue/` 独立模块，约 600-800 行。

---

## 五、ROI 表（每项升级单独决策）

| # | 升级项 | 收益 | 工作量 | 兼容性 | 优先级 |
|---|---|---|---|---|---|
| 1 | `.env` → `.env.example` + 删硬编码 IP | 安全 critical → 安全 | 0.2 天 | ✅ | **🔴 P0** |
| 2 | README 大重写 + 架构图 + quickstart | 项目"门面"，0→1 | 0.5 天 | ✅ | **🔴 P0** |
| 3 | docker-compose 一行起 SRS | 部署 1 天 → 1 分钟 | 0.5 天 | ✅ | **🔴 P0** |
| 4 | 录技术视频 pipeline (现有 MuseTalk + CosyVoice) | 立刻能录视频 | 1 天 | ✅ 增量 | **🔴 P0** |
| 5 | FunASR 替换 Whisper | 中文 CER 30% → 5%，延迟 ↓ | 1 天 | ⚠️ 配置切换 | 🟠 P1 |
| 6 | CosyVoice 3 替换现版本 | 流式 TTFT ↓ 50%，零样本质量大幅提升 | 1 天 | ⚠️ API 变更 | 🟠 P1 |
| 7 | DeepSeek-V4 Flash 替换 LLM 集合 | 价格 ↓ 70%, TTFT ↓ 40% | 0.5 天 | ✅ 增量 | 🟠 P1 |
| 8 | 移植 AdaptiveEnergyGate + 合并窗 + 版本号 | 玩具 → 生产级对话 | 3 天 | ⚠️ 重构 process_audio | 🟠 P1 |
| 9 | Qwen3.6-Plus 多模态替换 YOLO+DeepFace+OCR | 3 个模型 → 1 个 API, 代码 ↓ 70% | 1 天 | ⚠️ 移除 3 个模块 | 🟡 P2 |
| 10 | SoulX-FlashHead Lite 替换 MuseTalk | 实时 96fps / 6.4GB VRAM / 全表情 | 2 天 | ✅ 替换渲染模块 | 🟠 P1 |
| 11 | EchoMimicV3 接入录视频 pipeline | 半身+手势+1.3B 轻量 / AAAI 2026 | 2 天 | ✅ 新增模块 | 🟠 P1 |
| 12 | MuseTalk 1.5 作为低端 GPU 兜底 | 4GB VRAM 即可跑 | 1 天 | ✅ 权重升级 | 🟡 P2 |
| 13 | 豆包 Seed-ASR + SendTTS V3 接入 (闭源高端档) | 中文场景质量天花板 | 2 天 | ✅ 增量 | 🟢 P3 |
| 14 | 豆包 Realtime 端到端语音方案 | 端到端 < 500ms | 3 天 | ❌ 大改 | ⚪ P4 (探索) |

**推荐执行顺序**：P0 全做 → P1 按需 → P2 一周内 → P3 视目标决定 → P4 单独立项

---

## 六、淘汰 / 重组清单

| 当前路径 | 处理 | 理由 |
|---|---|---|
| `ernerf/` | 升级后移到 `legacy/ernerf/` | 训练成本高、效果不如 Diffusion |
| `wav2lip/` | 升级后移到 `legacy/wav2lip/` | 2020 模型，已被新一代方案超越 |
| `test/` | 改为 `tests/` 用 pytest 重组 | 当前是 ad-hoc 运行脚本 |
| `app.py` 单文件 | 拆 `api/` `service/` `worker/` | 单文件维护成本高 |
| `.env` | 提供 `.env.example` 模板 | 配置更清晰，避免误提交密钥 |
| 硬编码 IP | 替换为环境变量 `SRS_CANDIDATE` 等 | 安全 + 开箱即用 |

---

## 七、目标架构（升级后）

### 实时通话管道

```
浏览器 (麦克风 + 数字人视频)
    │  WebRTC
    ▼
SRS WebRTC 中转 (docker-compose 一行起)
    │
    ▼
mere-fusion Backend (Python)
    │
    ├─ AdaptiveEnergyGate (双层降噪 + HFER)
    │       │
    │       ▼
    ├─ ASR: FunASR / 豆包 Seed-ASR
    │   + 1.2s 条件合并窗
    │       │
    │       ▼
    ├─ 打断决策 (3 层保护 + 版本号)
    │       │
    │       ▼
    ├─ LLM: DeepSeek-V4 Flash 流式
    │   + Qwen3.6-Plus 多模态 (1Hz 截帧, 说完话图文合并)
    │       │
    │       ▼ 流式断句 (。！？)
    ├─ TTS: CosyVoice 3 / 豆包 SendTTS V3
    │       │
    │       ▼ wav 流
    └─ 数字人: MuseTalk 1.5 流式 (高质量挡可切 LatentSync 1.5)
            │
            ▼ RTC 推流
        SRS → 浏览器
```

### 离线录视频管道（独立）

```
tech_video.md (Markdown 讲稿)
    │
    ▼
CosyVoice 3 零样本克隆 (你自己音色的 .wav prompt)
    │
    ▼ wav
EchoMimic v2 加速版 (半身 + 手势)
    │
    ▼ mp4 (无字幕)
ffmpeg 合成 (讲稿字幕 SRT + BGM 可选)
    │
    ▼
tech_video.mp4
```

---

## 八、不在本计划范围内（明确不做）

- 数字人训练（自己采数据训自己的脸）—— 单独项目
- 多用户 session 维护 / 认证登录系统 —— 先单机能用
- 多语言（除中英外）—— 先把中英做好
- 移动端 SDK —— 先保证 Web 客户端能用
- 视频生成大模型（Sora / Kling）替代数字人 —— 延迟差 1000 倍，思路完全不同

---

## 九、相关参考

- SoulX-FlashHead: <https://github.com/Soul-AILab/SoulX-FlashHead>
- EchoMimicV3: <https://github.com/antgroup/echomimic_v3>
- EchoMimic v2: <https://github.com/antgroup/echomimic_v2>
- Ditto: <https://github.com/antgroup/ditto-talkinghead>
- MuseTalk 1.5: <https://github.com/TMElyralab/MuseTalk>
- LatentSync 1.5: <https://github.com/bytedance/LatentSync>
- Hallo3: <https://github.com/fudan-generative-vision/hallo3>
- OmniAvatar: <https://github.com/Omni-Avatar/OmniAvatar>
- FunASR: <https://github.com/modelscope/FunASR>
- CosyVoice 3: <https://github.com/FunAudioLLM/CosyVoice>
- DeepSeek API: <https://api-docs.deepseek.com/>
- 豆包大模型: <https://www.volcengine.com/product/doubao>
