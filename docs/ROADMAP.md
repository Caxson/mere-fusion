# Roadmap

## 当前状态 (v0.x)

- [x] 实时通话管道: WebRTC + SRS + Whisper + ChatGPT/Qwen/Gemini + EdgeTTS/CosyVoice + MuseTalk/ErNeRF/Wav2Lip
- [x] docker-compose 一行起 SRS
- [x] .env 模板化配置
- [x] README 重写
- [ ] 录技术视频 pipeline (Markdown → mp4)

## Phase 1: 基础升级 (P0)

- [ ] FunASR 替换 Whisper (中文 CER 5%, RTF 0.05)
- [ ] DeepSeek-V4 Flash 作为默认 LLM
- [ ] CosyVoice 3 流式 + 零样本克隆
- [ ] SoulX-FlashHead Lite 替换 MuseTalk (实时 96fps)
- [ ] EchoMimicV3 接入录视频 pipeline (半身+手势, AAAI 2026)

## Phase 2: 对话质量 (P1)

- [ ] 移植 AdaptiveEnergyGate 降噪 (HFER + 双门限)
- [ ] 实现 1.2s 条件合并窗
- [ ] 三层打断保护 (开场白/句首/LLM 二分类)
- [ ] 版本号控制 (旧 LLM/TTS 流自动停止)
- [ ] 豆包 Seed-ASR + SendTTS V3 闭源高端档

## Phase 3: 视觉 + 兜底 (P2)

- [ ] Qwen3.6-Plus 多模态替换 YOLO + DeepFace + EasyOCR
- [ ] MuseTalk 1.5 作为低端 GPU 兜底方案

## Phase 4: 探索 (P3+)

- [ ] 豆包 Realtime 端到端语音方案
- [ ] 多用户 session + 认证
- [ ] 前端 Vue 客户端升级
- [ ] CI/CD + 自动化测试

## 不做的事

- 数字人训练 (自己脸) — 单独项目
- 移动端 SDK
- 视频生成大模型 (Sora/Kling) 方向
- 多语言 (除中英外)
