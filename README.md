# mere-fusion
An AI digital person real-time streaming video voice call project, including picture and voice input and picture voice output.

### 架构介绍

**客户端（Vue.js）**：通过 WebRTC 推送视频和音频流到 SRS。

**SRS**：作为 WebRTC 中转服务器，转发流到后端（Python）。

**后端（Python）**：

- **拉流**：从 SRS 获取音视频流，进行处理（如 AI 分析）。
- **推流**：处理完成后，将音视频流重新推送到 SRS。

**客户端（Vue.js）**：拉取处理后的音视频流并展示。
