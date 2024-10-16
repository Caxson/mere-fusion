# mere-fusion
[English](#english-version) | [中文](#中文版本)

## English Version

An AI digital human real-time streaming video voice call project, including video voice stream input and large model processing and video voice stream output, with many large model open source projects built in.

### Introduction to the Architecture 
**Client (Vue.js)**: push video and audio streams to SRS via WebRTC. 
(**KMS**: Optional transcoding server that can be launched using Docker, with the KMS server receiving the stream and sending RTP streams to the backend gateway layer, which then sends the streams to the Python server.) 
**SRS**: As a WebRTC relay server, forward streams to the backend (Python). 

**Backend (Python)**: 

- **Streaming**: Receiving audio and video streams from the SRS and processing them.
- **Pushing**: After the processing is complete, pushing the audio and video streams back to the SRS. 

**Client (Vue.js)**:fetches and displays the processed audio and video streams. 

***

#### **Back-end Functions** 

##### Audio Stream Processing: 

- **Offline mode**: Processes entire audio files offline, suitable for testing and debugging. 
- **Computationally Unaware Pattern**: Divides the audio into segments and loads each block with `min_chunk_size` for processing pre-recorded long audio. 
- **Online mode**: simulates the processing of real-time audio streams, suitable for real-time transcription scenarios. 
  **VAD Support**: Voice activity detection can be enabled in any mode to optimize processing efficiency. 

**Optional audio processing tools:** 

+ **WhisperTimestamped**: A library based on. 
+ **FasterWhisper**: Optimized Whisper model for faster transcription. 
+ **OpenaiApiASR**: Uses OpenAI's cloud-based API service for speech-to-text transcription. 
+ **InsanelyFastWhisperASR**: An efficient Whisper pipeline implemented using Hugging Face's `transformers`. 

#### Video Stream Processing: 

+ **Local Video Files**
+ **RTSP Receive Video Streams**
+ **API Calls** 
  Overview of the Process: 

1. **Video stream input**: Read video frames from RTP streams or local video files, capture video using OpenCV or read RTP streams through a FFmpeg pipeline.
2. **Object detection**: Detect objects in each frame using a YOLO model, obtaining the object's category, confidence, and bounding box coordinates.
3. **Face detection**: If a "face" category is detected, use DeepFace to further analyze the face and extract age, gender, ethnicity, and emotional information.
4. **Text detection**: If a "text" category is detected, use EasyOCR to recognize text in the image and output the recognized text content. 

#### LLM Processing 

+ **ChatGPT**

+ **Qwen Open Source Model** 

+ **Gemini**

#### Digital Human Creation 

+ **ErNeRF**: It is suitable for generating highly efficient and realistic 3D scenes and virtual characters, with a focus on **multi-view rendering and optimizing computational efficiency**. 
+ **Musetalk**: It enables **voice and expression synchronization** through voice-driven virtual facial animation, suitable for real-time interactive virtual character scenarios. 
+ **Wav2Lip**: It specializes in driving mouth movements through speech, achieving **precise lip synchronization**, and is widely used in virtual face driving and video post-processing. 
  Audio Generation 
+ **EdgeTTS**: Microsoft's speech synthesis service, provided by Azure Cognitive Services to generate high-quality speech. 
+ **VoitsTTS**: Sovits models, supporting voice synthesis with reference vocal tone.
+ **CosyVoiceTTS**: Supports zero-sample voice synthesis, generating customized voice by referring to reference audio and input text. 
+ **XTTS**: A speech synthesis model that can clone the timbre of a reference audio and generate speech that matches that timbre based on input text.

### Deployment 

Start the SRS server.

```bash
export CANDIDATE='<server IP>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

Add your own openai TOKEN to the env file

vue project: https://github.com/Caxson/mere_web_client.git

Backend:

```bash
#create myenv
python3.10 -m venv myenv

#activate myenv
source myenv/bin/activate

#mac
brew install portaudio

#Linux
sudo apt-get update
sudo apt-get install libportaudio2 portaudio19-dev

#upgrade pip
pip install --upgrade pip

pip install -r requirements.txt

#museTalk
sudo xcodebuild -license

pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"
```

### TODO

+ Process problem testing and bug resolution ! ! ! 
+ Simplify and improve the startup process and parameter configuration and environment configuration
+ Optimize response time
+ Support voice interrupts speech
+ Digital human training model
+ yolo face and text recognition model training
+ Front-end multi-client support
+ Multi-user session maintenance and authentication, login, etc
+ etc.



## 中文版本

一个AI数字人实时流视频语音通话项目，包括视频语音流输入和大模型处理以及视频语音流输出，内置了许多大模型开源项目。

### 架构介绍

**客户端（Vue.js）**：通过 WebRTC 推送视频和音频流到 SRS。

（KMS：可选的中转服务器，可以用docker启动KMS然后后端网关层接收流并发送RTP流到python服务器）

**SRS**：作为 WebRTC 中转服务器，转发流到后端（Python）。

**后端（Python）**：

- **拉流**：从 SRS 获取音视频流，进行处理。
- **推流**：处理完成后，将音视频流重新推送到 SRS。

**客户端（Vue.js）**：拉取处理后的音视频流并展示。

### 后端功能

#### 音频流处理：

- **Offline 模式**：一次性处理完整音频文件，适合测试和调试。

- **Computationally Unaware 模式**：将音频分段处理，每个块按 `min_chunk_size` 加载，适合处理预先录制的长音频。

- **Online 模式**：模拟实时音频流的处理，适用于实时转录场景。

- **VAD 支持**：可以在任何模式下启用语音活动检测，优化处理效率。

  

  ##### **可选择的音频处理工具：**

  **WhisperTimestamped**：基于 OpenAI Whisper 模型并支持时间戳的库。

  **FasterWhisper**：性能优化后的 Whisper 模型，适用于更快速的转录。

  **OpenaiApiASR**：使用 OpenAI 的云端 API 服务进行语音转录。

  **InsanelyFastWhisperASR**：使用 Hugging Face `transformers` 实现的高效 Whisper 管道。

#### 视频流处理：

+ **本地视频文件**
+ **RTP接收视频流**
+ **API调用**

##### 处理流程概述：

1. **视频流输入**：从 RTP 流或本地视频文件中读取视频帧，使用 OpenCV 进行视频捕获或通过 FFmpeg 管道读取 RTP 流。
2. **物体检测**：每一帧通过 YOLO 模型进行物体检测，获取物体的类别、置信度和边界框坐标。
3. **人脸检测**：如果检测到 "人脸" 类别，则使用 DeepFace 对人脸进行进一步分析，提取年龄、性别、种族和情绪信息。
4. **文本检测**：如果检测到 "文字" 类别，使用 EasyOCR 对图像中的文本进行识别，并输出识别到的文字内容。

#### LLM处理

+ **ChatGPT**

+ **Qwen开源模型**

+ **Gemini**

#### 数字人生成

+ **ErNeRF**：适用于生成高效、逼真的 3D 场景和虚拟人物，重点在于**多视角渲染和优化计算效率**。

+ **Musetalk**：通过语音驱动虚拟人脸，实现**语音与表情同步**，适用于实时互动的虚拟角色场景。

+ **Wav2Lip**：专注于通过语音驱动嘴部运动，实现**精确的唇形同步**，广泛应用于虚拟人脸驱动和视频后期处理。

#### 音频生成

+ **EdgeTTS**：微软的语音合成服务，使用 Azure Cognitive Services 提供高质量的语音生成。

+ **VoitsTTS**：基于 GPT 和 Sovits 模型的语音合成服务，支持参考音色的语音合成。

+ **CosyVoiceTTS**：支持零样本语音合成，通过参考音频和输入文本生成定制语音。

+ **XTTS**：语音克隆模型，能够从参考音频中克隆音色，并根据输入文本生成与该音色匹配的语音。

### 部署

启动srs服务器

```bash
export CANDIDATE='<服务器外网IP>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

在env文件添加自己的openai的TOKEN

前端vue项目：https://github.com/Caxson/mere_web_client.git

后端：

```bash
#创建虚拟环境
python3.10 -m venv myenv

#激活虚拟环境
source myenv/bin/activate

#mac
brew install portaudio

#Linux
sudo apt-get update
sudo apt-get install libportaudio2 portaudio19-dev

#升级pip
pip install --upgrade pip

pip install -r requirements.txt

#museTalk
sudo xcodebuild -license

pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv>=2.0.1" 
mim install "mmdet>=3.1.0" 
mim install "mmpose>=1.1.0"
```

### TODO

+ 流程问题测试和bug解决
+ 简化和完善启动流程和参数配置以及环境配置
+ 优化响应时间
+ 支持声音打断讲话
+ 数字人的训练模型
+ yolo的人脸和文字识别模型训练
+ 前端的多客户端支持
+ 多用户session的维护和认证、登陆等
+ etc.
