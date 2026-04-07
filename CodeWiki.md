# ZLMediaKit Code Wiki

## 1. 项目整体架构 (Overall Project Architecture)
ZLMediaKit 是一个基于 C++11 的高性能、企业级流媒体服务框架。其核心架构采用了多线程、异步网络 I/O (Reactor 模型) 设计，具备极高的并发处理能力和低延迟特性。

整体架构自底向上可分为以下几层：
- **基础网络层 (ZLToolKit)**：提供跨平台的多路复用网络 I/O 框架，包括 `TcpServer`、`UdpServer`、`EventPoller` (事件轮询器)、定时器和线程池。
- **协议解析层**：实现了丰富的流媒体协议收发与解析，包括 RTSP、RTMP、HTTP-FLV、WebSocket-FLV、HLS、WebRTC、SRT、GB28181 等。
- **媒体处理层**：负责音视频轨道的管理 (`Track`)、媒体帧 (`Frame`) 的分发、解复用与复用。支持流媒体协议之间的相互转换 (例如 RTMP 转 RTSP/HLS/WebRTC)。
- **业务应用层**：包含完整的 `MediaServer` 服务程序，提供基于 HTTP 的 RESTful API 和 WebHook 事件回调机制，方便与第三方业务系统集成。同时提供 C API (`mk_api`) 供其他语言调用。

## 2. 主要模块职责 (Main Module Responsibilities)
项目的核心代码位于 `src/` 和顶层目录中，主要模块包括：

- **`src/Codec/`**：音视频编解码相关处理，支持 AAC、H264 等格式的封装与转码 (`Transcode`)。
- **`src/Common/`**：核心公共组件，包含流媒体源 (`MediaSource`)、媒体接收器 (`MediaSink`)、多协议复用器 (`MultiMediaSourceMuxer`)、配置文件解析和全局宏定义。
- **`src/Extension/`**：媒体扩展定义，抽象了音视频轨道 (`Track`)、媒体帧 (`Frame`)，以及用于协议转换的工厂类 (`Factory`)。
- **`src/Http/`**：HTTP 协议栈实现，包括 HTTP 服务端/客户端、WebSocket、HLS 解析与生成、HTTP-TS 和 HTTP-fMP4 的支持。
- **`src/Player/` & `src/Pusher/`**：分别实现了媒体流的拉取 (拉流代理 `PlayerProxy`) 和推送 (推流代理 `PusherProxy`)，支持多种协议。
- **`src/Record/`**：媒体录制模块，支持将直播流录制为 HLS、MP4 和 FLV 格式文件。
- **`src/Rtmp/`, `src/Rtsp/`, `src/Rtp/`, `src/Rtcp/`**：流媒体核心协议的具体实现。例如 `RtmpSession` 处理 RTMP 交互，`RtpServer` 处理 GB28181 的 RTP 流接收。
- **`srt/`**：SRT (Secure Reliable Transport) 协议的底层实现与集成。
- **`webrtc/`**：WebRTC 协议栈实现，包括 SDP 协商、DTLS 握手、ICE 打洞、SRTP 加密解密等。
- **`api/`**：C 语言风格的 API 接口 (`mk_mediakit.h`)，作为 SDK 供外部集成。
- **`server/`**：可独立运行的流媒体服务器主程序，整合了各种协议的 Server 端，并实现了 WebAPI 和 WebHook。

## 3. 关键类与函数说明 (Key Classes and Functions)
### 核心类
- **`EventPoller` / `EventPollerPool`** (来自 ZLToolKit)：事件轮询器和线程池。整个框架的异步 I/O 引擎，采用 epoll/kqueue 等系统调用实现事件驱动。
- **`TcpServer` / `UdpServer`**：网络服务器基类，负责监听端口并接受客户端连接。
- **`MediaSource`**：媒体源的抽象基类。每一种具体的流 (如 RTMP、RTSP) 都有对应的子类 (如 `RtmpMediaSource`)，负责管理流的状态、分发媒体数据。
- **`MultiMediaSourceMuxer`**：多媒体复用器。当有一路流输入时，它会负责将这路流的数据复用 (Mux) 转换为其他多种协议的格式，实现“一处推流，多处播放”。
- **`Track`**：音视频轨道抽象类，如 `H264Track`、`AACTrack`，包含了音视频的具体参数 (如 SPS/PPS、采样率等)。
- **`Frame`**：音视频帧抽象类，封装了原始音视频数据及时间戳信息。

### 关键函数
- **`start_main(int argc, char *argv[])`** (位于 `server/main.cpp`)：`MediaServer` 的程序入口。负责解析命令行参数、加载配置文件、初始化日志、设置事件线程池、加载 SSL 证书，并启动 RTSP、RTMP、HTTP、WebRTC、SRT 等各类服务器监听。
- **`api_regist(...)`** (位于 `server/WebApi.cpp`)：RESTful API 的注册函数。通过该函数将不同的 URL 路径映射到对应的 C++ 处理 Lambda 函数上。

## 4. 依赖关系 (Dependencies)
ZLMediaKit 采用 CMake 构建，依赖分为内部子模块和外部系统库：

### 内部依赖 (Git Submodules)
- **`ZLToolKit`**：核心基础库，提供网络、线程、日志等基础设施。
- **`media-server`**：第三方开源库，用于处理 TS、fMP4、MP4、PS 等容器格式的复用与解复用。
- **`jsoncpp`**：用于解析和生成 JSON 数据 (主要用于 RESTful API 和 WebHook)。
- **`webassist`**：Web 前端管理页面项目。

### 外部依赖 (系统级/第三方库)
- **`OpenSSL`**：(推荐必须) 用于支持 HTTPS、RTMPS、RTSPS，以及 WebRTC 的 DTLS 握手。
- **`FFmpeg` (libavcodec, libavutil, etc.)**：(可选) 开启转码功能或处理复杂拉流代理时需要。
- **`jemalloc`**：(可选/推荐) 用于替换 glibc 默认的 ptmalloc，有效减少高并发下的内存碎片问题。
- **`libsrt`**：(可选) 支持 SRT 协议。

## 5. 项目运行方式 (How to Run the Project)

### 源码编译运行
1. **克隆代码** (注意必须带上子模块)：
   ```bash
   git clone --depth 1 -b master --recursive https://github.com/ZLMediaKit/ZLMediaKit.git
   cd ZLMediaKit
   ```
2. **编译**：
   ```bash
   mkdir build && cd build
   cmake ..
   make -j4
   ```
3. **运行**：
   编译好的可执行文件位于 `release/linux/Debug/` (或 Release) 目录下。
   ```bash
   cd release/linux/Debug
   # 启动 MediaServer，-d 参数表示以守护进程(Daemon)运行
   ./MediaServer -d &
   ```
   **常用启动参数**：
   - `-c`：指定配置文件路径 (默认 `config.ini`)
   - `-s`：指定 SSL 证书文件路径 (默认 `default.pem`)
   - `-l`：设置日志等级 (0~4)

### Docker 运行
项目提供了官方维护的 Docker 镜像，可以通过一条命令快速拉起包含了所有功能的服务器：
```bash
docker run -id -p 1935:1935 -p 8080:80 -p 8443:443 -p 8554:554 -p 10000:10000 -p 10000:10000/udp -p 8000:8000/udp -p 9000:9000/udp zlmediakit/zlmediakit:master
```