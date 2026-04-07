# ZLMediaKit 外挂式 AI 推理服务部署指南

本项目为您提供了一种低耦合、高性能、且能与 **开源版 ZLMediaKit** 完美结合的 AI 视频分析方案。
通过采用外挂脚本的模式，您可以用 Python 快速实现**商业闭源版**中提到的各种 AI 目标检测、跟踪、画框和推流等高级业务功能。

## 方案架构优势

- **极低耦合**：ZLMediaKit 只负责高性能的流媒体分发和接收，不直接参与复杂的深度学习环境（避免依赖地狱和段错误崩溃）。
- **极速开发**：Python 生态极其丰富，集成 OpenCV、YOLOv8、ONNXRuntime 甚至 TensorRT 仅需几行代码。
- **高可用扩展**：支持多进程/多容器并发拉流推理，一台 GPU 机器可以启动数十个这样的 AI 服务进程，限流与熔断可在 Python 侧轻松实现。

## 环境依赖

请确保您的环境中安装了以下基础库：

```bash
pip install opencv-python paho-mqtt ultralytics
```

> **性能优化建议**：
> 1. 为了启用 GPU 加速，请确保安装了正确的 PyTorch CUDA 版本，或者使用 `onnxruntime-gpu`。
> 2. `ultralytics` 支持将模型导出为 `.engine` (TensorRT) 格式，这将大幅提升推理速度并降低延迟。
> 3. FFmpeg 是将处理后画面推流回 ZLMediaKit 的关键，确保系统已安装并可用：`apt install ffmpeg`。

## 核心代码解析 (`ai_service.py`)

您可以在当前工作区查看我为您编写的核心代码文件：[ai_service.py](file:///workspace/ai_service.py)。

该脚本完整实现了您的自研需求：
1. **视频流拉取**：通过 OpenCV 从 ZLMediaKit 的 HTTP-FLV 或 RTSP 接口实时拉流。
2. **YOLO 目标识别**：引入 `ultralytics.YOLO`，支持加载 `.pt`, `.onnx`, `.engine` 等多格式模型，进行帧级目标检测。
3. **实时 OSD 绘制**：利用 OpenCV 在原始画面上绘制检测框 (Bounding Box)、类别标签、置信度和帧率 (FPS)。
4. **事件点截图与 MQTT 推送**：当检测到特定目标（如 `person` 人员）时，将当前画面编码为 Base64 截图，并封装为 JSON 消息发布到 MQTT Broker。内置了冷却机制（Cooldown）以防止频繁报警。
5. **画框后推流 (可选)**：通过开辟子进程调用 `FFmpeg`，将带有 AI 绘制框的 RGB 裸数据流重新编码（支持硬件编码如 `h264_nvenc`），并推流回 ZLMediaKit 供前端业务系统播放。

## 使用示例

1. 启动您的 ZLMediaKit 服务器（假设运行在 `127.0.0.1`）。
2. 向 ZLMediaKit 推送一路测试流，例如：`rtsp://127.0.0.1/live/test`。
3. 运行 AI 推理脚本：

```bash
# 仅拉流分析并推送 MQTT 报警，不推流画面
python ai_service.py --input_url rtsp://127.0.0.1/live/test --model yolov8n.pt --device 0

# 拉流分析，画框后重新推流回 ZLMediaKit 的新地址
python ai_service.py --input_url rtsp://127.0.0.1/live/test \
                     --output_url rtsp://127.0.0.1/live/ai_test \
                     --model yolov8n.engine \
                     --device 0
```

## 进阶业务扩展思路

针对您提到的其他自研需求，您可以在此脚本基础上进一步扩展：

- **多边形布防**：在 OpenCV 绘制框之前，使用 `cv2.pointPolygonTest()` 判断 YOLO 输出的坐标中心点是否在您预设的多边形区域内。
- **目标跟踪**：引入 `DeepSORT` 或 `ByteTrack` 库，将 YOLO 的检测框送入跟踪器，为每个目标分配唯一 ID，实现轨迹跟踪。
- **OCR 识别**：对于检测到的车牌或文本区域，裁剪出 ROI（Region of Interest），送入 `PaddleOCR` 进行文本提取。
- **多实例限流**：使用 Python 的 `multiprocessing` 或 Docker 容器化部署该脚本，结合 Redis 队列控制最大并发推理路数。