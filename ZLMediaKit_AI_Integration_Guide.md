# ZLMediaKit 高性能 C++ AI 推理架构 (FFmpeg HW + TensorRT)

为了实现“又快又好”的商用级并发推理，我们必须抛弃低效的 Python 拉流和 CPU 内存拷贝，采用 **GPU 全链路加速方案**。

## 1. 极致性能的数据流设计 (Zero-Copy Architecture)

传统方案的瓶颈在于 `GPU -> CPU -> GPU` 的内存拷贝。本方案的核心数据流如下：

1. **获取原始帧**：ZLMediaKit (如 `PlayerProxy` 或 `RtmpSession`) 接收到 H.264/H.265 NALU，打包成 `Frame::Ptr`。
2. **NVDEC 硬件解码**：将 `Frame` 送入编译了 `--enable-cuvid` 的 FFmpeg 解码器。解码后的 `AVFrame` 格式为 `AV_PIX_FMT_CUDA` (直接在显存中)。
3. **CUDA 前处理**：调用自定义的 CUDA Kernel，将 NV12 的显存数据转换为 RGB 格式，并 Resize/Normalize 为 TensorRT 需要的 NCHW float32 张量格式。
4. **TensorRT 推理**：执行 YOLO 引擎，得出检测框。
5. **CUDA 画框 (OSD)**：在 GPU 上直接通过 CUDA Kernel 画检测框（避免下载回 CPU，如果赶时间可以用 OpenCV CPU 绘制作为过渡）。
6. **NVENC 硬件编码**：将画好框的 `AVFrame` 送入 FFmpeg 的 `h264_nvenc` 编码器。
7. **生成新流**：输出的 `AVPacket` 封装为 ZLMediaKit `H264Frame`，注入到新的 `MultiMediaSourceMuxer` 中，提供 HTTP-FLV/RTSP 播放。

## 2. 环境编译要求 (极度重要)

您必须重新编译依赖库，普通 `apt-get` 安装的库无法支持硬件加速：
- **CUDA Toolkit**: >= 11.4
- **TensorRT**: >= 8.4
- **OpenCV**: 需带 CUDA 支持编译 ( `-DWITH_CUDA=ON -DWITH_CUDNN=ON` )
- **FFmpeg**: 需带 NVIDIA 硬件加速编译 ( `--enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-nonfree` )
- **ZLMediaKit**: 修改 `CMakeLists.txt`，链接 `nvinfer`, `cudart`, `opencv_world` 等库。

## 3. 核心代码骨架

我为您编写了 `AIFilter` 的核心骨架（见 `src/Extension/AIFilter.h` 和 `src/Extension/AIFilter.cpp`）。
它继承自 `MediaSinkInterface`，可以直接作为 ZLMediaKit 内部流处理的一个 Filter 节点。