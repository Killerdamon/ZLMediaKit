//
// TrtYolo.cpp
// TensorRT 推理引擎封装 (YOLOv8)
//

#include "TrtYolo.h"
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Util/logger.h"

using namespace nvinfer1;

namespace mediakit {
namespace ai {

class Logger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 将 TensorRT 日志重定向到 ZLMediaKit 的日志系统，解决 Windows 无控制台程序报错问题
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                toolkit::ErrorL << "[TensorRT] " << msg;
                break;
            case Severity::kWARNING:
                toolkit::WarnL << "[TensorRT] " << msg;
                break;
            case Severity::kINFO:
                // toolkit::InfoL << "[TensorRT] " << msg; // 抑制过多的 Info 打印
                break;
            default:
                break;
        }
    }
} gLogger;

TrtYolo::TrtYolo() {
}

TrtYolo::~TrtYolo() {
    destroy();
}

bool TrtYolo::init(const std::string& engine_path, int input_width, int input_height) {
    _input_w = input_width;
    _input_h = input_height;

    // 读取编译好的 engine 文件
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "Failed to read engine file: " << engine_path << std::endl;
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> trtModelStream(size);
    file.read(trtModelStream.data(), size);
    file.close();

    _runtime = createInferRuntime(gLogger);
    if (!_runtime) return false;

    _engine = _runtime->deserializeCudaEngine(trtModelStream.data(), size);
    if (!_engine) return false;

    _context = _engine->createExecutionContext();
    if (!_context) return false;

    // 假设模型结构: input (1x3x640x640), output (1x84x8400)
    // 实际应使用 engine->getBindingDimensions(index) 获取动态输出大小
    // 这里以标准 YOLOv8n (640x640) 为例
    _num_classes = 80;
    _num_anchors = 8400; 
    _output_size = (_num_classes + 4) * _num_anchors; 

    // 分配输出 GPU 内存
    cudaError_t status = cudaMalloc((void**)&_d_output, _output_size * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate output memory: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    // 分配 CPU 锁定内存用于快速拷贝
    status = cudaMallocHost((void**)&_h_output, _output_size * sizeof(float));
    if (status != cudaSuccess) {
        std::cerr << "Failed to allocate pinned memory: " << cudaGetErrorString(status) << std::endl;
        return false;
    }

    return true;
}

bool TrtYolo::inferAsync(float* d_input, cudaStream_t stream) {
    if (!_context) return false;

    // 绑定输入(0) 和 输出(1) 指针
    _bindings[0] = d_input;
    _bindings[1] = _d_output;

    // TensorRT V2 异步执行 (新版 API 可能使用 enqueueV3)
    bool status = _context->enqueueV2(_bindings, stream, nullptr);
    if (!status) return false;

    // 异步将输出拷贝回 CPU (Host)
    cudaMemcpyAsync(_h_output, _d_output, _output_size * sizeof(float), cudaMemcpyDeviceToHost, stream);

    return true;
}

// 同步流，进行 NMS 后处理
std::vector<DetectBox> TrtYolo::postProcessSync(cudaStream_t stream, float scale, int pad_x, int pad_y) {
    std::vector<DetectBox> results;

    // 确保异步推理和拷贝已经完成
    cudaStreamSynchronize(stream);

    // YOLOv8 输出张量格式: (1, 84, 8400) -> (Batch, Class+BBox, Anchor)
    // 需要转置或直接按 stride 访问
    const float conf_thresh = 0.45f;
    const int rows = _num_classes + 4; // 84
    const int cols = _num_anchors;     // 8400

    for (int i = 0; i < cols; ++i) {
        float max_conf = 0.0f;
        int class_id = -1;

        // 查找置信度最高类别 (索引 4 到 83)
        for (int c = 0; c < _num_classes; ++c) {
            float conf = _h_output[(c + 4) * cols + i];
            if (conf > max_conf) {
                max_conf = conf;
                class_id = c;
            }
        }

        if (max_conf >= conf_thresh) {
            // 解析 Bounding Box (中心点 x, y, w, h)
            float cx = _h_output[0 * cols + i];
            float cy = _h_output[1 * cols + i];
            float w  = _h_output[2 * cols + i];
            float h  = _h_output[3 * cols + i];

            // 转为左上角，并还原到原始图像坐标
            float x1 = (cx - w / 2.0f - pad_x) / scale;
            float y1 = (cy - h / 2.0f - pad_y) / scale;
            float real_w = w / scale;
            float real_h = h / scale;

            results.push_back({x1, y1, real_w, real_h, max_conf, class_id});
        }
    }

    // --- 在此处应该添加 NMS (Non-Maximum Suppression) 非极大值抑制逻辑 ---
    // 为了简化代码，暂略 NMS 算法的详细实现，仅返回所有超过阈值的框。
    // 实际使用时，请使用 OpenCV 的 cv::dnn::NMSBoxes 或自定义 IOU 计算。

    return results;
}

void TrtYolo::destroy() {
    if (_context) { _context->destroy(); _context = nullptr; }
    if (_engine) { _engine->destroy(); _engine = nullptr; }
    if (_runtime) { _runtime->destroy(); _runtime = nullptr; }

    if (_d_output) { cudaFree(_d_output); _d_output = nullptr; }
    if (_h_output) { cudaFreeHost(_h_output); _h_output = nullptr; }
}

} // namespace ai
} // namespace mediakit