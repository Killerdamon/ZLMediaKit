//
// TrtYolo.h
// TensorRT 推理引擎封装 (YOLOv8)
// 

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>

// Forward declaration to avoid exposing TensorRT headers everywhere
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace mediakit {
namespace ai {

// 检测框结果
struct DetectBox {
    float x, y, width, height;
    float confidence;
    int class_id;
};

class TrtYolo {
public:
    TrtYolo();
    ~TrtYolo();

    /**
     * @brief 初始化 TensorRT 引擎
     * @param engine_path  编译好的 yolov8.engine 路径
     * @param input_width  YOLO 模型输入宽度 (通常 640)
     * @param input_height YOLO 模型输入高度 (通常 640)
     */
    bool init(const std::string& engine_path, int input_width = 640, int input_height = 640);

    /**
     * @brief 异步执行推理
     * @param d_input  由 CUDA kernel 转换好的 NCHW float32 设备内存指针
     * @param stream   异步执行流
     * @return true 成功入队推理任务
     */
    bool inferAsync(float* d_input, cudaStream_t stream);

    /**
     * @brief 同步获取结果
     * @param stream 确保推理流结束
     * @param scale  Letterbox缩放比，用于还原回原图坐标
     * @param pad_x  X轴 Padding
     * @param pad_y  Y轴 Padding
     * @return 解析后的检测框列表
     */
    std::vector<DetectBox> postProcessSync(cudaStream_t stream, float scale, int pad_x, int pad_y);

    int getInputWidth() const { return _input_w; }
    int getInputHeight() const { return _input_h; }
    float* getOutputBuffer() { return _d_output; }

private:
    void destroy();

private:
    int _input_w = 640;
    int _input_h = 640;
    int _output_size = 0; // YOLOv8 输出张量的大小 (如 1x84x8400 -> 84*8400)
    int _num_classes = 80;
    int _num_anchors = 8400; // YOLOv8n 的 anchor 数 (640x640)

    nvinfer1::IRuntime* _runtime = nullptr;
    nvinfer1::ICudaEngine* _engine = nullptr;
    nvinfer1::IExecutionContext* _context = nullptr;

    // GPU 输出缓存
    float* _d_output = nullptr;
    
    // CPU 接收缓存 (用于后处理)
    float* _h_output = nullptr;
    
    // TensorRT 执行时的 bindings 指针数组
    void* _bindings[2]; 
};

} // namespace ai
} // namespace mediakit
