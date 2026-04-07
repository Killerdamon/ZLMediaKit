//
// cuda_utils.h
// CUDA 核函数封装声明
//

#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

namespace mediakit {
namespace ai {

/**
 * @brief 将 FFmpeg NVDEC 解码出来的 NV12 显存数据转换为 YOLO 需要的 NCHW RGB Float32 格式
 *        同时包含 Letterbox (保持宽高比缩放) 操作。
 * 
 * @param d_y         NV12 的 Y 分量设备指针
 * @param d_uv        NV12 的 UV 分量设备指针
 * @param src_width   原图宽度
 * @param src_height  原图高度
 * @param src_pitch   Y分量的 pitch (每行字节数，FFmpeg 解码后通常需要对齐)
 * @param d_dst       输出的 float32 设备指针 (大小为 3 * dst_width * dst_height * sizeof(float))
 * @param dst_width   目标宽度 (如 YOLO 的 640)
 * @param dst_height  目标高度 (如 YOLO 的 640)
 * @param stream      CUDA 流，用于异步执行
 */
void nv12_to_nchw_rgb_letterbox(
    const uint8_t* d_y, 
    const uint8_t* d_uv, 
    int src_width, 
    int src_height, 
    int src_pitch,
    float* d_dst, 
    int dst_width, 
    int dst_height, 
    cudaStream_t stream
);

} // namespace ai
} // namespace mediakit
