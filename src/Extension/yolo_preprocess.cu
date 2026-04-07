//
// yolo_preprocess.cu
// 纯 CUDA 核函数实现，零拷贝(Zero-Copy)处理 NV12 数据到 YOLO 模型输入
//

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

namespace mediakit {
namespace ai {

// 限制宏
#define CLIP(x, a, b) ((x) < (a) ? (a) : ((x) > (b) ? (b) : (x)))

// CUDA NV12 到 RGB NCHW 并包含 Letterbox 缩放的 Kernel
__global__ void nv12_to_rgb_letterbox_kernel(
    const uint8_t* __restrict__ d_y,
    const uint8_t* __restrict__ d_uv,
    int src_w, int src_h, int src_pitch,
    float* __restrict__ d_dst,
    int dst_w, int dst_h,
    float scale, int pad_x, int pad_y) 
{
    // 获取当前线程处理的 dst_w 和 dst_h 坐标
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= dst_w || dy >= dst_h) return;

    // 映射回 src 坐标
    float sx = (dx - pad_x) / scale;
    float sy = (dy - pad_y) / scale;

    int isx = (int)sx;
    int isy = (int)sy;

    // 如果目标像素映射回原图时越界（落在 padding 区域）
    if (isx < 0 || isx >= src_w || isy < 0 || isy >= src_h) {
        // padding 颜色 114 (YOLO 标准)
        float pad_val = 114.0f / 255.0f;
        int area = dst_w * dst_h;
        // 写入 NCHW (R, G, B 三个通道平面)
        d_dst[0 * area + dy * dst_w + dx] = pad_val; // R
        d_dst[1 * area + dy * dst_w + dx] = pad_val; // G
        d_dst[2 * area + dy * dst_w + dx] = pad_val; // B
        return;
    }

    // 采样 NV12
    // Y 分量
    int y_idx = isy * src_pitch + isx;
    uint8_t y_val = d_y[y_idx];

    // UV 分量 (交错存储 U, V, U, V...)
    int uv_idx = (isy / 2) * src_pitch + (isx / 2) * 2;
    uint8_t u_val = d_uv[uv_idx];
    uint8_t v_val = d_uv[uv_idx + 1];

    // YUV to RGB (BT.601)
    int c = y_val - 16;
    int d = u_val - 128;
    int e = v_val - 128;

    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;

    r = CLIP(r, 0, 255);
    g = CLIP(g, 0, 255);
    b = CLIP(b, 0, 255);

    // Normalize 到 0.0 ~ 1.0
    float norm_r = r / 255.0f;
    float norm_g = g / 255.0f;
    float norm_b = b / 255.0f;

    // 写入 NCHW
    int area = dst_w * dst_h;
    d_dst[0 * area + dy * dst_w + dx] = norm_r;
    d_dst[1 * area + dy * dst_w + dx] = norm_g;
    d_dst[2 * area + dy * dst_w + dx] = norm_b;
}

void nv12_to_nchw_rgb_letterbox(
    const uint8_t* d_y, 
    const uint8_t* d_uv, 
    int src_width, 
    int src_height, 
    int src_pitch,
    float* d_dst, 
    int dst_width, 
    int dst_height, 
    cudaStream_t stream)
{
    // 计算缩放比和 padding
    float scale = std::min((float)dst_width / src_width, (float)dst_height / src_height);
    int pad_x = (dst_width - src_width * scale) / 2;
    int pad_y = (dst_height - src_height * scale) / 2;

    // 配置线程块
    dim3 block(32, 32);
    dim3 grid((dst_width + block.x - 1) / block.x, (dst_height + block.y - 1) / block.y);

    // 异步执行 CUDA Kernel
    nv12_to_rgb_letterbox_kernel<<<grid, block, 0, stream>>>(
        d_y, d_uv, src_width, src_height, src_pitch,
        d_dst, dst_width, dst_height,
        scale, pad_x, pad_y
    );
}

} // namespace ai
} // namespace mediakit
