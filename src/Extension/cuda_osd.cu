//
// cuda_osd.cu
// CUDA 核函数实现：直接在 NV12 显存中绘制检测框
//

#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <algorithm>

namespace mediakit {
namespace ai {

// 定义边框绘制的结构体，用于传入 GPU
struct BBox {
    int x1, y1, x2, y2;
    int thickness;
    uint8_t y, u, v; // 边框颜色的 YUV 分量
};

// --- RGB 转 YUV (BT.601) 工具函数 ---
__host__ __device__ void rgb2yuv(uint8_t r, uint8_t g, uint8_t b, uint8_t& y, uint8_t& u, uint8_t& v) {
    y = (uint8_t)(0.299f * r + 0.587f * g + 0.114f * b);
    u = (uint8_t)(-0.169f * r - 0.331f * g + 0.500f * b + 128);
    v = (uint8_t)(0.500f * r - 0.419f * g - 0.081f * b + 128);
}

// --- 绘制单个检测框的 Kernel ---
__global__ void draw_bbox_nv12_kernel(
    uint8_t* __restrict__ d_y,
    uint8_t* __restrict__ d_uv,
    int src_w, int src_h, int src_pitch,
    const BBox* __restrict__ boxes,
    int num_boxes)
{
    // 每个线程负责一个像素的绘制判断
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= src_w || py >= src_h) return;

    for (int i = 0; i < num_boxes; ++i) {
        const BBox& b = boxes[i];
        
        // 边界保护
        int bx1 = max(0, b.x1);
        int by1 = max(0, b.y1);
        int bx2 = min(src_w - 1, b.x2);
        int by2 = min(src_h - 1, b.y2);
        int t = b.thickness;

        // 判断当前像素是否在边框的四个“边”上
        bool on_border = false;
        if (px >= bx1 && px <= bx2 && py >= by1 && py <= by2) {
            // 左边 或 右边
            if ((px >= bx1 && px < bx1 + t) || (px <= bx2 && px > bx2 - t)) {
                on_border = true;
            }
            // 上边 或 下边
            if ((py >= by1 && py < by1 + t) || (py <= by2 && py > by2 - t)) {
                on_border = true;
            }
        }

        if (on_border) {
            // 修改 Y 分量
            int y_idx = py * src_pitch + px;
            d_y[y_idx] = b.y;

            // 修改 UV 分量 (NV12 交错存储，每 2x2 像素共享一对 UV)
            // 为了避免线程冲突，我们只让左上角的像素去写 UV，或者直接覆盖
            // NV12 中 UV 占据 1/4 的面积
            int uv_idx = (py / 2) * src_pitch + (px / 2) * 2;
            d_uv[uv_idx] = b.u;
            d_uv[uv_idx + 1] = b.v;
            
            // 因为可能同时属于多个框的边缘，画上颜色后即可返回，避免覆盖
            return;
        }
    }
}

void draw_bboxes_nv12(
    uint8_t* d_y, 
    uint8_t* d_uv, 
    int src_width, 
    int src_height, 
    int src_pitch,
    const std::vector<DetectBox>& boxes,
    cudaStream_t stream)
{
    if (boxes.empty()) return;

    // 准备传入 GPU 的数据
    std::vector<BBox> gpu_boxes;
    for (const auto& b : boxes) {
        BBox g_box;
        g_box.x1 = (int)b.x;
        g_box.y1 = (int)b.y;
        g_box.x2 = (int)(b.x + b.width);
        g_box.y2 = (int)(b.y + b.height);
        g_box.thickness = 2; // 默认线宽

        // 简易颜色分配：如果是人(id=0)，画绿色；其他画红色
        uint8_t r = (b.class_id == 0) ? 0 : 255;
        uint8_t g = (b.class_id == 0) ? 255 : 0;
        uint8_t b_c = 0;
        
        rgb2yuv(r, g, b_c, g_box.y, g_box.u, g_box.v);
        gpu_boxes.push_back(g_box);
    }

    // 分配显存存储 BBox 数组
    BBox* d_boxes = nullptr;
    size_t boxes_size = gpu_boxes.size() * sizeof(BBox);
    cudaMallocAsync(&d_boxes, boxes_size, stream);
    cudaMemcpyAsync(d_boxes, gpu_boxes.data(), boxes_size, cudaMemcpyHostToDevice, stream);

    // 配置线程块
    dim3 block(32, 32);
    dim3 grid((src_width + block.x - 1) / block.x, (src_height + block.y - 1) / block.y);

    // 异步执行绘制 Kernel
    draw_bbox_nv12_kernel<<<grid, block, 0, stream>>>(
        d_y, d_uv, src_width, src_height, src_pitch,
        d_boxes, gpu_boxes.size()
    );

    // 释放显存 (异步释放)
    cudaFreeAsync(d_boxes, stream);
}

} // namespace ai
} // namespace mediakit
