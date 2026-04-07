//
// AIFilter.cpp
// 核心 AI 处理节点实现框架
// 
#include "AIFilter.h"
#include "Util/logger.h"
#include "Extension/H264.h"

// 硬件加速相关，假设编译环境包含 CUDA 和 cuvid
extern "C" {
#include <libavutil/hwcontext_cuda.h>
}

using namespace toolkit;

namespace mediakit {

AIFilter::AIFilter(const std::string& stream_id, const std::string& model_path)
    : _stream_id(stream_id), _model_path(model_path) {
    InfoL << "AIFilter Created: Stream " << _stream_id << ", Model " << _model_path;
    
    // 初始化 TensorRT (此处为框架骨架)
    initTensorRT(_model_path);
}

AIFilter::~AIFilter() {
    _running = false;
    if (_ai_thread.joinable()) {
        _ai_thread.join();
    }
    
    if (_decoder_ctx) avcodec_free_context(&_decoder_ctx);
    if (_encoder_ctx) avcodec_free_context(&_encoder_ctx);
    if (_hw_device_ctx) av_buffer_unref(&_hw_device_ctx);
}

bool AIFilter::addTrack(const Track::Ptr &track) {
    if (track->getTrackType() == TrackVideo) {
        _video_track = track;
        initDecoder(track);
        initEncoder();
        
        // ZLMediaKit 将复用带有 AI 框的新流，并生成 RTSP/FLV 协议
        // 注意：新流名为 原流名_ai
        _muxer = std::make_shared<MultiMediaSourceMuxer>(
            "live", _stream_id + "_ai", 0, nullptr
        );
        _muxer->addTrack(_video_track); // 将编码后的 track 参数注册给 Muxer
        
        _running = true;
        _ai_thread = std::thread(&AIFilter::aiProcessLoop, this);
        return true;
    } else {
        // 音频轨道直接透传，不做 AI 处理，为了音视频同步
        if (_muxer) {
            _muxer->addTrack(track);
        }
        return true;
    }
}

void AIFilter::resetTracks() {
    if (_muxer) {
        _muxer->resetTracks();
    }
}

// 拦截原始 ZLMediaKit 网络流的每一帧 NALU
bool AIFilter::inputFrame(const Frame::Ptr &frame) {
    if (frame->getTrackType() == TrackVideo) {
        // 将 frame 放入 _decoder_ctx 进行解码
        // 伪代码：
        /*
        AVPacket *pkt = av_packet_alloc();
        pkt->data = (uint8_t*)frame->data();
        pkt->size = frame->size();
        pkt->pts = frame->pts();
        pkt->dts = frame->dts();
        
        avcodec_send_packet(_decoder_ctx, pkt);
        
        AVFrame *avframe = av_frame_alloc();
        while (avcodec_receive_frame(_decoder_ctx, avframe) == 0) {
            // 此时 avframe 是位于显存(CUDA)的硬解格式 (AV_PIX_FMT_CUDA)
            // 将其 push 到线程安全的队列 _frame_queue，由 aiProcessLoop() 处理
            _frame_queue.push(avframe); 
            avframe = av_frame_alloc(); // 准备下一个
        }
        av_packet_free(&pkt);
        */
        return true;
    } else {
        // 音频直接写回 Muxer 透传
        if (_muxer) {
            _muxer->inputFrame(frame);
        }
        return true;
    }
}

void AIFilter::initDecoder(const Track::Ptr &track) {
    // 1. 初始化 FFmpeg 的 CUVID (NVDEC) 硬件解码器
    // 伪代码：
    /*
    const AVCodec *codec = avcodec_find_decoder_by_name("h264_cuvid");
    _decoder_ctx = avcodec_alloc_context3(codec);
    av_hwdevice_ctx_create(&_hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, "0", nullptr, 0);
    _decoder_ctx->hw_device_ctx = av_buffer_ref(_hw_device_ctx);
    avcodec_open2(_decoder_ctx, codec, nullptr);
    */
    InfoL << "Initialized FFmpeg HW Decoder (NVDEC) for " << track->getCodecName();
}

void AIFilter::initEncoder() {
    // 2. 初始化 FFmpeg 的 NVENC 硬件编码器
    // 伪代码：
    /*
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    _encoder_ctx = avcodec_alloc_context3(codec);
    _encoder_ctx->hw_device_ctx = av_buffer_ref(_hw_device_ctx);
    _encoder_ctx->pix_fmt = AV_PIX_FMT_CUDA; // 接收显存中的帧
    _encoder_ctx->width = 1920;  // 实际由输入决定
    _encoder_ctx->height = 1080;
    _encoder_ctx->time_base = {1, 25};
    avcodec_open2(_encoder_ctx, codec, nullptr);
    */
    InfoL << "Initialized FFmpeg HW Encoder (NVENC)";
}

void AIFilter::initTensorRT(const std::string& model_path) {
    // 3. 使用 TensorRT C++ API 加载 yolov8.engine
    // 伪代码：
    /*
    IRuntime* runtime = createInferRuntime(gLogger);
    std::ifstream file(model_path, std::ios::binary);
    ...
    _engine = runtime->deserializeCudaEngine(modelData, modelSize);
    _context = _engine->createExecutionContext();
    */
    InfoL << "Initialized TensorRT Engine with " << model_path;
}

// 核心多线程推理与画框循环
void AIFilter::aiProcessLoop() {
    while (_running) {
        // 从 _frame_queue 中获取 NVDEC 解码出来的 NV12 显存帧
        // AVFrame* hw_frame = _frame_queue.pop();
        // if (!hw_frame) continue;

        // 4. CUDA 预处理 (NV12 -> RGB -> Resize -> NCHW Float32)
        // 这一步使用 CUDA 核函数(Kernel)直接在 GPU 完成，不经过 CPU
        // cudaPreprocess(hw_frame->data, input_tensor_ptr);

        // 5. 执行 TensorRT 推理 (异步)
        // _context->enqueueV2(bindings, stream, nullptr);
        // cudaStreamSynchronize(stream);
        
        // 6. 解析 YOLO 结果，如果发现目标(人/车)，用 CUDA Kernel 画框
        // 也可以回传极小的一块内存回 CPU 进行 OpenCV 画框，再送回 GPU
        // if (person_detected) {
        //     cudaDrawBox(hw_frame->data, bbox_x, bbox_y, w, h, color);
        // }

        // 7. 送入硬件编码器 (NVENC)
        // avcodec_send_frame(_encoder_ctx, hw_frame);
        /*
        AVPacket *pkt = av_packet_alloc();
        while (avcodec_receive_packet(_encoder_ctx, pkt) == 0) {
            onEncodedPacket(pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
        */
        
        // 释放解码帧
        // av_frame_free(&hw_frame);
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

// 将 NVENC 硬编出来的 H264 NALU 包装回 ZLMediaKit 的 Frame 对象并发布
void AIFilter::onEncodedPacket(AVPacket* pkt) {
    // 将 AVPacket 转为 H264Frame
    auto frame = std::make_shared<H264FrameNoCacheAble>(
        (char*)pkt->data, pkt->size, pkt->dts, pkt->pts, 0
    );
    
    // 送入复用器，生成最终的 RTMP/RTSP 流供前端播放
    if (_muxer) {
        _muxer->inputFrame(frame);
    }
}

} // namespace mediakit