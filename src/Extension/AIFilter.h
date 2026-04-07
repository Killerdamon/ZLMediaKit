//
// AIFilter.h
// 核心 AI 处理节点 (集成 FFmpeg NVDEC + TensorRT + NVENC)
// 适用于 ZLMediaKit 高性能深度定制流处理
//
#pragma once

#include <memory>
#include <mutex>
#include <thread>
#include <string>
#include "Common/MediaSink.h"
#include "Extension/Frame.h"
#include "Common/MultiMediaSourceMuxer.h"

// FFmpeg Headers
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
}

// TensorRT Headers (伪代码占位)
// #include <NvInfer.h>
// #include <NvOnnxParser.h>

namespace mediakit {

class AIFilter : public MediaSinkInterface, public std::enable_shared_from_this<AIFilter> {
public:
    using Ptr = std::shared_ptr<AIFilter>;

    AIFilter(const std::string& stream_id, const std::string& model_path);
    ~AIFilter() override;

    // 继承自 MediaSinkInterface
    // 拦截上游的音视频数据 (H264/H265/AAC)
    bool inputFrame(const Frame::Ptr &frame) override;
    bool addTrack(const Track::Ptr &track) override;
    void resetTracks() override;

private:
    void initDecoder(const Track::Ptr &track);
    void initEncoder();
    void initTensorRT(const std::string& model_path);
    
    // 异步解码、推理和编码的线程
    void aiProcessLoop();
    
    // 将带有 AI 结果的 H264 包重新推流
    void onEncodedPacket(AVPacket* pkt);

private:
    std::string _stream_id;
    std::string _model_path;
    
    // ZLMediaKit 负责将重新编码的音视频复用并发布为新的 HTTP-FLV/RTSP/RTMP 流
    MultiMediaSourceMuxer::Ptr _muxer;
    Track::Ptr _video_track;

    // FFmpeg 硬件解码上下文
    AVCodecContext* _decoder_ctx = nullptr;
    AVBufferRef* _hw_device_ctx = nullptr;
    
    // FFmpeg 硬件编码上下文 (NVENC)
    AVCodecContext* _encoder_ctx = nullptr;

    // TensorRT 推理引擎上下文 (占位)
    // nvinfer1::ICudaEngine* _engine = nullptr;
    // nvinfer1::IExecutionContext* _context = nullptr;

    bool _running = false;
    std::thread _ai_thread;
    std::mutex _mutex;
    
    // 此处可添加一个线程安全的队列，用于缓冲解码后的 AVFrame 供 TensorRT 处理
    // SafeQueue<AVFrame*> _frame_queue;
};

} // namespace mediakit