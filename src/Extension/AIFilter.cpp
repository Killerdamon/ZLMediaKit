//
// AIFilter.cpp
// 核心 AI 处理节点实现：完整集成 FFmpeg NVDEC (硬解) + NVENC (硬编) + TensorRT
// 
#include "AIFilter.h"
#include "Util/logger.h"
#include "Extension/H264.h"
#include "Extension/H265.h"

// 引入 CUDA 和 TensorRT 工具类
#include "cuda_utils.h"
#include "TrtYolo.h"

// 硬件加速相关，假设编译环境包含 CUDA 和 cuvid
extern "C" {
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
}

using namespace toolkit;

namespace mediakit {

AIFilter::AIFilter(const std::string& stream_id, const std::string& model_path)
    : _stream_id(stream_id), _model_path(model_path) {
    InfoL << "AIFilter Created: Stream " << _stream_id << ", Model " << _model_path;
    
    // 初始化 TensorRT 引擎
    initTensorRT(_model_path);
}

AIFilter::~AIFilter() {
    _running = false;
    // 发送一个空帧唤醒线程退出
    {
        std::lock_guard<std::mutex> lck(_mutex);
        _frame_queue.push_back(nullptr);
    }
    
    if (_ai_thread.joinable()) {
        _ai_thread.join();
    }
    
    if (_decoder_ctx) avcodec_free_context(&_decoder_ctx);
    if (_encoder_ctx) avcodec_free_context(&_encoder_ctx);
    if (_hw_device_ctx) av_buffer_unref(&_hw_device_ctx);
    
    if (_d_input_tensor) {
        cudaFree(_d_input_tensor);
        _d_input_tensor = nullptr;
    }
    if (_stream) {
        cudaStreamDestroy(_stream);
        _stream = nullptr;
    }
}

bool AIFilter::addTrack(const Track::Ptr &track) {
    if (track->getTrackType() == TrackVideo) {
        _video_track = track;
        initDecoder(track);
        initEncoder(); // 这里可以根据解码出来的第一帧宽高再初始化，简化起见这里先写骨架
        
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
        if (!_decoder_ctx) return false;

        AVPacket *pkt = av_packet_alloc();
        pkt->data = (uint8_t*)frame->data();
        pkt->size = frame->size();
        pkt->pts = frame->pts();
        pkt->dts = frame->dts();
        
        // 发送给解码器
        int ret = avcodec_send_packet(_decoder_ctx, pkt);
        if (ret < 0) {
            WarnL << "Error sending a packet for decoding: " << ret;
            av_packet_free(&pkt);
            return false;
        }
        
        // 从解码器接收显存帧
        while (ret >= 0) {
            AVFrame *avframe = av_frame_alloc();
            ret = avcodec_receive_frame(_decoder_ctx, avframe);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                av_frame_free(&avframe);
                break;
            } else if (ret < 0) {
                WarnL << "Error during decoding";
                av_frame_free(&avframe);
                break;
            }

            // 此时 avframe 是位于显存(CUDA)的硬解格式 (AV_PIX_FMT_CUDA)
            // 将其 push 到线程安全的队列 _frame_queue，由 aiProcessLoop() 处理
            std::lock_guard<std::mutex> lck(_mutex);
            _frame_queue.push_back(avframe); 
        }
        av_packet_free(&pkt);
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
    // 1. 根据编码格式选择对应的 CUDA 硬件解码器
    const AVCodec *codec = nullptr;
    if (track->getCodecId() == CodecH264) {
        codec = avcodec_find_decoder_by_name("h264_cuvid");
    } else if (track->getCodecId() == CodecH265) {
        codec = avcodec_find_decoder_by_name("hevc_cuvid");
    }

    if (!codec) {
        ErrorL << "Cannot find cuvid decoder for track!";
        return;
    }

    _decoder_ctx = avcodec_alloc_context3(codec);
    if (!_decoder_ctx) {
        ErrorL << "Failed to allocate decoder context";
        return;
    }

    // 创建 CUDA 硬件设备上下文
    int ret = av_hwdevice_ctx_create(&_hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, "0", nullptr, 0);
    if (ret < 0) {
        ErrorL << "Failed to create CUDA hardware device context";
        return;
    }

    _decoder_ctx->hw_device_ctx = av_buffer_ref(_hw_device_ctx);
    
    // 打开解码器
    ret = avcodec_open2(_decoder_ctx, codec, nullptr);
    if (ret < 0) {
        ErrorL << "Failed to open cuvid decoder";
        return;
    }

    InfoL << "Initialized FFmpeg HW Decoder (NVDEC) for " << track->getCodecName();
}

void AIFilter::initEncoder() {
    // 2. 初始化 FFmpeg 的 NVENC 硬件编码器
    const AVCodec *codec = avcodec_find_encoder_by_name("h264_nvenc");
    if (!codec) {
        ErrorL << "Cannot find h264_nvenc encoder!";
        return;
    }

    _encoder_ctx = avcodec_alloc_context3(codec);
    if (!_encoder_ctx) {
        ErrorL << "Failed to allocate encoder context";
        return;
    }

    // 这里硬编码为 1920x1080，实际生产中应在解码出第一帧时根据 AVFrame 的 width/height 动态初始化编码器
    _encoder_ctx->width = 1920;  
    _encoder_ctx->height = 1080;
    _encoder_ctx->time_base = {1, 25};
    _encoder_ctx->framerate = {25, 1};
    _encoder_ctx->gop_size = 50;
    _encoder_ctx->max_b_frames = 0;
    _encoder_ctx->pix_fmt = AV_PIX_FMT_CUDA; // 接收显存中的帧 (NV12)
    
    if (_hw_device_ctx) {
        _encoder_ctx->hw_device_ctx = av_buffer_ref(_hw_device_ctx);
        // 需要创建 frames context 告诉 NVENC 帧格式
        AVBufferRef* hw_frames_ref = av_hwframe_ctx_alloc(_hw_device_ctx);
        AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ref->data;
        frames_ctx->format = AV_PIX_FMT_CUDA;
        frames_ctx->sw_format = AV_PIX_FMT_NV12;
        frames_ctx->width = _encoder_ctx->width;
        frames_ctx->height = _encoder_ctx->height;
        av_hwframe_ctx_init(hw_frames_ref);
        _encoder_ctx->hw_frames_ctx = av_buffer_ref(hw_frames_ref);
        av_buffer_unref(&hw_frames_ref);
    }

    // 设置 nvenc 的低延迟预设
    av_opt_set(_encoder_ctx->priv_data, "preset", "p1", 0); 
    av_opt_set(_encoder_ctx->priv_data, "tune", "ull", 0);  // Ultra Low Latency

    int ret = avcodec_open2(_encoder_ctx, codec, nullptr);
    if (ret < 0) {
        ErrorL << "Failed to open h264_nvenc encoder";
        return;
    }

    InfoL << "Initialized FFmpeg HW Encoder (NVENC)";
}

void AIFilter::initTensorRT(const std::string& model_path) {
    // 3. 实例化 TensorRT 引擎
    _trt = std::make_shared<ai::TrtYolo>();
    if (_trt->init(model_path, 640, 640)) {
        cudaMalloc((void**)&_d_input_tensor, 1 * 3 * 640 * 640 * sizeof(float));
        cudaStreamCreate(&_stream);
        InfoL << "Initialized TensorRT Engine with " << model_path;
    } else {
        ErrorL << "Failed to initialize TensorRT Engine!";
        _trt = nullptr;
    }
}

// 核心多线程推理与画框循环
void AIFilter::aiProcessLoop() {
    while (_running) {
        AVFrame* hw_frame = nullptr;
        {
            std::lock_guard<std::mutex> lck(_mutex);
            if (!_frame_queue.empty()) {
                hw_frame = _frame_queue.front();
                _frame_queue.pop_front();
            }
        }

        if (!hw_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // 收到空帧代表退出
        if (!hw_frame->data[0]) {
            av_frame_free(&hw_frame);
            break;
        }
        
        // --- 串联推理与画框的核心代码 (零拷贝流水线) ---
        if (_trt && hw_frame->format == AV_PIX_FMT_CUDA) {
            // 1. 获取 FFmpeg 硬件解码出来的 NV12 显存指针
            uint8_t* d_y = hw_frame->data[0];
            uint8_t* d_uv = hw_frame->data[1];
            int src_w = hw_frame->width;
            int src_h = hw_frame->height;
            int src_pitch = hw_frame->linesize[0];

            // 2. CUDA 前处理 (NV12 -> NCHW RGB Float32)
            ai::nv12_to_nchw_rgb_letterbox(
                d_y, d_uv, src_w, src_h, src_pitch, 
                _d_input_tensor, 640, 640, _stream
            );

            // 3. TensorRT 异步推理
            _trt->inferAsync(_d_input_tensor, _stream);

            // 4. 同步获取检测结果
            float scale = std::min(640.0f / src_w, 640.0f / src_h);
            int pad_x = (640 - src_w * scale) / 2;
            int pad_y = (640 - src_h * scale) / 2;
            auto boxes = _trt->postProcessSync(_stream, scale, pad_x, pad_y);

            // 5. 如果检测到了目标，直接在显存中画框
            if (!boxes.empty()) {
                ai::draw_bboxes_nv12(d_y, d_uv, src_w, src_h, src_pitch, boxes, _stream);
                cudaStreamSynchronize(_stream); // 确保画完再送编码
            }
        }

        // 6. 送入硬件编码器 (NVENC)
        if (_encoder_ctx) {
            // 设置 PTS (简易设置，实际应与解码的 PTS 保持递增)
            hw_frame->pict_type = AV_PICTURE_TYPE_NONE; 
            
            int ret = avcodec_send_frame(_encoder_ctx, hw_frame);
            if (ret >= 0) {
                AVPacket *pkt = av_packet_alloc();
                while (avcodec_receive_packet(_encoder_ctx, pkt) == 0) {
                    // 将 pkt 打包为 H264Frame 发送回 ZLMediaKit
                    onEncodedPacket(pkt);
                    av_packet_unref(pkt);
                }
                av_packet_free(&pkt);
            }
        }
        
        // 释放解码帧
        av_frame_free(&hw_frame);
    }
}

// 将 NVENC 硬编出来的 H264 NALU 包装回 ZLMediaKit 的 Frame 对象并发布
void AIFilter::onEncodedPacket(AVPacket* pkt) {
    // 将 AVPacket 转为 H264Frame (无缓存，直接分发)
    auto frame = std::make_shared<H264FrameNoCacheAble>(
        (char*)pkt->data, pkt->size, pkt->dts, pkt->pts, 0
    );
    
    // 送入复用器，生成最终的 RTMP/RTSP 流供前端播放
    if (_muxer) {
        _muxer->inputFrame(frame);
    }
}

} // namespace mediakit