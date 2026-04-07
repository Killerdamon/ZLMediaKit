//
// AIPluginInjector.h
// 用于拦截 ZLMediaKit PlayerProxy 的流，挂载 AIFilter
// 
#pragma once

#include "Player/PlayerProxy.h"
#include "Extension/AIFilter.h"

namespace mediakit {

class AIPluginInjector {
public:
    // 工厂方法：拉起一路带 AI 处理的拉流代理
    static PlayerProxy::Ptr createAIProxy(
        const std::string& vhost,
        const std::string& app,
        const std::string& stream_id,
        const std::string& pull_url,
        const std::string& model_path) {
        
        // 1. 创建拉流代理
        auto proxy = std::make_shared<PlayerProxy>(vhost, app, stream_id, false, false);
        
        // 2. 创建核心 AI 滤镜
        auto ai_filter = std::make_shared<AIFilter>(stream_id, model_path);
        
        // 3. 注册回调：当拉流成功并解析出 Track 时，注入 AI Filter
        proxy->setOnPlayResult([ai_filter, proxy, pull_url](const toolkit::SockException &ex) {
            if (!ex) {
                toolkit::InfoL << "Play AI stream success: " << pull_url;
                
                // 将原流的 Track (如 H264Track, AACTrack) 添加到 AI 过滤器
                auto tracks = proxy->getTracks(false);
                for (auto& track : tracks) {
                    ai_filter->addTrack(track);
                    
                    // 拦截原流数据，送入 AIFilter 进行硬解和推理
                    track->addDelegate([ai_filter](const Frame::Ptr &frame) {
                        ai_filter->inputFrame(frame);
                        return true; // 返回 true 继续原流分发
                    });
                }
            } else {
                toolkit::WarnL << "Play AI stream failed: " << pull_url << " " << ex.what();
            }
        });
        
        // 4. 开始拉流
        proxy->play(pull_url);
        return proxy;
    }
};

} // namespace mediakit