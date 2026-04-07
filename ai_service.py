import cv2
import time
import argparse
import paho.mqtt.client as mqtt
import json
import base64
import os
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------
# ZLMediaKit 外挂式 AI 推理示例服务 (Python + OpenCV + YOLOv8 + MQTT)
# 
# 工作原理:
# 1. 从 ZLMediaKit 的 RTSP/RTMP/HTTP-FLV 地址拉取实时视频流。
# 2. 逐帧解码，将 RGB 图像送入 YOLO 模型进行目标检测(人员、车辆等)。
# 3. 使用 OpenCV 在画面上绘制检测框 (Bounding Box)、置信度、类别。
# 4. 如果检测到特定目标(如人)，触发事件：
#    - 截图并编码为 Base64。
#    - 通过 MQTT 客户端将事件信息(含截图)推送到业务系统。
# 5. (可选) 将画好框的图像通过 FFmpeg 管道重新编码推流回 ZLMediaKit。
# ----------------------------------------------------------------------

# MQTT 配置参数
MQTT_BROKER = "127.0.0.1"
MQTT_PORT = 1883
MQTT_TOPIC_ALARM = "ai/alarm/yolo"

def on_mqtt_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[MQTT] Connected to MQTT Broker successfully.")
    else:
        print(f"[MQTT] Failed to connect, return code {rc}")

def publish_alarm(mqtt_client, camera_id, class_name, confidence, image_base64):
    """
    发布报警事件到 MQTT Broker
    """
    if not mqtt_client:
        return
    
    payload = {
        "timestamp": int(time.time() * 1000),
        "camera_id": camera_id,
        "event_type": "object_detected",
        "object": class_name,
        "confidence": round(float(confidence), 3),
        "image_base64": image_base64
    }
    
    mqtt_client.publish(MQTT_TOPIC_ALARM, json.dumps(payload))
    print(f"[ALARM] Published alarm for {class_name} (conf: {confidence:.2f}) on camera {camera_id}")

def point_in_polygon(point, polygon):
    """
    使用 OpenCV 判断点是否在多边形内
    """
    if not polygon:
        return True # 如果没有定义多边形，则认为所有点都在区域内
    
    # cv2.pointPolygonTest 返回值:
    # > 0: 点在多边形内
    # = 0: 点在多边形边缘上
    # < 0: 点在多边形外
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), point, False) >= 0

def run_ai_service(args):
    # 1. 初始化 MQTT 客户端
    mqtt_client = mqtt.Client(client_id=f"ai_service_{args.camera_id}")
    mqtt_client.on_connect = on_mqtt_connect
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[Warning] MQTT Connection failed: {e}. Running without MQTT.")
        mqtt_client = None

    # 2. 加载 YOLO 模型 (这里以 YOLOv8n 为例，支持 ONNX / TensorRT 导出模型加速)
    # 你可以替换为 yolov8n.engine (TensorRT) 或 yolov8n.onnx (ONNXRuntime) 以获得极致性能
    print(f"[AI] Loading YOLO model: {args.model}")
    # 强制启用跟踪器参数如果传入
    if args.track:
        print("[AI] Target Tracking is enabled (using default tracker).")
        
    model = YOLO(args.model)

    # 3. 初始化视频流读取 (从 ZLMediaKit 拉流)
    print(f"[Stream] Connecting to input stream: {args.input_url}")
    cap = cv2.VideoCapture(args.input_url)
    
    if not cap.isOpened():
        print(f"[Error] Failed to open stream: {args.input_url}")
        return

    # 获取视频原始参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps == 0: fps = 25 # 默认兜底
    print(f"[Stream] Video Info: {width}x{height} @ {fps}fps")

    # 解析多边形布防区域
    polygon_points = []
    if args.polygon:
        try:
            # 格式例如: "100,100;500,100;500,500;100,500"
            points_str = args.polygon.split(';')
            for p in points_str:
                x, y = map(int, p.split(','))
                polygon_points.append((x, y))
            print(f"[AI] Polygon region defined: {polygon_points}")
        except Exception as e:
            print(f"[Error] Failed to parse polygon points: {e}")

    # 4. (可选) 配置 FFmpeg 推流管道 (推流回 ZLMediaKit)
    out_pipe = None
    if args.output_url:
        print(f"[Stream] Setting up output stream to: {args.output_url}")
        # 使用 FFmpeg 命令行，通过 stdin 接收原始 BGR 帧，硬编码(如 h264_nvenc)后推流
        # 如果没有显卡，可以将 h264_nvenc 换为 libx264
        ffmpeg_cmd = [
            'ffmpeg',
            '-y', '-an',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}",
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264', # CPU编码: libx264; GPU编码: h264_nvenc (Nvidia) / h264_qsv
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'rtsp', # 如果是 RTMP，这里换成 flv
            args.output_url
        ]
        import subprocess
        out_pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

    # 报警节流控制 (防止同一目标疯狂报警)
    last_alarm_time = 0
    alarm_cooldown = 5.0 # 秒

    # 5. 主循环：读取、推理、绘制、推流
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Stream] End of stream or connection lost. Reconnecting...")
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(args.input_url)
                continue
            
            # --- 绘制多边形布防区域 ---
            if polygon_points:
                cv2.polylines(frame, [np.array(polygon_points, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

            # --- YOLO 推理 (包含跟踪功能如果启用) ---
            # stream=True 适合视频流，半精度半张量
            if args.track:
                results = model.track(source=frame, persist=True, conf=args.conf, iou=args.iou, verbose=False, device=args.device)
            else:
                results = model.predict(source=frame, conf=args.conf, iou=args.iou, verbose=False, device=args.device)
            
            person_detected = False
            highest_conf = 0.0
            
            # --- 解析并绘制结果 ---
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # 坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 类别
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]
                    # 置信度
                    conf = float(box.conf[0])
                    
                    # 计算目标中心点
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # 多边形布防检测
                    in_zone = point_in_polygon((center_x, center_y), polygon_points)
                    
                    # 如果启用了跟踪，获取 Track ID
                    track_id = ""
                    if args.track and box.id is not None:
                        track_id = f" ID:{int(box.id[0])}"
                    
                    # 根据是否在区域内决定颜色 (绿色代表安全/未触发，红色代表报警/在区域内)
                    color = (0, 0, 255) if (in_zone and polygon_points) else (0, 255, 0)
                    
                    # 绘制边界框和中心点
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (center_x, center_y), 4, color, -1)
                    
                    label = f"{class_name}{track_id} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 业务逻辑：检测到人且在布防区域内
                    if (cls_id == 0 or class_name == 'person') and in_zone:
                        person_detected = True
                        if conf > highest_conf:
                            highest_conf = conf

            # --- 报警与截图逻辑 ---
            current_time = time.time()
            if person_detected and (current_time - last_alarm_time > alarm_cooldown):
                # 编码图像为 JPEG
                _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 推送 MQTT
                publish_alarm(mqtt_client, args.camera_id, "person_in_zone", highest_conf, img_base64)
                last_alarm_time = current_time

            # --- 帧率统计 (OSD 绘制) ---
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 1.0:
                current_fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()
                cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # --- 推流或本地显示 ---
            if out_pipe:
                # 写入 FFmpeg 管道
                out_pipe.stdin.write(frame.tobytes())
            
            # 如果在带界面的系统上，可以取消注释以本地查看
            # cv2.imshow("AI Detection", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    except KeyboardInterrupt:
        print("[Info] Interrupted by user.")
    finally:
        print("[Info] Cleaning up resources...")
        cap.release()
        if out_pipe:
            out_pipe.stdin.close()
            out_pipe.wait()
        if mqtt_client:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZLMediaKit External AI Inference Service")
    parser.add_argument("--camera_id", type=str, default="cam_001", help="Unique ID for this camera stream")
    parser.add_argument("--input_url", type=str, required=True, help="Input stream URL from ZLMediaKit (e.g., rtsp://127.0.0.1/live/test)")
    parser.add_argument("--output_url", type=str, default="", help="Output stream URL back to ZLMediaKit (e.g., rtsp://127.0.0.1/live/ai_test)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path (pt, onnx, engine)")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IOU threshold")
    parser.add_argument("--device", type=str, default="", help="Device to run on (e.g., 'cpu', '0' for CUDA GPU)")
    
    # 新增高级功能参数
    parser.add_argument("--track", action="store_true", help="Enable object tracking (ByteTrack/DeepSORT)")
    parser.add_argument("--polygon", type=str, default="", help="Define a polygon zone, format: 'x1,y1;x2,y2;x3,y3...'")
    
    args = parser.parse_args()
    run_ai_service(args)
