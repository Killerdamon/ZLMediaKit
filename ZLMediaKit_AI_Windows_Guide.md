# Windows 环境下 ZLMediaKit 高性能 C++ AI 推理架构部署指南

在 Windows 下编译并运行带有 **FFmpeg NVDEC/NVENC + CUDA + TensorRT** 的 ZLMediaKit 是一项极具挑战的任务，主要难点在于 MSVC 编译器、各种预编译库的依赖管理，以及 CUDA 工具链的配置。

以下是专为 Windows (Visual Studio 2019/2022) 打造的部署路线图：

## 1. 核心依赖准备 (Windows 版)

### 1.1 Visual Studio & CMake
- 安装 **Visual Studio 2019 或 2022**，必须勾选“使用 C++ 的桌面开发”工作负载。
- 安装最新的 **CMake** (确保将其添加到系统 PATH)。

### 1.2 CUDA Toolkit & cuDNN
- 下载并安装 **CUDA Toolkit 11.8 或 12.1** (Windows EXE 安装包)。
- 下载对应的 **cuDNN**，将其解压后，把 `bin`、`include`、`lib` 目录下的文件复制到 CUDA Toolkit 的安装目录 (如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)。

### 1.3 TensorRT (Windows zip)
- 从 NVIDIA 官网下载 **TensorRT 8.6** 的 Windows zip 包。
- 解压到例如 `C:\TensorRT` 目录。
- **关键**：将 `C:\TensorRT\lib` 添加到 Windows 系统的环境变量 `PATH` 中，否则运行时会报找不到 `nvinfer.dll` 的错误。

### 1.4 FFmpeg (NVIDIA 硬件加速版)
在 Windows 下自己从源码编译带 CUDA 支持的 FFmpeg 非常困难（需要 MSYS2 + YASM/NASM）。
**强烈建议直接下载预编译好的带有硬件加速的 FFmpeg (GPL-Shared 版)**：
- 推荐下载地址：[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases) (选择带有 `win64-gpl-shared` 字样的包)。
- 解压到 `C:\FFmpeg`。
- 将 `C:\FFmpeg\bin` 加入系统 `PATH`。

### 1.5 OpenSSL (可选但推荐)
- 使用 `vcpkg` 安装，或者下载预编译的 [Win32 OpenSSL](https://slproweb.com/products/Win32OpenSSL.html)。

---

## 2. 修改 CMakeLists.txt 以适配 Windows

在 Windows 下，CMake 经常找不到 TensorRT 和 FFmpeg 的路径。您需要在项目的根目录 `CMakeLists.txt` 中显式指定这些路径。

在执行 CMake 时，使用以下方式传递路径参数：

```bat
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DENABLE_AI_INFERENCE=ON ^
    -DTensorRT_ROOT="C:\TensorRT" ^
    -DFFMPEG_ROOT="C:\FFmpeg"
```

---

## 3. Windows 下 C++ 代码的适配点

我们之前写的核心代码 (`AIFilter.cpp`, `cuda_utils.h` 等) 大部分是跨平台的，但在 Windows 下需要注意以下几点：

### 3.1 动态链接库导出 (DLL Export)
如果 ZLMediaKit 以 DLL 形式编译，AI 插件的类可能需要 `__declspec(dllexport)` 宏修饰，但如果您将 AI 代码直接放入 ZLMediaKit 的 `src` 目录下静态编译，则无需修改。

### 3.2 CUDA 文件编译 (`.cu`)
MSVC 编译 CUDA 文件需要配置 `nvcc` 的特定参数。
在 `src/CMakeLists.txt` 中，我们需要为 MSVC 忽略一些严格的警告，并设置 C++ 标准：

```cmake
if(MSVC AND ENABLE_AI_INFERENCE)
    set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/Extension/yolo_preprocess.cu PROPERTIES COMPILE_FLAGS "-Xcompiler /wd4819")
endif()
```

### 3.3 TensorRT Logger
在 Windows 下，`std::cout` 可能会在没有控制台的守护进程模式下报错。建议将 `TrtYolo.cpp` 中的 `gLogger` 输出桥接到 ZLMediaKit 自带的 `InfoL` 宏上：

```cpp
#include "Util/logger.h"
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            toolkit::WarnL << "[TensorRT] " << msg;
    }
} gLogger;
```

---

## 4. 编译与运行

1. 打开 **x64 Native Tools Command Prompt for VS** (或者在 VS 中打开构建)。
2. 进入 `build` 目录执行：
   ```bat
   msbuild ZLMediaKit.sln /p:Configuration=Release /m
   ```
3. **极度重要**：将以下 DLL 文件拷贝到生成的 `MediaServer.exe` 同级目录下（或者确保它们在 `PATH` 中）：
   - `nvinfer.dll`, `nvparsers.dll` (来自 TensorRT/lib)
   - `avcodec-*.dll`, `avutil-*.dll`, `avformat-*.dll` (来自 FFmpeg/bin)
   - `cudart64_*.dll` (来自 CUDA/bin)
4. 运行 `MediaServer.exe`，如果控制台打印出：
   `Initialized FFmpeg HW Decoder (NVDEC)`
   `Initialized TensorRT Engine`
   说明 Windows 环境下 AI 硬件加速节点已成功挂载并运行！