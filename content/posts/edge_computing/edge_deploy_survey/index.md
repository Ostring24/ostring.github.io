


# 边缘端部署

## 一、部署效果最好的边缘模型（按任务场景）

| 应用领域 | 模型名称 | 优势 | 设备适配 |
|----------|----------|------|-----------|
| **视觉分类/检测** | MobileNetV2/V3 | 高速、轻量，广泛适配 | MCU / NPU / GPU |
| | EfficientNet-Lite | 高精度低功耗 | Edge TPU / 手机 |
| | YOLOv5-Nano / YOLOv6-N | 轻量检测，部署广泛 | Jetson / RK3588 |
| | PP-YOLOe Tiny | 高精度低计算量（百度） | ARM + NPU |
| **语音识别 / 唤醒** | DS-CNN / Keyword Spotting Model (KWS) | 用于离线语音命令 | Cortex-M4 / -M7 |
| | Whisper Tiny / DistilWhisper | 多语种识别（适配 GPU/NPU） | Edge AI SoC |
| **多模态推理** | MobileSAM / TinySAM | 适用于边缘设备的图像分割 | Jetson / RK3588 / NPU |
| **自然语言处理（NLP）** | DistilBERT / TinyBERT / MobileBERT | 微调灵活，适合分类、意图识别 | 边缘服务器 / 高性能 MCU |
| **嵌入式场景（超低功耗）** | TinyML models（如 uTensor、TFLM） | 量化后仅占数百KB | STM32 / K210 / Cortex-M |

---

## 二、边缘部署效果最好的平台（硬件+软件一体）

| 平台 | 描述 | 推荐部署框架 | 代表芯片 |
|-------|------|----------------|------------|
| **NVIDIA Jetson 系列** | 强大 GPU + CUDA + TensorRT，适合视觉、机器人 | TensorRT, DeepStream | Jetson Nano / Orin / Xavier |
| **Google Coral** | 支持 Edge TPU（INT8 模型） | TensorFlow Lite + Edge TPU Compiler | Coral Dev Board |
| **Qualcomm QCS/QCM 系列** | Snapdragon AI 引擎（Hexagon DSP） | SNPE（Snapdragon SDK） | QCS610 / 8250 |
| **Rockchip RK3588 / RV1106** | 高性价比 NPU SoC（支持 INT8/FP16） | RKNN Toolkit / ONNX | RK3588 / RV1126 |
| **Kneron / Horizon Robotics** | 面向安防、车载，低功耗 AI 芯片 | Kneron SDK / BPU SDK | KL720 / 旭日X3 |
| **Sipeed Maix / Kendryte** | RISC-V + NPU（TinyML场景） | KPU, nncase, TFLM | K210 / K510 |
| **ARM Cortex-M 系列** | 超低功耗 MCU + TinyML | TensorFlow Lite Micro / CMSIS-NN | STM32, nRF52 |

---

## 三、部署工具链推荐（跨平台）

| 工具 | 适用模型 | 特点 |
|------|----------|------|
| TensorFlow Lite / Lite Micro | TFLite / TinyML | 支持量化，适配 Android、MCU、TPU |
| ONNX Runtime + NNAPI / DirectML | ONNX 模型 | 跨平台通用，适配 Edge AI 芯片 |
| TensorRT | NVIDIA 模型 | 高性能优化，支持 FP16/INT8 加速 |
| TVM / Apache Relay | 任意模型 | 编译优化，适配异构硬件 |
| OpenVINO | Intel 硬件 | 支持 CPU、VPU（Myriad X）、FPGA |
| Edge Impulse | KWS / TinyML 分类 | 可视化模型训练 + MCU 部署 |

---

## 四、部署案例示意

### 🎯 示例：人脸检测模型部署比较

| 硬件平台 | 使用模型 | 平均延迟 | 功耗 | 工具链 |
|----------|----------|---------|------|--------|
| Jetson Nano | YOLOv5s | ~30ms | 5-10W | TensorRT |
| RK3588 | PP-YOLOe Tiny | ~25ms | 3-5W | RKNN Toolkit |
| STM32F746 | DS-CNN KWS | ~50ms | 0.5W | TFLite Micro |

---

## 五、实践建议

1. **用大模型训练，用小模型部署**（知识蒸馏 + 量化）
2. 使用 TensorFlow Lite / ONNX 导出部署模型
3. 利用平台 SDK 工具链做 INT8 / FP16 优化
4. 结合 Edge Impulse、TVM 等工具生成边缘可执行模型
5. 持续关注芯片厂商的工具链更新（如 RKNN、SNPE、TensorRT 版本）

---

如你有特定应用场景（如安防、医疗穿戴、工业边缘设备），可进一步定制推荐模型和部署策略。


# 📚 边缘智能 / AIoT 领域高质量 Survey 论文汇总

| 序号 | 论文标题 | 作者 | 发布时间 | 下载链接 | 主题与亮点 |
|------|---------|------|----------|-----------|------------|
| 1 | ^[**Empowering Edge Intelligence: A Comprehensive Survey on On‑Device AI Models**]({"attribution":{"attributableIndex":"0-1"}}) | ^[Xubin Wang 等]({"attribution":{"attributableIndex":"0-2"}}) | ^[2025‑03‑08]({"attribution":{"attributableIndex":"0-3"}}) | ^[[PDF (arXiv)](https://arxiv.org/pdf/2503.06027)]({"attribution":{"attributableIndex":"0-4"}}) | ^[系统梳理 on-device AI 核心难点：模型压缩、预处理、硬件加速与基础模型影响]({"attribution":{"attributableIndex":"0-5"}})  [oai_citation:0‡arxiv.org](https://arxiv.org/abs/2503.06027?utm_source=chatgpt.com) |
| 2 | **Optimizing Edge AI: A Comprehensive Survey on Data, Model, and System Strategies** | Xubin Wang & Weijia Jia | 2025‑01‑04 | [PDF (arXiv)](https://arxiv.org/pdf/2501.03265) | 提出“数据-模型-系统”三元优化框架，覆盖清洗、量化、推理加速 |
| 3 | **On Accelerating Edge AI: Optimizing Resource‑Constrained Environments** | Jacob Sander 等 | 2025‑01‑25 | [PDF (arXiv)](https://arxiv.org/abs/2501.15014) | 聚焦深度模型剪枝、NAS、编译框架（TVM/TensorRT/OpenVINO） |
| 4 | **Onboard Optimization and Learning: A Survey** | Monirul Islam Pavel 等 | 2025‑05‑07 | [PDF (arXiv)](https://arxiv.org/abs/2505.08793) | 重点关注边缘设备实时**在线训练**与推理：适应性、协作学习、安全性 |
