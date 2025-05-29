


# Nvidia Fusion 
Jenson Huang 今天抛出了绣球，接下来第一个问题：其他厂商家的黄花闺女能否看的上。
另一个问题：如果有一天真的成家了，亲家之间是否会闹的鸡飞狗跳？
这颗绣球就是今天刚刚在computex 上发布的 Nvidia Fusion。英伟达发布了一系列围绕数据中心和企业级 AI 计划的公告，其中包括新推出的 NVLink Fusion 计划。

![](nvlink_fusion.png)


## 先回答第一个问题
从公开的消息看，目前接下这颗绣球的有几家厂商：
>英伟达已为该计划聚集了众多合作伙伴，包括高通和富士通，它们将把这项技术集成到各自的 CPU 中。NVLink Fusion 还将扩展至定制 AI 加速器领域，因此英伟达已吸引包括 Marvell、联发科在内的多家芯片合作伙伴，以及芯片软件设计公司新思科技（Synopsys）和楷登电子（Cadence）加入 NVLink Fusion 生态系统。

>高通最近证实，其正在将自有定制服务器 CPU 推向市场，尽管细节仍不明确，但该公司与 NVLink 生态系统的合作将使其新 CPU 能够搭乘英伟达快速扩张的 AI 生态系统的顺风车。

>富士通也一直在致力于将其搭载 3D 堆叠 CPU 核心和内存的 144 核 Monaka CPU 推向市场。富士通 CTO Vivek Mahajan 表示：“富士通的下一代处理器 FUJITSU-MONAKA 是一款基于 Arm 架构的 2 纳米 CPU，旨在实现极高的能效。将我们的技术与英伟达架构直接连接，标志着我们在通过世界领先的计算技术推动 AI 发展的愿景中迈出了重要一步，为新型可扩展、自主且可持续的 AI 系统铺平了道路。”


从目前来看，Nvlink 可撬动的生态还只局限在Die2Die 的范围，在Die2Die 互联的选择上，Nvlink 不是唯一选择，另外也不是最优解。联发科已经和NVdia 有过深入合作，之前的`GB10`现在看来已经试水了Nvlink fusion生态合作模式，由联发科基于Arm SoC CPU 的功底，结合Blackwell 的GPU 通过Nvlink 互联。另外一家Marvell，基本是全开放生态，各种联盟都来者不拒，本就是CSP chip Maker，工具箱越丰富越好。说白了，通过联发科的合作先打个窝，吸引其他厂商，在D2D 范围内还处于刚起步的状态。

>NVIDIA GB10 Grace Blackwell 超级芯片：应用于 NVIDIA 的个人 AI 超级计算机 NVIDIA® Project DIGITS。由 Blackwell GPU 和 Grace CPU 组成，其中 Grace CPU 有 20 个 Arm 核心，配备 128GB LPDDR5X 内存和 4TB NVMe SSD，能运行超 2000 亿参数的大语言模型，两台 Project DIGITS 叠加可支持处理 4050 亿参数的大语言模型。由英伟达与联发科联合打造。联发科在基于 Arm 架构的 SoC 设计领域经验丰富，为 GB10 带来出色的能效和连接性能 ，助力其适用于桌面环境

再扩展到AI 加速器领域，这里指的就是ASIC 之间的C2C（Chip2Chip） 互联，也是老黄这次极力推销的。这里只从技术的角度来分析，在C2C 互联范围，可能遇到哪些问题。

## 再回答第二个问题


```shell
┌──────────────┐
│ NVLink 协议   │ ← 高级事务层（共享内存，一致性，GPU atomic）
├──────────────┤
│ NVLink Link  │  ← 可靠链路协议（flow control, retry）
├──────────────┤
│ Nvlink PHY   │  ← PAM4 PHY，支持 D2D/C2C 高速通信
└──────────────┘
```

NVLink 是封闭生态
- NVLink 并非开放标准，不像 PCIe、CXL 或 Ethernet 等是 industry standard。
- NVIDIA 没有公开 NVLink 的完整物理/协议/IP 规范。
- 所以目前 只有 NVIDIA 自家的 GPU（如 H100）、CPU（Grace）、NVSwitch、DPU（BlueField）等芯片可以原生支持 NVLink。
- 第三方芯片厂商无法直接适配 NVLink，除非与 NVIDIA 达成深度合作。

对比开放标准，UCIE 和UALINK，其中UCIE 主要用于Die2Die 互联，UALINK 主要用于C2C 互联，而NVLink 则是两者的结合。

```shell
UCIE
┌─────────────────────┐
│ Protocol Adapter     │ ← 支持 PCIe, CXL, AMBA AXI 等协议适配
├─────────────────────┤
│ Die-to-Die Adapter   │ ← 统一的物理接口 + 接收控制协议（类似 PIPE）
├─────────────────────┤
│ PHY                  │ ← 支持 NRZ/PAM4，Short/Long Reach，兼容多制程封装
└─────────────────────┘

```


PHY 的层面，都是通过差分线来实现高速串行通信SerDes，Nvlink 与其他标准的差别只在于调制方式，目前包括PCIe 1.0 - PCIe 5.0 PHY采用不归零调制（Non-Return-to-Zero，NRZ），调制方式简单，信息编码效率较低，物理层模型信号实现难度较低，只有到PCIE-Gen6 才开始采用PAM4 调制方式，PAM4 调制方式是一种基于相位调制（Phase Modulation，PM）的调制方式，它将数字信号转换为模拟信号，从而实现高速串行通信。PAM4 调制方式具有较高的信息编码效率和抗干扰能力，适用于高速数据传输和通信领域。在物理层由于Nvlink 的封闭，要和PCIE 阵营在phy 层面首先要进行适配，SerDes 工艺成熟度信号完整性要求高（如低抖动、低 BER）。

链路层方面，需要支持 NVLink 的 packet format、ack/nack、重传机制实现 flow control，在这个层级主要是对齐封包，做好协议前向和后向兼容，每一项单独开一份协议spec 都不为过。

事务层协议方面，需要支持统一虚拟地址空间（UVA）实现 cache coherence（如 NVIDIA 的 MESI-like 协议）支持 memory atomics、RMA-like 机制。

最后的最后，交付到cuda 工程师手中，目前NVIDIA 的驱动与 CUDA 都是闭源，任何第三方硬件即使“连上了”，也无法在软件上被 CUDA 识别和使用。

# 最后
对Nvidia 走这一步棋其实有点迷惑，三大护城河：NvLink 互联/CUDA/GPU 的传统就是闭源，如果NvLink 开了一道口子，是否意味着其感受到了威胁，是H 家的CloudMatrix 384 光互连，还是UALink 联盟（虽然好久没有消息了），未知。从今天的发布会来看，还只是个引子，另外UALink 和UCIE 的开放联盟能否经历的起这个考验，抵抗教主的这番引诱，我们后续拭目以待。





