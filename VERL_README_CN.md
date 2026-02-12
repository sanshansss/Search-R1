# veRL：火山引擎大型语言模型强化学习框架

<h1 style="text-align: center;">veRL：火山引擎强化学习（Volcano Engine Reinforcement Learning）</h1>

veRL 是一个灵活、高效且生产就绪的大型语言模型强化学习训练框架。

veRL 是 **[HybridFlow：灵活高效的 RLHF 框架](https://arxiv.org/abs/2409.19256v2)** 论文的开源版本。

veRL 具有灵活性和易用性：

- **轻松扩展多种强化学习算法**：混合编程模型结合了单控制器和多控制器范式的优势，能够灵活表示和高效执行复杂的后训练数据流。允许用户用几行代码构建强化学习数据流。

- **与现有大型语言模型基础设施的无缝集成，通过模块化 API**：解耦计算和数据依赖，实现与现有大型语言模型框架（如 PyTorch FSDP、Megatron-LM 和 vLLM）的无缝集成。此外，用户可以轻松扩展到其他大型语言模型训练和推理框架。

- **灵活的设备映射**：支持将模型放置在不同的 GPU 集合上，实现高效的资源利用和不同集群规模的可扩展性。

- 轻松集成流行的 HuggingFace 模型

veRL 具有高效性：

- **最先进的吞吐量**：通过无缝集成现有的最先进大型语言模型训练和推理框架，veRL 实现了高生成和训练吞吐量。

- **通过 3D-HybridEngine 实现高效的 actor 模型重分片**：消除内存冗余，显著减少训练和生成阶段之间的通信开销。

<p align="center">
| <a href="https://verl.readthedocs.io/en/latest/index.html"><b>文档</b></a> | <a href="https://arxiv.org/abs/2409.19256v2"><b>论文</b></a> | <a href="https://join.slack.com/t/verlgroup/shared_invite/zt-2w5p9o4c3-yy0x2Q56s_VlGLsJ93A6vA"><b>Slack</b></a> | <a href="https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/WeChat.JPG"><b>微信</b></a> | 
</p>

## 最新消息

- [2024/12] 团队在 NeurIPS 2024 上展示了[后训练大型语言模型：从算法到基础设施](https://neurips.cc/Expo/Conferences/2024/workshop/100677)。[幻灯片](https://github.com/eric-haibin-lin/verl-data/tree/neurips)和[视频](https://neurips.cc/Expo/Conferences/2024/workshop/100677)可用。
- [2024/10] veRL 在 Ray Summit 上展示。[YouTube 视频](https://www.youtube.com/watch?v=MrhMcXkXvJU&list=PLzTswPQNepXntmT8jr9WaNfqQ60QwW7-U&index=37)可用。
- [2024/08] HybridFlow (verl) 被 EuroSys 2025 接受。

## 主要特性

- **FSDP** 和 **Megatron-LM** 用于训练
- **vLLM** 和 **TGI** 用于 rollout 生成，**SGLang** 支持即将推出
- HuggingFace 模型支持
- 监督微调
- 奖励模型训练
- 基于人类反馈的强化学习（PPO）
- flash-attention 集成，序列打包
- 可扩展至 70B 模型和数百个 GPU
- 使用 wandb 和 mlflow 进行实验跟踪

## 快速入门

查看此 [Jupyter Notebook](https://github.com/volcengine/verl/tree/main/examples/ppo_trainer/verl_getting_started.ipynb)，开始使用单个 24GB L4 GPU 进行 PPO 训练（**免费** GPU 配额由 [Lighting Studio](https://lightning.ai/hlin-verl/studios/verl-getting-started) 提供）！

**快速开始：**
- [安装](https://verl.readthedocs.io/en/latest/start/install.html)
- [快速开始](https://verl.readthedocs.io/en/latest/start/quickstart.html)

**逐步运行 PPO 示例：**
- 数据和奖励准备
  - [准备后训练数据（Parquet 格式）](https://verl.readthedocs.io/en/latest/preparation/prepare_data.html)
  - [为数据集实现奖励函数](https://verl.readthedocs.io/en/latest/preparation/reward_function.html)
- 理解 PPO 示例
  - [PPO 示例架构](https://verl.readthedocs.io/en/latest/examples/ppo_code_architecture.html)
  - [配置说明](https://verl.readthedocs.io/en/latest/examples/config.html)
  - [运行 GSM8K 示例](https://verl.readthedocs.io/en/latest/examples/gsm8k_example.html)

**可复现的算法基线：**
- [PPO](https://verl.readthedocs.io/en/latest/experiment/ppo.html)

**代码说明和高级用法（扩展）：**
- PPO 训练器和 Worker
  - [PPO Ray 训练器](https://verl.readthedocs.io/en/latest/workers/ray_trainer.html)
  - [PyTorch FSDP 后端](https://verl.readthedocs.io/en/latest/workers/fsdp_workers.html)
  - [Megatron-LM 后端](https://verl.readthedocs.io/en/latest/index.html)
- 高级用法和扩展
  - [Ray API 设计教程](https://verl.readthedocs.io/en/latest/advance/placement.html)
  - [扩展到其他强化学习（人工反馈）算法](https://verl.readthedocs.io/en/latest/advance/dpo_extension.html)
  - [使用 FSDP 后端添加模型](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)
  - [使用 Megatron-LM 后端添加模型](https://verl.readthedocs.io/en/latest/advance/megatron_extension.html)

## 引用和致谢

如果本项目对您有帮助，请引用：
- [HybridFlow：灵活高效的 RLHF 框架](https://arxiv.org/abs/2409.19256v2)
- [通过近端策略优化训练大型语言模型进行代码生成的框架](https://i.cs.hku.hk/~cwu/papers/gmsheng-NL2Code24.pdf)

```tex
@article{sheng2024hybridflow,
  title   = {HybridFlow：灵活高效的 RLHF 框架},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv 预印本 arXiv: 2409.19256}
}
```

veRL 的设计受到了 Nemo-Aligner、Deepspeed-chat 和 OpenRLHF 的启发。该项目得到 Anyscale、ByteDance、LMSys.org、上海人工智能实验室、清华大学、加州大学伯克利分校、加州大学洛杉矶分校、伊利诺伊大学香槟分校和香港大学的支持和采用。

## 使用 veRL 的出版物

- [通过直接 Q 函数优化增强语言模型的多步推理能力](https://arxiv.org/abs/2410.09302)
- [大型语言模型的火热初始化与规则执行采样](https://arxiv.org/abs/2410.21236)
- [通过隐式奖励进行过程强化学习](https://github.com/PRIME-RL/PRIME/)

## 招聘

我们正在招聘！如果您对大型语言模型推理/多模态对齐领域的实习或全职机会感兴趣，请给我们发送[电子邮件](mailto:haibin.lin@bytedance.com)。
