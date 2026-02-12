## 实验日志

### 初步结果

资源：[wandb](https://wandb.ai/peterjin/Search-R1-open)

初步实验仅在自然问题（NQ）数据集上使用 PPO 方法进行，训练步数较少。

### v0.1

资源：[wandb](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.1)

我们将实验从 NQ 扩展到七个数据集，同时使用 PPO 和 GRPO 方法。实验仍在较少的训练步数下进行，但使用了较大的学习率预热比例。

### v0.2

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.2)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.2)、[论文](https://arxiv.org/abs/2503.09516)

我们修复了多个 bug，包括[检索标记遮蔽](https://github.com/PeterGriffinJin/Search-R1/pull/21)和 [GRPO 样本索引](https://github.com/PeterGriffinJin/Search-R1/commit/9ec2fa9892fbf0315d0c67b4dc08ae8f6cf5f378)。

前者可以大幅提高强化学习训练的稳定性。然后我们调整了训练脚本，增加训练步数并降低学习率预热比例，以获得更好的性能，并在不同规模的大型语言模型（3B、7B、14B）上进行实验。

### v0.3

资源：[wandb](https://wandb.ai/peterjin/Search-R1-v0.3)、[文档](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa)、[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/scripts/nq_hotpotqa/v0.3)、[论文](https://arxiv.org/abs/2505.15117)

我们对以下方面进行了研究：
- (1) 奖励设计
- (2) 大型语言模型骨干网络
- (3) 搜索引擎

- **奖励设计**
  - 格式奖励
  - 中间检索奖励
- **大型语言模型骨干网络**
  - 模型类型（如通用大型语言模型或推理型大型语言模型）
  - 模型规模（3B/7B/14B/32B）
- **搜索引擎**
  - 强化学习训练动态
  - 推理时的泛化能力
- **数据规模**

详情请参见[论文](https://arxiv.org/abs/2505.15117)。
