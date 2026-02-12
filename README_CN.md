# Search-R1：通过强化学习训练大型语言模型进行推理和搜索引擎调用

<div align="center">
  <img src="https://raw.githubusercontent.com/PeterGriffinJin/Search-R1/main/public/logo.png" alt="logo" width="300"/>
</div>

<p align="center">
  <a href="https://arxiv.org/abs/2503.09516">
    <img src="https://img.shields.io/badge/Paper1-blue?style=for-the-badge" alt="Button1"/>
  </a>
  <a href="https://arxiv.org/abs/2505.15117">
    <img src="https://img.shields.io/badge/Paper2-green?style=for-the-badge" alt="Button2"/>
  </a>
  <a href="https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5">
    <img src="https://img.shields.io/badge/Resources-orange?style=for-the-badge" alt="Button3"/>
  </a>
  <a href="https://x.com/BowenJin13/status/1895544294473109889">
    <img src="https://img.shields.io/badge/Tweet-red?style=for-the-badge" alt="Button4"/>
  </a>
  <a href="https://wandb.ai/peterjin/Search-R1-v0.2">
    <img src="https://img.shields.io/badge/Logs-purple?style=for-the-badge" alt="Button5"/>
  </a>
</p>

**Search-R1** 是一个强化学习框架，专门用于训练**推理与搜索交织的大型语言模型**——即能够以协调方式学习推理和调用工具（如搜索引擎）的语言模型。

基于 [veRL](https://github.com/volcengine/verl) 构建，Search-R1 扩展了 **DeepSeek-R1(-Zero)** 的思想，将搜索引擎访问交织其中，并提供了一个完全开源的强化学习训练管道。它可以作为 **OpenAI DeepResearch** 的替代开源解决方案，实现工具增强型大型语言模型推理的研究与开发。

我们支持不同的强化学习方法（如 PPO、GRPO、REINFORCE）、不同的大型语言模型（如 llama3、Qwen2.5 等）以及不同的搜索引擎（如本地稀疏/稠密检索器和在线搜索引擎）。

论文：[链接1](https://arxiv.org/pdf/2503.09516)、[链接2](https://arxiv.org/abs/2505.15117)；模型和数据：[链接](https://huggingface.co/collections/PeterJinGo/search-r1-67d1a021202731cb065740f5)；Twitter 讨论：[链接](https://x.com/BowenJin13/status/1895544294473109889)；完整实验日志：[预实验](https://wandb.ai/peterjin/Search-R1-open)、[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)、[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)。关于这些日志和方法的详细信息请参见[此处](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/experiment_log.md)。

![single-turn](public/main.png)

## 最新消息

- [2025.10] Search-R1 被 Thinking Machines Lab 的首款产品 [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook) 采用！详情：[文档](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/tool_use/search)
- [2025.7] Search-R1 现已支持 [SkyRL](https://github.com/NovaSky-AI/SkyRL)！详细说明：[代码](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-train/examples/search)、[文档](https://novasky-ai.notion.site/skyrl-searchr1)
- [2025.6] Search-R1 已集成到最新版本的 veRL 中，可以利用其最新功能！详细说明：[veRL](https://verl.readthedocs.io/en/latest/sglang_multurn/search_tool_example.html)、[英文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like.md)、[中文文档](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/tool_examples/verl-multiturn-searchR1-like_ZH.md)
- [2025.5] 第二篇[论文](https://arxiv.org/abs/2505.15117)发布，详细实证研究附日志：[v0.3](https://wandb.ai/peterjin/Search-R1-v0.3)
- [2025.4] 我们支持 30B+ 大型语言模型的[多节点](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/multinode.md)训练！
- [2025.4] 我们支持不同的[搜索引擎](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)，包括稀疏本地检索器、带有 ANN 索引的稠密本地检索器和在线搜索引擎！
- [2025.3] 首篇 Search-R1 [论文](https://arxiv.org/pdf/2503.09516)发布，附日志：[v0.1](https://wandb.ai/peterjin/Search-R1-nq_hotpotqa_train)、[v0.2](https://wandb.ai/peterjin/Search-R1-v0.2)
- [2025.2] 我们开源了 Search-R1 代码库，附[初步结果](https://wandb.ai/peterjin/Search-R1-open)

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [初步结果](#初步结果)
- [推理](#推理)
- [使用自己的数据集](#使用自己的数据集)
- [使用自己的搜索引擎](#使用自己的搜索引擎)
- [功能特性](#功能特性)
- [致谢](#致谢)
- [引用](#引用)

## 安装

### Search-r1 环境

```bash
conda create -n searchr1 python=3.9
conda activate searchr1
# 安装 torch [或者可以跳过此步骤，让 vllm 为您安装正确版本]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# 安装 vllm
pip3 install vllm==0.6.3 # 或者可以安装 0.5.4、0.4.2 和 0.3.1

# verl
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation
pip install wandb
```

### 检索器环境（可选）

如果您想将本地检索器作为搜索引擎调用，可以按以下方式安装环境。（建议使用单独的环境。）

```bash
conda create -n retriever python=3.10
conda activate retriever

# 建议使用 conda 安装 torch 以支持 faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## 安装 GPU 版本 faiss 以保证高效的强化学习 rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API 功能
pip install uvicorn fastapi
```

## 快速开始

使用 e5 作为检索器和 wikipedia 作为语料库，在 NQ 数据集上训练推理 + 搜索大型语言模型。

(1) 下载索引和语料库。
```bash
save_path=/要保存的路径
python scripts/download.py --save_path $save_path
cat $save_path/part_* > $save_path/e5_Flat.index
gzip -d $save_path/wiki-18.jsonl.gz
```

(2) 处理 NQ 数据集。
```bash
python scripts/data_process/nq_search.py
```

(3) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(4) 运行强化学习训练（PPO），使用 Llama-3.2-3b-base。
```bash
conda activate searchr1
bash train_ppo.sh
```

## 初步结果

(1) 基础模型（llama3.2-3b-base）学习调用搜索引擎并获得性能提升。

![llama-3b](public/llama32-3b.png)

(2) 基础模型（Qwen2.5-7b-base）可以通过强化学习学习多轮搜索引擎调用和推理。

![multi-turn](public/multi-turn.png)

## 推理

#### 您可以使用自己的问题与训练好的 Search-R1 模型进行交互。

(1) 启动本地检索服务器。
```bash
conda activate retriever
bash retrieval_launch.sh
```

(2) 运行推理。
```bash
conda activate searchr1
python infer.py
```

您可以修改第 7 行的 ```question``` 为您感兴趣的问题。

## 使用自己的数据集

### 问答数据

对于每个问答样本，它应该是一个包含以下内容的字典：

```
data = {
        "data_source": 数据源,
        "prompt": [{
            "role": "user",
            "content": 问题,
        }],
        "ability": "fact-reasoning",
        "reward_model": {
            "style": "rule",
            "ground_truth": 答案
        },
        "extra_info": {
            'split': 数据集划分,
            'index': 索引,
        }
    }
```

您可以参考 ```scripts/data_process/nq_search.py``` 查看具体的数据处理示例。

### 语料库

建议将您的语料库制作成 jsonl 文件，每行（一个包含 "id" 键和 "contents" 键的字典）对应一篇文章。您可以参考 ```example/corpus.jsonl``` 查看示例。

"id" 键对应文章 ID，而 "contents" 键对应文章内容（'"' + 标题 + '"\n' + 正文）。

例如：
```
{"id": "0", "contents": "Evan Morris Evan L. Morris (1977年1月26日 - 2015年7月9日) 是 Genentech 及其母公司 Roche 在华盛顿的游说者。"}
...
{"id": "100", "contents": "三年后，当美国探险远征队组织起来探索地球未知地区时，Hale 还在本科阶段就被推荐加入。"}

```

**为您的语料库建立索引（可选）。**
如果您想将本地检索器作为搜索引擎使用，可以通过以下方式为您的语料库建立索引：
```
bash search_r1/search/build_index.sh
```
您可以将 ```retriever_name``` 和 ```retriever_model``` 更改为您感兴趣的现成检索器。

## 使用自己的搜索引擎

我们的代码库支持本地稀疏检索器（如 BM25）、本地稠密检索器（支持 GPU 平面索引和 CPU ANN 索引）以及在线搜索引擎（如 Google、Bing、Brave 等）。更多详情请参见[此处](https://github.com/PeterGriffinJin/Search-R1/tree/main/docs/retriever.md)。

主要理念是从主强化学习训练管道中单独启动本地或远程搜索引擎服务器。

大型语言模型可以通过调用搜索 API 来调用搜索引擎（例如 "http://127.0.0.1:8000/retrieve"）。

您可以参考 ```search_r1/search/retriever_server.py``` 查看启动本地检索服务器的示例。

## 功能特性

- 支持本地稀疏检索器（如 BM25）。✅
- 支持本地稠密检索器（平面索引和 ANN 索引）。✅
- 支持 Google 搜索 / Bing 搜索 / Brave 搜索 API 等。✅
- 支持现成的神经重排序器。✅
- 支持不同的强化学习方法（如 PPO、GRPO、REINFORCE）。✅
- 支持不同的大型语言模型（如 llama3、Qwen2.5 等）。✅

## 致谢

Search-R1 的概念受到了 [Deepseek-R1](https://github.com/deepseek-ai/DeepSeek-R1) 和 [TinyZero](https://github.com/Jiayi-Pan/TinyZero/tree/main) 的启发。
其实施基于 [veRL](https://github.com/volcengine/verl) 和 [RAGEN](https://github.com/ZihanWang314/RAGEN/tree/main)。
我们真诚地感谢这些团队对开源研究和开发的贡献。

## 受 Search-R1 启发或采用的开源项目

- [DeepResearcher](https://github.com/GAIR-NLP/DeepResearcher)：通过真实环境中的强化学习扩展深度研究。[![代码](https://img.shields.io/github/stars/GAIR-NLP/DeepResearcher)](https://github.com/GAIR-NLP/DeepResearcher)
- [Multimodal-Search-R1](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)：激励多模态大型语言模型进行搜索。[![代码](https://img.shields.io/github/stars/EvolvingLMMs-Lab/multimodal-search-r1)](https://github.com/EvolvingLMMs-Lab/multimodal-search-r1)
- [OTC](https://arxiv.org/pdf/2504.14870)：通过强化学习实现最优工具调用。
- [ZeroSearch](https://github.com/Alibaba-NLP/ZeroSearch)：在不进行搜索的情况下激励大型语言模型的搜索能力。[![代码](https://img.shields.io/github/stars/Alibaba-NLP/ZeroSearch)](https://github.com/Alibaba-NLP/ZeroSearch)
- [IKEA](https://github.com/hzy312/knowledge-r1)：通过强化学习实现内部-外部知识协同推理，实现高效自适应搜索代理。[![代码](https://img.shields.io/github/stars/hzy312/knowledge-r1)](https://github.com/hzy312/knowledge-r1)
- [Scent of Knowledge](https://arxiv.org/abs/2505.09316)：通过信息觅食优化搜索增强推理。
- [AutoRefine](https://www.arxiv.org/pdf/2505.11277)：在思考中搜索和精炼。[![代码](https://img.shields.io/github/stars/syr-cn/AutoRefine)](https://github.com/syr-cn/AutoRefine)
- [O^2-Searcher](https://arxiv.org/pdf/2505.16582)：基于搜索的开放域开放性问答代理模型。[![代码](https://img.shields.io/github/stars/Acade-Mate/O2-Searcher)](https://github.com/Acade-Mate/O2-Searcher)
- [MaskSearch](https://arxiv.org/pdf/2505.20285)：增强代理搜索能力的通用预训练框架。[![代码](https://img.shields.io/github/stars/Alibaba-NLP/MaskSearch)](https://github.com/Alibaba-NLP/MaskSearch)
- [VRAG-RL](https://arxiv.org/abs/2505.22019)：基于视觉理解的检索增强生成。[![代码](https://img.shields.io/github/stars/Alibaba-NLP/VRAG)](https://github.com/Alibaba-NLP/VRAG)
- [R1-Code-Interpreter](https://arxiv.org/abs/2505.21668)：通过监督学习和强化学习训练大型语言模型进行代码推理。[![代码](https://img.shields.io/github/stars/yongchao98/R1-Code-Interpreter)](https://github.com/yongchao98/R1-Code-Interpreter)
- [R-Search](https://arxiv.org/abs/2506.04185)：通过多奖励强化学习增强大型语言模型的搜索推理能力。[![代码](https://img.shields.io/github/stars/QingFei1/R-Search)](https://github.com/QingFei1/R-Search)
- [StepSearch](https://arxiv.org/pdf/2505.15107)：通过逐步近端策略优化激发大型语言模型的搜索能力。[![代码](https://img.shields.io/github/stars/Zillwang/StepSearch)](https://github.com/Zillwang/StepSearch)
- [SimpleTIR](https://simpletir.notion.site/report)：多轮工具集成推理的稳定端到端强化学习。[![代码](https://img.shields.io/github/stars/ltzheng/SimpleTIR)](https://github.com/ltzheng/SimpleTIR)
- [Router-R1](https://arxiv.org/pdf/2506.09033)：通过强化学习训练大型语言模型进行多轮路由和聚合。[![代码](https://img.shields.io/github/stars/ulab-uiuc/Router-R1)](https://github.com/ulab-uiuc/Router-R1)
- [SkyRL](https://skyrl.readthedocs.io/en/latest/)：大型语言模型的模块化全栈强化学习库。[![代码](https://img.shields.io/github/stars/NovaSky-AI/SkyRL)](https://github.com/NovaSky-AI/SkyRL)
- [ASearcher](https://arxiv.org/abs/2508.07976)：大型语言模型搜索代理的大规模强化学习。[![代码](https://img.shields.io/github/stars/inclusionAI/ASearcher)](https://github.com/inclusionAI/ASearcher)
- [ParallelSearch](https://www.arxiv.org/abs/2508.09303)：通过强化学习分解查询和并行搜索子查询。[![代码](https://img.shields.io/github/stars/Tree-Shu-Zhao/ParallelSearch)](https://github.com/Tree-Shu-Zhao/ParallelSearch)
- [AutoTIR](https://arxiv.org/pdf/2507.21836)：通过强化学习实现自主工具集成推理。[![代码](https://img.shields.io/github/stars/weiyifan1023/AutoTIR)](https://github.com/weiyifan1023/AutoTIR)
- [verl-tool](https://arxiv.org/pdf/2509.01055)：支持多种工具使用的 verl 版本。[![代码](https://img.shields.io/github/stars/TIGER-AI-Lab/verl-tool)](https://github.com/TIGER-AI-Lab/verl-tool)
- [Tree-GRPO](https://arxiv.org/abs/2509.21240)：大型语言模型代理强化学习的树搜索。[![代码](https://img.shields.io/github/stars/AMAP-ML/Tree-GRPO)](https://github.com/AMAP-ML/Tree-GRPO)
- [EviNote-RAG](https://arxiv.org/abs/2509.00877)：通过答案支持性证据笔记增强检索增强生成模型。[![代码](https://img.shields.io/github/stars/Da1yuqin/EviNoteRAG)](https://github.com/Da1yuqin/EviNoteRAG)
- [GlobalRAG](https://arxiv.org/pdf/2510.20548v1)：GlobalRAG：通过强化学习增强多跳问答中的全局推理。[![代码](https://img.shields.io/github/stars/CarnegieBin/GlobalRAG)](https://github.com/CarnegieBin/GlobalRAG)

## 引用

```bibtex
@article{jin2025search,
  title={Search-r1: 通过强化学习训练大型语言模型进行推理和搜索引擎调用},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv 预印本 arXiv:2503.09516},
  year={2025}
}
```

```bibtex
@article{jin2025empirical,
  title={关于推理-搜索交织型大型语言模型代理的强化学习实证研究},
  author={Jin, Bowen and Yoon, Jinsung and Kargupta, Priyanka and Arik, Sercan O and Han, Jiawei},
  journal={arXiv 预印本 arXiv:2505.15117},
  year={2025}
}
```
