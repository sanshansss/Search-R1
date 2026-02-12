## 多节点训练

我们的代码库支持大型语言模型的多节点训练。实现主要基于 [Ray](https://github.com/ray-project/ray)。

在进行 Ray 多节点训练时，有两种类型的节点：(1) 主节点 (head node) 和 (2) 工作节点 (worker nodes)。

主节点只有一个，用于启动 Ray 集群并提交任务。
其他节点都是工作节点，只需要启动并注册到 Ray 集群即可。

### 步骤 1：设置多节点 Ray 集群（来自[链接](https://verl.readthedocs.io/en/latest/start/multinode.html#set-up-multinode-ray-cluster)）

a. 使用 ```ray start --head --dashboard-host=0.0.0.0``` 启动**主**节点，有两个地址需要注意：

- GCS 地址：```ray start --address=<address>```，**工作**节点需要连接到这个地址。

- 仪表盘地址：```<address>:8265```，这是提交任务到集群的地址。

![head](../public/head.png)

b. 启动**工作节点**，并使用上面获得的 ```ray start --address=<address>``` 将其注册到 Ray 集群。

![worker](../public/worker.png)

c. 使用 ```ray status``` 检查集群状态。

例如，如果集群中有两个节点（每个节点有 8 个 GPU），应该能看到类似以下内容：

![status](../public/status.png)

### 步骤 2：在每个节点上启动检索服务器

为了稳定强化学习训练，我们建议在所有节点（包括主节点和工作节点）上启动**相同**的检索服务器。启动不同检索器的详细信息请参见：[文档](https://github.com/PeterGriffinJin/Search-R1/blob/main/docs/retriever.md)和[脚本](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/retriever)。

例如，如果要在每个节点上启动带有平面索引的本地稠密检索器，请在**每个**节点上运行以下命令：

```
bash retrieval_launch.sh
```

### 步骤 3：启动任务

启动检索器后，您可以开始训练任务。只需要在**主**节点上启动任务。

示例脚本如下所示。将 ```RAY_DASHBOARD_ADDRESS``` 和 ```N_NODES``` 更改为步骤 1 中找到的仪表盘地址和节点数量。

更多脚本示例请参见[此处](https://github.com/PeterGriffinJin/Search-R1/tree/main/example/multinode)。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DATA_DIR='data/nq_search'

WAND_PROJECT="Search-R1-release"
RAY_DASHBOARD_ADDRESS="<address>:8265"
N_NODES=2

export BASE_MODEL='Qwen/Qwen2.5-7B'
export EXPERIMENT_NAME=${train_data}-${test_data}-search-r1-ppo-qwen2.5-7b-em-multinode-$N_NODES

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

ulimit -n 65535

ray job submit --address=$RAY_DASHBOARD_ADDRESS \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.actor.state_masking=true \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size=16 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.grad_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1005 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee $EXPERIMENT_NAME.log
```
