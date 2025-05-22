#!/bin/bash
# 使用绝对路径激活conda环境
export PATH="/data/xiaobei/anaconda3/bin:$PATH"
eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
conda activate AgenticRAG

# 确保安装必要的依赖
pip install python-dotenv rich 'swanlab[dashboard]' --quiet

# 设置分布式环境变量
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=0
export NCCL_P2P_LEVEL=NVL
# 添加或修改以下环境变量
export NCCL_SOCKET_NTHREADS=8            # 增加通信线程数
export NCCL_NSOCKS_PERTHREAD=8           # 每个线程的套接字数
export NCCL_SHM_DISABLE=0                # 启用共享内存
export NCCL_P2P_LEVEL=NVL                # NVLink优先(已设置)
export NCCL_P2P_DISABLE=0                # 启用P2P(已设置)
export NCCL_IB_DISABLE=1                 # 在非InfiniBand环境下禁用IB
export NCCL_BUFFSIZE=4194304             # 增大缓冲区大小
export NCCL_CROSS_NIC=0                  # 禁用跨NIC通信
export NCCL_NET_GDR_LEVEL=PIX            # 设置GDR级别为PIX
# 添加以下两行，使用通配符允许任何可用接口
export NCCL_SOCKET_IFNAME="^lo,docker,bond,dummy,virbr"  # 排除这些接口

# 使用accelerate启动DeepSpeed ZeRO-3配置的训练
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero3.yaml \
    --main_process_port 12347  \
    --num_processes 4 \
    --mixed_precision "fp16" \
    ./hhhdoctor_train.py 