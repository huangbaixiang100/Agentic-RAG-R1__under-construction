#!/bin/bash
# Token级奖励分配训练脚本
# 支持细粒度奖励分配，结合过程奖励和结果奖励

# 使用绝对路径激活conda环境
export PATH="/data/xiaobei/anaconda3/bin:$PATH"
eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
conda activate AgenticRAG

# 确保安装必要的依赖
pip install python-dotenv rich 'swanlab[dashboard]' --quiet

# 设置SwanLab环境变量
export SWANLAB_DISABLE_RICH=0
export SWANLAB_DISABLE_DEBUG=1
export SWANLAB_API_KEY="fT8QlkzJr5kY9syLiIdSr"  # SwanLab API密钥
export SWANLAB_ENTITY="xiaobei"  

# 设置DeepSpeed和分布式环境变量
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

# NCCL优化设置
export NCCL_SOCKET_NTHREADS=8            # 增加通信线程数
export NCCL_NSOCKS_PERTHREAD=8           # 每个线程的套接字数
export NCCL_SHM_DISABLE=0                # 启用共享内存
export NCCL_P2P_DISABLE=0                # 启用P2P
export NCCL_IB_DISABLE=1                 # 在非InfiniBand环境下禁用IB
export NCCL_BUFFSIZE=4194304             # 增大缓冲区大小
export NCCL_CROSS_NIC=0                  # 禁用跨NIC通信
export NCCL_NET_GDR_LEVEL=PIX            # 设置GDR级别为PIX

# 网络接口设置，排除不需要的接口
export NCCL_SOCKET_IFNAME="^lo,docker,bond,dummy,virbr"

# 设置CUDA相关环境变量以优化性能
export CUDA_DEVICE_MAX_CONNECTIONS=1     # 限制每个设备的连接数
export CUDA_LAUNCH_BLOCKING=0            # 禁用CUDA启动阻塞

# 确保项目目录正确
PROJECT_DIR=$(pwd)
echo "当前工作目录: $PROJECT_DIR"

# 检查DeepSpeed配置文件
CONFIG_FILE="$PROJECT_DIR/src/config/accelerate_config/train_zero2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: DeepSpeed配置文件不存在: $CONFIG_FILE"
    exit 1
fi
echo "使用DeepSpeed配置文件: $CONFIG_FILE"

# 启用Token级奖励分配训练！
echo "=========================================="
echo "🎯 启动Token级奖励分配医疗对话训练"
echo "=========================================="
echo "✓ 启用Token级奖励分配"
echo "✓ 结合过程奖励和结果奖励"
echo "✓ 支持Shapley值加权信息增益"
echo "✓ 每个问题获得与其贡献相匹配的奖励"
echo "=========================================="

# 使用accelerate启动DeepSpeed ZeRO-2配置的Token级奖励训练
CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12348 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    ./hhhdoctor_train.py \
    # --use_shapley=True \
    # --shapley_max_samples 50 \
    # --shapley_min_samples 3 \
    --use_token_level=True \
    --alpha_reward=2.0 \
    --beta_reward=1.0 \
    --gamma_reward=3.0

echo "🎉 Token级奖励分配训练完成！"
echo "📊 奖励分配统计信息已保存到日志中"
echo "🔍 可通过SwanLab查看详细训练指标" 