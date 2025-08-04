#!/bin/bash

# 激活conda环境
# conda activate AgenticRAG
pip install python-dotenv rich 'swanlab[dashboard]' --quiet

# 设置SwanLab环境变量
export SWANLAB_DISABLE_RICH=0
export SWANLAB_DISABLE_DEBUG=1
export SWANLAB_API_KEY="fT8QlkzJr5kY9syLiIdSr"
export SWANLAB_ENTITY="xiaobei"

# 设置DeepSpeed和分布式环境变量
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

# NCCL优化设置
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SHM_DISABLE=0
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_BUFFSIZE=4194304
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_SOCKET_IFNAME=""

# 设置CUDA相关环境变量
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

# 确保项目目录正确
PROJECT_DIR=$(pwd)
echo "当前工作目录: $PROJECT_DIR"

# 检查配置文件
CONFIG_FILE="$PROJECT_DIR/src/config/accelerate_config/train_zero2.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

echo "=========================================="
echo "🎯 启动Rollout级别Shapley值训练"
echo "=========================================="
echo "🏆 训练模式: Rollout级别"
echo "   🔹 使用overall_reward函数"
echo "   🔹 启用Shapley值加权事实分数"
echo "   🔹 每个rollout获得综合分数"
echo "   🔹 Group Baseline在rollout级别计算"
echo "=========================================="

# 简化的训练配置
USE_SHAPLEY="True"
USE_TOKEN_LEVEL="False"

echo "🔧 训练配置:"
echo "   🧮 USE_SHAPLEY: $USE_SHAPLEY"
echo "   📏 USE_TOKEN_LEVEL: $USE_TOKEN_LEVEL"
echo "=========================================="

# 启动训练
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12349 \
    --num_processes 1 \
    --mixed_precision "bf16" \
    ./hhhdoctor_train.py \
    --use_token_level="$USE_TOKEN_LEVEL" \
    --use_shapley="$USE_SHAPLEY"

echo ""
echo "🎉 Rollout级别Shapley值训练完成！"
echo "📊 训练特点:"
echo "   🎯 每个rollout获得综合分数"
echo "   📈 Shapley值加权事实获取"
echo "   🚀 Group Baseline在rollout级别"
echo "🔍 监控指标:"
echo "   📊 rewards/total_scores"
echo "   📊 rewards/correctness_scores"
echo "   📊 rewards/format_scores"
echo "   📊 rewards/fact_scores" 