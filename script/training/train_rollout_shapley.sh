#!/bin/bash

# 激活conda环境
# conda activate AgenticRAG
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

# 网络接口设置，使用默认接口
export NCCL_SOCKET_IFNAME=""

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

# 启用Rollout级别Shapley值训练！
echo "=========================================="
echo "🎯 启动Rollout级别Shapley值训练"
echo "=========================================="
echo "🏆 Rollout奖励系统:"
echo "   🔹 使用overall_reward函数"
echo "   🔹 启用Shapley值加权事实分数"
echo "   🔹 每个rollout获得综合分数"
echo "   🔹 分数包括: 正确性 + 格式 + Shapley加权事实分数"
echo "📊 Shapley值特点:"
echo "   ✅ 对所有事实平等计算Shapley值"
echo "   ✅ 蒙特卡洛方法确保公平性"
echo "   ✅ 基于事实对答案的边际贡献"
echo "   ✅ 权重归一化后加权事实获取奖励"
echo "🚀 训练特色:"
echo "   🎯 Rollout级别Group Baseline"
echo "   📊 每个rollout的综合得分"
echo "   📈 SwanLab记录各维度奖励"
echo "   🎪 整体对话质量优化"
echo "=========================================="

# Rollout Shapley训练配置
USE_SHAPLEY=${USE_SHAPLEY:-"True"}                    # 启用Shapley值加权
SHAPLEY_MAX_SAMPLES=${SHAPLEY_MAX_SAMPLES:-"50"}     # Shapley蒙特卡洛最大采样数
SHAPLEY_MIN_SAMPLES=${SHAPLEY_MIN_SAMPLES:-"5"}      # Shapley蒙特卡洛最小采样数
SHAPLEY_NORMALIZATION=${SHAPLEY_NORMALIZATION:-"softmax"}  # Shapley权重归一化方法
SHAPLEY_TEMPERATURE=${SHAPLEY_TEMPERATURE:-"2.0"}    # Shapley softmax温度

echo "🔧 Rollout Shapley配置:"
echo "   🧮 USE_SHAPLEY: $USE_SHAPLEY"
echo "   📊 SHAPLEY_MAX_SAMPLES: $SHAPLEY_MAX_SAMPLES"
echo "   📊 SHAPLEY_MIN_SAMPLES: $SHAPLEY_MIN_SAMPLES"
echo "   ⚖️ SHAPLEY_NORMALIZATION: $SHAPLEY_NORMALIZATION"
echo "   🌡️ SHAPLEY_TEMPERATURE: $SHAPLEY_TEMPERATURE"
echo "=========================================="



# 使用accelerate启动DeepSpeed ZeRO-2配置的Rollout Shapley训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12349 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    ./hhhdoctor_train.py \
    --use_token_level=False \
    --use_shapley="$USE_SHAPLEY" \
    --shapley_max_samples="$SHAPLEY_MAX_SAMPLES" \
    --shapley_min_samples="$SHAPLEY_MIN_SAMPLES" \
    --shapley_normalization="$SHAPLEY_NORMALIZATION" \
    --shapley_temperature="$SHAPLEY_TEMPERATURE" \
  

echo ""
echo "🎉 Rollout级别Shapley值训练完成！"
echo "📊 训练特点总结:"
echo "   🎯 每个rollout获得综合分数(正确性+格式+Shapley事实分数)"
echo "   📈 Shapley值基于事实对答案的边际贡献计算"
echo "   🚀 Group Baseline在rollout级别计算"
echo "   💨 每个rollout_advantage = rollout_reward - group_baseline"
echo "🔍 SwanLab监控指标:"
echo "   📊 rewards/total_scores - 总体奖励"
echo "   📊 rewards/correctness_scores - 正确性奖励"
echo "   📊 rewards/format_scores - 格式奖励"
echo "   📊 rewards/fact_scores - Shapley加权事实奖励"
echo "   📊 shapley/avg_shapley_values - 平均Shapley值"
echo "   📊 shapley/shapley_convergence - Shapley收敛情况"
echo "📈 检查Shapley加权是否提升事实获取质量！" 