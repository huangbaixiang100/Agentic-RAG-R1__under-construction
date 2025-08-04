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

# 启用纯Token级奖励训练！
echo "=========================================="
echo "🎯 启动纯Token级奖励训练"
echo "=========================================="
echo "🏆 新奖励系统 (满分4分):"
echo "   🔹 Question tokens: Shapley奖励(0-3) + 格式奖励(0-1)"
echo "   🔹 Answer tokens: 正确性奖励(0-3) + 格式奖励(0-1)"
echo "   🔹 其他tokens: 0分"
echo "📊 格式奖励规则:"
echo "   ✅ 以question:开头的句子 → 所有token +1分"
echo "   ✅ 以answer:开头的句子 → 所有token +1分"
echo "🚀 训练特色:"
echo "   🎯 Token级Group Baseline Advantage计算"
echo "   📊 每个token的advantage = token_reward - group_baseline"
echo "   📈 SwanLab记录各类token奖励均值"
echo "   🎪 精准token级优化"
echo "=========================================="

# 纯Token奖励训练配置
USE_SHAPLEY=${USE_SHAPLEY:-"True"}              # 是否使用Shapley值加权
ALPHA_REWARD=${ALPHA_REWARD:-"2.0"}             # Question Shapley奖励权重
BETA_REWARD=${BETA_REWARD:-"1.0"}               # Question结果奖励权重
GAMMA_REWARD=${GAMMA_REWARD:-"3.0"}             # Answer正确性奖励权重
FORMAT_REWARD_WEIGHT=${FORMAT_REWARD_WEIGHT:-"1.0"}  # 格式奖励权重

echo "🔧 纯Token奖励配置:"
echo "   🧮 USE_SHAPLEY: $USE_SHAPLEY"
echo "   📝 ALPHA_REWARD: $ALPHA_REWARD (Question Shapley权重)"
echo "   🎯 BETA_REWARD: $BETA_REWARD (Question结果权重)"
echo "   🏆 GAMMA_REWARD: $GAMMA_REWARD (Answer正确性权重)"
echo "   📋 FORMAT_REWARD_WEIGHT: $FORMAT_REWARD_WEIGHT (格式权重)"
echo "=========================================="

# 使用accelerate启动DeepSpeed ZeRO-2配置的纯Token奖励训练
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12349 \
    --num_processes 2 \
    --mixed_precision "bf16" \
    ./hhhdoctor_train.py \
    --use_token_level=True \
    --use_shapley="$USE_SHAPLEY" \
    --shapley_max_samples=50 \
    --shapley_min_samples=3 \
    --alpha_reward="$ALPHA_REWARD" \
    --beta_reward="$BETA_REWARD" \
    --gamma_reward="$GAMMA_REWARD" \
    --format_reward_weight="$FORMAT_REWARD_WEIGHT"

echo ""
echo "🎉 纯Token级奖励训练完成！"
echo "📊 训练特点总结:"
echo "   🎯 每个token获得精确奖励(0-4分)"
echo "   📈 Question/Answer/Format奖励分离统计"
echo "   🚀 Token级Group Baseline Advantage计算"
echo "   💨 每个token_advantage = token_reward - group_baseline"
echo "🔍 SwanLab监控指标:"
echo "   📊 token_rewards/question_token_mean"
echo "   📊 token_rewards/answer_token_mean"
echo "   📊 token_rewards/format_token_mean"
echo "   📊 token_rewards/total_token_mean"
echo "📈 检查各类token奖励分布是否合理！" 