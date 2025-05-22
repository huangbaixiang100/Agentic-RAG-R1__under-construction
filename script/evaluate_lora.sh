#!/bin/bash
# 使用绝对路径激活conda环境
export PATH="/data/xiaobei/anaconda3/bin:$PATH"
eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
conda activate AgenticRAG

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0  

# 基础模型路径
BASE_MODEL="/home/xiaobei/小北健康-qwen2.5-7b"

# LoRA权重路径
LORA_WEIGHTS="/home/xiaobei/hbx/HBXAgentic-RAG-R1__under-construction/checkpoints/qwen2.5-32b-medqa-4GPU/2025-05-22/step-0005"

# 数据集和评估配置
DATASET_SPLIT="test"
NUM_SAMPLES=1935  # 测试的数量
BATCH_SIZE=20
MAX_NEW_TOKENS=2048
MAX_GENERATE_ITERATIONS=10
TEMPERATURE=0.7

# 输出目录
OUTPUT_DIR="evaluation_results/lora"
mkdir -p $OUTPUT_DIR

echo "开始评估LoRA模型..."
echo "基础模型: $BASE_MODEL"
echo "LoRA权重: $LORA_WEIGHTS"

# 运行评估脚本
python src/evaluate_lora.py \
    --base_model_path "$BASE_MODEL" \
    --lora_weights_path "$LORA_WEIGHTS" \
    --dataset_split "$DATASET_SPLIT" \
    --num_samples "$NUM_SAMPLES" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_generate_iterations "$MAX_GENERATE_ITERATIONS" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --log_file "$OUTPUT_DIR/evaluation_lora_$(date +%Y%m%d_%H%M%S).log"

echo "评估完成！结果保存在 $OUTPUT_DIR 目录" 