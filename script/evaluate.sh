#!/bin/bash

# 环境配置
export PATH="/data/xiaobei/anaconda3/bin:$PATH"
eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
conda activate AgenticRAG

# 确保安装必要的依赖
pip install python-dotenv rich 'swanlab[dashboard]' --quiet

# 设置环境变量
export CUDA_VISIBLE_DEVICES=2,3,4,5,7

# 设置工作目录
cd /home/xiaobei/hbx/HHHAgentic-RAG-R1__under-construction

# 配置变量
MODEL_PATH="/home/xiaobei/hbx/HHHAgentic-RAG-R1__under-construction/checkpoints/promed-qwen25-1.5b-correctness-reward/2025-05-23/step-0015"
DATASET_SPLIT="test"  
NUM_SAMPLES=100        # 设置为null使用全部样本
BATCH_SIZE=4
MAX_NEW_TOKENS=2048
MAX_GENERATE_ITERATIONS=10
TEMPERATURE=0.7
OUTPUT_DIR="evaluation_results/$(date +%Y-%m-%d)"
LOG_FILE="$OUTPUT_DIR/evaluation_$(date +%Y-%m-%d_%H-%M-%S).log"

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行评估脚本
python src/evaluate_cmb.py \
  --model_path $MODEL_PATH \
  --dataset_split $DATASET_SPLIT \
  --num_samples $NUM_SAMPLES \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --max_generate_iterations $MAX_GENERATE_ITERATIONS \
  --temperature $TEMPERATURE \
  --output_dir $OUTPUT_DIR \
  --log_file $LOG_FILE \
  --local_model