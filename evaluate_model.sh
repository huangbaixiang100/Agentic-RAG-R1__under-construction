#!/bin/bash

# 评估模型脚本 - 复制Qwen2.5-1.5B配置并运行评估
# 使用方法: ./evaluate_model.sh

set -e  # 遇到错误时退出

# 配置参数
#MODEL_PATH="/data/xiaobei/dhx/LLaMA-Factory-main-new/models/promed-qwen2.5-1.5b-sft-3epoch-merged"
#MODEL_PATH="/home/xiaobei/hbx/HHHAgentic-RAG-R1__under-construction/checkpoints/promed-qwen25-1.5b-correctness-reward73-softmax/2025-07-03/step-0100"
#MODEL_PATH="/home/xiaobei/hbx/HHHAgentic-RAG-R1__under-construction/checkpoints/promed-qwen25-1.5b-correctness-reward72-softmax/2025-07-03/step-0200"
MODEL_PATH="/data/xiaobei/Common_LLM_Base/Qwen2.5-1.5B-Instruct"  # 基础模型名称，不变
QWEN_MODEL_NAME="Qwen/Qwen2.5-1.5B"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALUATE_SCRIPT="${SCRIPT_DIR}/src/evaluate_cmb.py"

echo "=========================================="
echo "模型评估脚本开始执行"
echo "模型路径: ${MODEL_PATH}"
echo "=========================================="

# 检查模型目录是否存在
if [ ! -d "${MODEL_PATH}" ]; then
    echo "错误: 模型目录不存在: ${MODEL_PATH}"
    exit 1
fi

# 检查evaluate_cmb.py是否存在
if [ ! -f "${EVALUATE_SCRIPT}" ]; then
    echo "错误: 评估脚本不存在: ${EVALUATE_SCRIPT}"
    exit 1
fi

# 检查是否已有config.json
if [ -f "${MODEL_PATH}/config.json" ]; then
    echo "检测到已存在config.json文件，跳过下载步骤"
else
    echo "步骤1: 下载Qwen2.5-1.5B的配置文件..."
    
    # 创建临时目录下载配置
    TEMP_DIR="/tmp/qwen25_config_$$"
    mkdir -p "${TEMP_DIR}"
    
    echo "正在下载Qwen2.5-1.5B配置文件..."
    
    # 使用Python下载配置文件
    python3 -c "
import os
from transformers import AutoConfig, AutoTokenizer
import shutil

temp_dir = '${TEMP_DIR}'
model_path = '${MODEL_PATH}'
qwen_model = '${QWEN_MODEL_NAME}'

print(f'下载配置文件到临时目录: {temp_dir}')

try:
    # 下载配置文件
    config = AutoConfig.from_pretrained(qwen_model, trust_remote_code=True)
    config.save_pretrained(temp_dir)
    print('配置文件下载成功')
    
    # 复制config.json到模型目录
    config_src = os.path.join(temp_dir, 'config.json')
    config_dst = os.path.join(model_path, 'config.json')
    
    if os.path.exists(config_src):
        shutil.copy2(config_src, config_dst)
        print(f'配置文件已复制到: {config_dst}')
    else:
        print('错误: 未找到config.json文件')
        exit(1)
        
except Exception as e:
    print(f'下载配置文件时出错: {e}')
    exit(1)
"
    
    # 清理临时目录
    rm -rf "${TEMP_DIR}"
    
    if [ -f "${MODEL_PATH}/config.json" ]; then
        echo "✓ 配置文件复制成功"
    else
        echo "✗ 配置文件复制失败"
        exit 1
    fi
fi

echo "步骤2: 验证模型文件完整性..."

# 检查必要文件是否存在
required_files=("config.json" "tokenizer.json" "tokenizer_config.json")
for file in "${required_files[@]}"; do
    if [ -f "${MODEL_PATH}/${file}" ]; then
        echo "✓ ${file} 存在"
    else
        echo "✗ ${file} 缺失"
        exit 1
    fi
done

# 检查模型权重文件（支持pytorch_model.bin或safetensors格式）
if [ -f "${MODEL_PATH}/pytorch_model.bin" ]; then
    echo "✓ pytorch_model.bin 存在"
elif [ -f "${MODEL_PATH}/model.safetensors.index.json" ] && [ -f "${MODEL_PATH}/model-00001-of-00002.safetensors" ]; then
    echo "✓ safetensors格式模型文件存在"
elif [ -f "${MODEL_PATH}/model.safetensors" ]; then
    echo "✓ model.safetensors 存在"
else
    echo "✗ 未找到模型权重文件（pytorch_model.bin 或 safetensors）"
    exit 1
fi

echo "步骤3: 开始模型评估..."

# 设置Python路径
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# 运行评估
cd "${SCRIPT_DIR}"

echo "运行命令: python ${EVALUATE_SCRIPT} --model_path ${MODEL_PATH} --local_model --dataset_split test --num_samples 1935 --batch_size 20 --max_new_tokens 2048 --max_generate_iterations 10 --temperature 0.3 --output_dir evaluation_results"

python3 "${EVALUATE_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --local_model \
    --dataset_split test \
    --num_samples 1935 \
    --batch_size 20 \
    --max_new_tokens 2048 \
    --max_generate_iterations 10 \
    --temperature 0.3 \
    --output_dir evaluation_results \
    --log_file evaluation_$(date +%Y%m%d_%H%M%S).log

echo "=========================================="
echo "模型评估完成！"
echo "结果保存在: evaluation_results/ 目录下"
echo "==========================================" 