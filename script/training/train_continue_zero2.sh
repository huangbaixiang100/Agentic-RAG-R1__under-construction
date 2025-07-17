# #!/bin/bash
# # 使用绝对路径激活conda环境
# export PATH="/data/xiaobei/anaconda3/bin:$PATH"
# eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
# conda activate AgenticRAG

# # 确保安装必要的依赖
# pip install python-dotenv rich 'swanlab[dashboard]' --quiet

# # 设置SwanLab环境变量
# export SWANLAB_DISABLE_RICH=0
# export SWANLAB_DISABLE_DEBUG=1
# export SWANLAB_API_KEY="fT8QlkzJr5kY9syLiIdSr"  # SwanLab API密钥
# export SWANLAB_ENTITY="xiaobei"  

# # 设置DeepSpeed和分布式环境变量
# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO
# export NCCL_P2P_LEVEL=NVL

# # NCCL优化设置
# export NCCL_SOCKET_NTHREADS=8            # 增加通信线程数
# export NCCL_NSOCKS_PERTHREAD=8           # 每个线程的套接字数
# export NCCL_SHM_DISABLE=0                # 启用共享内存
# export NCCL_P2P_DISABLE=0                # 启用P2P
# export NCCL_IB_DISABLE=1                 # 在非InfiniBand环境下禁用IB
# export NCCL_BUFFSIZE=4194304             # 增大缓冲区大小
# export NCCL_CROSS_NIC=0                  # 禁用跨NIC通信
# export NCCL_NET_GDR_LEVEL=PIX            # 设置GDR级别为PIX

# # 网络接口设置，排除不需要的接口
# export NCCL_SOCKET_IFNAME="^lo,docker,bond,dummy,virbr"

# # 设置CUDA相关环境变量以优化性能
# export CUDA_DEVICE_MAX_CONNECTIONS=1     # 限制每个设备的连接数
# export CUDA_LAUNCH_BLOCKING=0            # 禁用CUDA启动阻塞

# # 确保项目目录正确
# PROJECT_DIR=$(pwd)
# echo "当前工作目录: $PROJECT_DIR"

# # 检查DeepSpeed配置文件
# CONFIG_FILE="$PROJECT_DIR/src/config/accelerate_config/train_zero2.yaml"
# if [ ! -f "$CONFIG_FILE" ]; then
#     echo "错误: DeepSpeed配置文件不存在: $CONFIG_FILE"
#     exit 1
# fi
# echo "使用DeepSpeed配置文件: $CONFIG_FILE"

# # 备份原始配置文件
# echo "备份原始配置文件..."
# cp src/config/config.yaml src/config/config.yaml.bak

# # 创建临时配置文件
# cat > src/config/config.yaml << EOF
# project:
#   name: "xiaobeir1"
#   description: "under-construction"

# experiment:
#   name: "promed-qwen25-1.5b-continue"  # 实验名称，对应 experiments/ 下的目录
#   random_seed: 42

# # SwanLab配置
# swanlab: true

# model:
#   name: "Qwen/Qwen2.5-1.5B-Instruct"  # 基础模型名称，不变
#   torch_dtype: "bfloat16"
#   device_map: null

# dataset:
#   name: "cmb"
#   num_eval: 100

# training:
#   continue_training: true
#   current_step: 645

#   use_lora: true
#   use_quant: true
#   batch_size: 1
#   learning_rate: 0.000005
#   num_iterations: 10 # epoch
#   steps_per_iteration: 5 # in one epoch
#   save_interval: 5 # steps

#   generation:
#     num_generations: 4
#     max_new_tokens: 200
#     max_length_for_gather: 100000
#     max_generate_iterations: 10
#     temperature: 0.7
#     do_sample: True
  
#   optimizer:
#     beta: 0.04
#     mu: 1
#     epsilon: 0.1

# lora:
#   r: 8
#   lora_alpha: 32
#   target_modules:
#     - "q_proj"    # qwen
#     - "v_proj"    # qwen
#   lora_dropout: 0.1
#   bias: "none"
#   task_type: "CAUSAL_LM"

# qlora:
#   load_in_4bit: True           # zero 2 可以为True
#   bnb_4bit_quant_type: "nf4"
#   bnb_4bit_compute_dtype: "bfloat16"
#   bnb_4bit_use_double_quant: True   # zero 2 可以为True
#   load_in_8bit: False    # enable 8bit quantization
#   llm_int8_threshold: 6.0   # if load_in_8bit is True
# EOF

# # 创建临时补丁脚本，解决LLaMA-Factory模型加载问题
# cat > /tmp/fix_peft_loading.py << EOF
# #!/usr/bin/env python3
# # 创建一个补丁脚本，修改doctor_trainer.py中的build_model函数

# import re

# # 读取文件
# with open("src/models/doctor_trainer.py", "r") as f:
#     content = f.read()

# # 修改build_model函数，适配LLaMA-Factory训练的模型
# pattern = r"if continue_training:(.*?)else:(.*?)# 验证LoRA是否正确应用"
# replacement = r'''if continue_training:
#             weights_path = f"/data/xiaobei/dhx/LLaMA-Factory-main-new/models/promed_qwen2_5_1_5b_sft/checkpoint-645"
#             logging.info(f"从检查点加载LoRA权重: {weights_path}")
            
#             # 先加载基础模型
#             model = AutoModelForCausalLM.from_pretrained(
#                 config.model.name,  # 使用原始模型名称
#                 torch_dtype=getattr(torch, config.model.torch_dtype),
#                 trust_remote_code=True,
#             ).to(device)
            
#             # 然后应用LoRA
#             model = PeftModel.from_pretrained(model, weights_path, is_trainable=True)
#         else:\2# 验证LoRA是否正确应用'''

# # 应用修改
# modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# # 写入文件
# with open("src/models/doctor_trainer.py", "w") as f:
#     f.write(modified_content)

# print("已修改doctor_trainer.py以适配LLaMA-Factory训练的模型")
# EOF

# # 执行补丁脚本
# chmod +x /tmp/fix_peft_loading.py
# python /tmp/fix_peft_loading.py

# # 使用accelerate启动DeepSpeed ZeRO-2配置的训练
# echo "启动DeepSpeed ZeRO-2训练..."
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
#     --config_file ./src/config/accelerate_config/train_zero2.yaml \
#     --main_process_port 12348 \
#     --num_processes 8 \
#     --mixed_precision "fp16" \
#     ./hhhdoctor_train.py

# # 恢复配置文件
# echo "恢复原始配置文件..."
# mv src/config/config.yaml.bak src/config/config.yaml

# echo "训练完成！" 


#!/bin/bash
# 使用绝对路径激活conda环境
export PATH="/data/xiaobei/anaconda3/bin:$PATH"
eval "$(/data/xiaobei/anaconda3/bin/conda shell.bash hook)"
conda activate AgenticRAG

# 确保安装必要的依赖
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
export NCCL_SOCKET_IFNAME="^lo,docker,bond,dummy,virbr"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0

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

# 备份原始配置文件
echo "备份原始配置文件..."
cp src/config/config.yaml src/config/config.yaml.bak

# 创建临时配置文件
cat > src/config/config.yaml << EOF
project:
  name: "xiaobeir1"
  description: "under-construction"

experiment:
  name: "promed-qwen25-1.5b-merged"  # 修改了实验名称
  random_seed: 42

# SwanLab配置
swanlab: true

model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"  # 基础模型名称，不变
  torch_dtype: "bfloat16"
  device_map: null

dataset:
  name: "cmb"
  num_eval: 100

training:
  continue_training: false  # 修改为false，不继续训练LoRA
  current_step: 0           # 重置步数

  use_lora: true            # 仍然使用LoRA，但是会在新模型上应用
  use_quant: true
  batch_size: 1
  learning_rate: 0.000005
  num_iterations: 10
  steps_per_iteration: 5
  save_interval: 5

  generation:
    num_generations: 4
    max_new_tokens: 200
    max_length_for_gather: 100000
    max_generate_iterations: 10
    temperature: 0.7
    do_sample: True
  
  optimizer:
    beta: 0.04
    mu: 1
    epsilon: 0.1

lora:
  r: 8
  lora_alpha: 32
  target_modules:
    - "q_proj"    # qwen
    - "v_proj"    # qwen
  lora_dropout: 0.1
  bias: "none"
  task_type: "CAUSAL_LM"

qlora:
  load_in_4bit: True
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: True
  load_in_8bit: False
  llm_int8_threshold: 6.0
EOF

# 创建LoRA合并脚本
cat > /tmp/merge_lora.py << EOF
#!/usr/bin/env python3
# 创建一个脚本，合并LoRA权重和基础模型

import os
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import shutil

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 路径设置
base_model_path = "Qwen/Qwen2.5-1.5B-Instruct"
lora_weights_path = "/data/xiaobei/dhx/LLaMA-Factory-main-new/models/promed_qwen2_5_1_5b_sft/checkpoint-645"
merged_model_path = "merged_qwen25_model"

# 创建输出目录
os.makedirs(merged_model_path, exist_ok=True)
logging.info(f"创建合并模型目录: {merged_model_path}")

# 加载基础模型
logging.info(f"加载基础模型: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 加载LoRA权重
logging.info(f"加载LoRA权重: {lora_weights_path}")
model = PeftModel.from_pretrained(model, lora_weights_path)

# 合并LoRA权重到基础模型
logging.info("合并LoRA权重到基础模型...")
model = model.merge_and_unload()

# 保存合并后的模型
logging.info(f"保存合并后的模型到: {merged_model_path}")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

# 为了确保后续步骤能找到这个合并模型，也可以复制到config中指定的路径
config_model_path = "/data/xiaobei/hbx/merged_qwen25_model"
if os.path.exists(config_model_path):
    shutil.rmtree(config_model_path)
shutil.copytree(merged_model_path, config_model_path)
logging.info(f"复制合并模型到全局路径: {config_model_path}")

logging.info("模型合并完成！")
EOF

# 创建修改后的doctor_trainer.py使用合并模型
cat > /tmp/modify_trainer.py << EOF
#!/usr/bin/env python3
# 创建一个补丁脚本，修改doctor_trainer.py中的build_model函数使用合并模型

import re

# 读取文件
with open("src/models/doctor_trainer.py", "r") as f:
    content = f.read()

# 修改build_model函数，使用合并后的模型
pattern = r"if continue_training:(.*?)else:(.*?)# 验证LoRA是否正确应用"
replacement = r'''if continue_training:
            # 原先的continue_training逻辑不再使用
            logging.warning("continue_training设置为True，但我们现在使用合并后的模型，忽略此设置")
            
            # 直接加载合并后的模型
            merged_model_path = "/data/xiaobei/hbx/merged_qwen25_model"
            logging.info(f"加载合并后的模型: {merged_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                merged_model_path,
                torch_dtype=getattr(torch, config.model.torch_dtype),
                trust_remote_code=True,
            ).to(device)
        else:\2# 验证LoRA是否正确应用'''

# 应用修改
modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# 写入文件
with open("src/models/doctor_trainer.py", "w") as f:
    f.write(modified_content)

print("已修改doctor_trainer.py以使用合并后的模型")
EOF

# 执行模型合并
echo "开始合并LoRA权重到基础模型..."
python /tmp/merge_lora.py

# 执行补丁脚本修改trainer
chmod +x /tmp/modify_trainer.py
python /tmp/modify_trainer.py

# 使用accelerate启动DeepSpeed ZeRO-2配置的训练
echo "启动DeepSpeed ZeRO-2训练..."
CUDA_VISIBLE_DEVICES=0,1,2,3,7 accelerate launch \
    --config_file ./src/config/accelerate_config/train_zero2.yaml \
    --main_process_port 12348 \
    --num_processes 5 \
    --mixed_precision "fp16" \
    ./hhhdoctor_train.py

# 恢复配置文件
echo "恢复原始配置文件..."
mv src/config/config.yaml.bak src/config/config.yaml

echo "训练完成！"