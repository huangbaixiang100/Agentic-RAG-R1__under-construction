#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练启动脚本，内部预处理以禁用DTensor
"""

import os
import sys
import subprocess
from pathlib import Path

# 确保我们处于正确的工作目录
project_root = Path(__file__).parent.parent.parent
os.chdir(project_root)

# 设置必要的环境变量
os.environ['PYTORCH_NO_DTENSOR'] = '1'
os.environ['TORCH_DTENSOR_DISABLED'] = '1'
os.environ['ACCELERATE_DISABLE_DTENSOR'] = '1'

# 准备训练命令
gpu_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
num_gpus = len(gpu_devices.split(','))

# 构建命令
cmd = [
    "accelerate", "launch",
    "--config_file", "./src/config/accelerate_config/train_zero2.yaml",
    "--main_process_port", "12348",
    "--num_processes", str(num_gpus),
    "./hbxdoctor_train.py"
]

# 打印即将执行的命令
print(f"执行命令: {' '.join(cmd)}")

# 执行命令
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"命令执行失败: {e}")
    sys.exit(1)
except Exception as e:
    print(f"发生错误: {e}")
    sys.exit(1) 