from dotenv import load_dotenv
from rich.traceback import install
from rich.logging import RichHandler
from rich import print
load_dotenv()
install()
import datetime
import json
import logging
import os
import time
import sys
import argparse
from pathlib import Path
import deepspeed
import swanlab
import torch
from accelerate import Accelerator, init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from dataclasses import dataclass, field
from typing import Optional
# 初始化rich traceback
load_dotenv()
install()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
from src.data.prepare_dataset import prepare_dataset
from src.models.doctor_trainer import train_with_grpo, overall_reward, build_model
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)


@dataclass
class DeepSpeedArguments:
    """用于DeepSpeed的命令行参数"""
    deepspeed: Optional[bool] = field(default=False, metadata={"help": "是否使用DeepSpeed"})
    deepspeed_config: Optional[str] = field(default=None, metadata={"help": "DeepSpeed配置文件路径"})
    local_rank: Optional[int] = field(default=-1, metadata={"help": "本地GPU编号"})


def custom_collate_fn(batch):
    """
    将批次中的字典进行整合
    """
    collated = {key: [sample[key] for sample in batch] for key in batch[0]}
    return collated


def main():
    # 解析DeepSpeed参数
    parser = HfArgumentParser(DeepSpeedArguments)
    ds_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # 设置环境
    config = load_config("src/config/config.yaml")

    if ds_args.deepspeed and ds_args.deepspeed_config:
        # 当使用DeepSpeed时，加载DeepSpeed配置
        with open(ds_args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
        # 确保本地排名设置正确
        local_rank = ds_args.local_rank if ds_args.local_rank != -1 else int(os.environ.get('LOCAL_RANK', '0'))
    else:
        ds_config = None
        local_rank = -1

    # 设置分布式配置
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        is_main_process = local_rank == 0
    else:
        is_main_process = True

    # 创建accelerator
    accelerator = Accelerator()
    #初始化swanlab
    if accelerator.is_local_main_process and config.swanlab:
        swanlab.init(
            project=config.project.name,
            experiment_name=config.experiment.name,
            config=config.__dict__,
            api_key="fT8QlkzJr5kY9syLiIdSr"
        )
        logging.info("SwanLab已初始化")

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    checkpoint_dir = Path(f"checkpoints/{config.experiment.name}/{today}")
    output_dir = Path(f"experiments/training/{config.experiment.name}/{time_str}")
    
    if is_main_process:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(output_dir, level=logging.INFO)
        with open(output_dir / "config.json", "w") as f:
            json.dump(config.__dict__, f, indent=2)
        logging.info(f"保存配置到 {output_dir / 'config.json'}")

    set_random_seed(config.experiment.random_seed)
    if is_main_process:
        logging.info(f"设置随机种子为 {config.experiment.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process:
        logging.info(f"使用设备: {device}")

    # 准备数据集
    train_dataset, eval_dataset = prepare_dataset("train", config.dataset.name, eval_size=config.dataset.num_eval)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    if is_main_process:
        logging.info(f"训练数据加载器: {len(train_dataloader)}, 评估数据加载器: {len(eval_dataloader)}")

    # 初始化模型和分词器
    if is_main_process:
        logging.info("加载模型...")

    # 构建基础模型和分词器
    base_model, tokenizer = build_model(config, device)
    reference_base_model, _ = build_model(config, device)
    
    if is_main_process:
        logging.info("基础模型和分词器加载成功")
        
    # 确保tokenizer设置正确
    tokenizer.pad_token = tokenizer.eos_token
    if hasattr(base_model, 'config'):
        base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id
        reference_base_model.config.pad_token_id = reference_base_model.config.eos_token_id = tokenizer.eos_token_id
    
    # 模型内存优化
    base_model = optimize_model_memory(base_model)
    reference_base_model = optimize_model_memory(reference_base_model)

    # 训练配置
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "num_generations": config.training.generation.num_generations,
        "max_new_tokens": config.training.generation.max_new_tokens,
        "max_length_for_gather": config.training.generation.max_length_for_gather,
        "max_generate_iterations": config.training.generation.max_generate_iterations,
        "temperature": config.training.generation.temperature,
        "do_sample": config.training.generation.do_sample,
        "beta": config.training.optimizer.beta,
        "learning_rate": config.training.learning_rate,
        "mu": config.training.optimizer.mu,
        "epsilon": config.training.optimizer.epsilon,
        "reward_function": overall_reward,
        "save_interval": config.training.save_interval,
    }
    if is_main_process:
        logging.info(f"训练配置: {training_config}")

    if config.training.continue_training:
        current_step = config.training.current_step
    else:
        current_step = 0

    # 检测DeepSpeed配置
    zero_stage = None
    if ds_config is not None and "zero_optimization" in ds_config:
        zero_stage = ds_config["zero_optimization"].get("stage", 0)
        if is_main_process:
            logging.info(f"检测到DeepSpeed ZeRO-{zero_stage}配置")

    # 使用GRPO训练
    train_with_grpo(
        config=config,
        device=device,
        policy_model=base_model,
        ref_base_model=reference_base_model,
        tokenizer=tokenizer,
        accelerator=accelerator,
        dataloader=train_dataloader,
        checkpoint_dir=checkpoint_dir,
        current_step=current_step,
        **training_config,
    )
    if is_main_process:
        logging.info("训练完成")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main() 
