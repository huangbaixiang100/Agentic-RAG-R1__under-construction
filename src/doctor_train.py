from dotenv import load_dotenv
from rich.traceback import install

load_dotenv()
install()

import datetime
import json
import logging
import os
import pdb
import time
from pathlib import Path
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import deepspeed
import swanlab
import torch
from accelerate import Accelerator, init_empty_weights
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from src.data.prepare_dataset import prepare_dataset
from src.models.doctor_reward import overall_reward
from src.models.doctor_trainer import train_with_grpo
from src.utils.utils import (
    load_config,
    optimize_model_memory,
    set_random_seed,
    setup_logging,
)
from src.utils.patient_model import PatientModel

# 添加collate_fn函数
def custom_collate_fn(batch):
    """
    Collate a batch of dicts with potentially non-tensor and variable-length fields.
    This version preserves lists and dicts as-is without stacking.
    """
    if not batch:
        return {}
    collated = {key: [sample.get(key) for sample in batch] for key in batch[0].keys()}
    return collated

# 定义AgenticRAGModel类替代原始代码中的缺失类
class AgenticRAGModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        # 设置默认生成参数
        self.generate_configs = {
            "repetition_penalty": 1.5,
            "no_repeat_ngram_size": 4,
            "min_length": 10,
            "top_p": 0.9
        }
        # 添加zero_optimization属性以便和deepspeed兼容
        self.zero_optimization = {'stage': 0}
    
    @property
    def config(self):
        """返回模型配置，如果配置没有zero_optimization属性，则添加一个"""
        if hasattr(self.model, 'config'):
            if not hasattr(self.model.config, 'zero_optimization'):
                self.model.config.zero_optimization = {'stage': 0}
            return self.model.config
        return {'zero_optimization': {'stage': 0}}
    
    def save_pretrained(self, save_directory):
        """保存模型到指定目录"""
        if hasattr(self.model, 'save_pretrained'):
            return self.model.save_pretrained(save_directory)
        else:
            logging.warning(f"模型不支持save_pretrained方法，跳过保存")
            return None
    
    def generate(self, *args, **kwargs):
        # 合并配置，但保留kwargs中的优先级
        valid_params = {}
        for k, v in self.generate_configs.items():
            # 只传递有效的生成参数，过滤掉非生成参数
            if k in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 
                    'top_k', 'repetition_penalty', 'no_repeat_ngram_size', 
                    'min_length', 'num_beams', 'num_return_sequences', 
                    'length_penalty', 'early_stopping']:
                valid_params[k] = v
        
        all_kwargs = {**valid_params, **kwargs}
        return self.model.generate(*args, **all_kwargs)
    
    def forward(self, *args, **kwargs):
        # 过滤无效参数
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['input_ids', 'attention_mask', 'labels', 'position_ids', 'token_type_ids']}
        return self.model.forward(*args, **valid_kwargs)
    
    def __call__(self, *args, **kwargs):
        # 过滤无效参数
        valid_kwargs = {k: v for k, v in kwargs.items() 
                       if k in ['input_ids', 'attention_mask', 'labels', 'position_ids', 'token_type_ids']}
        return self.model(*args, **valid_kwargs)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self
    
    def __getattr__(self, name):
        if hasattr(self.model, name):
            return getattr(self.model, name)
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")


def main():
    # Setup environment
    config = load_config("src/config/config.yaml")

    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        swanlab.init(
            project=config.project.name,
            experiment_name=config.experiment.name,
            config=config.__dict__,
        )

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    today = now.strftime("%Y-%m-%d")
    checkpoint_dir = Path(f"checkpoints/{config.experiment.name}/{today}")
    output_dir = Path(f"experiments/training/{config.experiment.name}/{time_str}")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, level=logging.INFO)

    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)
    logging.info(f"Saving config to {output_dir / 'config.json'}")

    set_random_seed(config.experiment.random_seed)
    logging.info(f"Set random seed to {config.experiment.random_seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 将批量大小从config.training.batch_size减小到1或2
    batch_size = min(2, config.training.batch_size)
    
    # 指定训练集文件路径
    train_file_path = "/root/cmb_atomic_patient_train.json"
    logging.info(f"使用自定义训练集文件: {train_file_path}")
    
    train_dataset, eval_dataset = prepare_dataset(
        "train", 
        config.dataset.name, 
        eval_size=config.dataset.num_eval,
        train_file=train_file_path
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)
    logging.info(f"Train dataloader: {len(train_dataloader)}, Eval dataloader: {len(eval_dataloader)}")

    # Initialize model and tokenizer
    logging.info("Loading model...")

    # with init_empty_weights():
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
        # empty_init=True
    )

    reference_base_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
        # empty_init=True
    )

    base_model = base_model.to(device)
    reference_base_model = reference_base_model.to(device)
    logging.info("Base model loaded successfully")

    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    base_model.config.pad_token_id = base_model.config.eos_token_id = tokenizer.eos_token_id
    reference_base_model.config.pad_token_id = reference_base_model.config.eos_token_id = tokenizer.eos_token_id
    logging.info("Tokenizer loaded successfully")

    # lora
    if config.training.use_lora:
        lora_config = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        if not config.training.continue_training:
            base_model = get_peft_model(base_model, lora_config)
            reference_base_model = get_peft_model(reference_base_model, lora_config)
        else:
            weights_path = f"checkpoints/{config.experiment.name}/step-{config.training.current_step:04d}"
            base_model = PeftModel.from_pretrained(base_model, weights_path, config=lora_config, is_trainable=True)
            reference_base_model = PeftModel.from_pretrained(
                reference_base_model, weights_path, config=lora_config, is_trainable=True
            )
            logging.info(f"Continue training from {weights_path}")
        logging.info(f"Using lora:\n {lora_config}")

        base_model.print_trainable_parameters()
        reference_base_model.print_trainable_parameters()

    else:
        logging.info("Not using LoRA")

    # quant
    if config.training.use_quant:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),  # optional
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,         # optional
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,               # optional
            load_in_8bit= config.qlora.load_in_8bit,  # enable 8bit quantization
            llm_int8_threshold = config.qlora.llm_int8_threshold, # if load_in_8bit is True
        )
        
        base_model = load_and_quantize_model(
            base_model,
            bnb_quantization_config=bnb_quantization_config,
            device_map = "auto"
        )
        
        reference_base_model = load_and_quantize_model(
            reference_base_model,
            bnb_quantization_config=bnb_quantization_config,
            device_map = "auto"
        )
        
        logging.info(f"Using Quant: {config.qlora}")
    else:
        bnb_quantization_config = None
        logging.info("Not using Quant")
    
    
    # GRPO fine-tuning
    logging.info("Starting GRPO fine-tuning...")
    # 分离生成参数和训练参数
    generation_config = {
        "max_new_tokens": config.training.generation.max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 4,
        "min_length": 10,
        "top_p": 0.9
    }
    
    # 降低学习率和beta参数，提高稳定性
    training_config = {
        "num_iterations": config.training.num_iterations,
        "steps_per_iteration": config.training.steps_per_iteration,
        "num_generations": config.training.generation.num_generations,
        "max_new_tokens": config.training.generation.max_new_tokens,
        "max_length_for_gather": config.training.generation.max_length_for_gather,
        "max_generate_iterations": config.training.generation.max_generate_iterations,
        "temperature": 0.7,
        "do_sample": True,
        "beta": 0.1,  # KL惩罚系数
        "learning_rate": 5*1e-6,  # 学习率
        "mu": 1,
        "epsilon": 0.05,  # 剪切参数
        "reward_function": overall_reward,
        "save_interval": config.training.save_interval,
    }
    logging.info(f"Training config: {training_config}")
    logging.info(f"Generation config: {generation_config}")
    
    # Optimize model memory usage
    base_model = optimize_model_memory(base_model)
    reference_base_model = optimize_model_memory(reference_base_model)

    # 对模型进行初始化和预热
    # 添加这段代码确保模型初始化正确
    if not config.training.continue_training:
        # 运行一次前向传播确保权重初始化
        dummy_input = tokenizer("这是一个测试输入", return_tensors="pt").to(device)
        with torch.no_grad():
            _ = base_model(**dummy_input)
            _ = reference_base_model(**dummy_input)
    
   
    
    # 6. 包装模型时添加更多自定义属性
    policy_model = AgenticRAGModel(base_model, tokenizer)
    reference_model = AgenticRAGModel(reference_base_model, tokenizer)
    
    # 添加generate配置到模型，只添加生成相关参数
    policy_model.generate_configs = generation_config
    reference_model.generate_configs = generation_config
    
    # 添加zero_optimization配置到模型配置
    # 解决'Qwen2Config' object is not subscriptable错误
    if not hasattr(policy_model.model, 'zero_optimization'):
        # 为模型添加zero_optimization属性
        policy_model.zero_optimization = {'stage': 0}  # 默认设置stage为0
        policy_model.config.zero_optimization = {'stage': 0}  # 同时设置到config对象
        reference_model.zero_optimization = {'stage': 0}
        reference_model.config.zero_optimization = {'stage': 0}
    
    # 在main函数中，添加current_step的定义（第224行之后，在policy_model定义之前）
    if config.training.continue_training:
        current_step = config.training.current_step
    else:
        current_step = 0

    # 修改train_with_grpo调用，让它与trainer实际接口匹配
    try:
        train_with_grpo(
            config=config,
            device=device,
            policy_model=policy_model,
            ref_base_model=reference_model,
            tokenizer=tokenizer,
            accelerator=accelerator,
            dataloader=train_dataloader,
            checkpoint_dir=checkpoint_dir,
            current_step=current_step,
            **training_config,  # 只传递train_with_grpo支持的参数
        )
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise
    finally:
        # 保存最终模型，即使训练失败
        if accelerator.is_local_main_process:
            final_ckpt = f"{checkpoint_dir}/final"
            os.makedirs(final_ckpt, exist_ok=True)
            try:
                policy_model.save_pretrained(final_ckpt)
                tokenizer.save_pretrained(final_ckpt)
                logging.info(f"Saved final model to {final_ckpt}")
            except Exception as e:
                logging.error(f"Failed to save final model: {e}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()