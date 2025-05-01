import copy
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from deepspeed import DeepSpeedEngine
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.doctor_reward import overall_reward
#from src.utils.extractor import analyze_completions
#from src.utils.utils import optimize_model_memory
from src.utils.patient_model import PatientModel


def create_completion_mask(
        completion_ids: torch.LongTensor,
        start_ids: List[int],
        end_ids: List[int],
) -> torch.LongTensor:
    """
    Create a binary mask for the completion part.
    Only tokens between <|im_start|>assistant and the next <|im_start|>, or after the last assistant, are counted.

    Args:
        completion_ids: (seq_len,) completion token ids
        eos_token_id: tokenizer.eos_token_id
        start_ids: token ids for "<|im_start|>assistant"
        end_ids: token ids for "<|im_start|>"

    Returns:
        mask: (seq_len,) 0/1 tensor
    """
    seq_len = completion_ids.size(0)
    mask = torch.zeros(seq_len, dtype=torch.long, device=completion_ids.device)

    i = 0
    while i < seq_len:
        # 找到下一个 "<|im_start|>assistant"
        if i + len(start_ids) <= seq_len and torch.all(
                completion_ids[i:i + len(start_ids)] == torch.tensor(start_ids, device=completion_ids.device)):
            i += len(start_ids)
            start_pos = i

            # 向后找下一个 "<|im_start|>"
            while i < seq_len:
                if i + len(end_ids) <= seq_len and torch.all(
                        completion_ids[i:i + len(end_ids)] == torch.tensor(end_ids, device=completion_ids.device)):
                    break
                i += 1
            end_pos = i

            # 标记[start_pos, end_pos)之间是1
            mask[start_pos:end_pos] = 1
        else:
            i += 1

    return mask


def generate_completions_multi_round(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2048,
    temperature: float = 0.7,
    do_sample: bool = True,
    max_generate_iterations: int = 8,
    patient_models: List['PatientModel'] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Multi-round generation with patient_models per sample.
    """
    device = next(model.parameters()).device
    tokenizer.padding_side = "left"

    # Step 1: Tokenize initial prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    prompt_ids = inputs["input_ids"].to(device)
    prompt_mask = inputs["attention_mask"].to(device)

    # Repeat for multiple generations
    prompt_ids = prompt_ids.repeat_interleave(num_generations, dim=0)
    prompt_mask = prompt_mask.repeat_interleave(num_generations, dim=0)

    # Expand patient_models if needed
    if patient_models is not None and num_generations > 1:
        expanded_patient_models = []
        for model_i in patient_models:
            expanded_patient_models.extend([model_i] * num_generations)
        patient_models = expanded_patient_models

    batch_size = prompt_ids.size(0)

    current_ids = prompt_ids.clone()
    current_mask = prompt_mask.clone()

    should_gen = torch.ones(batch_size, dtype=torch.bool, device=device)
    final_outputs: List[Optional[torch.LongTensor]] = [None] * batch_size #完整的prompt+所有生成内容
    completion_texts=[""] * batch_size

    #同一个batch里不同sample一起decode，需要注意padding，每轮生成完之后重新pad
    for round_idx in range(max_generate_iterations):
        print("=" * 80)
        print(f"[Round {round_idx + 1}/{max_generate_iterations}] Start")
        print(f"  should_gen: {should_gen.tolist()}")
        print(f"  current_ids shape: {current_ids.shape}")

        if not should_gen.any():
            break

        active = torch.nonzero(should_gen).squeeze(1) #获得需要继续生成的sample id
        print(f"[Generation] active batch indices: {active.tolist()}")

        #对active samples做生成
        outputs = model.generate(
            input_ids=current_ids,
            attention_mask=current_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        old_len = current_ids.size(1)
        history_ids = outputs[:, :old_len]
        new_generated_ids = outputs[:, old_len:]

        history_texts = tokenizer.batch_decode(history_ids, skip_special_tokens=False)
        generated_texts = tokenizer.batch_decode(new_generated_ids, skip_special_tokens=False)
        history_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in history_texts
        ]
        generated_texts = [
            text.replace(tokenizer.pad_token, "").strip()
            for text in generated_texts
        ]

        next_prompts = []

        for idx, text in enumerate(generated_texts):
            b = active[idx].item()
            print(f"\n[Sample {b}] Generated text: {repr(text)}")

            merged_text = history_texts[idx] #merged_text中没有额外的padding

            # Step2: Check answer or no question
            if "answer:" in text:
                completion_texts[b] += text
                merged_text += text
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False
                continue

            # Step3: Handle question
            if "question:" in text and (round_idx < max_generate_iterations-1):
                start = text.index("question:")
                question = text[start:].strip()
                try:
                    # 每个样本用自己的 patient_models[b]
                    answer = patient_models[b].get_answer(question) if patient_models is not None else "No answer available."
                except Exception as exc:
                    answer = f"Get Patient Answer Error: {exc}"

                new_text=text + '\n<|im_start|>user\n' + answer + '<|im_end|>\n<|im_start|>assistant\n'
                completion_texts[b] += new_text
                merged_text += new_text

                next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                next_prompts.append(next_prompt_ids)
            else:
                merged_text += text
                completion_texts[b] += new_text
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False

        if next_prompts:
            texts = [tokenizer.decode(t, skip_special_tokens=False) for t in next_prompts]
            tokenizer.padding_side = "left"
            enc = tokenizer(texts, add_special_tokens=False,return_tensors="pt", padding=True)
            current_ids = enc.input_ids.to(device)
            current_mask = enc.attention_mask.to(device)

    tokenizer.padding_side = "right"
    completion_ids=tokenizer(completion_texts,add_special_tokens=False,return_tensors="pt", padding=True).input_ids.to(
                    device)
    completion_masks=[]
    prompt_len = prompt_ids.size(1)  # 统一固定的prompt长度
    allowed_completion_len = max_length_for_gather - prompt_len

    if completion_ids.size(1) > allowed_completion_len:
        # 统一裁剪到 allowed_completion_len
        completion_ids = completion_ids[:, :allowed_completion_len]

    for b in range(batch_size):
        mask = create_completion_mask(
            completion_ids[b],
            start_ids=tokenizer.encode("<|im_start|>assistant", add_special_tokens=False),
            end_ids=tokenizer.encode("<|im_start|>", add_special_tokens=False),
        )
        completion_masks.append(mask)
    completion_masks = torch.stack(completion_masks, dim=0)

    return prompt_ids, prompt_mask, completion_ids,  completion_masks


def selective_log_softmax(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute log probabilities only for specified token IDs.

    Args:
        logits (torch.Tensor): Raw model logits (batch, seq_len, vocab_size).
        input_ids (torch.Tensor): Token IDs to select (batch, seq_len).

    Returns:
        torch.Tensor: Log probabilities for each input_id (batch, seq_len).
    """
    log_probs = F.log_softmax(logits, dim=-1)
    selected = log_probs.gather(dim=-1, index=input_ids.unsqueeze(-1))
    return selected.squeeze(-1)


def compute_log_probabilities(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
) -> torch.Tensor:
    """
    Compute log probabilities for the last logits_to_keep tokens.

    Args:
        model (torch.nn.Module): Model supporting logits_to_keep & obtain_logits flags.
        input_ids (torch.Tensor): Combined prompt+completion IDs.
        attention_mask (torch.Tensor): Corresponding attention mask.
        logits_to_keep (int): Number of final tokens to evaluate.

    Returns:
        torch.Tensor: Log probabilities of shape (batch, logits_to_keep).

    Raises:
        RuntimeError: If model does not support required kwargs.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,
    ).logits
    logits = outputs[:, :-1, :]
    ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, ids)


def generate_rollout_data(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_samples: Dict[str, List[Any]],
    num_generations: int,
    max_new_tokens: int,
    max_length_for_gather: int,
    temperature: float,
    do_sample: bool,
    max_generate_iterations: int,
) -> Dict[str, Any]:
    """
    Generate completions and compute log-probabilities for rollouts.

    Args:
        model (torch.nn.Module): Current policy model.
        ref_model (torch.nn.Module): Reference (static) model.
        tokenizer (AutoTokenizer): Tokenizer for decoding.
        batch_samples (Dict[str, List[Any]]): Contains "prompt", "question", "answer" lists.
        num_generations (int): Completions per prompt.
        max_new_tokens (int): Maximum new tokens.
        max_length_for_gather (int): Maximum total length.
        temperature (float): Sampling temperature.
        do_sample (bool): Sampling flag.
        max_generate_iterations (int): Maximum generate iterations.
    Returns:
        Dict[str, Any]: Rollout data including IDs, masks, log-probs, completions, etc.
    """
    prompts = batch_samples["prompt"]
    answers = batch_samples["answer"]
    batch_facts=batch_samples['facts']

    patient_model_list = []
    for facts in batch_facts:  # batch_facts: List[List[str]]
        patient_model = PatientModel(facts)
        patient_model_list.append(patient_model)

    with torch.no_grad():
        p_ids, p_mask, c_ids, c_mask = generate_completions_multi_round(
            model,
            tokenizer,
            prompts,
            num_generations,
            max_new_tokens,
            max_length_for_gather,
            temperature,
            do_sample,
            max_generate_iterations,
            patient_model_list
        )
        input_ids = torch.cat([p_ids, c_ids], dim=1)
        attention_mask = torch.cat([p_mask, c_mask], dim=1)
        k = c_ids.size(1)

        old_log_probs = compute_log_probabilities(model, input_ids, attention_mask, k)
        ref_log_probs = compute_log_probabilities(ref_model, input_ids, attention_mask, k)

    formatted = [[{"content": tokenizer.decode(ids, skip_special_tokens=False).replace(tokenizer.pad_token, "").strip()}] for ids in c_ids]
    repeated_facts = [f for f in batch_facts for _ in range(num_generations)]
    repeated_options = [o for o in batch_samples['option'] for _ in range(num_generations)]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": formatted,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "repeated_facts": repeated_facts,
        "repeated_options":repeated_options,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }


def compute_group_relative_advantages(
    rewards: torch.Tensor,
    num_generations: int,
) -> torch.Tensor:
    """
    Normalize rewards within each prompt group and handle degenerate cases.

    Args:
        rewards (torch.Tensor): Flat tensor of rewards (batch*num_gen,).
        num_generations (int): Number of completions per prompt.

    Returns:
        torch.Tensor: Advantages of shape (batch*num_gen, 1).
    """
    groups = rewards.view(-1, num_generations)
    means = groups.mean(dim=1)
    stds = groups.std(dim=1)
    mins = groups.min(dim=1).values
    maxs = groups.max(dim=1).values

    degenerate = (means == mins) | (means == maxs)
    exp_means = means.repeat_interleave(num_generations)
    exp_stds = stds.repeat_interleave(num_generations)
    mask = degenerate.repeat_interleave(num_generations)

    adv = (rewards - exp_means) / (exp_stds + 1e-4)
    # Random ±1 for degenerate groups
    rand = (torch.randint(0, 2, rewards.shape, device=rewards.device) * 2 - 1).float()
    adv[mask] = rand[mask]
    return adv.unsqueeze(1)


def maximize_grpo_objective(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    rollout_data: Dict[str, Any],
    tokenizer: AutoTokenizer,
    reward_function: Callable[..., Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    beta: float,
    epsilon: float,
    accelerator: Accelerator,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform a single GRPO update step, computing loss and backpropagating.

    Args:
        model (torch.nn.Module): Policy model.
        ref_model (torch.nn.Module): Reference model.
        rollout_data (Dict[str, Any]): Output from generate_rollout_data.
        tokenizer (AutoTokenizer): For decoding completions.
        reward_function (Callable): Function to compute rewards.
        optimizer (torch.optim.Optimizer): Optimizer instance.
        beta (float): KL penalty coefficient.
        epsilon (float): Clipping parameter.
        accelerator (Accelerator): For distributed training.

    Returns:
        Tuple[float, float, Dict[str, Any]]: Loss value, average reward, full reward dict.
    """
    input_ids = rollout_data["input_ids"]
    attention_mask = rollout_data["attention_mask"]
    comp_mask = rollout_data["completion_mask"]
    old_lp = rollout_data["old_log_probs"]
    ref_lp = rollout_data["ref_log_probs"]
    k = rollout_data["logits_to_keep"]

    # Current policy log probs
    curr_lp = compute_log_probabilities(model, input_ids, attention_mask, k)
    ratio = torch.exp(curr_lp - old_lp)

    rewards_dict = reward_function(
        model=model,
        tokenizer=tokenizer,
        facts=rollout_data["repeated_facts"],
        completions=rollout_data["formatted_completions"],
        options=rollout_data["repeated_options"],
        answers=rollout_data["repeated_answers"]
    )
    rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
    avg_reward = float(rewards.mean())

    adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * adv
    surr = torch.min(surr1, surr2)

    kl = torch.exp(ref_lp - curr_lp) - (ref_lp - curr_lp) - 1
    per_token = surr - beta * kl
    loss = -((per_token * comp_mask).sum(dim=1) / comp_mask.sum(dim=1)).mean()

    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    return float(loss), avg_reward, rewards_dict


from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.utils.utils import optimize_model_memory


def build_model(
    config,
    device: torch.device,
):
    """
    Build and return a language model based on the provided config and device.
    This function handles tokenizer loading, (Q)LoRA application, and memory optimization.
    """
    continue_training = config.training.continue_training
    checkpoint_step = config.training.current_step
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
    ).to(device)
    
    # Apply LoRA if needed
    if config.training.use_lora:
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        if continue_training:
            weights_path = f"checkpoints/{config.experiment.name}/step-{checkpoint_step:04d}"
            model = PeftModel.from_pretrained(model, weights_path, config=lora_cfg, is_trainable=True)
        else:
            model = get_peft_model(model, lora_cfg)
            
    # Quantization config (if using Quant)
    if config.training.use_quant:
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=config.qlora.load_in_4bit,
            bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),  # optional
            bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,         # optional
            bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,               # optional
            load_in_8bit= config.qlora.load_in_8bit,  # enable 8bit quantization
            llm_int8_threshold = config.qlora.llm_int8_threshold, # if load_in_8bit is True
        )
        
        model = load_and_quantize_model(
            model,
            bnb_quantization_config=bnb_quantization_config,
            device_map = "auto"
        )
        
        logging.info(f"Using Quant: {config.qlora}")
    else:
        bnb_quantization_config = None
        logging.info("Not using Quant")
    
    # Optimize memory usage
    model = optimize_model_memory(model)
    # Wrap and return AgenticRAGModel
    return model


def train_with_grpo(
    config: Dict[str, Any],
    device: torch.device,
    policy_model: torch.nn.Module,
    ref_base_model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    accelerator: Optional[Accelerator] = None,
    dataloader: Optional[torch.utils.data.DataLoader] = None,
    num_iterations: int = 1,
    steps_per_iteration: int = 500,
    num_generations: int = 4,
    max_new_tokens: int = 128,
    max_length_for_gather: int = 2000,
    max_generate_iterations: int = 8,
    temperature: float = 0.7,
    do_sample: bool = True,
    beta: float = 0.1,
    learning_rate: float = 5e-6,
    mu: int = 1,
    epsilon: float = 0.2,
    reward_function: Callable[..., Dict[str, Any]] = overall_reward,
    checkpoint_dir: Optional[str] = None,
    current_step: int = 0,
    save_interval: int = 5,
) -> None:
    """
    Train policy model using GRPO fine-tuning with periodic checkpointing.

    Args:
        policy_model (torch.nn.Module): The policy network.
        base_reference_model (torch.nn.Module): Base model for reference rollouts.
        tokenizer (AutoTokenizer): Tokenizer for data processing.
        accelerator (Optional[Accelerator]): Accelerator for distributed training.
        dataloader (Optional[DataLoader]): Training data loader.
        num_iterations (int): Number of outer iterations.
        steps_per_iteration (int): Max steps per iteration.
        num_generations (int): Completions per prompt.
        max_new_tokens (int): Max tokens to generate.
        max_length_for_gather (int): Max sequence length.
        temperature (float): Sampling temperature.
        do_sample (bool): Whether to sample or greedy.
        beta (float): KL penalty coefficient.
        learning_rate (float): Optimizer learning rate.
        mu (int): GRPO updates per batch.
        epsilon (float): Clipping epsilon.
        reward_function (Callable): Reward computation function.
        checkpoint_dir (Optional[str]): Path to save checkpoints.
        current_step (int): Starting training step.
        save_interval (int): Steps between saves.

    Raises:
        RuntimeError: On training failures or save errors.
    """    
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    
    zero_stage = policy_model.config['zero_optimization']['stage']
    
    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        logging.info(f"start GRPO iteration {it}/{num_iterations}")
        torch.cuda.empty_cache()

        ref_model = build_model(config, device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # Sync LoRA weights to reference
        lora_params = [p for n, p in policy_model.named_parameters() if "lora" in n]
        with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
            sd = policy_model.state_dict()
            lora_sd = {k: v for k, v in sd.items() if "lora" in k}
            ref_model.load_state_dict(lora_sd, strict=False)
            ref_model.to(accelerator.device)
            
        if zero_stage != 2:
            ref_model = accelerator.prepare(ref_model)  # 如果是 Stage 3，准备模型
        else:   # 如果是Stage 2，因为ref model不需要优化，所以ref model不需要用zero 2的优化optimizer
            pass

        step = 0
        for batch in dataloader:
            logging.info(f"start to generate rollout data, step {step+1}/{min(steps_per_iteration, len(dataloader))}")
            with torch.no_grad():
                rollout = generate_rollout_data(
                    policy_model,
                    ref_model,
                    tokenizer,
                    batch,
                    num_generations,
                    max_new_tokens,
                    max_length_for_gather,
                    temperature,
                    do_sample,
                    max_generate_iterations,
                )
            logging.info(f"success to generate rollout data")
            for _ in range(mu):
                loss_val, avg_r, rdict = maximize_grpo_objective(
                    policy_model, ref_model, rollout, tokenizer, reward_function, optimizer, beta, epsilon, accelerator
                )
            logging.info(f"success to maximize grpo objective")

            print(
                f"Iteration {it}/{num_iterations}, Step {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"Loss: {loss_val:.6f}, Avg Reward: {avg_r:.2f}"
            )
            if accelerator.is_local_main_process:
                swanlab.log(
                    {
                        "Iteration": it,
                        "Step": step+1,
                        "Loss": loss_val,
                        "Avg Reward": avg_r,
                    }
                )

            # Logging and checkpointing omitted for brevity
            sum_steps += 1
            step += 1
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    policy_model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
            if step >= steps_per_iteration:
                break

            accelerator.wait_for_everyone()

        del ref_model       # 清除历史的ref model，节约内存
        torch.cuda.empty_cache()


if __name__ == '__main__':
    import json
    from src.data.doctor_patient_prompts import *
    from torch.utils.data import DataLoader
    from src.data.prepare_dataset import prepare_dataset
    from accelerate import Accelerator, init_empty_weights


    def custom_collate_fn(batch):
        """
        Collate a batch of dicts with potentially non-tensor and variable-length fields.
        This version preserves lists and dicts as-is without stacking.
        """
        collated = {key: [sample[key] for sample in batch] for key in batch[0]}
        return collated


    dataset = prepare_dataset("train", 'cmb', eval_size=2)
    train_dataloader=DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)
    accelerator = Accelerator()

    model_name_or_path = "/data1/dhx/llmbase/hub/Qwen/Qwen2___5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    )
    num_generations=2
    max_new_tokens=512
    max_length_for_gather=2048
    temperature=0.7
    do_sample=True
    max_generate_iterations=4

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    for batch in train_dataloader:
        with torch.no_grad():
            rollout = generate_rollout_data(
                model,
                model,
                tokenizer,
                batch,
                num_generations,
                max_new_tokens,
                max_length_for_gather,
                temperature,
                do_sample,
                max_generate_iterations,
            )

        print("="*80)
        print("Final Results:")
        for completion in rollout['formatted_completions']:
            print("*"*80)
            print(completion)

        print("="*80)
        print("Rewards:")
        rewards_dict = overall_reward(
            model=model,
            tokenizer=tokenizer,
            facts=rollout["repeated_facts"],
            completions=rollout["formatted_completions"],
            options=rollout["repeated_options"],
            answers=rollout["repeated_answers"]
        )
        print(rewards_dict)
        break
