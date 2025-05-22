import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dotmap import DotMap
from peft import PeftModel, get_peft_model_state_dict, LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.doctor_reward import overall_reward
from src.utils.utils import optimize_model_memory
from src.utils.patient_model import PatientModel
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
try:
    from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
except ImportError:
    BnbQuantizationConfig = None
    load_and_quantize_model = None



def create_completion_mask(
        completion_ids: torch.LongTensor,
        tokenizer: AutoTokenizer,
) -> torch.LongTensor:
    """
    创建一个二进制掩码，标记医生模型生成的所有内容。
    
    规则：
    1. 所有非padding内容默认标记为1
    2. 用户输入部分（<|im_start|>user到<|im_end|>之间）标记为0
    3. <|endoftext|>后的所有标记都设为0
    4. 所有<|im_start|>assistant和<|im_end|>标记都设为0
    
    Args:
        completion_ids: (seq_len,) 完成部分的token IDs
        tokenizer: 使用的tokenizer，用于编码特殊标记

    Returns:
        mask: (seq_len,) 0/1张量，1表示参与训练的token
    """
    seq_len = completion_ids.size(0)
    mask = torch.zeros(seq_len, dtype=torch.long, device=completion_ids.device)
    
    # 找到第一个非padding token的位置
    start_pos = 0
    while start_pos < seq_len and completion_ids[start_pos] == 0:
        start_pos += 1
    
    # 默认将所有非padding内容标记为1
    mask[start_pos:] = 1
    
    # 排除用户输入部分
    user_start_ids = tokenizer.encode("<|im_start|>user", add_special_tokens=False)
    user_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    
    # 排除assistant标记部分
    assistant_start_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    
    # 排除<|endoftext|>后的所有内容
    eos_ids = tokenizer.encode("<|endoftext|>", add_special_tokens=False)
    
    i = 0
    while i < seq_len:
        # 检查是否是用户输入开始
        if i + len(user_start_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(user_start_ids)] == torch.tensor(user_start_ids, device=completion_ids.device)):
            user_start_pos = i  # 包括<|im_start|>user标记
            i += len(user_start_ids)
            
            # 查找用户输入结束
            while i < seq_len:
                if i + len(user_end_ids) <= seq_len and torch.all(
                        completion_ids[i:i+len(user_end_ids)] == torch.tensor(user_end_ids, device=completion_ids.device)):
                    user_end_pos = i + len(user_end_ids)  # 包括<|im_end|>标记
                    break
                i += 1
                
            if i < seq_len:  # 找到了用户输入结束
                # 将整个用户输入部分标记为0
                mask[user_start_pos:user_end_pos] = 0
        
        # 检查是否是assistant标记开始
        elif i + len(assistant_start_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(assistant_start_ids)] == torch.tensor(assistant_start_ids, device=completion_ids.device)):
            # 将assistant标记部分标记为0
            assistant_end_pos = i + len(assistant_start_ids)
            mask[i:assistant_end_pos] = 0
            i = assistant_end_pos
        
        # 检查是否是独立的<|im_end|>标记
        elif i + len(user_end_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(user_end_ids)] == torch.tensor(user_end_ids, device=completion_ids.device)):
            # 将<|im_end|>标记设为0
            mask[i:i+len(user_end_ids)] = 0
            i += len(user_end_ids)
        
        # 检查是否是EOS标记
        elif i + len(eos_ids) <= seq_len and torch.all(
                completion_ids[i:i+len(eos_ids)] == torch.tensor(eos_ids, device=completion_ids.device)):
            # 找到第一个EOS标记，将它及之后的所有标记设为0
            mask[i:] = 0
            break
            
        else:
            i += 1
            
    return mask


def _unwrap_peft(model):
    """
    Sequentially unwrap DeepSpeedEngine / model, and return the PeftModel.
    如果不是PEFT模型，返回None而不是引发异常。
    """
    if isinstance(model, deepspeed.DeepSpeedEngine):
        model = model.module  # --> 基础模型

    if hasattr(model, "model"):
        model = model.model  # --> PeftModel

    if not isinstance(model, PeftModel):
        logging.warning("底层模型不是PeftModel，可能未应用LoRA或使用了其他方式")
        return None

    return model


def save_lora_only_in_zero2(engine, tokenizer, ckpt_dir):
    """
    save lora only for ZeRO-2
    如果模型不是PEFT模型，则使用常规方式保存
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    peft_model = _unwrap_peft(engine)
    if peft_model is None:
        logging.warning("模型不是PEFT模型，使用常规方式保存")
        if isinstance(engine, deepspeed.DeepSpeedEngine):
            state_dict = engine.module.state_dict()
        else:
            state_dict = engine.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)
        return

    lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n]
    if not lora_params:
        logging.warning("未找到任何LoRA参数，使用常规方式保存")
        if isinstance(engine, deepspeed.DeepSpeedEngine):
            state_dict = engine.module.state_dict()
        else:
            state_dict = engine.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(ckpt_dir)
        return

    enabled = isinstance(engine, deepspeed.DeepSpeedEngine) and engine.zero_optimization_stage() == 2

    with deepspeed.zero.GatheredParameters(lora_params, enabled=enabled):
        lora_state = get_peft_model_state_dict(peft_model)

    peft_model.save_pretrained(ckpt_dir, state_dict=lora_state)
    tokenizer.save_pretrained(ckpt_dir)



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

                new_text = text + '\n<|im_start|>user\n' + answer + '<|im_end|>\n<|im_start|>assistant\n'
                completion_texts[b] += new_text
                merged_text += new_text

                next_prompt_ids = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                next_prompts.append(next_prompt_ids)
            else:
                merged_text += text
                completion_texts[b] += text
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
            tokenizer,
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
    计算最后 logits_to_keep 个token的对数概率。
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        logits_to_keep=logits_to_keep + 1,
        obtain_logits=True,
    )
    
    # 正确提取logits - 如果outputs是元组，取第一个元素（通常是logits）
    if isinstance(outputs, tuple):
        logits = outputs[0]
    else:
        # 如果输出是TransformerOutput对象
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        
    # 确保logits是张量后再进行切片
    logits = logits[:, :-1, :]
    ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:, :]
    return selective_log_softmax(logits, ids)



def parse_dialog(completion: str) -> List[Dict[str, str]]:
    """解析对话内容为结构化格式"""
    dialog = []
    pattern = re.compile(r"<\|im_start\|>(user|assistant)\s*(.*?)(?=<\|im_start\|>|$)", re.DOTALL)
    
    # 处理可能的初始内容
    initial_parts = completion.split("<|im_start|>", 1)
    if initial_parts[0].strip():
        dialog.append({
            "role": "assistant", 
            "content": initial_parts[0].strip()
        })
    
    for match in pattern.finditer("<|im_start|>" + completion):
        role = match.group(1).strip()
        content = match.group(2).strip()
        dialog.append({"role": role, "content": content})
    
    return dialog


def parse_dialog_simple(completion: str) -> List[Dict[str, str]]:
    """
    简单解析对话内容为结构化格式，支持中英文标记
    
    Args:
        completion: 包含对话内容的字符串
        
    Returns:
        List[Dict[str, str]]: 包含角色和内容的字典列表
    """
    dialog = []
    # 首先清理可能存在的HTML标签
    completion = re.sub(r'<br\s*/?>', '\n', completion)
    completion = re.sub(r'</?(?:p|ol|ul|li|div|span|h\d|strong|em)[^>]*>', '', completion)
    
    # 按特殊标记分割对话
    parts = completion.split("<|im_start|>")
    
    # 处理第一部分（如果不是空的）
    if parts[0].strip():
        # 默认第一部分为医生回复
        dialog.append({
            "role": "assistant",
            "content": parts[0].strip()
        })
    
    # 处理其余部分
    for part in parts[1:]:
        if not part.strip():
            continue
            
        try:
            # 提取角色和内容
            if part.startswith("user"):
                role = "user"
                content = part[4:].strip()  # 4 = len("user")
            elif part.startswith("assistant"):
                role = "assistant"
                content = part[9:].strip()  # 9 = len("assistant")
            else:
                # 无法识别的角色，默认为assistant
                role = "assistant"
                content = part.strip()
                
            # 处理内容中的结束标记
            if "<|im_end|>" in content:
                content = content.split("<|im_end|>")[0].strip()
                
            # 移除空对话
            if content.strip():
                dialog.append({"role": role, "content": content})
        except Exception as e:
            logging.warning(f"解析对话部分时出错: {e}, 部分内容: {part[:50]}...")
    
    # 如果没有提取到对话，尝试使用问题/回答格式解析
    if not dialog:
        try:
            # 尝试识别问题和回答格式
            qa_parts = re.split(r'(问题:|question:|回答:|answer:|答案:)', completion, flags=re.IGNORECASE)
            current_role = "assistant"
            current_content = ""
            
            for i, part in enumerate(qa_parts):
                part = part.strip()
                if not part:
                    continue
                
                lower_part = part.lower()
                if lower_part in ['问题:', 'question:']:
                    # 保存之前的内容
                    if current_content:
                        dialog.append({"role": current_role, "content": current_content.strip()})
                    current_role = "assistant"  # 问题由医生提出
                    current_content = "question: "  # 新内容前缀
                elif lower_part in ['回答:', 'answer:', '答案:']:
                    # 保存之前的内容
                    if current_content:
                        dialog.append({"role": current_role, "content": current_content.strip()})
                    current_role = "assistant"  # 回答由医生给出
                    current_content = "answer: "  # 新内容前缀
                else:
                    # 添加内容到当前部分
                    if current_content or not dialog:
                        current_content += part
                    else:
                        # 如果没有明确标记且已有对话，则添加为新的回复
                        dialog.append({"role": "assistant", "content": part})
            
            # 添加最后一部分
            if current_content:
                dialog.append({"role": current_role, "content": current_content.strip()})
        except Exception as e:
            logging.warning(f"尝试问答格式解析失败: {e}")
            # 如果所有解析都失败，至少返回整个内容作为一个对话轮次
            if not dialog:
                dialog.append({"role": "assistant", "content": completion.strip()})
    
    return dialog



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

    # 修改生成的内容显示，确保格式一致性
    completions = []
    for ids in c_ids:
        raw_text = tokenizer.decode(ids, skip_special_tokens=False).replace(tokenizer.pad_token, "").strip()
        # 清理HTML标签
        clean_text = re.sub(r'<br\s*/?>', '\n', raw_text)
        clean_text = re.sub(r'</?[a-zA-Z][^>]*>', '', clean_text)
        completions.append([{"content": clean_text}])
    
    # 记录对话内容
    logging.info("="*80)
    logging.info("生成的对话内容详情：")
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        logging.info(f"Sample {i} Content:")
        logging.info(f"{content}")
        logging.info("-"*50)
        
        # 记录对话内容的mask信息
        mask_sum = c_mask[i].sum().item()
        mask_percentage = (mask_sum / c_mask[i].size(0)) * 100
        logging.info(f"Mask信息: 总和={mask_sum}, 占比={mask_percentage:.2f}%, 总长度={c_mask[i].size(0)}")
        
        # 分析对话结构
        try:
            # 更健壮的对话解析
            dialog = parse_dialog_simple(content)
            
            # 查找对话中的问题和回答标记
            has_assistant_tag = "<|im_start|>assistant" in content
            has_question = "question:" in content.lower() or "问题:" in content.lower()
            has_answer = "answer:" in content.lower() or "回答:" in content.lower() or "答案:" in content.lower()
            
            # 如果对话中有多个交互，显示完整对话
            if len(dialog) > 1:
                logging.info("完整对话交互:")
                for turn in dialog:
                    role = "医生" if turn["role"] == "assistant" else "患者"
                    turn_content = turn["content"]
                    logging.info(f"{role}: {turn_content}")
        except Exception as e:
            logging.warning(f"解析对话时出错: {e}")
        
        # 记录标记检查
        logging.info(f"标记检查: assistant标记={has_assistant_tag}, question标记={has_question}, answer标记={has_answer}")
    
    logging.info("="*80)
    
    repeated_facts = [f for f in batch_facts for _ in range(num_generations)]
    repeated_options = [o for o in batch_samples['option'] for _ in range(num_generations)]
    repeated_prompts = [p for p in prompts for _ in range(num_generations)]
    repeated_answers = [a for a in answers for _ in range(num_generations)]

    # 打印rollout数据结构信息
    print("\n" + "="*80)
    print("Rollout数据结构信息:")
    for key, value in {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "repeated_facts": repeated_facts,
        "repeated_options": repeated_options,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
    }.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: 形状={value.shape}, 类型={value.dtype}")
        elif isinstance(value, list):
            print(f"{key}: 类型=列表, 长度={len(value)}")
            if value and hasattr(value[0], 'keys'):
                print(f"  - 首个元素键: {list(value[0].keys())}")
        else:
            print(f"{key}: 类型={type(value)}")

    # 简单的内容有效性检查函数
    def simple_valid_content_check(content):
        # 检查内容是否包含必要的标记或文本
        if not content or len(content) < 10:  # 内容太短
            return False
        # 检查是否包含问题或回答
        has_q = "question:" in content.lower() or "问题:" in content.lower()
        has_a = "answer:" in content.lower() or "回答:" in content.lower() or "答案:" in content.lower()
        return has_q or has_a

    # 在generate_rollout_data函数中添加质量检查
    valid_completions = []
    for i, completion in enumerate(completions):
        content = completion[0]["content"]
        is_valid = simple_valid_content_check(content)
        if not is_valid:
            logging.warning(f"Sample {i} failed quality check: {content[:50]}...")
            # 可以在这里重新生成或使用替代内容
        valid_completions.append(is_valid)

    # 更新返回值中的标志
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": c_mask,
        "old_log_probs": old_log_probs,
        "ref_log_probs": ref_log_probs,
        "formatted_completions": completions,
        "repeated_prompts": repeated_prompts,
        "repeated_answers": repeated_answers,
        "repeated_facts": repeated_facts,
        "repeated_options":repeated_options,
        "logits_to_keep": k,
        "batch_size": len(prompts),
        "num_generations": num_generations,
        "valid_completions": valid_completions
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

    # rewards_dict = reward_function(
    #     prompts=rollout_data["repeated_prompts"],
    #     completions=rollout_data["formatted_completions"],
    #     answers=rollout_data["repeated_answers"],
    # )
    rewards_dict = reward_function(
        model=model,  # 添加模型参数
        tokenizer=tokenizer,  # 添加tokenizer参数
        facts=rollout_data["repeated_facts"],  # 使用facts而不是prompts
        completions=rollout_data["formatted_completions"],
        options=rollout_data["repeated_options"],  # 添加options参数
        answers=rollout_data["repeated_answers"],
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


def build_model(
    config,
    device: torch.device,
):
    """
    构建并返回基于提供的配置和设备的语言模型。
    该函数处理分词器加载、(Q)LoRA应用和内存优化。
    支持DeepSpeed ZeRO-2和ZeRO-3分布式训练。
    
    Returns:
        Tuple[torch.nn.Module, AutoTokenizer]: 返回(model, tokenizer)元组
    """
    continue_training = config.training.continue_training
    checkpoint_step = config.training.current_step
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        torch_dtype=getattr(torch, config.model.torch_dtype),
        trust_remote_code=True,
    ).to(device)
    
    logging.info(f"基础模型加载完成，类型: {type(model)}")
    
    # 应用LoRA（如果需要）
    if config.training.use_lora:
        logging.info("开始应用LoRA配置...")
        lora_cfg = LoraConfig(
            r=config.lora.r,
            lora_alpha=config.lora.lora_alpha,
            target_modules=config.lora.target_modules,
            lora_dropout=config.lora.lora_dropout,
            bias=config.lora.bias,
            task_type=config.lora.task_type,
        )
        logging.info(f"LoRA配置: {lora_cfg}")
        
        if continue_training:
            weights_path = f"checkpoints/{config.experiment.name}/step-{checkpoint_step:04d}"
            logging.info(f"从检查点加载LoRA权重: {weights_path}")
            model = PeftModel.from_pretrained(model, weights_path, config=lora_cfg, is_trainable=True)
        else:
            logging.info("应用LoRA配置到模型")
            model = get_peft_model(model, lora_cfg)
        
        # 验证LoRA是否正确应用
        logging.info(f"应用LoRA后的模型类型: {type(model)}")
        lora_params = [n for n, _ in model.named_parameters() if "lora" in n]
        logging.info(f"模型包含 {len(lora_params)} 个LoRA参数")
        if lora_params:
            logging.info(f"LoRA参数示例: {lora_params[:5]}")
        else:
            logging.warning("警告: 未找到任何LoRA参数!")
            
    # 量化配置（如果使用量化）
    if config.training.use_quant:
        # 优先使用accelerate的量化配置
        if BnbQuantizationConfig is not None and load_and_quantize_model is not None:
            logging.info("使用accelerate的量化配置")
            bnb_quantization_config = BnbQuantizationConfig(
                load_in_4bit=config.qlora.load_in_4bit,
                bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
                load_in_8bit=config.qlora.load_in_8bit,
                llm_int8_threshold=config.qlora.llm_int8_threshold,
            )
            
            model = load_and_quantize_model(model, bnb_quantization_config=bnb_quantization_config, device_map="auto")
            logging.info(f"使用量化: {config.qlora}")
        # 回退到transformers的BitsAndBytesConfig
        elif BitsAndBytesConfig is not None:
            logging.info("使用transformers的量化配置")
            try:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=config.qlora.load_in_4bit,
                    bnb_4bit_compute_dtype=getattr(torch, config.qlora.bnb_4bit_compute_dtype),
                    bnb_4bit_use_double_quant=config.qlora.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=config.qlora.bnb_4bit_quant_type,
                    load_in_8bit=config.qlora.load_in_8bit,
                    llm_int8_threshold=config.qlora.llm_int8_threshold,
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    config.model.name,
                    quantization_config=bnb_config,
                    torch_dtype=getattr(torch, config.model.torch_dtype),
                    trust_remote_code=True,
                    device_map="auto"
                )
                
                logging.info(f"使用量化: {config.qlora}")
            except Exception as e:
                logging.error(f"量化过程中出错: {e}")
                logging.warning("回退到非量化模型")
        else:
            logging.warning("量化配置不可用，请安装transformers>=4.30.0或bitsandbytes>=0.39.0")
            logging.warning("跳过量化，使用原始模型")
    else:
        logging.info("不使用量化")
    
    # 优化内存使用
    model = optimize_model_memory(model)
    
    return model, tokenizer


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
    使用GRPO微调训练策略模型，支持DeepSpeed ZeRO-2和ZeRO-3分布式训练
    """    
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)
    policy_model.train()
    policy_model, optimizer, dataloader = accelerator.prepare(policy_model, optimizer, dataloader)
    
    # 获取zero_stage - 采用多种方法尝试获取
    zero_stage = None
    try:
        if hasattr(policy_model, 'config') and isinstance(policy_model.config, dict) and 'zero_optimization' in policy_model.config:
            zero_stage = policy_model.config['zero_optimization']['stage']
        elif isinstance(policy_model, deepspeed.DeepSpeedEngine):
            zero_stage = policy_model.zero_optimization_stage()
        else:
            # 检查accelerator中的配置
            deepspeed_plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
            if deepspeed_plugin is not None and hasattr(deepspeed_plugin, 'zero_stage'):
                zero_stage = deepspeed_plugin.zero_stage
    except Exception as e:
        logging.warning(f"无法获取zero_stage值: {str(e)}")
    
    if zero_stage is None:
        zero_stage = 3  # 默认值
    
    logging.info(f"使用DeepSpeed ZeRO-{zero_stage}进行训练")
    
    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        logging.info(f"开始GRPO迭代 {it}/{num_iterations}")
        torch.cuda.empty_cache()

        # 构建参考模型 - 这里不使用build_model函数避免再次应用LoRA
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.torch_dtype),
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()
        
        # 确保参考模型不需要梯度
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # 同步LoRA权重到参考模型 - 仅在policy_model是带LoRA的模型时执行
        if config.training.use_lora:
            try:
                # 获取policy_model的LoRA参数
                if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                    policy_unwrapped = policy_model.module
                else:
                    policy_unwrapped = policy_model
                
                # 如果原始模型被DeepSpeed包装，需要额外的解包
                while hasattr(policy_unwrapped, "module"):
                    policy_unwrapped = policy_unwrapped.module
                
                # 应用LoRA配置到参考模型
                lora_cfg = LoraConfig(
                    r=config.lora.r,
                    lora_alpha=config.lora.lora_alpha,
                    target_modules=config.lora.target_modules,
                    lora_dropout=config.lora.lora_dropout,
                    bias=config.lora.bias,
                    task_type=config.lora.task_type,
                )
                ref_model = get_peft_model(ref_model, lora_cfg)
                
                # 获取LoRA参数并同步
                lora_params = [p for n, p in policy_unwrapped.named_parameters() if "lora" in n]
                with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
                    sd = policy_unwrapped.state_dict()
                    lora_sd = {k: v for k, v in sd.items() if "lora" in k}
                    ref_model.load_state_dict(lora_sd, strict=False)
                
                logging.info("成功将LoRA权重同步到参考模型")
            except Exception as e:
                logging.error(f"同步LoRA权重失败: {e}")
                logging.warning("继续使用未同步的参考模型")
        
        # 将参考模型移至正确的设备但不使用DeepSpeed包装
        ref_model = ref_model.to(accelerator.device)

        step = 0
        for batch in dataloader:
            logging.info(f"开始生成rollout数据, 步骤 {step+1}/{min(steps_per_iteration, len(dataloader))}")
            with torch.no_grad():
                rollout = generate_rollout_data(
                    policy_model,
                    ref_model,
                    tokenizer,  # 使用传入的tokenizer，保持原有行为一致
                    batch,
                    num_generations,
                    max_new_tokens,
                    max_length_for_gather,
                    temperature,
                    do_sample,
                    max_generate_iterations,
                )
            logging.info("成功生成rollout数据")
            
            # 执行mu次GRPO更新
            for _ in range(mu):
                # 确保参数顺序与maximize_grpo_objective的定义匹配
                loss_val, avg_r, rdict = maximize_grpo_objective(
                    model=policy_model, 
                    ref_model=ref_model, 
                    rollout_data=rollout, 
                    tokenizer=tokenizer, 
                    reward_function=reward_function, 
                    optimizer=optimizer, 
                    beta=beta, 
                    epsilon=epsilon, 
                    accelerator=accelerator
                )
            logging.info("成功最大化GRPO目标函数")

            print(
                f"迭代 {it}/{num_iterations}, 步骤 {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"损失: {loss_val:.6f}, 平均奖励: {avg_r:.2f}"
            )
            if accelerator.is_local_main_process:
                try:
                    swanlab.log(
                        {
                            "Iteration": it,
                            "Step": step+1,
                            "Loss": loss_val,
                            "Avg Reward": avg_r,
                        }
                    )
                except Exception as e:
                    logging.warning(f"记录SwanLab日志失败: {str(e)}")

            sum_steps += 1
            step += 1
            
            # 保存检查点
            if sum_steps % save_interval == 0 and sum_steps > current_step:
                if accelerator.is_local_main_process:
                    logging.info(f"保存检查点，步骤 {sum_steps}")
                    ckpt = f"{checkpoint_dir}/step-{sum_steps:04d}"
                    os.makedirs(ckpt, exist_ok=True)
                    
                    # 根据模型类型和ZeRO级别使用不同的保存方法
                    try:
                        # 检查模型是否应用了LoRA
                        is_peft_model = False
                        if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                            model_module = policy_model.module
                            if hasattr(model_module, "model") and "PeftModel" in str(type(model_module.model)):
                                is_peft_model = True
                        else:
                            if hasattr(policy_model, "model") and "PeftModel" in str(type(policy_model.model)):
                                is_peft_model = True
                        
                        logging.info(f"模型是否应用了LoRA: {is_peft_model}")
                        
                        if is_peft_model and zero_stage == 2:
                            logging.info("使用save_lora_only_in_zero2保存LoRA权重")
                            save_lora_only_in_zero2(policy_model, tokenizer, ckpt)
                        else:
                            logging.info("使用常规方法保存模型")
                            if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                                # 使用DeepSpeed的保存方法
                                state_dict = policy_model.module.state_dict() if hasattr(policy_model, "module") else policy_model.state_dict()
                                torch.save(state_dict, os.path.join(ckpt, "pytorch_model.bin"))
                            else:
                                policy_model.save_pretrained(ckpt)
                            tokenizer.save_pretrained(ckpt)
                    except Exception as e:
                        logging.error(f"保存检查点时出错: {e}")
                        import traceback
                        logging.error(traceback.format_exc())
                            
            if step >= steps_per_iteration:
                break

            # 等待所有进程
            accelerator.wait_for_everyone()

        # 清除参考模型释放内存
        del ref_model
        torch.cuda.empty_cache()

    # 在训练结束时调用swanlab.finish()
    if accelerator.is_local_main_process:
        try:
            swanlab.finish()
            logging.info("SwanLab实验已完成")
        except Exception as e:
            logging.warning(f"调用swanlab.finish()失败: {str(e)}")









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

    train_dataset, eval_dataset = prepare_dataset("train", 'cmb', eval_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    accelerator = Accelerator()


    # dataset = prepare_dataset("train", 'cmb', eval_size=2)
    # train_dataloader=DataLoader(dataset, batch_size=2, shuffle=True,collate_fn=custom_collate_fn)
    # accelerator = Accelerator()

    model_name_or_path = "/root/小北健康-模型"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    )
    num_generations=3
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

        # 打印rollout数据结构信息
        print("\n" + "="*80)
        print("Rollout数据结构信息:")
        for key, value in rollout.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: 形状={value.shape}, 类型={value.dtype}")
            elif isinstance(value, list):
                print(f"{key}: 类型=列表, 长度={len(value)}")
                if value and hasattr(value[0], 'keys'):
                    print(f"  - 首个元素键: {list(value[0].keys())}")
            else:
                print(f"{key}: 类型={type(value)}")
        
        # 打印completion_mask的内容和统计信息
        print("\n" + "="*80)
        print("Completion Mask 详细信息:")
        c_mask = rollout["completion_mask"]
        
        # 打印整体统计
        total_mask_sum = c_mask.sum().item()
        total_elements = c_mask.numel()
        print(f"总体Mask统计: 非零元素总数={total_mask_sum}, 总元素数={total_elements}, 占比={(total_mask_sum/total_elements)*100:.2f}%")
        
        # 每个样本的统计
        for i in range(c_mask.size(0)):
            mask_sum = c_mask[i].sum().item()
            mask_percentage = (mask_sum / c_mask[i].size(0)) * 100
            print(f"样本 {i}: 非零元素数量={mask_sum}, 总长度={c_mask[i].size(0)}, 占比={mask_percentage:.2f}%")
            
            # 打印第一个1和最后一个1的位置
            if mask_sum > 0:
                first_one = (c_mask[i] == 1).nonzero()[0].item()
                last_one = (c_mask[i] == 1).nonzero()[-1].item()
                print(f"  第一个1的位置: {first_one}, 最后一个1的位置: {last_one}")
                
                # 直接从input_ids的后半部分获取对应的token文本
                completion_length = rollout["logits_to_keep"]
                
                # 注意：completion_ids是单独的tensor，不是input_ids的一部分
                # 我们需要检查c_mask[i]中的1对应于completion_ids[i]中的哪些token
                text = tokenizer.decode(rollout["input_ids"][i][-completion_length:])
                print(f"  完整的completion文本: {text}")
                
                # 获取第一个标记为1的token及其周围上下文
                c_ids = rollout["input_ids"][i, -completion_length:]
                start_idx = max(0, first_one - 5)
                end_idx = min(first_one + 10, c_ids.size(0))
                context_ids = c_ids[start_idx:end_idx]
                context_text = tokenizer.decode(context_ids)
                print(f"  第一个mask=1处的上下文: {context_text}")
                
                # 打印前10个被标记为1的token
                ones_indices = (c_mask[i] == 1).nonzero().squeeze().tolist()
                if not isinstance(ones_indices, list):
                    ones_indices = [ones_indices]  # 处理只有一个元素的情况
                ones_indices = ones_indices[:10]  # 只取前10个
                ones_tokens = [tokenizer.decode(c_ids[idx:idx+1]) for idx in ones_indices]
                print(f"  mask=1的前10个token: {ones_tokens}")
                
                # 查找特定模式
                assistant_pattern = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
                start_pattern = tokenizer.encode("<|im_start|>", add_special_tokens=False)
                
                # 寻找这些pattern在completion_ids中的位置
                for j in range(len(c_ids) - len(assistant_pattern) + 1):
                    if torch.all(c_ids[j:j+len(assistant_pattern)] == torch.tensor(assistant_pattern, device=c_ids.device)):
                        print(f"  找到<|im_start|>assistant在completion第{j}位")
                        # 检查这个位置的mask值
                        if j < len(c_mask[i]):
                            print(f"  该位置的mask值: {c_mask[i][j:j+len(assistant_pattern)].tolist()}")
                
                for j in range(len(c_ids) - len(start_pattern) + 1):
                    if torch.all(c_ids[j:j+len(start_pattern)] == torch.tensor(start_pattern, device=c_ids.device)):
                        print(f"  找到<|im_start|>在completion第{j}位")
                        # 检查这个位置的mask值
                        if j < len(c_mask[i]):
                            print(f"  该位置的mask值: {c_mask[i][j:j+len(start_pattern)].tolist()}")
                
                # 使用create_completion_mask函数重新生成mask并对比
                print("\n  重新计算mask以验证原始计算是否正确:")
                recomputed_mask = create_completion_mask(
                    c_ids,
                    tokenizer,
                )
                
                # 比较两个mask
                original_sum = c_mask[i].sum().item()
                recomputed_sum = recomputed_mask.sum().item()
                match_ratio = (recomputed_mask == c_mask[i]).sum().item() / len(c_mask[i])
                
                print(f"  原始mask总和: {original_sum}, 重新计算的mask总和: {recomputed_sum}")
                print(f"  两个mask的匹配率: {match_ratio*100:.2f}%")
                
                # 如果不匹配，找出不匹配的位置并查看原因
                if match_ratio < 1.0:
                    diff_indices = torch.nonzero(recomputed_mask != c_mask[i]).squeeze().tolist()
                    if not isinstance(diff_indices, list):
                        diff_indices = [diff_indices]  # 处理只有一个不匹配的情况
                    
                    print(f"  发现{len(diff_indices)}个不匹配的位置")
                    for diff_idx in diff_indices[:5]:  # 只显示前5个
                        original_val = c_mask[i][diff_idx].item()
                        recomputed_val = recomputed_mask[diff_idx].item()
                        token = tokenizer.decode(c_ids[diff_idx:diff_idx+1])
                        print(f"    位置{diff_idx}: 原值={original_val}, 新值={recomputed_val}, token='{token}'")
                    
                    # 解码附近区域
                    for diff_idx in diff_indices[:2]:  # 只为前两个不匹配的位置提供上下文
                        start = max(0, diff_idx - 10)
                        end = min(diff_idx + 10, len(c_ids))
                        context = tokenizer.decode(c_ids[start:end])
                        print(f"    位置{diff_idx}周围的上下文: '{context}'")

        print("="*80)
        
        # 分析对话结构
        print("\n" + "="*80)
        print("对话结构分析:")
        for i, completion in enumerate(rollout['formatted_completions']):
            content = completion[0]["content"]
            print(f"\n样本 {i} 的对话结构分析:")
            
            # 解析对话
            dialog = parse_dialog_simple(content)
            print(f"对话轮次数: {len(dialog)}")
            
            # 输出每个轮次的角色和内容摘要
            for j, turn in enumerate(dialog):
                role = turn["role"]
                turn_content = turn["content"]
                # 截取内容摘要
                content_preview = turn_content[:50] + "..." if len(turn_content) > 50 else turn_content
                print(f"  轮次 {j+1}: 角色={role}, 内容摘要='{content_preview}'")
                
                # 检查特殊标记
                has_question = "question:" in turn_content.lower() or "问题:" in turn_content.lower()
                has_answer = "answer:" in turn_content.lower() or "回答:" in turn_content.lower() or "答案:" in turn_content.lower()
                
                if has_question:
                    print(f"    含有问题标记")
                if has_answer:
                    print(f"    含有回答标记")
                    
            # 检查对话中的标记格式
            has_im_start = "<|im_start|>" in content
            has_im_end = "<|im_end|>" in content
            has_assistant = "<|im_start|>assistant" in content
            has_user = "<|im_start|>user" in content
            
            print(f"特殊格式检查: im_start={has_im_start}, im_end={has_im_end}, assistant={has_assistant}, user={has_user}")
        
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

        # 在打印完mask的统计信息后添加以下代码，显示所有mask=1的token组成的完整文本
        print("\n" + "="*80)
        print("Mask=1 Token分析 - 测试mask是否正确标记了医生模型生成的内容:")
        for i in range(c_mask.size(0)):
            # 获取当前样本的completion_ids和mask
            c_ids = rollout["input_ids"][i, -completion_length:]
            cur_mask = c_mask[i]
            
            # 找出所有mask=1的位置
            masked_positions = (cur_mask == 1).nonzero().squeeze().tolist()
            if not isinstance(masked_positions, list):
                masked_positions = [masked_positions]  # 处理只有一个位置的情况
            
            # 获取所有mask=1的token
            masked_tokens = [c_ids[pos].item() for pos in masked_positions]
            
            # 将所有mask=1的token解码为文本
            masked_text = tokenizer.decode(masked_tokens)
            
            # 获取原始完整文本
            full_text = tokenizer.decode(c_ids)
            
            print(f"\n样本 {i} 的Mask=1部分:")
            print(f"完整文本长度: {len(full_text)}, Mask=1文本长度: {len(masked_text)}")
            print(f"Mask=1的token数量: {len(masked_tokens)}/{len(c_ids)}")
            
            # 对比mask=1的文本和原始文本中的医生回复部分
            try:
                # 解析对话，分离医生和患者部分
                dialog = parse_dialog_simple(full_text)
                doctor_parts = [turn["content"] for turn in dialog if turn["role"] == "assistant"]
                doctor_text = " ".join(doctor_parts)
                
                print(f"医生部分长度: {len(doctor_text)}")
                
                # 计算mask=1文本与医生部分的相似度 (简单字符重叠率)
                common_chars = sum(1 for c in masked_text if c in doctor_text)
                similarity = common_chars / len(masked_text) if masked_text else 0
                
                print(f"医生部分与Mask=1部分的字符重叠率: {similarity:.2f}")
                
                # 显示mask=1的文本（截断以避免过长）
                print("\nMask=1文本 (前500字符):")
                print(masked_text[:500] + ("..." if len(masked_text) > 500 else ""))
                
                # 检查mask=1部分是否包含关键标记
                user_markers = ["<|im_start|>user", "<|im_end|>"]
                for marker in user_markers:
                    if marker in masked_text:
                        print(f"警告: Mask=1部分包含用户标记 '{marker}'")

                # 检查question:/answer:标记是否被包含在mask=1中
                if "question:" in masked_text.lower() or "问题:" in masked_text.lower():
                    print("✓ 问题标记已被正确包含在mask=1中")
                else:
                    print("✗ 问题标记未被包含在mask=1中")
                    
                if "answer:" in masked_text.lower() or "回答:" in masked_text.lower() or "答案:" in masked_text.lower():
                    print("✓ 回答标记已被正确包含在mask=1中")
                else:
                    print("✗ 回答标记未被包含在mask=1中")
                    
            except Exception as e:
                print(f"解析对话时出错: {e}")
            
            print("-"*50)

        break

