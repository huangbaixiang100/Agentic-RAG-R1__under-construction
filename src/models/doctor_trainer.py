import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple
import deepspeed
import swanlab
import torch
import torch.nn.functional as F
from accelerate import Accelerator

from peft import (PeftModel, get_peft_model_state_dict, LoraConfig, 
                  get_peft_model)
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.doctor_reward import overall_reward, overall_reward_with_token_allocation
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


def _is_poor_quality_generation(text: str) -> bool:
    """
    检测生成文本质量是否过差（包含过多重复内容）
    """
    if not text or len(text.strip()) == 0:
        return True
    
    # 检测重复字符的比例
    char_counts = {}
    for char in text:
        char_counts[char] = char_counts.get(char, 0) + 1
    
    # 如果某个字符出现次数超过总长度的30%，认为质量差
    max_char_ratio = max(char_counts.values()) / len(text)
    if max_char_ratio > 0.3:
        return True
    
    # 检测重复短语（连续相同的词）
    words = text.split()
    if len(words) > 10:
        consecutive_same = 0
        max_consecutive = 0
        for i in range(1, len(words)):
            if words[i] == words[i-1]:
                consecutive_same += 1
                max_consecutive = max(max_consecutive, consecutive_same)
            else:
                consecutive_same = 0
        
        # 如果连续相同词超过5个，认为质量差
        if max_consecutive > 5:
            return True
    
    # 检测过度重复的短语或句子
    if len(text) > 50:
        common_repeat_patterns = [
            r'(.{2,10})\1{3,}',  # 短语重复3次以上
            r'([，。？！])\1{5,}',  # 标点符号重复5次以上
            r'([a-zA-Z]+)\s+\1\s+\1',  # 英文单词重复3次
        ]
        
        import re
        for pattern in common_repeat_patterns:
            if re.search(pattern, text):
                return True
    
    return False


def extract_atomic_questions_from_batch(
    batch_samples: Dict[str, List[Any]], 
    num_generations: int
) -> List[str]:
    """
    从batch数据中提取原子问题，用于Shapley值计算
    
    这个函数支持用户期望的Shapley流程：
    在对话生成之前就确定要评估的原子问题
    
    Args:
        batch_samples: 包含prompt、question、answer等的batch数据
        num_generations: 每个prompt的生成数量
        
    Returns:
        重复扩展后的原子问题列表
    """
    try:
        # 🔍 调试：打印batch_samples的所有字段
        print(f"🔍 ===== extract_atomic_questions_from_batch调试信息 =====")
        print(f"📊 batch_samples中的所有字段: {list(batch_samples.keys())}")
        
        for key, value in batch_samples.items():
            if isinstance(value, list):
                print(f"  {key}: 类型=列表, 长度={len(value)}")
                if value and len(value) > 0:
                    print(f"    首个元素类型: {type(value[0])}")
                    if isinstance(value[0], str):
                        print(f"    首个元素内容: '{value[0][:100]}...'")
                    else:
                        print(f"    首个元素内容: {str(value[0])[:100]}...")
            else:
                print(f"  {key}: 类型={type(value)}, 内容={str(value)[:50]}...")
        
        # 尝试从batch中获取question字段
        if 'question' in batch_samples:
            questions = batch_samples['question']
            print(f"✅ 找到question字段，内容: {questions}")
        elif 'atomic_question' in batch_samples:
            questions = batch_samples['atomic_question']
            print(f"✅ 找到atomic_question字段，内容: {questions}")
        else:
            # 如果没有明确的question字段，尝试从prompt中解析
            logging.warning(
                "未找到question字段，尝试从prompt中解析原子问题"
            )
            questions = []
            for prompt in batch_samples.get('prompt', []):
                # 简单的问题提取逻辑，可以根据实际数据格式调整
                if '问题：' in prompt:
                    question_part = (
                        prompt.split('问题：')[-1].split('\n')[0].strip()
                    )
                    questions.append(question_part)
                elif '问题:' in prompt:
                    question_part = (
                        prompt.split('问题:')[-1].split('\n')[0].strip()
                    )
                    questions.append(question_part)
                else:
                    # 默认问题
                    questions.append("请根据患者信息进行诊断")
        
        # 根据num_generations重复扩展
        repeated_questions = []
        for question in questions:
            repeated_questions.extend([question] * num_generations)
        
        logging.info(
            f"成功提取{len(questions)}个原子问题，"
            f"扩展为{len(repeated_questions)}个"
        )
        return repeated_questions
        
    except Exception as e:
        logging.error(f"提取原子问题失败: {e}")
        # 返回默认问题
        default_question = "请根据患者信息进行诊断"
        prompt_count = len(batch_samples.get('prompt', []))
        total_questions = prompt_count * num_generations
        return [default_question] * total_questions


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
    # 完整的prompt+所有生成内容
    final_outputs: List[Optional[torch.LongTensor]] = [None] * batch_size
    completion_texts = [""] * batch_size

    # 同一个batch里不同sample一起decode，需要注意padding，每轮生成完之后重新pad
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
            repetition_penalty=1.1,  # 添加重复惩罚
            no_repeat_ngram_size=3,  # 防止3-gram重复
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

            # 检测生成质量 - 如果包含过多重复内容，标记为无效
            if _is_poor_quality_generation(text):
                print(f"[Warning] Sample {b} generated poor quality text, stopping generation")
                completion_texts[b] += "<invalid_generation>"
                merged_text = history_texts[idx] + "<invalid_generation>"
                final_outputs[b] = tokenizer.encode(merged_text, add_special_tokens=False, return_tensors="pt").to(
                    device).squeeze(0)
                should_gen[b] = False
                continue

            merged_text = history_texts[idx]

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

    print("🔍 ===== generate_completions_multi_round调试信息 =====")
    print(f"📏 prompt_len: {prompt_len}")
    print(f"📏 max_length_for_gather: {max_length_for_gather}")
    print(f"📏 allowed_completion_len: {allowed_completion_len}")
    print(f"📏 原始completion_ids形状: {completion_ids.shape}")
    
    # 显示每个样本的completion_texts信息
    for i, text in enumerate(completion_texts):
        original_tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"📄 Sample {i}: completion_text长度={len(text)}字符, 原始token数={len(original_tokens)}")

    if completion_ids.size(1) > allowed_completion_len:
        # 统一裁剪到 allowed_completion_len
        print(f"🔧 裁剪completion_ids从{completion_ids.size(1)}到{allowed_completion_len}")
        completion_ids = completion_ids[:, :allowed_completion_len]
        print(f"✅ 裁剪后completion_ids形状: {completion_ids.shape}")
    else:
        print(f"✅ completion_ids无需裁剪，保持形状: {completion_ids.shape}")

    for b in range(batch_size):
        print(f"🎭 为样本{b}创建completion_mask...")
        mask = create_completion_mask(
            completion_ids[b],
            tokenizer,
        )
        print(f"📊 样本{b}的mask: 长度={len(mask)}, 非零数量={mask.sum().item()}")
        completion_masks.append(mask)
    completion_masks = torch.stack(completion_masks, dim=0)
    
    print(f"🎯 最终completion_masks形状: {completion_masks.shape}")
    print("=" * 60)

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
    # 兼容两种数据集的字段名：统一使用options
    options_key = 'options' if 'options' in batch_samples else 'option'
    repeated_options = [o for o in batch_samples[options_key] for _ in range(num_generations)]
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
    use_shapley: bool = False,  # 新增参数：是否使用Shapley值加权
    atomic_questions: List[str] = None,  # 新增参数：原子问题列表
    use_token_level: bool = False,  # 新增参数：是否使用token级奖励分配
    token_reward_mode: str = "token_baseline",  # 新增参数：token奖励模式
    alpha: float = 2.0,  # Question Shapley奖励权重
    beta_reward: float = 1.0,  # Question结果奖励权重
    gamma: float = 3.0,  # Answer正确性奖励权重
    format_reward_weight: float = 1.0,  # 格式奖励权重
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Perform a single GRPO update step, computing loss and backpropagating.
    支持token级奖励分配的增强版本。

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
        use_shapley (bool): Whether to use Shapley value weighting for fact scores.
        atomic_questions (List[str]): List of atomic questions for Shapley calculation.
        use_token_level (bool): Whether to use token-level reward allocation.
        token_reward_mode (str): Token reward calculation mode:
                                - "token_baseline": 在每个token位置计算baseline（原方式）
                                - "rollout_baseline": 每个rollout计算总分，然后组内比较
        alpha (float): Process reward weight.
        beta_reward (float): Result reward weight. 
        gamma (float): Final answer reward weight.

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

    # 根据配置调用不同的奖励函数
    if use_token_level:
        # 使用token级奖励分配
        from src.models.doctor_reward import overall_reward_with_token_allocation
        
        # 🔧 修复：传递截断长度以保持与completion_mask一致
        completion_length = rollout_data["completion_mask"].size(1) if rollout_data["completion_mask"].dim() > 1 else len(rollout_data["completion_mask"])
        
        print("🔍 ===== Token级奖励分配调试信息 =====")
        print(f"📏 completion_mask形状: {rollout_data['completion_mask'].shape}")
        print(f"📏 input_ids形状: {rollout_data['input_ids'].shape}")
        print(f"📏 logits_to_keep: {k}")
        print(f"📏 completion_length: {completion_length}")
        print(f"📋 batch数量: {len(rollout_data['formatted_completions'])}")
        print(f"🎯 Token奖励模式: {token_reward_mode}")
        
        # 检查每个completion的原始文本长度
        for i, completion_list in enumerate(rollout_data['formatted_completions']):
            for j, completion_dict in enumerate(completion_list):
                content = completion_dict.get('content', '')
                tokens = tokenizer.encode(content, add_special_tokens=False)
                print(f"📄 Sample {i}-{j}: 原始文本长度={len(tokens)}, 内容前50字符='{content[:50]}...'")
        
        # 🎯 纯Token级奖励计算（整合格式奖励）
        rewards_dict = overall_reward_with_token_allocation(
            model=model,
            tokenizer=tokenizer,
            facts=rollout_data["repeated_facts"],
            completions=rollout_data["formatted_completions"],
            options=rollout_data["repeated_options"],
            answers=rollout_data["repeated_answers"],
            use_shapley=use_shapley,
            atomic_questions=atomic_questions,
            use_token_level=True,
            alpha=alpha,
            beta=beta_reward,
            gamma=gamma,
            format_reward_weight=format_reward_weight,  # 🎯 格式奖励权重
            max_completion_length=completion_length
        )
        
        # 处理token级奖励
        if "token_rewards" in rewards_dict:
            print("🎯 使用纯token级奖励分配模式")
            token_rewards_list = rewards_dict["token_rewards"]
            
            # 🎯 记录各类token奖励的统计信息到SwanLab
            if "question_token_rewards" in rewards_dict:
                question_token_mean = sum(rewards_dict["question_token_rewards"]) / len(rewards_dict["question_token_rewards"]) if rewards_dict["question_token_rewards"] else 0.0
                print(f"📊 Question tokens平均奖励: {question_token_mean:.4f}")
                # SwanLab记录 - 将在train_with_grpo中统一记录
            
            if "answer_token_rewards" in rewards_dict:
                answer_token_mean = sum(rewards_dict["answer_token_rewards"]) / len(rewards_dict["answer_token_rewards"]) if rewards_dict["answer_token_rewards"] else 0.0
                print(f"📊 Answer tokens平均奖励: {answer_token_mean:.4f}")
            
            if "format_token_rewards" in rewards_dict:
                format_token_mean = sum(rewards_dict["format_token_rewards"]) / len(rewards_dict["format_token_rewards"]) if rewards_dict["format_token_rewards"] else 0.0
                print(f"📊 Format tokens平均奖励: {format_token_mean:.4f}")
            
            if "token_rewards_mean" in rewards_dict:
                total_token_mean = sum(rewards_dict["token_rewards_mean"]) / len(rewards_dict["token_rewards_mean"]) if rewards_dict["token_rewards_mean"] else 0.0
                print(f"📊 Total tokens平均奖励: {total_token_mean:.4f}")
            
            print(f"📊 token_rewards_list长度: {len(token_rewards_list)}")
            for i, token_rewards in enumerate(token_rewards_list):
                print(f"📊 Sample {i}: token_rewards长度={len(token_rewards)}")
            
            # 🎯 Token级Group Baseline模式：每个token的advantage = token_reward - group_baseline
            print("🎯 Token级Group Baseline模式：token_advantage = token_reward - group_baseline")
            
            # 使用新的group baseline advantage计算
            adv = compute_token_level_group_advantages(
                token_rewards_list, rollout_data["num_generations"], comp_mask
            )
            
            print(f"📊 Token级Group Baseline Advantage统计:")
            print(f"  均值: {adv.mean():.4f}, 标准差: {adv.std():.4f}")
            print(f"  形状: {adv.shape}, 非零元素数: {(adv != 0).sum().item()}")
            print(f"  正advantage数: {(adv > 0).sum().item()}")
            print(f"  负advantage数: {(adv < 0).sum().item()}")
                
            # Token级模式下的avg_reward使用所有token奖励的平均值
            avg_reward = float(adv.mean().item())
                
        else:
            print("⚠️ Token级奖励计算失败，回退到全局奖励")
            # 回退到全局奖励
            rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
            avg_reward = float(rewards.mean())
            adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
            
    else:
        # 传统rollout级别奖励计算
        print("🎯 使用传统rollout级别奖励计算")
        
        # 根据配置调用不同的奖励函数
        if use_shapley:
            # 使用Shapley值加权的overall_reward
            from src.models.doctor_reward import overall_reward_with_token_allocation
            rewards_dict = overall_reward_with_token_allocation(
                model=model,
                tokenizer=tokenizer,
                facts=rollout_data["repeated_facts"],
                completions=rollout_data["formatted_completions"],
                options=rollout_data["repeated_options"],
                answers=rollout_data["repeated_answers"],
                use_shapley=True,
                atomic_questions=atomic_questions,
                use_token_level=False,  # 明确设置为False
            )
        else:
            # 使用传统的overall_reward
            rewards_dict = reward_function(
                model=model,
                tokenizer=tokenizer,
                facts=rollout_data["repeated_facts"],
                completions=rollout_data["formatted_completions"],
                options=rollout_data["repeated_options"],
                answers=rollout_data["repeated_answers"],
                use_shapley=False,
            )
        
        # 传统的rollout级别advantage计算
        rewards = torch.tensor(rewards_dict["total_scores"], dtype=torch.float32, device=curr_lp.device)
        avg_reward = float(rewards.mean())
        adv = compute_group_relative_advantages(rewards, rollout_data["num_generations"])
    
    # GRPO loss计算
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


def compute_token_level_advantages(
    token_rewards: torch.Tensor, 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算token级别的advantage
    
    Args:
        token_rewards: [batch_size, seq_len] token级奖励
        num_generations: 每个prompt的生成数量
        completion_mask: [batch_size, seq_len] completion掩码
        
    Returns:
        torch.Tensor: token级advantage
    """
    print("🔍 ===== compute_token_level_advantages调试信息 =====")
    print(f"📏 输入token_rewards形状: {token_rewards.shape}")
    print(f"📏 输入completion_mask形状: {completion_mask.shape}")
    print(f"📏 num_generations: {num_generations}")
    
    batch_size_rewards, seq_len_rewards = token_rewards.shape
    batch_size_mask, seq_len_mask = completion_mask.shape
    
    print(f"📊 token_rewards: batch_size={batch_size_rewards}, seq_len={seq_len_rewards}")
    print(f"📊 completion_mask: batch_size={batch_size_mask}, seq_len={seq_len_mask}")
    
    # 确保batch_size匹配
    if batch_size_rewards != batch_size_mask:
        raise ValueError(f"token_rewards和completion_mask的batch_size不匹配: {batch_size_rewards} vs {batch_size_mask}")
    
    # 如果序列长度不匹配，需要对齐
    if seq_len_rewards != seq_len_mask:
        print(f"⚠️ 检测到长度不匹配: token_rewards({seq_len_rewards}) vs completion_mask({seq_len_mask})")
        
        # 使用较短的长度来截断
        min_seq_len = min(seq_len_rewards, seq_len_mask)
        
        print(f"🔧 将长度对齐到: {min_seq_len}")
        
        # 截断到相同长度
        token_rewards_orig = token_rewards.clone()
        completion_mask_orig = completion_mask.clone()
        
        token_rewards = token_rewards[:, :min_seq_len]
        completion_mask = completion_mask[:, :min_seq_len]
        
        print(f"✅ 对齐后token_rewards形状: {token_rewards.shape}")
        print(f"✅ 对齐后completion_mask形状: {completion_mask.shape}")
        
        # 显示截断信息
        for i in range(min(3, batch_size_rewards)):  # 只显示前3个样本
            orig_reward_nonzero = (token_rewards_orig[i] != 0).sum().item()
            new_reward_nonzero = (token_rewards[i] != 0).sum().item()
            orig_mask_nonzero = (completion_mask_orig[i] != 0).sum().item() 
            new_mask_nonzero = (completion_mask[i] != 0).sum().item()
            print(f"📊 Sample {i}: 非零奖励 {orig_reward_nonzero}->{new_reward_nonzero}, 非零mask {orig_mask_nonzero}->{new_mask_nonzero}")
        
        seq_len = min_seq_len
    else:
        print("✅ 长度匹配，无需对齐")
        seq_len = seq_len_rewards
    
    batch_size = batch_size_rewards
    
    print(f"🎯 最终处理形状: batch_size={batch_size}, seq_len={seq_len}")
    
    # 重塑为[num_prompts, num_generations, seq_len]
    num_prompts = batch_size // num_generations
    print(f"📊 计算参数: num_prompts={num_prompts}, num_generations={num_generations}")
    
    reshaped_rewards = token_rewards.view(num_prompts, num_generations, seq_len)
    reshaped_mask = completion_mask.view(num_prompts, num_generations, seq_len)
    
    print(f"📏 重塑后rewards形状: {reshaped_rewards.shape}")
    print(f"📏 重塑后mask形状: {reshaped_mask.shape}")
    
    # 计算每个token位置的baseline（平均奖励）
    masked_rewards = reshaped_rewards * reshaped_mask
    token_count = reshaped_mask.sum(dim=1, keepdim=True)  # [num_prompts, 1, seq_len]
    token_count = torch.clamp(token_count, min=1)  # 避免除零
    
    baseline = masked_rewards.sum(dim=1, keepdim=True) / token_count  # [num_prompts, 1, seq_len]
    
    print(f"📊 计算baseline: masked_rewards总和={masked_rewards.sum():.4f}")
    print(f"📊 token_count范围: min={token_count.min()}, max={token_count.max()}")
    print(f"📊 baseline统计: mean={baseline.mean():.4f}, std={baseline.std():.4f}")
    
    # 计算advantage
    advantage = reshaped_rewards - baseline  # [num_prompts, num_generations, seq_len]
    
    print(f"📊 advantage统计: mean={advantage.mean():.4f}, std={advantage.std():.4f}")
    
    # 重塑回原始形状
    advantage = advantage.view(batch_size, seq_len)
    
    print(f"✅ 最终advantage形状: {advantage.shape}")
    print("=" * 60)
    
    return advantage


def compute_rollout_total_advantages(
    token_rewards_list: List[List[float]], 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    Token级Rollout Baseline计算方式：
    1. 每个token保持自己的具体奖励值（0-3分）
    2. Group baseline = 该组所有token reward的平均值
    3. 每个token的advantage = 自己的奖励 - group baseline
    
    Args:
        token_rewards_list: List[List[float]] - 每个rollout的token级奖励
        num_generations: 每个prompt的生成数量
        completion_mask: [batch_size, seq_len] completion掩码
        
    Returns:
        torch.Tensor: [batch_size, seq_len] 形状的advantage
    """
    print("🔍 ===== Token级Rollout Baseline Advantage计算 =====")
    print(f"📏 输入token_rewards_list长度: {len(token_rewards_list)}")
    print(f"📏 输入completion_mask形状: {completion_mask.shape}")
    print(f"📏 num_generations: {num_generations}")
    
    batch_size, seq_len = completion_mask.shape
    num_prompts = batch_size // num_generations
    
    print(f"📊 计算参数: batch_size={batch_size}, seq_len={seq_len}")
    print(f"📊 计算参数: num_prompts={num_prompts}, num_generations={num_generations}")
    
    # 验证输入数据一致性
    if len(token_rewards_list) != batch_size:
        print(f"⚠️ token_rewards_list长度({len(token_rewards_list)})与batch_size({batch_size})不匹配")
        while len(token_rewards_list) < batch_size:
            token_rewards_list.append([])
    
    # 第一步：将token rewards转换为tensor
    token_rewards_tensor = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for i, token_rewards in enumerate(token_rewards_list):
        if i < batch_size and len(token_rewards) > 0:
            # 将token rewards填入tensor，但不超过seq_len
            for j, reward in enumerate(token_rewards):
                if j < seq_len:
                    token_rewards_tensor[i, j] = reward
    
    print(f"📊 Token rewards tensor形状: {token_rewards_tensor.shape}")
    
    # 第二步：按组计算baseline
    # 重塑为[num_prompts, num_generations, seq_len]
    grouped_token_rewards = token_rewards_tensor.view(num_prompts, num_generations, seq_len)
    grouped_completion_mask = completion_mask.view(num_prompts, num_generations, seq_len)
    
    print(f"📊 分组后token rewards形状: {grouped_token_rewards.shape}")
    print(f"📊 分组后completion mask形状: {grouped_completion_mask.shape}")
    
    # 计算每组的token reward baseline
    group_baselines = torch.zeros(num_prompts, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        # 获取该组所有rollout的有效token rewards
        group_rewards = grouped_token_rewards[group_idx]  # [num_generations, seq_len]
        group_mask = grouped_completion_mask[group_idx]   # [num_generations, seq_len]
        
        # 收集该组所有有效token的奖励值
        valid_rewards = []
        for gen_idx in range(num_generations):
            for token_idx in range(seq_len):
                if group_mask[gen_idx, token_idx] == 1:  # 有效token
                    reward = group_rewards[gen_idx, token_idx].item()
                    valid_rewards.append(reward)
        
        # 计算该组所有token的平均奖励作为baseline
        if len(valid_rewards) > 0:
            group_baseline = sum(valid_rewards) / len(valid_rewards)
        else:
            group_baseline = 0.0
            
        group_baselines[group_idx] = group_baseline
        
        print(f"📊 Group {group_idx}: {len(valid_rewards)}个有效token, baseline={group_baseline:.4f}")
        print(f"    有效奖励分布: min={min(valid_rewards) if valid_rewards else 0:.2f}, "
              f"max={max(valid_rewards) if valid_rewards else 0:.2f}, "
              f"mean={group_baseline:.2f}")
    
    print(f"🎯 各组baseline: {group_baselines}")
    
    # 第三步：计算每个token的advantage = token_reward - group_baseline
    token_advantages = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        group_baseline = group_baselines[group_idx]
        
        for gen_idx in range(num_generations):
            # 计算在flattened batch中的索引
            batch_idx = group_idx * num_generations + gen_idx
            
            for token_idx in range(seq_len):
                if completion_mask[batch_idx, token_idx] == 1:  # 有效token
                    token_reward = token_rewards_tensor[batch_idx, token_idx]
                    token_advantage = token_reward - group_baseline
                    token_advantages[batch_idx, token_idx] = token_advantage
                    
        print(f"📊 Group {group_idx} token advantages计算完成")
    
    # 调试信息：显示一些具体的advantage值
    print("📊 Token Advantage详细分析:")
    for i in range(min(4, batch_size)):  # 显示前4个rollout的信息
        valid_mask = completion_mask[i] == 1
        if valid_mask.any():
            valid_rewards = token_rewards_tensor[i][valid_mask]
            valid_advantages = token_advantages[i][valid_mask]
            group_idx = i // num_generations
            
            print(f"  Rollout {i} (Group {group_idx}):")
            print(f"    Token rewards: {valid_rewards[:5].tolist()}...")  # 显示前5个
            print(f"    Token advantages: {valid_advantages[:5].tolist()}...")
            print(f"    Group baseline: {group_baselines[group_idx]:.4f}")
    
    print(f"✅ 最终token_advantages形状: {token_advantages.shape}")
    print(f"📊 Advantage统计: mean={token_advantages.mean():.4f}, std={token_advantages.std():.4f}")
    print(f"📊 非零advantage数量: {(token_advantages != 0).sum().item()}")
    print("=" * 60)
    
    return token_advantages


def compute_token_level_group_advantages(
    token_rewards_list: List[List[float]], 
    num_generations: int,
    completion_mask: torch.Tensor
) -> torch.Tensor:
    """
    计算Token级Group Baseline的Advantage
    
    逻辑：
    1. 每个token有自己的奖励 (token_reward)
    2. Group baseline = 该组内所有有效token的平均奖励
    3. 每个token的advantage = token_reward - group_baseline
    
    Args:
        token_rewards_list: List[List[float]] - 每个rollout的token级奖励
        num_generations: 每个prompt的生成数量
        completion_mask: [batch_size, seq_len] completion掩码
        
    Returns:
        torch.Tensor: [batch_size, seq_len] 形状的advantage
    """
    print("🔍 ===== Token级Group Baseline Advantage计算 =====")
    print(f"📏 输入token_rewards_list长度: {len(token_rewards_list)}")
    print(f"📏 输入completion_mask形状: {completion_mask.shape}")
    print(f"📏 num_generations: {num_generations}")
    
    batch_size, seq_len = completion_mask.shape
    num_prompts = batch_size // num_generations
    
    print(f"📊 计算参数: batch_size={batch_size}, seq_len={seq_len}")
    print(f"📊 计算参数: num_prompts={num_prompts}, num_generations={num_generations}")
    
    # 验证输入数据一致性
    if len(token_rewards_list) != batch_size:
        print(f"⚠️ token_rewards_list长度({len(token_rewards_list)})与batch_size({batch_size})不匹配")
        while len(token_rewards_list) < batch_size:
            token_rewards_list.append([])
    
    # 第一步：将token rewards转换为tensor
    token_rewards_tensor = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for i, token_rewards in enumerate(token_rewards_list):
        if i < batch_size and len(token_rewards) > 0:
            # 将token rewards填入tensor，但不超过seq_len
            for j, reward in enumerate(token_rewards):
                if j < seq_len:
                    token_rewards_tensor[i, j] = reward
    
    print(f"📊 Token rewards tensor形状: {token_rewards_tensor.shape}")
    
    # 第二步：按组计算group baseline
    group_baselines = torch.zeros(num_prompts, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        # 获取该组的rollout索引范围
        start_idx = group_idx * num_generations
        end_idx = (group_idx + 1) * num_generations
        
        # 收集该组所有有效token的奖励值
        group_token_rewards = []
        
        for rollout_idx in range(start_idx, min(end_idx, batch_size)):
            # 获取该rollout的有效token奖励
            mask = completion_mask[rollout_idx]  # [seq_len]
            rewards = token_rewards_tensor[rollout_idx]  # [seq_len]
            
            # 只收集有效token的奖励
            valid_positions = (mask == 1).nonzero().squeeze(-1)
            for pos in valid_positions:
                 pos_idx = pos.item()
                 if pos_idx < rewards.size(0):
                     group_token_rewards.append(rewards[pos_idx].item())
        
        # 计算该组所有token的平均奖励作为baseline
        if len(group_token_rewards) > 0:
            group_baseline = sum(group_token_rewards) / len(group_token_rewards)
        else:
            group_baseline = 0.0
            
        group_baselines[group_idx] = group_baseline
        
        print(f"📊 Group {group_idx}: {len(group_token_rewards)}个有效token")
        print(f"    Token奖励分布: min={min(group_token_rewards) if group_token_rewards else 0:.3f}, "
              f"max={max(group_token_rewards) if group_token_rewards else 0:.3f}, "
              f"baseline={group_baseline:.3f}")
    
    print(f"🎯 各组baseline: {group_baselines}")
    
    # 第三步：计算每个token的advantage = token_reward - group_baseline
    token_advantages = torch.zeros(batch_size, seq_len, device=completion_mask.device)
    
    for group_idx in range(num_prompts):
        group_baseline = group_baselines[group_idx]
        start_idx = group_idx * num_generations
        end_idx = (group_idx + 1) * num_generations
        
        for rollout_idx in range(start_idx, min(end_idx, batch_size)):
            for token_idx in range(seq_len):
                if completion_mask[rollout_idx, token_idx] == 1:  # 有效token
                    token_reward = token_rewards_tensor[rollout_idx, token_idx]
                    token_advantage = token_reward - group_baseline
                    token_advantages[rollout_idx, token_idx] = token_advantage
                    
        print(f"📊 Group {group_idx} (baseline={group_baseline:.3f}) token advantages计算完成")
    
    # 调试信息：显示一些具体的advantage值
    print("📊 Token Advantage详细分析:")
    for i in range(min(3, batch_size)):  # 显示前3个rollout的信息
        valid_mask = completion_mask[i] == 1
        if valid_mask.any():
            valid_rewards = token_rewards_tensor[i][valid_mask]
            valid_advantages = token_advantages[i][valid_mask]
            group_idx = i // num_generations
            
            print(f"  Rollout {i} (Group {group_idx}):")
            print(f"    前5个Token rewards: {valid_rewards[:5].tolist()}")
            print(f"    前5个Token advantages: {valid_advantages[:5].tolist()}")
            print(f"    Group baseline: {group_baselines[group_idx]:.3f}")
    
    print(f"✅ 最终token_advantages形状: {token_advantages.shape}")
    print(f"📊 Advantage统计: mean={token_advantages.mean():.4f}, std={token_advantages.std():.4f}")
    print(f"📊 正advantage token数: {(token_advantages > 0).sum().item()}")
    print(f"📊 负advantage token数: {(token_advantages < 0).sum().item()}")
    print(f"📊 零advantage token数: {(token_advantages == 0).sum().item()}")
    print("=" * 60)
    
    return token_advantages


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
            # 原先的continue_training逻辑不再使用
            logging.warning("continue_training设置为True，但我们现在使用合并后的模型，忽略此设置")
            
            # 直接加载合并后的模型
            merged_model_path = "/ssd/hbx_llm/stepAblation-rolloutreward-0090"
            logging.info(f"加载合并后的模型: {merged_model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                merged_model_path,
                torch_dtype=getattr(torch, config.model.torch_dtype),
                trust_remote_code=True,
            ).to(device)
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
    use_shapley: bool = False,  # 新增：是否使用Shapley值加权
    extract_atomic_questions_fn: Optional[Callable] = None,  # 新增：提取原子问题的函数
    # Token级奖励分配参数
    use_token_level: bool = False,  # 是否启用token级奖励分配
    token_reward_mode: str = "token_baseline",  # 新增：token奖励计算模式
    alpha_reward: float = 1.0,  # Question Shapley奖励权重
    beta_reward: float = 1.0,  # Question结果奖励权重
    gamma_reward: float = 3.0,  # Answer正确性奖励权重
    format_reward_weight: float = 1.0,  # 格式奖励权重
) -> None:
    """
    使用GRPO微调训练策略模型，支持DeepSpeed ZeRO-2和ZeRO-3分布式训练
    
    新增参数:
        token_reward_mode (str): Token奖励计算模式，可选：
                                - "token_baseline": 在每个token位置计算baseline（原方式）
                                - "rollout_baseline": 每个rollout计算总分，然后组内比较
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
    logging.info(f"Token奖励模式: {token_reward_mode}")
    
    # 修复：构建参考模型（仅在训练开始时创建一次）
    if current_step == 0 or not hasattr(train_with_grpo, '_ref_model'):
        logging.info("创建参考模型（保持原始基础模型状态，不加载LoRA权重）...")
        ref_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=getattr(torch, config.model.torch_dtype),
            trust_remote_code=True,
        ).to(device)
        ref_model.eval()
        
        # 确保参考模型不需要梯度
        for p in ref_model.parameters():
            p.requires_grad_(False)

        # GRPO参考模型保持原始基础模型状态，不应用LoRA权重
        # 这样可以确保参考模型代表未经训练的原始状态
        logging.info("参考模型保持基础状态，未应用LoRA配置或权重")
        
        # 将参考模型移至正确的设备但不使用DeepSpeed包装
        ref_model = ref_model.to(accelerator.device)
        
        # 缓存参考模型，避免重复创建
        train_with_grpo._ref_model = ref_model
        logging.info("参考模型已缓存，后续迭代将重用此模型")
    else:
        # 重用已缓存的参考模型
        ref_model = train_with_grpo._ref_model
        logging.info("重用已缓存的参考模型")

    sum_steps = current_step
    for it in range(1, num_iterations + 1):
        logging.info(f"开始GRPO迭代 {it}/{num_iterations}")
        torch.cuda.empty_cache()

        step = 0
        for batch in dataloader:
            logging.info(f"开始生成rollout数据, 步骤 {step+1}/{min(steps_per_iteration, len(dataloader))}")
            # 确保模型处于评估模式进行生成
            was_training = policy_model.training
            policy_model.eval()
            
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
            
            # 恢复之前的训练状态
            if was_training:
                policy_model.train()
            logging.info("成功生成rollout数据")
            
            # 执行mu次GRPO更新
            for _ in range(mu):
                # 提取原子问题（如果启用Shapley）
                atomic_questions = None
                if use_shapley and extract_atomic_questions_fn is not None:
                    try:
                        print(f"🎯 开始提取原子问题，batch字段: {list(batch.keys())}")
                        atomic_questions = extract_atomic_questions_fn(batch, num_generations)
                        logging.info(f"提取到 {len(atomic_questions)} 个原子问题用于Shapley计算")
                        print(f"📋 提取的原子问题: {atomic_questions[:3]}...")  # 显示前3个
                    except Exception as e:
                        logging.warning(f"提取原子问题失败: {e}, 将使用传统奖励模式")
                        import traceback
                        print(f"📋 详细错误信息: {traceback.format_exc()}")
                        atomic_questions = None
                else:
                    if not use_shapley:
                        print(f"⚠️ use_shapley=False，跳过原子问题提取")
                    elif extract_atomic_questions_fn is None:
                        print(f"⚠️ extract_atomic_questions_fn为None，跳过原子问题提取")
                
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
                    accelerator=accelerator,
                    use_shapley=use_shapley,
                    atomic_questions=atomic_questions,
                    use_token_level=use_token_level,
                    token_reward_mode=token_reward_mode,  # 新增参数传递
                    alpha=alpha_reward,
                    beta_reward=beta_reward,
                    gamma=gamma_reward,
                    format_reward_weight=format_reward_weight
                )
            logging.info("成功最大化GRPO目标函数")

            print(
                f"迭代 {it}/{num_iterations}, 步骤 {step+1}/{min(steps_per_iteration, len(dataloader))}, "
                f"损失: {loss_val:.6f}, 平均奖励: {avg_r:.2f}, Token模式: {token_reward_mode}"
            )
            if accelerator.is_local_main_process:
                try:
                    # 正确计算各类奖励分数的平均值
                    def safe_avg(scores_list):
                        """安全地计算分数列表的平均值"""
                        if scores_list and len(scores_list) > 0:
                            return sum(scores_list) / len(scores_list)
                        return 0.0
                    
                    format_reward = safe_avg(rdict.get("format_scores", []))
                    answer_reward = safe_avg(rdict.get("correctness_scores", []))
                    fact_reward = safe_avg(rdict.get("fact_scores", []))
                    
                    print(f"📊 SwanLab记录 - Format: {format_reward:.3f}, Answer: {answer_reward:.3f}, Fact: {fact_reward:.3f}")
                    print(f"📋 原始数据 - Format scores: {rdict.get('format_scores', [])}")
                    
                    # 构建SwanLab记录字典
                    swanlab_data = {
                        "Iteration": it,
                        "Step": step+1,
                        "Loss": loss_val,
                        "Avg Reward": avg_r,
                        "Format Reward": format_reward,
                        "Answer Reward": answer_reward,
                        "Fact Score Reward": fact_reward,
                        "Token Reward Mode": token_reward_mode,  # 新增：记录token奖励模式
                    }
                    
                    # 🎯 添加token级奖励统计到SwanLab
                    if use_token_level and "question_token_rewards" in rdict:
                        question_token_mean = sum(rdict["question_token_rewards"]) / len(rdict["question_token_rewards"]) if rdict["question_token_rewards"] else 0.0
                        swanlab_data["token_rewards/question_token_mean"] = question_token_mean
                    
                    if use_token_level and "answer_token_rewards" in rdict:
                        answer_token_mean = sum(rdict["answer_token_rewards"]) / len(rdict["answer_token_rewards"]) if rdict["answer_token_rewards"] else 0.0
                        swanlab_data["token_rewards/answer_token_mean"] = answer_token_mean
                    
                    if use_token_level and "format_token_rewards" in rdict:
                        format_token_mean = sum(rdict["format_token_rewards"]) / len(rdict["format_token_rewards"]) if rdict["format_token_rewards"] else 0.0
                        swanlab_data["token_rewards/format_token_mean"] = format_token_mean
                    
                    if use_token_level and "token_rewards_mean" in rdict:
                        total_token_mean = sum(rdict["token_rewards_mean"]) / len(rdict["token_rewards_mean"]) if rdict["token_rewards_mean"] else 0.0
                        swanlab_data["token_rewards/total_token_mean"] = total_token_mean
                    
                    # 记录到SwanLab
                    swanlab.log(swanlab_data)
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
                    
                    # 改进的LoRA检测和保存逻辑
                    try:
                        # 获取模型进行检查
                        model_to_check = policy_model
                        if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                            model_to_check = policy_model.module
                        
                        # 多种方式检测是否为PEFT模型
                        is_peft_model = False
                        peft_model = None
                        
                        # 方法1: 直接检查类型
                        if "PeftModel" in str(type(model_to_check)):
                            is_peft_model = True
                            peft_model = model_to_check
                            logging.info("检测方法1: 直接类型检查发现PEFT模型")
                        
                        # 方法2: 检查是否有model属性且为PeftModel
                        elif hasattr(model_to_check, "model") and "PeftModel" in str(type(model_to_check.model)):
                            is_peft_model = True
                            peft_model = model_to_check.model
                            logging.info("检测方法2: 通过model属性发现PEFT模型")
                        
                        # 方法3: 检查是否有peft相关属性
                        elif hasattr(model_to_check, "peft_config") or hasattr(model_to_check, "get_peft_model"):
                            is_peft_model = True
                            peft_model = model_to_check
                            logging.info("检测方法3: 通过PEFT属性发现PEFT模型")
                        
                        # 方法4: 检查named_parameters中是否有lora参数
                        else:
                            lora_param_names = [n for n, _ in model_to_check.named_parameters() if "lora" in n.lower()]
                            if lora_param_names:
                                is_peft_model = True
                                peft_model = model_to_check
                                logging.info(f"检测方法4: 通过参数名发现LoRA参数，共{len(lora_param_names)}个")
                        
                        logging.info(f"最终检测结果 - 是否为PEFT模型: {is_peft_model}")
                        logging.info(f"模型类型: {type(model_to_check)}")
                        
                        # 如果是PEFT模型，保存LoRA权重
                        if is_peft_model and peft_model is not None:
                            logging.info("开始保存LoRA权重...")
                            
                            # 获取LoRA参数
                            lora_params = [p for n, p in peft_model.named_parameters() if "lora" in n.lower()]
                            logging.info(f"找到{len(lora_params)}个LoRA参数")
                            
                            if lora_params:
                                # 根据ZeRO级别决定是否需要gather参数
                                need_gather = isinstance(policy_model, deepspeed.DeepSpeedEngine) and zero_stage >= 2
                                logging.info(f"是否需要gather参数: {need_gather} (ZeRO级别: {zero_stage})")
                                
                                if need_gather:
                                    with deepspeed.zero.GatheredParameters(lora_params, enabled=True):
                                        lora_state = get_peft_model_state_dict(peft_model)
                                else:
                                    lora_state = get_peft_model_state_dict(peft_model)
                                
                                # 保存LoRA权重
                                peft_model.save_pretrained(ckpt, state_dict=lora_state)
                                logging.info(f"LoRA权重已保存到: {ckpt}")
                                
                                # 确保保存config.json
                                if hasattr(peft_model, 'config') and hasattr(peft_model.config, 'to_dict'):
                                    import json
                                    config_path = os.path.join(ckpt, "adapter_config.json")
                                    if not os.path.exists(config_path):
                                        with open(config_path, 'w') as f:
                                            json.dump(peft_model.config.to_dict(), f, indent=2)
                                        logging.info(f"adapter_config.json已保存到: {config_path}")
                            else:
                                logging.warning("虽然检测到PEFT模型，但未找到LoRA参数，使用常规保存方法")
                                raise ValueError("No LoRA parameters found")
                        else:
                            logging.warning("未检测到PEFT模型，使用常规保存方法")
                            raise ValueError("Not a PEFT model")
                            
                    except Exception as e:
                        logging.warning(f"LoRA保存失败: {e}，尝试使用常规方法保存")
                        # 回退到常规保存方法
                        if isinstance(policy_model, deepspeed.DeepSpeedEngine):
                            state_dict = policy_model.module.state_dict()
                            torch.save(state_dict, os.path.join(ckpt, "pytorch_model.bin"))
                        else:
                            policy_model.save_pretrained(ckpt)
                        logging.info("使用常规方法保存了完整模型")
                    
                    # 始终保存tokenizer
                    tokenizer.save_pretrained(ckpt)
                    logging.info(f"tokenizer已保存到: {ckpt}")
                            
            if step >= steps_per_iteration:
                break

            # 等待所有进程
            accelerator.wait_for_everyone()

        # 注意：不删除参考模型，在多个迭代中重用它
        # 只在最后一个迭代结束时清理内存
        if it == num_iterations:
            logging.info("训练完成，清理参考模型缓存")
            if hasattr(train_with_grpo, '_ref_model'):
                del train_with_grpo._ref_model
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

    # 使用配置文件中的模型路径，而不是硬编码
    model_name_or_path = ("/data/xiaobei/dhx/LLaMA-Factory-main-new/"
                         "models/promed-qwen2.5-1.5b-sft-3epoch-merged")
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
        print("标准奖励计算:")
        rewards_dict = overall_reward(
            model=model,
            tokenizer=tokenizer,
            facts=rollout["repeated_facts"],
            completions=rollout["formatted_completions"],
            options=rollout["repeated_options"],
            answers=rollout["repeated_answers"]
        )
        print(rewards_dict)

        # 🚀 新增：测试两种token级奖励计算模式
        print("\n" + "="*80)
        print("🚀 Token级奖励分配模式对比测试:")
        
        # 模式1: token_baseline (原方式)
        print("\n🎯 模式1: token_baseline - 每个token位置计算baseline")
        from src.models.doctor_reward import overall_reward_with_token_allocation
        
        completion_length = rollout["completion_mask"].size(1)
        
        token_rewards_1 = overall_reward_with_token_allocation(
            model=model,
            tokenizer=tokenizer,
            facts=rollout["repeated_facts"],
            completions=rollout["formatted_completions"],
            options=rollout["repeated_options"],
            answers=rollout["repeated_answers"],
            use_shapley=False,
            atomic_questions=None,
            use_token_level=True,
            max_completion_length=completion_length
        )
        
        print(f"Token_baseline模式结果: {list(token_rewards_1.keys())}")
        if 'token_rewards' in token_rewards_1:
            print(f"Token rewards数量: {len(token_rewards_1['token_rewards'])}")
            for i, tr in enumerate(token_rewards_1['token_rewards'][:2]):  # 只显示前2个
                nonzero_count = sum(1 for r in tr if r != 0)
                total_reward = sum(tr)
                print(f"  Sample {i}: 非零token数={nonzero_count}/{len(tr)}, 总奖励={total_reward:.3f}")
        
        # 模式2: rollout_baseline (新方式)
        print("\n🎯 模式2: rollout_baseline - 每个rollout计算总分后组内比较")
        
        # 使用compute_rollout_total_advantages直接测试
        if 'token_rewards' in token_rewards_1:
            token_rewards_list = token_rewards_1['token_rewards']
            
            # 测试新的advantage计算方法
            rollout_advantages = compute_rollout_total_advantages(
                token_rewards_list, 
                rollout["num_generations"], 
                rollout["completion_mask"]
            )
            
            print(f"Rollout_baseline模式Advantage形状: {rollout_advantages.shape}")
            print(f"Advantage统计: mean={rollout_advantages.mean():.4f}, std={rollout_advantages.std():.4f}")
        
      

        break


