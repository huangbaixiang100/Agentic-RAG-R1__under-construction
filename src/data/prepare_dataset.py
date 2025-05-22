from datasets import load_dataset

#from src.data.prompt import SYSTEM_PROMPT_TOOLS as SYSTEM_PROMPT
#from src.data.prompt import build_prompt, build_system_tools
from src.data.doctor_patient_prompts import *
import json
import logging
import os

from datasets import load_dataset, Dataset


def prepare_dataset(split="train", name="gsm8k", eval_size=10, train_file=None, test_file=None):
    if name == "gsm8k":
        return prepare_dataset_gsm8k(split, eval_size)
    elif name == "cmb":
        return prepare_dataset_cmb(split, eval_size, train_file, test_file)
    elif name == "medmcqa":
        return prepare_dataset_medmcqa(split, eval_size)
    elif name == "medqa":
        return prepare_dataset_medqa(split, eval_size)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def prepare_dataset_cmb(split="train", eval_size=10, train_file=None, test_file=None):
    """
    加载CMB数据集
    
    Args:
        split: 指定加载"train"训练集还是"test"测试集
        eval_size: 已废弃参数，保留以兼容旧代码
        train_file: 训练集文件路径，如果为None则使用默认路径
        test_file: 测试集文件路径，如果为None则使用默认路径
        
    Returns:
        tuple: 如果split为"train"，返回(train_dataset, empty_dataset)，
              如果split为"test"，返回(empty_dataset, test_dataset)
    """
    # 设置默认数据文件路径
    default_train_file = 'src/data/cmb_atomic_patient_train.json'
    default_test_file = 'src/data/cmb_atomic_patient_test.json'
    
    # 使用提供的文件路径或默认路径
    train_data_file = train_file if train_file else default_train_file
    test_data_file = test_file if test_file else default_test_file
    
    # 根据split参数确定加载哪个数据文件
    if split == "train" or split == "all":
        data_file = train_data_file
        logging.info(f"加载训练集数据: {data_file}")
    elif split == "test" or split == "eval":
        data_file = test_data_file
        logging.info(f"加载测试集数据: {data_file}")
    else:
        raise ValueError(f"无效的split参数: {split}, 必须是'train'或'test'")
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        # 如果找不到指定的训练/测试文件，尝试使用默认文件
        fallback_file = default_test_file if os.path.exists(default_test_file) else default_train_file
        logging.warning(f"找不到数据文件: {data_file}, 回退使用: {fallback_file}")
        data_file = fallback_file
        
    # 检查回退文件是否存在
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"无法找到数据文件: {data_file}")
        
    # 加载数据文件
    logging.info(f"正在加载数据文件: {data_file}")
    with open(data_file) as f:
        data = json.load(f)

    formatted_data = []

    for idx, example in enumerate(data):
        partial_question = '，'.join(example['facts'][:int(len(example['facts']) / 2)]) + '。' + example['atomic_question']
        option_str = "\n".join([f"{key}: {value}" for key, value in example['option'].items()])
        prompt_str = doctor_system_prompt.format(question_type=example['question_type'], question=partial_question,
                                                    option_str=option_str)
        final_prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + prompt_str + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        formatted_example = {
            "id": idx + 1,
            'prompt': final_prompt,
            'facts': example['facts'],
            'answer': example['answer'],
            'option': example['option']
        }
        formatted_data.append(formatted_example)

    dataset = Dataset.from_list(formatted_data)

    # 创建空数据集，以保持接口兼容性
    empty_dataset = Dataset.from_list([])
    
    # 根据split参数决定返回哪个数据集
    if split == "train" or split == "all":
        logging.info(f"训练集大小: {len(dataset)}")
        return dataset, empty_dataset
    else:
        logging.info(f"测试集大小: {len(dataset)}")
        return empty_dataset, dataset


def prepare_dataset_gsm8k(split="train", eval_size=10):
    """Load and prepare the GSM8K dataset for training with string prompts."""
    data = load_dataset("openai/gsm8k", "main")[split]
    formatted_data = []

    for example in data:
        # Convert list of messages to a single string prompt.
        prompt_str = build_prompt(
            [
                {"role": "system", "content": build_system_tools(SYSTEM_PROMPT)},
                {"role": "user", "content": example["question"]},
            ]
        )
        formatted_example = {
            "prompt": prompt_str,  # Now a string rather than a list.
            "answer": extract_answer_from_dataset(example["answer"]),
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medmcqa(split="train"):
    # 加载医学数据集（以 medmcqa 为例）
    data = load_dataset("medmcqa", split=split)
    formatted_data = []

    for example in data:
        # 构造 prompt，假设 SYSTEM_PROMPT 是一个医学相关的提示
        question = f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}"""

        prompt_str = "\n".join(
            [
                build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {example["question"]}
            Options:
            A. {example["opa"]}
            B. {example["opb"]}
            C. {example["opc"]}
            D. {example["opd"]}""",
            ]
        )
        # 提取正确答案（假设答案在 "correct_answer" 字段中）
        # 构造格式化数据
        correct_answer_index = example["cop"]
        options = [example["opa"], example["opb"], example["opc"], example["opd"]]
        correct_answer = options[correct_answer_index]

        formatted_example = {
            "prompt": prompt_str,
            "question": question,
            "answer": str(correct_answer),  # 将答案转换为字符串
        }
        formatted_data.append(formatted_example)

    return formatted_data


def prepare_dataset_medqa(split="train", eval_size=10):
    # med_qa_zh_4options_bigbio_qa_train 这个 subset
    data = load_dataset("fzkuji/MedQA", "med_qa_zh_4options_bigbio_qa")[split]

    formatted_data = []

    for idx, example in enumerate(data):
        question = example["question"]
        choices = example["choices"]
        answer = example["answer"][0]

        # 将选项拼接成 A [0] B [1] ... 的格式
        options_text = ""
        for j, choice in enumerate(choices):  # 使用j而不是i作为循环变量，避免覆盖外层循环的i
            option_letter = chr(65 + j)  # 65 是 ASCII 中 'A' 的编码
            options_text += f"{option_letter}. {choice}\n"

        prompt_str = "\n".join(
            [
                #build_system_tools(SYSTEM_PROMPT).strip(),
                f"""Question: {question}f
            Options:
            {options_text}""",
            ]
        )

        formatted_data.append(
            {
                "id": idx + 1,
                "prompt": prompt_str,
                "question": question + "\n" + options_text,
                "answer": str(answer),
            }
        )

    eval_data = formatted_data[:eval_size]
    train_data = formatted_data[eval_size:]  # fixme here

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    # for i, item in enumerate(eval_dataset):
    #     print(item["id"])

    return train_dataset, eval_dataset


if __name__ == "__main__":
    data = prepare_dataset_medqa(split="train")
