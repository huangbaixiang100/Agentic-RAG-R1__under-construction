import json
import random
import logging
from pathlib import Path


def shuffle_cmb_dataset(input_file, output_file=None, seed=42):
    """
    重新打乱CMB数据集
    
    Args:
        input_file (str): 输入数据文件路径
        output_file (str): 输出文件路径，如果为None则覆盖原文件
        seed (int): 随机种子，确保可重现性
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取原始数据
    logging.info(f"正在读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = len(data)
    logging.info(f"原始数据集大小: {original_size}")
    
    # 打乱数据
    logging.info("正在打乱数据...")
    random.shuffle(data)
    
    # 更新ID（保持数据完整性）
    for idx, item in enumerate(data):
        item['id'] = idx + 1
    
    # 确定输出文件路径
    if output_file is None:
        # 创建备份
        backup_file = input_file.replace('.json', '_backup.json')
        logging.info(f"创建备份文件: {backup_file}")
        with open(backup_file, 'w', encoding='utf-8') as f:
            original_data = json.load(open(input_file, 'r', encoding='utf-8'))
            json.dump(original_data, f, ensure_ascii=False, indent=2)
        
        output_file = input_file
    
    # 保存打乱后的数据
    logging.info(f"正在保存打乱后的数据到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logging.info("数据集重新打乱完成！")
    logging.info(f"打乱后数据集大小: {len(data)}")
    
    # 验证数据完整性
    verify_dataset_integrity(data, original_size)
    
    return output_file


def verify_dataset_integrity(data, expected_size):
    """验证数据集完整性"""
    logging.info("正在验证数据集完整性...")
    
    # 检查数据大小
    if len(data) != expected_size:
        msg = f"数据大小不匹配! 期望: {expected_size}, 实际: {len(data)}"
        logging.error(msg)
        return False
    
    # 检查必要字段
    required_fields = ['facts', 'atomic_question', 'option', 'answer', 
                       'question_type']
    for idx, item in enumerate(data[:10]):  # 检查前10条数据
        for field in required_fields:
            if field not in item:
                logging.error(f"数据第{idx}条缺少必要字段: {field}")
                return False
    
    logging.info("数据集完整性验证通过!")
    return True


def create_train_eval_split(input_file, train_ratio=0.9, seed=42):
    """
    将数据集分割为训练集和验证集
    
    Args:
        input_file (str): 输入数据文件路径
        train_ratio (float): 训练集比例
        seed (int): 随机种子
    """
    random.seed(seed)
    
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 打乱数据
    random.shuffle(data)
    
    # 分割数据
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    
    train_data = data[:train_size]
    eval_data = data[train_size:]
    
    # 更新ID
    for idx, item in enumerate(train_data):
        item['id'] = idx + 1
    for idx, item in enumerate(eval_data):
        item['id'] = idx + 1
    
    # 保存分割后的数据
    base_path = Path(input_file).parent
    train_file = base_path / "cmb_atomic_patient_train_shuffled.json"
    eval_file = base_path / "cmb_atomic_patient_eval_shuffled.json"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    
    logging.info(f"训练集保存到: {train_file} (大小: {len(train_data)})")
    logging.info(f"验证集保存到: {eval_file} (大小: {len(eval_data)})")
    
    return str(train_file), str(eval_file)


if __name__ == "__main__":
    # 配置日志
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    
    # 数据文件路径
    data_file = "src/data/cmb_atomic_patient_train.json"
    
    print("选择操作:")
    print("1. 仅重新打乱现有数据集")
    print("2. 重新打乱并分割为训练集和验证集")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        # 仅重新打乱
        seed = int(input("请输入随机种子 (默认42): ") or "42")
        output_file = shuffle_cmb_dataset(data_file, seed=seed)
        print(f"数据集已重新打乱并保存到: {output_file}")
        
    elif choice == "2":
        # 重新打乱并分割
        train_ratio = float(input("请输入训练集比例 (默认0.9): ") or "0.9")
        seed = int(input("请输入随机种子 (默认42): ") or "42")
        
        train_file, eval_file = create_train_eval_split(
            data_file, train_ratio=train_ratio, seed=seed)
        print(f"训练集保存到: {train_file}")
        print(f"验证集保存到: {eval_file}")
        
    else:
        print("无效选择，程序退出") 