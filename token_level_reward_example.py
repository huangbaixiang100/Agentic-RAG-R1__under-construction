"""
Token级奖励分配系统使用示例
展示如何使用新的细粒度奖励分配机制
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.doctor_reward import (
    compute_token_level_rewards, 
    overall_reward_with_token_allocation
)

def demo_token_level_rewards():
    """
    演示token级奖励分配的工作原理
    """
    print("🎯 Token级奖励分配演示")
    print("=" * 50)
    
    # 模拟数据
    facts = [
        [
            "患者年龄65岁",           # 事实1 - 基础信息
            "有高血压病史",          # 事实2 - 重要信息
            "胸痛症状持续2小时",     # 事实3 - 关键症状
            "血压160/90mmHg",        # 事实4 - 具体数值
            "心电图显示ST段抬高"     # 事实5 - 诊断依据
        ]
    ]
    
    completions = [[{
        "content": """<|im_start|>assistant
你好，我是医生。请告诉我你的主要症状是什么？
<|im_end|>
<|im_start|>user
我胸痛已经持续2小时了，很难受。
<|im_end|>
<|im_start|>assistant
除了胸痛，还有其他症状吗？比如头晕、恶心？
<|im_end|>
<|im_start|>user
有点头晕，血压测量是160/90mmHg。
<|im_end|>
<|im_start|>assistant
我需要为您做心电图检查。根据您的症状和检查结果，最可能的诊断是急性心肌梗死。答案是A。
<|im_end|>"""
    }]]
    
    options = [{"A": "急性心肌梗死", "B": "心绞痛", "C": "高血压", "D": "心律失常"}]
    answers = ["A"]
    atomic_questions = ["根据患者症状和检查结果，最可能的诊断是什么？"]
    
    # 模拟模型和分词器（实际使用时应该加载真实模型）
    print("📚 准备模拟环境...")
    
    # 演示不同模式的对比
    print("\n📊 奖励计算对比分析")
    print("-" * 50)
    
    # 1. 传统全局奖励模式
    print("🔸 传统全局奖励模式:")
    print("  - 所有token获得相同的平均奖励")
    print("  - 无法区分不同问题的贡献")
    print("  - 示例：总奖励=6.0，所有token平均分配")
    
    # 2. Token级奖励分配模式  
    print("\n🎯 Token级奖励分配模式:")
    print("  - 每个问题根据其信息增益获得不同奖励")
    print("  - 结合过程奖励和结果奖励")
    print("  - 示例奖励分配:")
    
    # 模拟问题级奖励分配
    question_rewards = {
        "问题1: 主要症状": {"shapley_gain": 0.7, "process_reward": 0.7, "result_bonus": 0.7, "total": 1.4},
        "问题2: 其他症状": {"shapley_gain": 0.3, "process_reward": 0.3, "result_bonus": 0.3, "total": 0.6},
        "问题3: 最终诊断": {"shapley_gain": 0.0, "process_reward": 0.0, "result_bonus": 0.0, "total": 3.0}
    }
    
    for question, rewards in question_rewards.items():
        print(f"    {question}:")
        print(f"      - Shapley增益: {rewards['shapley_gain']}")
        print(f"      - 过程奖励: {rewards['process_reward']}")
        print(f"      - 结果奖励: {rewards['result_bonus']}")
        print(f"      - 总奖励: {rewards['total']}")
    
    print(f"\n  总奖励: {sum(r['total'] for r in question_rewards.values())}")
    
    # 3. 参数配置说明
    print("\n⚙️ 参数配置说明:")
    print("  - α (alpha): 过程奖励权重，控制信息获取的奖励强度")
    print("  - β (beta): 结果奖励权重，控制正确答案额外奖励")
    print("  - γ (gamma): 最终答案奖励，保持与原系统一致")
    
    # 4. 使用场景
    print("\n🎭 适用场景:")
    print("  ✓ 多轮医疗问诊对话")
    print("  ✓ 需要精确信用分配的任务")
    print("  ✓ 结合过程和结果的评估")
    print("  ✓ 引导模型问出高价值问题")


def demo_configuration_options():
    """
    演示不同配置选项的效果
    """
    print("\n🔧 配置选项演示")
    print("=" * 50)
    
    configs = [
        {
            "name": "传统模式",
            "use_token_level": False,
            "use_shapley": False,
            "description": "完全兼容现有系统"
        },
        {
            "name": "Shapley模式",
            "use_token_level": False,
            "use_shapley": True,
            "description": "加权事实重要性，全局奖励"
        },
        {
            "name": "Token级基础模式",
            "use_token_level": True,
            "use_shapley": False,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 3.0,
            "description": "Token级分配，平均分配事实分数"
        },
        {
            "name": "Token级Shapley模式",
            "use_token_level": True,
            "use_shapley": True,
            "alpha": 1.0,
            "beta": 1.0,
            "gamma": 3.0,
            "description": "Token级分配+Shapley加权，最佳效果"
        },
        {
            "name": "偏重过程奖励",
            "use_token_level": True,
            "use_shapley": True,
            "alpha": 2.0,
            "beta": 0.5,
            "gamma": 3.0,
            "description": "更强调信息获取过程"
        },
        {
            "name": "偏重结果奖励",
            "use_token_level": True,
            "use_shapley": True,
            "alpha": 0.5,
            "beta": 2.0,
            "gamma": 3.0,
            "description": "更强调最终答案正确性"
        }
    ]
    
    for config in configs:
        print(f"\n📋 {config['name']}:")
        print(f"  描述: {config['description']}")
        if 'alpha' in config:
            print(f"  参数: α={config['alpha']}, β={config['beta']}, γ={config['gamma']}")


def demo_training_commands():
    """
    演示不同训练命令
    """
    print("\n💻 训练命令示例")
    print("=" * 50)
    
    commands = [
        {
            "name": "传统训练",
            "command": "./script/training/train_hhhzero2_shapley.sh",
            "description": "保持现有训练方式不变"
        },
        {
            "name": "Token级奖励训练",
            "command": "./script/training/train_token_level_reward.sh",
            "description": "使用新的token级奖励分配"
        },
        {
            "name": "自定义参数训练",
            "command": """
accelerate launch ./hhhdoctor_train.py \\
    --use_shapley=True \\
    --use_token_level=True \\
    --alpha_reward=1.5 \\
    --beta_reward=1.0 \\
    --gamma_reward=3.0
""",
            "description": "自定义奖励权重参数"
        }
    ]
    
    for cmd in commands:
        print(f"\n🚀 {cmd['name']}:")
        print(f"  描述: {cmd['description']}")
        print(f"  命令: {cmd['command']}")


if __name__ == "__main__":
    demo_token_level_rewards()
    demo_configuration_options() 
    demo_training_commands()
    
    print("\n🎉 演示完成！")
    print("📚 更多详细信息请查看相关文档文件") 