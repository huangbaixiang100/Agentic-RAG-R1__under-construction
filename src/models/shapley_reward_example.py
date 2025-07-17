"""
Shapley值加权奖励函数使用示例

这个示例展示了如何在训练过程中使用Shapley值来改进fact score奖励。
主要特性：
1. 支持传统模式和Shapley值加权模式的切换
2. 动态计算Shapley值（使用当前策略模型）
3. 只对医生一开始不知道的事实（len(fact_list) // 2之后的部分）计算Shapley值
4. 高效的在线计算优化
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.doctor_reward import overall_reward
from src.models.doctor_trainer import train_with_grpo, maximize_grpo_objective
from accelerate import Accelerator
from torch.utils.data import DataLoader

# ================== 配置示例 ==================

def get_shapley_config():
    """获取Shapley值加权的配置"""
    return {
        "use_shapley": True,           # 是否启用Shapley值加权
        "max_samples": 15,             # Shapley值计算的最大采样次数（在线计算用较少次数）
        "min_samples": 3,              # 最小采样次数
        "tolerance": 5e-2,             # 收敛容忍值
        "early_stop_eps": 1e-2,        # 提前停止阈值
    }

def get_traditional_config():
    """获取传统模式的配置"""
    return {
        "use_shapley": False,          # 禁用Shapley值加权
    }

# ================== 使用示例 ==================

def example_shapley_reward_training():
    """
    展示如何在训练中使用Shapley值加权奖励
    """
    # 1. 初始化模型和数据
    model_name = "/root/小北健康-qwen2.5-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    
    accelerator = Accelerator()
    # 假设已经准备好了dataloader
    # dataloader = prepare_your_dataloader()
    
    # 2. 配置Shapley值参数
    shapley_config = get_shapley_config()
    
    # 3. 定义自定义奖励函数（带Shapley值支持）
    def custom_reward_function(model, tokenizer, facts, completions, options, answers, 
                             use_shapley=False, atomic_questions=None):
        """
        自定义奖励函数，支持Shapley值加权
        """
        return overall_reward(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            completions=completions,
            options=options,
            answers=answers,
            use_shapley=use_shapley,
            atomic_questions=atomic_questions
        )
    
    # 4. 训练配置
    training_config = {
        "num_iterations": 5,
        "steps_per_iteration": 100,
        "num_generations": 4,
        "max_new_tokens": 256,
        "temperature": 0.7,
        "beta": 0.1,
        "learning_rate": 5e-6,
        "epsilon": 0.2
    }
    
    # 5. 启动训练（使用Shapley值加权）
    print("开始使用Shapley值加权的训练...")
    
    # 注意：需要从数据中提取atomic_questions
    # 这通常在数据预处理阶段完成
    sample_atomic_questions = [
        "患者的主要症状是什么？",
        "患者的病史中有哪些重要信息？",
        "应该推荐什么治疗方案？"
    ]
    
    # train_with_grpo(
    #     config=training_config,
    #     device=accelerator.device,
    #     policy_model=model,
    #     ref_base_model=None,  # 会自动创建
    #     tokenizer=tokenizer,
    #     accelerator=accelerator,
    #     dataloader=dataloader,
    #     reward_function=custom_reward_function,
    #     use_shapley=shapley_config["use_shapley"],
    #     atomic_questions=sample_atomic_questions,
    #     **training_config
    # )

def example_traditional_training():
    """
    展示如何使用传统模式训练（不使用Shapley值）
    """
    # 配置为传统模式
    traditional_config = get_traditional_config()
    
    # 使用相同的训练代码，但设置use_shapley=False
    print("开始使用传统模式的训练...")
    
    # train_with_grpo(
    #     # ... 其他参数 ...
    #     use_shapley=traditional_config["use_shapley"],
    #     atomic_questions=None,  # 传统模式不需要
    # )

def example_batch_processing_with_shapley():
    """
    展示如何在批处理中处理Shapley值奖励
    """
    # 模拟批处理数据
    batch_facts = [
        ["患者年龄65岁", "有高血压病史", "胸痛症状持续2小时", "血压160/90mmHg"],
        ["患者女性", "30岁", "头痛症状", "伴有恶心呕吐", "发热38.5°C"],
    ]
    
    batch_completions = [
        [{"content": "question: 您的胸痛是什么性质的？<|im_end|>\n<|im_start|>user\n压榨性疼痛<|im_end|>\n<|im_start|>assistant\nanswer: 建议进行心电图检查"}],
        [{"content": "question: 头痛多长时间了？<|im_end|>\n<|im_start|>user\n2天了<|im_end|>\n<|im_start|>assistant\nanswer: 可能是感冒引起的"}]
    ]
    
    batch_options = [
        {"A": "心肌梗死", "B": "心绞痛", "C": "胃痛", "D": "肌肉疼痛"},
        {"A": "偏头痛", "B": "感冒", "C": "脑炎", "D": "低血糖"}
    ]
    
    batch_answers = ["A", "B"]
    
    batch_atomic_questions = [
        "根据症状判断最可能的诊断是什么？",
        "患者的症状提示什么疾病？"
    ]
    
    # 模拟模型和tokenizer
    # model = load_your_model()
    # tokenizer = load_your_tokenizer()
    
    print("处理批处理数据...")
    print(f"批处理大小: {len(batch_facts)}")
    
    # 计算Shapley值加权奖励
    # rewards = overall_reward(
    #     model=model,
    #     tokenizer=tokenizer,
    #     facts=batch_facts,
    #     completions=batch_completions,
    #     options=batch_options,
    #     answers=batch_answers,
    #     use_shapley=True,
    #     atomic_questions=batch_atomic_questions
    # )
    
    # print("奖励分数:")
    # print(f"总分数: {rewards['total_scores']}")
    # print(f"正确性分数: {rewards['correctness_scores']}")
    # print(f"格式分数: {rewards['format_scores']}")
    # print(f"事实分数: {rewards['fact_scores']}")

# ================== 配置切换示例 ==================

class RewardModeManager:
    """奖励模式管理器，支持运行时切换传统模式和Shapley模式"""
    
    def __init__(self):
        self.current_mode = "traditional"
        self.shapley_config = get_shapley_config()
        self.traditional_config = get_traditional_config()
    
    def set_shapley_mode(self):
        """切换到Shapley值加权模式"""
        self.current_mode = "shapley"
        print("已切换到Shapley值加权模式")
    
    def set_traditional_mode(self):
        """切换到传统模式"""
        self.current_mode = "traditional"
        print("已切换到传统模式")
    
    def get_reward_kwargs(self, atomic_questions=None):
        """获取当前模式的奖励函数参数"""
        if self.current_mode == "shapley":
            return {
                "use_shapley": True,
                "atomic_questions": atomic_questions
            }
        else:
            return {
                "use_shapley": False,
                "atomic_questions": None
            }
    
    def compute_reward(self, model, tokenizer, facts, completions, options, answers, atomic_questions=None):
        """根据当前模式计算奖励"""
        kwargs = self.get_reward_kwargs(atomic_questions)
        
        return overall_reward(
            model=model,
            tokenizer=tokenizer,
            facts=facts,
            completions=completions,
            options=options,
            answers=answers,
            **kwargs
        )

# ================== 性能监控示例 ==================

def monitor_shapley_performance():
    """监控Shapley值计算的性能"""
    import time
    
    print("Shapley值计算性能监控:")
    print("=" * 50)
    
    # 模拟不同规模的数据
    test_sizes = [1, 5, 10, 20]
    
    for size in test_sizes:
        print(f"\n测试批处理大小: {size}")
        
        # 模拟数据
        facts = [["事实1", "事实2", "事实3", "事实4"] for _ in range(size)]
        completions = [[{"content": "test completion"}] for _ in range(size)]
        options = [{"A": "选项A", "B": "选项B"} for _ in range(size)]
        answers = ["A" for _ in range(size)]
        atomic_questions = ["测试问题？" for _ in range(size)]
        
        # 传统模式计时
        start_time = time.time()
        # traditional_rewards = overall_reward(model, tokenizer, facts, completions, options, answers, use_shapley=False)
        traditional_time = time.time() - start_time
        print(f"传统模式耗时: {traditional_time:.3f}s")
        
        # Shapley模式计时
        start_time = time.time()
        # shapley_rewards = overall_reward(model, tokenizer, facts, completions, options, answers, use_shapley=True, atomic_questions=atomic_questions)
        shapley_time = time.time() - start_time
        print(f"Shapley模式耗时: {shapley_time:.3f}s")
        
        # 计算性能比较
        # overhead = (shapley_time - traditional_time) / traditional_time * 100
        # print(f"Shapley模式额外开销: {overhead:.1f}%")

if __name__ == "__main__":
    print("Shapley值加权奖励函数示例")
    print("=" * 50)
    
    # 运行示例
    print("\n1. 批处理示例:")
    example_batch_processing_with_shapley()
    
    print("\n2. 模式管理示例:")
    manager = RewardModeManager()
    manager.set_shapley_mode()
    manager.set_traditional_mode()
    
    print("\n3. 性能监控示例:")
    # monitor_shapley_performance()
    
    print("\n所有示例完成！") 