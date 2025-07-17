# Shapley值加权事实奖励函数使用指南

## 概述

本文档介绍了基于Shapley值加权的事实奖励函数，这是对原有`fact_score_reward`函数的重要改进。该功能支持动态计算Shapley值来加权医生模型获取的事实信息，提供更精确的奖励信号。

## 主要特性

1. **动态Shapley值计算**：使用当前策略模型实时计算每个事实的Shapley值
2. **选择性事实评估**：只对`len(fact_list) // 2`之后的事实计算Shapley值（医生一开始不知道的事实）
3. **兼容性设计**：支持传统模式和Shapley值加权模式的无缝切换
4. **高效在线计算**：优化的算法参数，适合训练过程中的实时计算

## 核心函数说明

### 1. `compute_shapley_weighted_fact_score`

```python
def compute_shapley_weighted_fact_score(
    model, tokenizer, fact_list, formatted_dialog, 
    atomic_question, target_answer, use_shapley=True
):
    """
    计算基于Shapley值加权的事实分数
    
    Args:
        model: 当前策略模型
        tokenizer: 分词器
        fact_list: 事实列表
        formatted_dialog: 格式化的对话内容
        atomic_question: 原子问题
        target_answer: 目标答案
        use_shapley: 是否使用Shapley值加权
    
    Returns:
        加权后的事实分数
    """
```

### 2. 改进的`fact_score_reward`

```python
def fact_score_reward(model, tokenizer, facts: List[List[str]], 
                     completions: List[List[Dict[str, Any]]],
                     use_shapley: bool = False,
                     atomic_questions: List[str] = None,
                     target_answers: List[str] = None) -> List[float]:
    """
    支持Shapley值加权的事实奖励函数
    """
```

### 3. 更新的`overall_reward`

```python
def overall_reward(model, tokenizer, facts: List[str], 
                  completions: List[List[Dict[str, Any]]],
                  options: List[Dict[str, str]], answers: List[str],
                  use_shapley: bool = False,
                  atomic_questions: List[str] = None) -> Dict[str, List[float]]:
    """
    支持Shapley值加权的综合奖励函数
    """
```

## 使用方法

### 1. 传统模式（现有行为）

```python
from src.models.doctor_reward import overall_reward

# 使用传统模式，与原有代码兼容
rewards = overall_reward(
    model=model,
    tokenizer=tokenizer,
    facts=batch_facts,
    completions=batch_completions,
    options=batch_options,
    answers=batch_answers,
    use_shapley=False  # 传统模式
)
```

### 2. Shapley值加权模式

```python
# 准备原子问题列表
atomic_questions = [
    "患者的主要症状是什么？",
    "应该如何诊断这个病例？",
    "推荐的治疗方案是什么？"
]

# 使用Shapley值加权模式
rewards = overall_reward(
    model=model,
    tokenizer=tokenizer,
    facts=batch_facts,
    completions=batch_completions,
    options=batch_options,
    answers=batch_answers,
    use_shapley=True,  # 启用Shapley值加权
    atomic_questions=atomic_questions
)
```

### 3. 在训练中使用

```python
from src.models.doctor_trainer import train_with_grpo, maximize_grpo_objective

# 定义自定义奖励函数
def shapley_reward_function(model, tokenizer, facts, completions, options, answers, 
                           use_shapley=False, atomic_questions=None):
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

# 在train_with_grpo中，maximize_grpo_objective会自动处理Shapley参数
# 只需要确保数据中包含atomic_questions信息
```

## 配置参数

### Shapley值计算参数

```python
shapley_config = {
    "max_samples": 15,      # 最大采样次数（在线计算建议15-20）
    "min_samples": 3,       # 最小采样次数
    "tolerance": 5e-2,      # 收敛容忍值
    "early_stop_eps": 1e-2  # 提前停止阈值
}
```

这些参数在`estimate_atomic_shapley_fast`函数中控制计算效率和精度的平衡。

## 数据准备

### 1. 事实列表结构

```python
batch_facts = [
    [
        "患者年龄65岁",        # 已知事实（前半部分）
        "有高血压病史",        # 已知事实
        "胸痛症状持续2小时",   # 未知事实（后半部分，需要计算Shapley值）
        "血压160/90mmHg"       # 未知事实
    ],
    # ... 更多样本
]
```

### 2. 原子问题

原子问题应该与医疗决策相关，例如：
- "根据症状判断最可能的诊断是什么？"
- "患者的症状提示哪种疾病？"
- "应该进行什么检查？"

## 性能考虑

### 1. 计算复杂度

- **传统模式**：O(1) - 与原有实现相同
- **Shapley模式**：O(n × k) - 其中n是未知事实数量，k是采样次数

### 2. 优化策略

1. **在线计算优化**：
   - 减少采样次数（max_samples=15而非100）
   - 提前停止机制
   - 缓存优化

2. **批处理优化**：
   - 并行计算多个样本的Shapley值
   - 重用KV缓存

### 3. 内存使用

Shapley值计算会增加GPU内存使用，建议：
- 监控GPU内存使用情况
- 必要时减少batch_size
- 使用gradient checkpointing

## 错误处理

函数包含完善的错误处理机制：

```python
try:
    # Shapley值计算
    shapley_scores = estimate_atomic_shapley_fast(...)
    # 处理计算结果
except Exception as e:
    print(f"Shapley值计算出错: {e}, 回退到传统模式")
    # 自动回退到传统模式
    return compute_shapley_weighted_fact_score(..., use_shapley=False)
```

## 最佳实践

### 1. 模式选择

- **训练初期**：使用传统模式，确保基本功能正常
- **训练中后期**：切换到Shapley模式，获得更精确的奖励信号
- **调试阶段**：使用传统模式，减少复杂性

### 2. 参数调优

```python
# 快速测试配置
fast_config = {
    "max_samples": 10,
    "min_samples": 2,
    "tolerance": 1e-1
}

# 精确计算配置
precise_config = {
    "max_samples": 25,
    "min_samples": 5,
    "tolerance": 1e-2
}
```

### 3. 监控建议

```python
# 记录奖励分数分解
if use_shapley:
    logging.info(f"Shapley模式 - 事实分数: {rewards['fact_scores']}")
    logging.info(f"事实数量: {len(facts[0])}, 未知事实: {len(facts[0])//2}")
else:
    logging.info(f"传统模式 - 事实分数: {rewards['fact_scores']}")
```

## 示例代码

完整示例请参考：`src/models/shapley_reward_example.py`

该文件包含：
- 基本使用示例
- 批处理示例
- 模式切换示例
- 性能监控示例

## 故障排除

### 常见问题

1. **Shapley值全为负数**
   - 检查target_answer格式是否正确
   - 确认atomic_question与任务匹配

2. **计算时间过长**
   - 减少max_samples参数
   - 增加tolerance值
   - 检查事实列表长度

3. **内存不足**
   - 减少batch_size
   - 启用gradient checkpointing
   - 监控GPU内存使用

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.INFO)

# 监控Shapley值分布
shapley_scores = estimate_atomic_shapley_fast(...)
print(f"Shapley值分布: min={min(shapley_scores):.3f}, max={max(shapley_scores):.3f}")
print(f"Shapley值: {shapley_scores}")
```

## 总结

Shapley值加权奖励函数提供了更精确的事实获取评估，特别适合医疗对话场景中的知识获取任务。通过合理配置和使用，可以显著提升模型的训练效果和对话质量。 