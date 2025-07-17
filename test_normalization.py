#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Shapley值归一化功能
演示z-score归一化（默认）和softmax归一化的差异
"""

import numpy as np
import sys
import os

# 添加src路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.doctor_reward import normalize_shapley_weights

def test_normalization_methods():
    """测试不同归一化方法的效果"""
    
    print("🧪 测试Shapley值归一化方法")
    print("=" * 60)
    
    # 测试案例1: 差异较大的Shapley值
    print("\n📊 测试案例1: 差异较大的Shapley值")
    shapley_scores_1 = np.array([0.8, 0.3, -0.2, 0.1, -0.5])
    print(f"原始Shapley值: {shapley_scores_1}")
    
    # Z-score归一化（默认）
    print("\n🔹 Z-score归一化（默认方法）:")
    weights_zscore_1 = normalize_shapley_weights(shapley_scores_1, method="z_score")
    
    # Softmax归一化
    print("\n🔹 Softmax归一化:")
    weights_softmax_1 = normalize_shapley_weights(shapley_scores_1, method="softmax")
    
    print(f"\n📈 结果对比:")
    print(f"  Z-score归一化权重: {weights_zscore_1}")
    print(f"  Softmax归一化权重: {weights_softmax_1}")
    print(f"  权重差异: {np.abs(weights_zscore_1 - weights_softmax_1)}")
    
    # 测试案例2: 相近的Shapley值
    print("\n" + "=" * 60)
    print("📊 测试案例2: 相近的Shapley值")
    shapley_scores_2 = np.array([0.12, 0.15, 0.13, 0.11, 0.14])
    print(f"原始Shapley值: {shapley_scores_2}")
    
    # Z-score归一化
    print("\n🔹 Z-score归一化:")
    weights_zscore_2 = normalize_shapley_weights(shapley_scores_2, method="z_score")
    
    # Softmax归一化
    print("\n🔹 Softmax归一化:")
    weights_softmax_2 = normalize_shapley_weights(shapley_scores_2, method="softmax")
    
    print(f"\n📈 结果对比:")
    print(f"  Z-score归一化权重: {weights_zscore_2}")
    print(f"  Softmax归一化权重: {weights_softmax_2}")
    print(f"  权重差异: {np.abs(weights_zscore_2 - weights_softmax_2)}")
    
    # 测试案例3: 包含正负值的Shapley值
    print("\n" + "=" * 60)
    print("📊 测试案例3: 包含正负值的Shapley值")
    shapley_scores_3 = np.array([0.5, -0.3, 0.8, -0.1, 0.2])
    print(f"原始Shapley值: {shapley_scores_3}")
    
    # Z-score归一化
    print("\n🔹 Z-score归一化:")
    weights_zscore_3 = normalize_shapley_weights(shapley_scores_3, method="z_score")
    
    # Softmax归一化
    print("\n🔹 Softmax归一化:")
    weights_softmax_3 = normalize_shapley_weights(shapley_scores_3, method="softmax")
    
    print(f"\n📈 结果对比:")
    print(f"  Z-score归一化权重: {weights_zscore_3}")
    print(f"  Softmax归一化权重: {weights_softmax_3}")
    print(f"  权重差异: {np.abs(weights_zscore_3 - weights_softmax_3)}")
    
    # 测试不同温度参数的softmax
    print("\n" + "=" * 60)
    print("📊 测试案例4: 不同温度参数的Softmax归一化")
    shapley_scores_4 = np.array([0.8, 0.3, 0.1, 0.05])
    print(f"原始Shapley值: {shapley_scores_4}")
    
    temperatures = [0.5, 1.0, 2.0]
    for temp in temperatures:
        print(f"\n🔹 Softmax归一化 (temperature={temp}):")
        weights_temp = normalize_shapley_weights(shapley_scores_4, method="softmax", temperature=temp)
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("\n💡 使用建议:")
    print("  - 默认使用z_score方法，它能更好地处理正负Shapley值")
    print("  - z_score方法基于绝对值重要性，不受正负符号影响")
    print("  - softmax方法保留原始值的相对关系，适合明确正值场景")
    print("  - 可通过method参数选择: method='z_score'(默认) 或 method='softmax'")

if __name__ == "__main__":
    test_normalization_methods() 