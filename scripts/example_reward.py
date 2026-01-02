#!/usr/bin/env python3
"""
示例: R1-RAG的基于DAG的奖励计算

演示核心奖励机制:
1. 解析模型响应（plan、sub-answers、answer）
2. 计算结构相似度（GED）
3. 计算语义相似度（E5）
4. 结合渐进式退火

用法:
    python scripts/example_reward.py
"""

import sys
sys.path.insert(0, '.')

from r1_rag.reward import DAGRewardEvaluator, RewardConfig


def main():
    print("=" * 70)
    print("R1-RAG: 基于DAG的奖励计算示例")
    print("=" * 70)
    
    # 初始化奖励评估器
    config = RewardConfig(
        embedding_model_path="intfloat/e5-base-v2",
        node_match_threshold=0.7,
        format_weight=0.1,
        plan_sim_weight=0.5,
        structure_weight=0.5,
        step_weight=0.5,
        ged_beta=1.0,
        annealing_total_steps=50,
        annealing_center=0.9,
        annealing_temperature=10.0,
    )
    
    print("\n[配置]")
    print(f"  嵌入模型: {config.embedding_model_path}")
    print(f"  节点匹配阈值: {config.node_match_threshold}")
    print(f"  退火步数: {config.annealing_total_steps}")
    
    evaluator = DAGRewardEvaluator(config)
    
    # ==================== 测试案例1: 完美响应 ====================
    print("\n" + "=" * 70)
    print("测试案例1: 完美响应")
    print("=" * 70)
    
    response_perfect = """
<think> 这个问题问的是《泰坦尼克号》导演的第一部电影。这是一个2跳问题。
首先，我需要找出《泰坦尼克号》的导演。然后，我需要找出该导演的第一部电影。 </think>
<plan>
{"Q1": ["谁执导了《泰坦尼克号》？", "#1"], "Q2": ["#1的第一部电影是什么？", "#2"]}
</plan>

<subPlan>
    <think> 让我搜索《泰坦尼克号》的导演。 </think>
    <search> 泰坦尼克号 导演 </search>
    <information> 《泰坦尼克号》是1997年上映的史诗级电影，由詹姆斯·卡梅隆执导。 </information>
    <think> 导演是詹姆斯·卡梅隆。 </think>
    <subAnswer> #1 = 詹姆斯·卡梅隆 </subAnswer>
</subPlan>

<subPlan>
    <think> 现在我需要找詹姆斯·卡梅隆的第一部电影。 </think>
    <search> 詹姆斯·卡梅隆 第一部电影 导演处女作 </search>
    <information> 詹姆斯·卡梅隆的导演处女作是1982年的《食人鱼2》。 </information>
    <think> 他的第一部电影是《食人鱼2》。 </think>
    <subAnswer> #2 = 食人鱼2 </subAnswer>
</subPlan>

<think> 我已经收集了所有必要的信息。 </think>
<answer> 食人鱼2 </answer>
    """.strip()
    
    gold_answer = ["食人鱼2", "Piranha II: The Spawning"]
    gold_plan = {
        "Q1": ["谁执导了《泰坦尼克号》？", "<A1>"],
        "Q2": ["<A1>的第一部电影是什么？", "<A2>"],
    }
    gold_graph = [{
        "Q1": {"answer": "詹姆斯·卡梅隆"},
        "Q2": {"answer": "食人鱼2"},
    }]
    
    scores = evaluator.compute_reward(
        response=response_perfect,
        gold_answer=gold_answer,
        gold_plan=gold_plan,
        gold_graph=gold_graph,
        training_step=25,  # 训练中期
        verbose=False,
    )
    
    print("\n[分数]")
    print(f"  格式合规:    {scores['format_score']:.3f}")
    print(f"  语义相似度:  {scores['semantic_score']:.3f}")
    print(f"  结构匹配:    {scores['structure_score']:.3f}")
    print(f"  步骤完成:    {scores['step_score']:.3f}")
    print(f"  答案正确:    {scores['answer_score']:.3f}")
    print(f"  退火权重:    {scores.get('annealing_weight', 1.0):.3f}")
    print(f"  最终奖励:    {scores['final_reward']:.3f}")
    
    # ==================== 测试案例2: 错误结构 ====================
    print("\n" + "=" * 70)
    print("测试案例2: 错误结构（缺少步骤）")
    print("=" * 70)
    
    response_wrong = """
<think> 我直接搜索答案。 </think>
<plan>
{"Q1": ["《泰坦尼克号》导演的第一部电影是什么？", "#1"]}
</plan>

<subPlan>
    <think> 直接搜索。 </think>
    <search> 泰坦尼克号导演第一部电影 </search>
    <information> 詹姆斯·卡梅隆导演了《泰坦尼克号》，他的第一部电影是《食人鱼2》。 </information>
    <think> 找到答案了。 </think>
    <subAnswer> #1 = 食人鱼2 </subAnswer>
</subPlan>

<think> 完成。 </think>
<answer> 食人鱼2 </answer>
    """.strip()
    
    scores_wrong = evaluator.compute_reward(
        response=response_wrong,
        gold_answer=gold_answer,
        gold_plan=gold_plan,
        gold_graph=gold_graph,
        training_step=25,
    )
    
    print("\n[分数]")
    print(f"  格式合规:    {scores_wrong['format_score']:.3f}")
    print(f"  语义相似度:  {scores_wrong['semantic_score']:.3f}")
    print(f"  结构匹配:    {scores_wrong['structure_score']:.3f}")
    print(f"  步骤完成:    {scores_wrong['step_score']:.3f}")
    print(f"  答案正确:    {scores_wrong['answer_score']:.3f}")
    print(f"  最终奖励:    {scores_wrong['final_reward']:.3f}")
    
    print("\n[分析]")
    print(f"  模型答案正确但结构错误（1步 vs 2步）。")
    print(f"  结构惩罚: {scores['structure_score'] - scores_wrong['structure_score']:.3f}")
    print(f"  最终奖励差异: {scores['final_reward'] - scores_wrong['final_reward']:.3f}")
    
    # ==================== 测试案例3: 退火效果 ====================
    print("\n" + "=" * 70)
    print("测试案例3: 渐进式退火效果")
    print("=" * 70)
    
    print("\n[退火权重调度]")
    print("  步数 | 权重   | 关注点")
    print("  -----|--------|------")
    for step in [0, 10, 25, 40, 45, 50]:
        weight = config.get_annealing_weight(step)
        focus = "过程" if weight > 0.5 else "结果"
        print(f"  {step:4d} | {weight:.3f}  | {focus}")
    
    print("\n[相同响应在不同训练阶段]")
    print("  步数 | 最终奖励")
    print("  -----|--------")
    for step in [0, 25, 45, 50]:
        scores_step = evaluator.compute_reward(
            response=response_wrong,  # 使用不完美的响应
            gold_answer=gold_answer,
            gold_plan=gold_plan,
            gold_graph=gold_graph,
            training_step=step,
        )
        print(f"  {step:4d} | {scores_step['final_reward']:.3f}")
    
    print("\n[洞察]")
    print("  训练早期: 过程奖励更重要（惩罚错误结构）")
    print("  训练后期: 结果更重要（奖励正确答案）")
    
    print("\n" + "=" * 70)
    print("示例完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
