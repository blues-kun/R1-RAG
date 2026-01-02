#!/usr/bin/env python3
"""
Example: DAG-based Reward Computation for R1-RAG

Demonstrates the core reward mechanism:
1. Parse model response (plan, sub-answers, answer)
2. Compute structural similarity (GED)
3. Compute semantic similarity (E5)
4. Combine with progressive annealing

Usage:
    python scripts/example_reward.py
"""

import sys
sys.path.insert(0, '.')

from r1_rag.reward import DAGRewardEvaluator, RewardConfig


def main():
    print("=" * 70)
    print("R1-RAG: DAG-based Reward Computation Example")
    print("=" * 70)
    
    # Initialize reward evaluator
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
    
    print("\n[Config]")
    print(f"  Embedding model: {config.embedding_model_path}")
    print(f"  Node match threshold: {config.node_match_threshold}")
    print(f"  Annealing steps: {config.annealing_total_steps}")
    
    evaluator = DAGRewardEvaluator(config)
    
    # ==================== Test Case 1: Perfect Response ====================
    print("\n" + "=" * 70)
    print("Test Case 1: Perfect Response")
    print("=" * 70)
    
    response_perfect = """
<think> This question asks about the population of Einstein's birthplace. It's a 2-hop question.
First, I need to find where Einstein was born. Then, I need to find the population of that city. </think>
<plan>
{"Q1": ["Where was Albert Einstein born?", "#1"], "Q2": ["What is the population of #1?", "#2"]}
</plan>

<subPlan>
    <think> Let me search for Einstein's birthplace. </think>
    <search> Albert Einstein birthplace city </search>
    <information> Albert Einstein was born on March 14, 1879, in Ulm, Germany. </information>
    <think> Einstein was born in Ulm, Germany. </think>
    <subAnswer> #1 = Ulm </subAnswer>
</subPlan>

<subPlan>
    <think> Now I need to find the population of Ulm. </think>
    <search> population of Ulm Germany </search>
    <information> Ulm has a population of approximately 126,000 people. </information>
    <think> The population is about 126,000. </think>
    <subAnswer> #2 = 126,000 </subAnswer>
</subPlan>

<think> I have all the information needed. </think>
<answer> 126,000 </answer>
    """.strip()
    
    gold_answer = ["126,000", "126000", "about 126,000"]
    gold_plan = {
        "Q1": ["Where was Albert Einstein born?", "<A1>"],
        "Q2": ["What is the population of <A1>?", "<A2>"],
    }
    gold_graph = [{
        "Q1": {"answer": "Ulm"},
        "Q2": {"answer": "126,000"},
    }]
    
    scores = evaluator.compute_reward(
        response=response_perfect,
        gold_answer=gold_answer,
        gold_plan=gold_plan,
        gold_graph=gold_graph,
        training_step=25,  # Middle of training
        verbose=False,
    )
    
    print("\n[Scores]")
    print(f"  Format compliance:    {scores['format_score']:.3f}")
    print(f"  Semantic similarity:  {scores['semantic_score']:.3f}")
    print(f"  Structure match:      {scores['structure_score']:.3f}")
    print(f"  Step completion:      {scores['step_score']:.3f}")
    print(f"  Answer correctness:   {scores['answer_score']:.3f}")
    print(f"  Annealing weight:     {scores.get('annealing_weight', 1.0):.3f}")
    print(f"  Final reward:         {scores['final_reward']:.3f}")
    
    # ==================== Test Case 2: Wrong Structure ====================
    print("\n" + "=" * 70)
    print("Test Case 2: Wrong Structure (Missing Step)")
    print("=" * 70)
    
    response_wrong = """
<think> I'll directly search for the answer. </think>
<plan>
{"Q1": ["What is the population of Einstein's birthplace?", "#1"]}
</plan>

<subPlan>
    <think> Searching directly. </think>
    <search> population of Einstein birthplace </search>
    <information> Albert Einstein was born in Ulm, which has about 126,000 people. </information>
    <think> Found the answer. </think>
    <subAnswer> #1 = 126,000 </subAnswer>
</subPlan>

<think> Done. </think>
<answer> 126,000 </answer>
    """.strip()
    
    scores_wrong = evaluator.compute_reward(
        response=response_wrong,
        gold_answer=gold_answer,
        gold_plan=gold_plan,
        gold_graph=gold_graph,
        training_step=25,
    )
    
    print("\n[Scores]")
    print(f"  Format compliance:    {scores_wrong['format_score']:.3f}")
    print(f"  Semantic similarity:  {scores_wrong['semantic_score']:.3f}")
    print(f"  Structure match:      {scores_wrong['structure_score']:.3f}")
    print(f"  Step completion:      {scores_wrong['step_score']:.3f}")
    print(f"  Answer correctness:   {scores_wrong['answer_score']:.3f}")
    print(f"  Final reward:         {scores_wrong['final_reward']:.3f}")
    
    print("\n[Analysis]")
    print(f"  The model gets the answer right but has wrong structure (1 step vs 2 steps).")
    print(f"  Structure penalty: {scores['structure_score'] - scores_wrong['structure_score']:.3f}")
    print(f"  Final reward difference: {scores['final_reward'] - scores_wrong['final_reward']:.3f}")
    
    # ==================== Test Case 3: Annealing Effect ====================
    print("\n" + "=" * 70)
    print("Test Case 3: Progressive Annealing Effect")
    print("=" * 70)
    
    print("\n[Annealing Weight Schedule]")
    print("  Step | Weight | Focus")
    print("  -----|--------|------")
    for step in [0, 10, 25, 40, 45, 50]:
        weight = config.get_annealing_weight(step)
        focus = "Process" if weight > 0.5 else "Outcome"
        print(f"  {step:4d} | {weight:.3f}  | {focus}")
    
    print("\n[Same Response at Different Training Stages]")
    print("  Step | Final Reward")
    print("  -----|-------------")
    for step in [0, 25, 45, 50]:
        scores_step = evaluator.compute_reward(
            response=response_wrong,  # Use imperfect response
            gold_answer=gold_answer,
            gold_plan=gold_plan,
            gold_graph=gold_graph,
            training_step=step,
        )
        print(f"  {step:4d} | {scores_step['final_reward']:.3f}")
    
    print("\n[Insight]")
    print("  Early training: Process rewards matter more (penalize wrong structure)")
    print("  Late training: Outcome matters more (reward correct answer)")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

