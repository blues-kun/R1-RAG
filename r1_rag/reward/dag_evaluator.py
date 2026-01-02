"""
R1-RAG DAG奖励评估器

核心奖励计算模块，集成:
1. 语义相似度（E5嵌入）
2. 结构相似度（图编辑距离）
3. 子目标完成度（F1分数）
4. 答案正确性（精确匹配）

结合渐进式权重退火实现平衡训练。
"""

import re
import json
import string
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .config import RewardConfig
from .semantic_scorer import SemanticScorer, SubGoalScorer
from .structure_scorer import StructureScorer


class DAGRewardEvaluator:
    """R1-RAG训练的主奖励评估器
    
    计算多维度奖励，引导模型:
    1. 生成结构良好的推理规划（结构奖励）
    2. 提出语义正确的子问题（语义奖励）
    3. 正确执行子目标（步骤奖励）
    4. 得出正确的最终答案（结果奖励）
    
    渐进式退火策略确保:
    - 训练早期: 关注学习良好的规划习惯
    - 训练后期: 关注获得正确答案
    """
    
    def __init__(self, config: RewardConfig):
        """初始化DAG奖励评估器
        
        Args:
            config: 包含权重和阈值的奖励配置
        """
        self.config = config
        self.semantic_scorer = SemanticScorer(config.embedding_model_path)
        self.structure_scorer = StructureScorer()
        self.subgoal_scorer = SubGoalScorer()
        
        # 解析模型输出的正则表达式模式
        self.plan_pattern = re.compile(r'<plan>(.+?)</plan>', re.DOTALL)
        self.subplan_pattern = re.compile(r'<subPlan>(.+?)</subPlan>', re.DOTALL)
        self.subanswer_pattern = re.compile(r'<subAnswer>\s*#(\d+)\s*=\s*(.*?)\s*</subAnswer>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
    # ==================== 答案提取 ====================
    
    def extract_final_answer(self, response: str) -> Optional[str]:
        """从模型响应中提取最终答案
        
        Args:
            response: 完整的模型响应文本
            
        Returns:
            提取的答案字符串，未找到则返回None
        """
        matches = list(self.answer_pattern.finditer(response))
        if not matches:
            return None
        # 返回最后一个匹配（如果有多次尝试）
        return matches[-1].group(1).strip()
    
    def extract_plan(self, response: str) -> Optional[Dict[str, List[str]]]:
        """从模型响应中提取规划DAG
        
        响应中的预期格式:
        <plan>
        {"Q1": ["子问题1", "#1"], "Q2": ["使用#1的子问题2", "#2"]}
        </plan>
        
        Args:
            response: 完整的模型响应
            
        Returns:
            解析的规划字典，解析失败返回None
        """
        match = self.plan_pattern.search(response)
        if not match:
            return None
            
        try:
            plan_text = match.group(1).strip()
            # 清理常见格式问题
            plan_text = plan_text.replace("json", "").replace("```", "")
            plan = json.loads(plan_text)
            
            # 规范化占位符: #1 -> <A1>
            normalized = {}
            for key, value in plan.items():
                new_value = []
                for item in value:
                    # 将 #N 替换为 <AN>
                    normalized_item = re.sub(r"#(\d+)", r"<A\1>", str(item))
                    new_value.append(normalized_item)
                normalized[key] = new_value
            
            return normalized
        except (json.JSONDecodeError, Exception) as e:
            return None
    
    def extract_execution_graph(self, response: str) -> Dict[str, Dict]:
        """从subPlan块中提取执行结果
        
        解析每个subPlan的中间答案，构建执行图，
        显示每个子问题找到的答案。
        
        Args:
            response: 完整的模型响应
            
        Returns:
            执行图 {"Q1": {"answer": "..."}, "Q2": {...}}
        """
        graph = {}
        
        subplans = self.subplan_pattern.findall(response)
        for subplan in subplans:
            # 提取 <subAnswer> #N = answer </subAnswer>
            match = self.subanswer_pattern.search(subplan)
            if match:
                q_idx = match.group(1)
                answer = match.group(2).strip()
                graph[f"Q{q_idx}"] = {"answer": answer}
        
        return graph
    
    # ==================== 答案评分 ====================
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """规范化答案文本用于精确匹配比较"""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(c for c in text if c not in string.punctuation)
        text = " ".join(text.split())
        return text
    
    def exact_match(self, prediction: str, gold_answers: List[str]) -> float:
        """检查预测是否与任一黄金答案精确匹配
        
        Args:
            prediction: 预测的答案
            gold_answers: 可接受的黄金答案列表
            
        Returns:
            匹配返回1.0，否则返回0.0
        """
        if prediction is None:
            return 0.0
            
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
            
        norm_pred = self.normalize_answer(prediction)
        
        for gold in gold_answers:
            if self.normalize_answer(gold) == norm_pred:
                return 1.0
        return 0.0
    
    # ==================== 格式检查 ====================
    
    def check_format(self, response: str) -> float:
        """检查响应是否遵循要求的格式
        
        要求的结构:
        1. <plan>前有<think>...</think>
        2. <plan>...</plan>块包含有效JSON
        3. 一个或多个<subPlan>...</subPlan>块
        4. 每个subPlan包含<search>, <information>, <subAnswer>
        5. <answer>前有最终的<think>...</think>
        
        Args:
            response: 要检查的模型响应
            
        Returns:
            格式有效返回1.0，否则返回0.0
        """
        errors = []
        
        # 检查plan块及其前面的think
        if not re.search(r"<think>.*?</think>\s*<plan>.*?</plan>", response, re.DOTALL):
            errors.append("缺少<plan>块或前面的<think>")
        
        # 检查answer块及其前面的think
        if not re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", response, re.DOTALL):
            errors.append("缺少<answer>块或前面的<think>")
        
        # 检查subPlan块
        subplans = re.findall(r"<subPlan>.*?</subPlan>", response, re.DOTALL)
        if not subplans:
            errors.append("未找到<subPlan>块")
        
        # 验证每个subPlan结构
        for i, sp in enumerate(subplans, 1):
            if not re.search(r"<think>.*?</think>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: 缺少<think>")
            if not re.search(r"<search>.*?</search>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: 缺少<search>")
            if not re.search(r"<information>.*?</information>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: 缺少<information>")
            if not re.search(r"<subAnswer>.*?</subAnswer>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: 缺少<subAnswer>")
        
        return 1.0 if not errors else 0.0
    
    # ==================== 主奖励计算 ====================
    
    def compute_reward(
        self,
        response: str,
        gold_answer: List[str],
        gold_plan: Optional[Dict[str, List[str]]] = None,
        gold_graph: Optional[List[Dict]] = None,
        training_step: int = 0,
        verbose: bool = False
    ) -> Dict[str, float]:
        """计算模型响应的综合奖励
        
        使用渐进式退火组合多个奖励信号:
        
        R_total = α(t) * R_process + R_outcome
        
        其中:
        - α(t) 随训练递减（退火）
        - R_process = w_f * format + w_s * semantic + w_g * structure + w_step * step
        - R_outcome = answer_score（精确匹配）
        
        Args:
            response: 模型响应文本
            gold_answer: 可接受的黄金答案列表
            gold_plan: 黄金规划DAG（可选）
            gold_graph: 黄金执行图（可选）
            training_step: 当前训练步数，用于退火
            verbose: 是否打印详细分数
            
        Returns:
            包含所有分量分数和最终奖励的字典
        """
        # 初始化分数
        scores = {
            "format_score": 0.0,
            "semantic_score": 0.0,
            "structure_score": 0.0,
            "step_score": 0.0,
            "answer_score": 0.0,
            "final_reward": 0.0
        }
        
        # 1. 格式合规性
        scores["format_score"] = self.check_format(response)
        
        # 2. 答案正确性（结果奖励）
        pred_answer = self.extract_final_answer(response)
        scores["answer_score"] = self.exact_match(pred_answer, gold_answer)
        
        # 3. 过程奖励（仅当有黄金标注时）
        if gold_plan and gold_graph:
            # 提取预测的规划和执行
            pred_plan = self.extract_plan(response)
            pred_graph = self.extract_execution_graph(response)
            
            if pred_plan:
                # 语义相似度
                semantic_score, mapping = self.semantic_scorer.compute_semantic_score(
                    pred_plan, gold_plan, threshold=self.config.node_match_threshold
                )
                scores["semantic_score"] = semantic_score
                
                # 结构相似度
                ged, norm_ged, struct_match = self.structure_scorer.compute_structure_score(
                    pred_plan, gold_plan, mapping,
                    alpha=self.config.plan_alpha,
                    beta=self.config.ged_beta
                )
                scores["structure_score"] = struct_match
                scores["ged_raw"] = ged
                scores["ged_normalized"] = norm_ged
                
                # 子目标完成度
                if pred_graph and gold_graph:
                    gold_graph_dict = gold_graph[0] if isinstance(gold_graph, list) else gold_graph
                    scores["step_score"] = self.subgoal_scorer.compute_step_score(
                        pred_graph, gold_graph_dict, mapping
                    )
        
        # 4. 计算带退火的最终奖励
        if self.config.validation_mode:
            # 验证: 只关注答案分数
            scores["final_reward"] = scores["answer_score"]
        else:
            # 训练: 组合过程和结果奖励
            annealing_weight = self.config.get_annealing_weight(training_step)
            
            process_reward = (
                self.config.format_weight * scores["format_score"] +
                self.config.plan_sim_weight * scores["semantic_score"] +
                self.config.structure_weight * scores["structure_score"] +
                self.config.step_weight * scores["step_score"]
            )
            
            scores["process_reward"] = process_reward
            scores["annealing_weight"] = annealing_weight
            scores["final_reward"] = annealing_weight * process_reward + scores["answer_score"]
        
        # 详细日志
        if verbose or random.random() < 0.02:  # 记录约2%的样本
            print(f"\n{'='*50}")
            print(f"[DAG评估器] 训练步数: {training_step}")
            print(f"  格式:    {scores['format_score']:.3f}")
            print(f"  语义:    {scores['semantic_score']:.3f}")
            print(f"  结构:    {scores['structure_score']:.3f}")
            print(f"  步骤:    {scores['step_score']:.3f}")
            print(f"  答案:    {scores['answer_score']:.3f}")
            print(f"  退火:    {scores.get('annealing_weight', 1.0):.3f}")
            print(f"  最终:    {scores['final_reward']:.3f}")
            print(f"  黄金: {gold_answer}")
            print(f"  预测: {pred_answer}")
        
        return scores


class RewardManager:
    """管理批量训练的奖励计算
    
    与veRL训练框架集成，在GRPO优化期间提供奖励信号。
    """
    
    def __init__(
        self,
        tokenizer,
        config: RewardConfig,
        num_examine: int = 1
    ):
        """初始化奖励管理器
        
        Args:
            tokenizer: HuggingFace tokenizer用于解码
            config: 奖励配置
            num_examine: 每个数据源打印的样本数
        """
        self.tokenizer = tokenizer
        self.evaluator = DAGRewardEvaluator(config)
        self.config = config
        self.num_examine = num_examine
        
    def __call__(self, data, training_step: int = 0):
        """计算批量样本的奖励
        
        Args:
            data: 来自veRL的DataProto批次
            training_step: 当前训练步数
            
        Returns:
            形状与responses匹配的奖励张量
        """
        import torch
        
        reward_tensor = torch.zeros_like(
            data.batch['responses'], dtype=torch.float32
        )
        
        examined = {}
        
        for i in range(len(data)):
            item = data[i]
            
            # 获取prompt和response长度
            prompt_ids = item.batch['prompts']
            prompt_len = prompt_ids.shape[-1]
            
            valid_prompt_len = item.batch['attention_mask'][:prompt_len].sum()
            
            response_ids = item.batch['responses']
            valid_response_len = item.batch['attention_mask'][prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]
            
            # 解码响应
            response_text = self.tokenizer.decode(valid_response_ids)
            
            # 获取ground truth和元数据
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            gold_answer = ground_truth.get('target', [])
            
            metadata = item.non_tensor_batch.get('metadata', {})
            gold_plan = metadata.get('plan', None)
            gold_graph = metadata.get('graph', None)
            
            # 修复numpy数组问题
            if gold_plan:
                gold_plan = {k: [str(x) for x in v] for k, v in gold_plan.items()}
            if gold_graph and isinstance(gold_graph, np.ndarray):
                gold_graph = gold_graph.tolist()
            
            # 计算奖励
            scores = self.evaluator.compute_reward(
                response=response_text,
                gold_answer=gold_answer,
                gold_plan=gold_plan,
                gold_graph=gold_graph,
                training_step=training_step
            )
            
            # 将奖励赋给最后一个token
            reward_tensor[i, valid_response_len - 1] = scores['final_reward']
            
            # 打印示例
            data_source = item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in examined:
                examined[data_source] = 0
            if examined[data_source] < self.num_examine:
                examined[data_source] += 1
                print(f"\n[来自 {data_source} 的样本]")
                print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        return reward_tensor
