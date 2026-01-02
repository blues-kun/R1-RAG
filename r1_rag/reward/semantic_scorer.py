"""
R1-RAG 语义评分器

使用E5嵌入模型计算语义相似度:
- 预测子问题与黄金子问题的相似度
- 预测中间答案与黄金答案的相似度

为强化学习训练提供"内容奖励"信号。
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, util


class SemanticScorer:
    """使用嵌入计算规划DAG之间的语义相似度
    
    评分器使用预训练的E5模型来:
    1. 将子问题编码为稠密向量
    2. 计算成对余弦相似度
    3. 在预测和黄金规划之间找到最优节点映射
    
    核心洞察: 语义相似度能捕捉模型是否提出了"正确的问题"，
    即使措辞与黄金规划不同。
    """
    
    def __init__(self, model_path: str = "intfloat/e5-base-v2"):
        """初始化语义评分器
        
        Args:
            model_path: sentence transformer模型路径
        """
        self.model = SentenceTransformer(model_path)
        self.placeholder_pattern = re.compile(r"<A(\d+)>")
        
    def encode_questions(self, questions: List[str]) -> np.ndarray:
        """将问题编码为稠密嵌入
        
        Args:
            questions: 问题字符串列表
            
        Returns:
            形状为 (N, embedding_dim) 的嵌入矩阵
        """
        # 将占位符规范化为通用的"entity"以获得更好的匹配
        normalized = []
        for q in questions:
            normalized_q = self.placeholder_pattern.sub("entity", q)
            normalized.append(normalized_q)
        
        return self.model.encode(normalized)
    
    def compute_similarity_matrix(
        self, 
        pred_questions: List[str], 
        gold_questions: List[str]
    ) -> np.ndarray:
        """计算问题集之间的成对余弦相似度
        
        Args:
            pred_questions: 预测的子问题
            gold_questions: 黄金子问题
            
        Returns:
            形状为 (len(gold), len(pred)) 的相似度矩阵
        """
        pred_emb = self.encode_questions(pred_questions)
        gold_emb = self.encode_questions(gold_questions)
        
        return util.cos_sim(gold_emb, pred_emb).numpy()
    
    def find_optimal_mapping(
        self,
        similarity_matrix: np.ndarray,
        gold_nodes: List[str],
        pred_nodes: List[str],
        threshold: float = 0.7
    ) -> Tuple[Dict[str, str], List[float]]:
        """使用贪婪匹配找到最优节点映射
        
        使用余弦相似度将预测节点匹配到黄金节点。
        只有超过阈值的匹配才被视为有效。
        
        Args:
            similarity_matrix: 成对相似度 (gold x pred)
            gold_nodes: 黄金节点ID
            pred_nodes: 预测节点ID
            threshold: 有效匹配的最小相似度
            
        Returns:
            (映射字典, 匹配节点的相似度分数) 元组
        """
        mapping = {}
        similarities = []
        used_pred = set()
        
        # 贪婪匹配：遍历黄金节点
        for i, gold_node in enumerate(gold_nodes):
            best_sim = -1
            best_pred_idx = -1
            
            for j, pred_node in enumerate(pred_nodes):
                if pred_node in used_pred:
                    continue
                    
                sim = similarity_matrix[i, j]
                if sim >= threshold and sim > best_sim:
                    best_sim = sim
                    best_pred_idx = j
            
            if best_pred_idx >= 0:
                mapping[gold_node] = pred_nodes[best_pred_idx]
                similarities.append(best_sim)
                used_pred.add(pred_nodes[best_pred_idx])
        
        return mapping, similarities
    
    def compute_semantic_score(
        self,
        pred_plan: Dict[str, List[str]],
        gold_plan: Dict[str, List[str]],
        threshold: float = 0.7
    ) -> Tuple[float, Dict[str, str]]:
        """计算规划之间的整体语义相似度分数
        
        Args:
            pred_plan: 预测规划 {"Q1": ["问题", "<A1>"], ...}
            gold_plan: 黄金规划，格式相同
            threshold: 匹配的相似度阈值
            
        Returns:
            (平均相似度分数, 节点映射) 元组
        """
        if not pred_plan or not gold_plan:
            return 0.0, {}
        
        # 从规划中提取问题
        gold_questions = [v[0] for v in gold_plan.values()]
        pred_questions = [v[0] for v in pred_plan.values()]
        
        gold_nodes = list(gold_plan.keys())
        pred_nodes = list(pred_plan.keys())
        
        # 计算相似度矩阵
        sim_matrix = self.compute_similarity_matrix(pred_questions, gold_questions)
        
        # 找到最优映射
        mapping, similarities = self.find_optimal_mapping(
            sim_matrix, gold_nodes, pred_nodes, threshold
        )
        
        # 按黄金规划大小归一化的平均相似度
        if similarities:
            avg_score = sum(similarities) / len(gold_plan)
        else:
            avg_score = 0.0
            
        return avg_score, mapping


class SubGoalScorer:
    """使用token级F1评估子目标完成度
    
    对于每个匹配的子问题，使用F1指标比较
    预测的中间答案与黄金答案。
    """
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """规范化答案用于比较"""
        import string
        
        text = text.lower()
        # 移除冠词
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # 移除标点
        text = "".join(c for c in text if c not in string.punctuation)
        # 规范化空白
        text = " ".join(text.split())
        
        return text
    
    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        """计算token级F1分数"""
        pred_tokens = set(SubGoalScorer.normalize_answer(pred).split())
        gold_tokens = set(SubGoalScorer.normalize_answer(gold).split())
        
        if not pred_tokens or not gold_tokens:
            return 0.0
        
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
    
    def compute_step_score(
        self,
        pred_graph: Dict[str, Dict],
        gold_graph: Dict[str, Dict],
        mapping: Dict[str, str]
    ) -> float:
        """计算子目标完成度分数
        
        Args:
            pred_graph: 预测执行结果 {"Q1": {"answer": "..."}, ...}
            gold_graph: 黄金执行结果
            mapping: 从黄金到预测的节点映射
            
        Returns:
            匹配子目标的平均F1分数
        """
        if not mapping or not gold_graph:
            return 0.0
        
        scores = []
        for gold_node, pred_node in mapping.items():
            if gold_node not in gold_graph or pred_node not in pred_graph:
                continue
                
            gold_answer = gold_graph[gold_node].get("answer", "")
            pred_answer = pred_graph[pred_node].get("answer", "")
            
            if gold_answer and pred_answer:
                f1 = self.token_f1(pred_answer, gold_answer)
                scores.append(f1)
        
        if not scores:
            return 0.0
            
        return sum(scores) / len(gold_graph)
