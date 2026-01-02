"""
R1-RAG 结构评分器

使用图编辑距离（GED）测量预测和黄金规划DAG之间的结构相似度。

为多跳推理中正确的依赖建模提供"结构奖励"信号。
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher


class StructureScorer:
    """计算规划DAG之间的结构相似度
    
    核心洞察: 多跳推理需要正确的依赖结构：
    Q2应该依赖于Q1的答案，等等。
    
    GED捕捉:
    - 缺失/多余的节点（错误的子问题数量）
    - 缺失/多余的边（错误的依赖关系）
    - 不正确的依赖顺序
    """
    
    def __init__(self, placeholder_pattern: str = r"<A(\d+)>"):
        """初始化结构评分器
        
        Args:
            placeholder_pattern: 答案占位符的正则表达式模式
        """
        self.placeholder_pattern = re.compile(placeholder_pattern)
        
    def plan_to_dag(self, plan: Dict[str, List[str]]) -> nx.DiGraph:
        """将规划字典转换为有向无环图
        
        规划格式: {"Q1": ["问题文本", "<A1>"], "Q2": ["包含<A1>的文本", "<A2>"]}
        
        创建的DAG中:
        - 每个节点是一个子问题（Q1, Q2, ...）
        - 边表示依赖关系（如果Q2使用<A1>，则Q1 → Q2）
        
        Args:
            plan: 规划字典
            
        Returns:
            表示规划结构的NetworkX有向图
        """
        dag = nx.DiGraph()
        
        for node_id, (question, _) in plan.items():
            # 添加节点，问题作为属性
            dag.add_node(node_id, template=question)
            
            # 从问题中的占位符找到依赖关系
            placeholders = self.placeholder_pattern.findall(question)
            for placeholder_idx in placeholders:
                parent_node = f"Q{placeholder_idx}"
                if parent_node in plan:
                    dag.add_edge(parent_node, node_id)
        
        return dag
    
    def compute_ged(
        self,
        pred_dag: nx.DiGraph,
        gold_dag: nx.DiGraph,
        mapping: Dict[str, str],
        beta: float = 1.0
    ) -> Tuple[int, float]:
        """计算DAG之间的图编辑距离
        
        GED = |节点差异| + |边差异|
        
        使用提供的节点映射在计算差异前对齐图。
        
        Args:
            pred_dag: 预测的规划DAG
            gold_dag: 黄金规划DAG
            mapping: 从黄金到预测节点的映射
            beta: 指数衰减的归一化因子
            
        Returns:
            (原始GED, 归一化的GED分数[0,1]) 元组
        """
        # 基于映射重新标记节点
        gold_relabel = {}
        pred_relabel = {}
        common_idx = 1
        
        # 匹配的节点获得共同标签
        for gold_node, pred_node in mapping.items():
            common_label = f"C{common_idx}"
            gold_relabel[gold_node] = common_label
            pred_relabel[pred_node] = common_label
            common_idx += 1
        
        # 未匹配的节点获得唯一标签
        for idx, node in enumerate(gold_dag.nodes()):
            if node not in gold_relabel:
                gold_relabel[node] = f"Gg{idx}"
                
        for idx, node in enumerate(pred_dag.nodes()):
            if node not in pred_relabel:
                pred_relabel[node] = f"Gp{idx}"
        
        # 应用重新标记
        gold_relabeled = nx.relabel_nodes(gold_dag, gold_relabel)
        pred_relabeled = nx.relabel_nodes(pred_dag, pred_relabel)
        
        # 计算对称差异
        node_union = set(gold_relabeled.nodes()) | set(pred_relabeled.nodes())
        node_intersection = set(gold_relabeled.nodes()) & set(pred_relabeled.nodes())
        
        edge_union = set(gold_relabeled.edges()) | set(pred_relabeled.edges())
        edge_intersection = set(gold_relabeled.edges()) & set(pred_relabeled.edges())
        
        # GED = 节点差异 + 边差异
        ged = len(node_union - node_intersection) + len(edge_union - edge_intersection)
        
        # 使用指数衰减归一化
        normalized_ged = np.exp(-beta * ged)
        
        return ged, round(normalized_ged, 6)
    
    def compute_structure_score(
        self,
        pred_plan: Dict[str, List[str]],
        gold_plan: Dict[str, List[str]],
        mapping: Dict[str, str],
        alpha: float = 0.9,
        beta: float = 1.0
    ) -> Tuple[int, float, float]:
        """计算整体结构相似度分数
        
        组合:
        - 原始GED（用于可解释性）
        - 归一化GED分数（用于奖励）
        - 二值结构匹配（GED == 0）
        
        Args:
            pred_plan: 预测规划
            gold_plan: 黄金规划
            mapping: 来自语义评分器的节点映射
            alpha: 语义与结构分量的权重
            beta: GED归一化因子
            
        Returns:
            (原始GED, 归一化分数, 二值匹配指示器) 元组
        """
        if not pred_plan or not gold_plan:
            return 0, 0.0, 0.0
        
        pred_dag = self.plan_to_dag(pred_plan)
        gold_dag = self.plan_to_dag(gold_plan)
        
        ged, normalized_ged = self.compute_ged(pred_dag, gold_dag, mapping, beta)
        
        # 二值指示器: 结构完全匹配（或接近）时为1
        structure_match = 1.0 if ged < 1 else 0.0
        
        return ged, normalized_ged, structure_match
    
    def find_subgraph_isomorphism(
        self,
        pred_dag: nx.DiGraph,
        gold_dag: nx.DiGraph
    ) -> List[str]:
        """在预测和黄金DAG之间找到同构子图
        
        当预测规划捕捉到黄金推理结构的子集时，
        用于给予部分分数。
        
        Args:
            pred_dag: 预测DAG
            gold_dag: 黄金DAG（要匹配的模式）
            
        Returns:
            pred_dag中形成同构子图的节点列表
        """
        matcher = DiGraphMatcher(pred_dag, gold_dag)
        matched_nodes = []
        
        for mapping in matcher.subgraph_isomorphisms_iter():
            matched_nodes.extend(mapping.keys())
        
        return list(set(matched_nodes))
