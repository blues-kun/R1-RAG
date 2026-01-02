"""
R1-RAG 奖励配置

定义双重奖励机制的超参数:
- 结构奖励权重
- 语义相似度阈值
- 渐进式退火调度
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RewardConfig:
    """DAG奖励计算配置类
    
    奖励信号包含三个部分:
    1. 答案正确性（结果导向）
    2. 规划质量（过程导向，结构）
    3. 子目标完成度（过程导向，语义）
    
    渐进式退火逐渐将焦点从过程转向结果。
    """
    
    # 语义相似度的嵌入模型
    embedding_model_path: str = "intfloat/e5-base-v2"
    
    # 相似度阈值
    node_match_threshold: float = 0.7  # 节点匹配的最小余弦相似度
    
    # 奖励分量权重
    format_weight: float = 0.1      # 格式合规权重
    plan_sim_weight: float = 0.5    # 规划语义相似度权重
    structure_weight: float = 0.5   # DAG结构匹配权重（GED）
    step_weight: float = 0.5        # 子目标完成度权重
    
    # 图编辑距离参数
    ged_beta: float = 1.0           # GED归一化因子
    plan_alpha: float = 0.9         # 语义与结构的平衡系数
    
    # 渐进式退火调度
    annealing_total_steps: int = 50     # 退火总步数
    annealing_center: float = 0.9       # 中心点（总步数的90%）
    annealing_temperature: float = 10.0 # 控制过渡陡峭程度
    
    # 验证模式（仅使用答案分数）
    validation_mode: bool = False
    
    def get_annealing_weight(self, current_step: int) -> float:
        """计算渐进式退火权重
        
        训练早期（权重 ≈ 1）: 关注规划质量
        训练后期（权重 → 0）: 关注答案正确性
        
        公式: w(t) = 1 / (1 + exp((t - center*T) / k))
        """
        import numpy as np
        t = float(current_step)
        T = float(self.annealing_total_steps)
        k = self.annealing_temperature
        center = self.annealing_center
        
        return 1.0 / (1.0 + np.exp((t - center * T) / k))


@dataclass  
class DAGNodeConfig:
    """DAG节点表示配置"""
    
    # 依赖项的占位符模式
    placeholder_pattern: str = r"<A(\d+)>"
    
    # 节点ID前缀
    node_prefix: str = "Q"
    
    # 答案占位符格式
    answer_placeholder: str = "<A{index}>"
