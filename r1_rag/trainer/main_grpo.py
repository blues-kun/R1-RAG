"""
R1-RAG GRPO训练入口

实现带有以下特性的组相对策略优化（GRPO）:
1. 基于DAG的过程监督（结构 + 语义）
2. 渐进式权重退火
3. 多轮搜索增强生成

架构:
    问题 → 规划DAG → 子目标执行 → 答案
              ↓            ↓
         R_structure   R_semantic
              ↓            ↓
         R_total = α(t) * R_process + R_outcome
"""

import os
import torch
import numpy as np
import ray
import hydra
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


@dataclass
class TrainingMetrics:
    """用于日志记录的聚合指标"""
    answer_accuracy: float
    format_compliance: float
    semantic_similarity: float  
    structure_match: float
    step_completion: float
    annealing_weight: float


class DAGRewardComputer:
    """计算R1-RAG训练的DAG奖励
    
    集成多个奖励信号:
    - 格式合规: 输出是否遵循预期结构?
    - 语义相似度: 子问题是否正确?（E5嵌入）
    - 结构匹配: 依赖图是否正确?（GED）
    - 步骤完成: 中间答案是否正确?（F1）
    - 答案正确性: 最终答案是否正确?（EM）
    
    使用渐进式退火来平衡过程和结果奖励。
    """
    
    def __init__(
        self, 
        embedding_model_path: str = "intfloat/e5-base-v2",
        format_weight: float = 0.1,
        semantic_weight: float = 0.5,
        structure_weight: float = 0.5,
        step_weight: float = 0.5,
        node_threshold: float = 0.7,
        ged_beta: float = 1.0,
        annealing_steps: int = 50,
        annealing_center: float = 0.9,
        annealing_temp: float = 10.0,
    ):
        """初始化DAG奖励计算器
        
        Args:
            embedding_model_path: 用于语义评分的E5模型路径
            format_weight: 格式合规奖励权重
            semantic_weight: 语义相似度奖励权重
            structure_weight: 结构匹配奖励权重
            step_weight: 子目标完成奖励权重
            node_threshold: 节点匹配的最小余弦相似度
            ged_beta: GED归一化因子
            annealing_steps: 退火调度的总步数
            annealing_center: sigmoid的中心点（总数的分数）
            annealing_temp: sigmoid陡峭度的温度
        """
        # 存储配置
        self.format_weight = format_weight
        self.semantic_weight = semantic_weight
        self.structure_weight = structure_weight
        self.step_weight = step_weight
        self.node_threshold = node_threshold
        self.ged_beta = ged_beta
        self.annealing_steps = annealing_steps
        self.annealing_center = annealing_center
        self.annealing_temp = annealing_temp
        
        # 延迟初始化评分器以避免导入问题
        self._semantic_scorer = None
        self._structure_scorer = None
        self._embedding_model_path = embedding_model_path
        
    @property
    def semantic_scorer(self):
        """延迟初始化语义评分器"""
        if self._semantic_scorer is None:
            from r1_rag.reward import SemanticScorer
            self._semantic_scorer = SemanticScorer(self._embedding_model_path)
        return self._semantic_scorer
    
    @property
    def structure_scorer(self):
        """延迟初始化结构评分器"""
        if self._structure_scorer is None:
            from r1_rag.reward import StructureScorer
            self._structure_scorer = StructureScorer()
        return self._structure_scorer
    
    def get_annealing_weight(self, step: int) -> float:
        """计算渐进式退火权重
        
        使用sigmoid调度:
            α(t) = 1 / (1 + exp((t - center*T) / k))
        
        训练早期（α ≈ 1）: 关注过程奖励
        训练后期（α → 0）: 关注结果奖励
        
        Args:
            step: 当前训练步数
            
        Returns:
            退火权重，范围[0, 1]
        """
        t = float(step)
        T = float(self.annealing_steps)
        k = self.annealing_temp
        center = self.annealing_center
        
        return 1.0 / (1.0 + np.exp((t - center * T) / k))
    
    def compute(
        self,
        response: str,
        gold_answer: list,
        gold_plan: Optional[Dict] = None,
        gold_graph: Optional[list] = None,
        step: int = 0,
        verbose: bool = False,
    ) -> Dict[str, float]:
        """计算单个响应的综合奖励
        
        Args:
            response: 模型生成的响应文本
            gold_answer: 可接受的黄金答案列表
            gold_plan: 黄金规划DAG（可选）
            gold_graph: 黄金执行图（可选）
            step: 当前训练步数，用于退火
            verbose: 是否打印详细分数
            
        Returns:
            包含分量分数和最终奖励的字典
        """
        from r1_rag.reward import DAGRewardEvaluator, RewardConfig
        
        # 为此次计算创建配置
        config = RewardConfig(
            embedding_model_path=self._embedding_model_path,
            node_match_threshold=self.node_threshold,
            format_weight=self.format_weight,
            plan_sim_weight=self.semantic_weight,
            structure_weight=self.structure_weight,
            step_weight=self.step_weight,
            ged_beta=self.ged_beta,
            annealing_total_steps=self.annealing_steps,
            annealing_center=self.annealing_center,
            annealing_temperature=self.annealing_temp,
        )
        
        evaluator = DAGRewardEvaluator(config)
        
        return evaluator.compute_reward(
            response=response,
            gold_answer=gold_answer,
            gold_plan=gold_plan,
            gold_graph=gold_graph,
            training_step=step,
            verbose=verbose,
        )


class RewardManager:
    """管理veRL训练批次的奖励计算
    
    此类与veRL训练框架接口:
    1. 从token张量解码模型响应
    2. 从批次中提取ground truth和元数据
    3. 计算多维度奖励
    4. 处理验证vs训练模式
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        e5_model_path: str = "intfloat/e5-base-v2",
        format_weight: float = 0.1,
        validation: bool = False,
    ):
        """初始化奖励管理器
        
        Args:
            tokenizer: HuggingFace tokenizer用于解码
            num_examine: 每个数据源打印的样本数
            e5_model_path: E5嵌入模型路径
            format_weight: 格式合规权重
            validation: 如果True，只使用答案分数（无过程奖励）
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.validation = validation
        
        # 初始化DAG奖励计算器
        self.reward_computer = DAGRewardComputer(
            embedding_model_path=e5_model_path,
            format_weight=format_weight,
        )
        
        print(f"[奖励管理器] 已初始化，E5模型: {e5_model_path}")
        print(f"[奖励管理器] 模式: {'验证' if validation else '训练'}")
    
    def __call__(self, data: DataProto, step: int = 0) -> torch.Tensor:
        """计算批量样本的奖励
        
        Args:
            data: 来自veRL的DataProto批次，包含:
                - batch['prompts']: Prompt token IDs
                - batch['responses']: Response token IDs  
                - batch['attention_mask']: 有效token掩码
                - non_tensor_batch['reward_model']: Ground truth信息
                - non_tensor_batch['metadata']: Plan和graph标注
            step: 当前训练步数，用于退火
            
        Returns:
            形状与responses匹配的奖励张量
        """
        # 检查预计算的奖励
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        # 初始化奖励张量
        reward_tensor = torch.zeros_like(
            data.batch['responses'], 
            dtype=torch.float32
        )
        
        # 跟踪每个数据源打印的样本
        examined = {}
        
        for i in range(len(data)):
            item = data[i]
            
            # 获取prompt/response边界
            prompt_ids = item.batch['prompts']
            prompt_len = prompt_ids.shape[-1]
            
            valid_prompt_len = item.batch['attention_mask'][:prompt_len].sum()
            
            response_ids = item.batch['responses']
            valid_response_len = item.batch['attention_mask'][prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]
            
            # 解码响应
            response_text = self.tokenizer.decode(valid_response_ids)
            
            # 获取ground truth
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            gold_answer = ground_truth.get('target', [])
            
            # 获取元数据（plan和graph标注）
            metadata = item.non_tensor_batch.get('metadata', {})
            gold_plan = metadata.get('plan', None)
            gold_graph = metadata.get('graph', None)
            
            # 处理元数据中的numpy数组
            if gold_plan:
                gold_plan = {k: [str(x) for x in v] for k, v in gold_plan.items()}
            if gold_graph and isinstance(gold_graph, np.ndarray):
                gold_graph = gold_graph.tolist()
            
            # 计算奖励
            scores = self.reward_computer.compute(
                response=response_text,
                gold_answer=gold_answer,
                gold_plan=gold_plan,
                gold_graph=gold_graph,
                step=step,
            )
            
            # 将奖励赋给最后一个有效token
            if self.validation:
                # 验证: 只有答案正确性重要
                reward_tensor[i, valid_response_len - 1] = scores['answer_score']
            else:
                # 训练: 使用带退火的完整奖励
                reward_tensor[i, valid_response_len - 1] = scores['final_reward']
            
            # 打印样本示例
            data_source = item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in examined:
                examined[data_source] = 0
            
            if examined[data_source] < self.num_examine:
                examined[data_source] += 1
                self._print_sample(data_source, response_text, scores)
        
        return reward_tensor
    
    def _print_sample(
        self, 
        data_source: str, 
        response: str, 
        scores: Dict[str, float]
    ):
        """打印带分数的样本响应用于调试"""
        print(f"\n{'='*60}")
        print(f"[来自 {data_source} 的样本]")
        print(f"{'='*60}")
        
        # 截断过长响应
        if len(response) > 800:
            response = response[:400] + "\n...[已截断]...\n" + response[-400:]
        print(response)
        
        print(f"\n[分数]")
        print(f"  格式:    {scores.get('format_score', 0):.3f}")
        print(f"  语义:    {scores.get('semantic_score', 0):.3f}")
        print(f"  结构:    {scores.get('structure_score', 0):.3f}")
        print(f"  步骤:    {scores.get('step_score', 0):.3f}")
        print(f"  答案:    {scores.get('answer_score', 0):.3f}")
        print(f"  最终:    {scores.get('final_reward', 0):.3f}")


class GRPOTrainer:
    """R1-RAG GRPO训练的高级训练器类
    
    包装veRL的RayPPOTrainer，带有R1-RAG特定的:
    - 基于DAG的奖励函数
    - 多轮搜索生成
    - 渐进式退火调度
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        e5_model_path: str = "intfloat/e5-base-v2",
    ):
        """初始化GRPO训练器
        
        Args:
            config: Hydra/OmegaConf配置
            tokenizer: HuggingFace tokenizer
            e5_model_path: E5嵌入模型路径
        """
        self.config = config
        self.tokenizer = tokenizer
        self.e5_model_path = e5_model_path
        
        # 创建奖励函数
        self.reward_fn = RewardManager(
            tokenizer=tokenizer,
            num_examine=0,
            e5_model_path=e5_model_path,
            validation=False,
        )
        
        self.val_reward_fn = RewardManager(
            tokenizer=tokenizer,
            num_examine=1,
            e5_model_path=e5_model_path,
            validation=True,
        )
        
        self.trainer = None
    
    def setup_workers(self):
        """设置Ray worker用于分布式训练"""
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        # 根据策略选择worker类
        if self.config.actor_rollout_ref.actor.strategy == 'fsdp':
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
            
        elif self.config.actor_rollout_ref.actor.strategy == 'megatron':
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise ValueError(f"未知策略: {self.config.actor_rollout_ref.actor.strategy}")
        
        # 设置角色-worker映射
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }
        
        # 资源池配置
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        # 处理可选的奖励模型
        if self.config.reward_model.enable:
            if self.config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif self.config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # 创建训练器
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=mapping
        )
        
        self.trainer = RayPPOTrainer(
            config=self.config,
            tokenizer=self.tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=self.reward_fn,
            val_reward_fn=self.val_reward_fn,
        )
        
        return self.trainer
    
    def train(self):
        """运行训练循环"""
        if self.trainer is None:
            self.setup_workers()
        
        self.trainer.init_workers()
        self.trainer.fit()


# ============== 入口点 ==============

@hydra.main(config_path='../../../configs', config_name='grpo_qwen_3b', version_base=None)
def main(config):
    """GRPO训练的主入口"""
    if not ray.is_initialized():
        ray.init(
            runtime_env={
                'env_vars': {
                    'TOKENIZERS_PARALLELISM': 'true',
                    'NCCL_DEBUG': 'WARN'
                }
            }
        )
    
    ray.get(train_task.remote(config))


@ray.remote
def train_task(config):
    """Ray远程训练任务"""
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    
    # 打印配置
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # 如需要下载模型检查点
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    
    # 初始化tokenizer
    tokenizer = hf_tokenizer(local_path)
    
    # 获取E5模型路径
    e5_model_path = getattr(config, 'e5_model_path', 'intfloat/e5-base-v2')
    
    # 创建并运行训练器
    trainer = GRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        e5_model_path=e5_model_path,
    )
    trainer.train()


def test_reward_computation():
    """使用样本数据测试奖励计算"""
    from r1_rag.reward import DAGRewardEvaluator, RewardConfig
    
    config = RewardConfig()
    evaluator = DAGRewardEvaluator(config)
    
    # 样本响应
    response = """
<think> 这是一个多跳问题，我需要分解它。 </think>
<plan>
{"Q1": ["谁执导了《泰坦尼克号》？", "#1"], "Q2": ["#1的第一部电影是什么？", "#2"]}
</plan>

<subPlan>
    <think> 让我搜索导演信息。 </think>
    <search> 泰坦尼克号 导演 </search>
    <information> 《泰坦尼克号》由詹姆斯·卡梅隆执导。 </information>
    <think> 找到了 - 詹姆斯·卡梅隆。 </think>
    <subAnswer> #1 = 詹姆斯·卡梅隆 </subAnswer>
</subPlan>

<subPlan>
    <think> 现在我需要找詹姆斯·卡梅隆导演的第一部电影。 </think>
    <search> 詹姆斯·卡梅隆 第一部电影 </search>
    <information> 詹姆斯·卡梅隆的导演处女作是《食人鱼2》。 </information>
    <think> 第一部电影是《食人鱼2》。 </think>
    <subAnswer> #2 = 食人鱼2 </subAnswer>
</subPlan>

<think> 现在我有答案了。 </think>
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
        response=response,
        gold_answer=gold_answer,
        gold_plan=gold_plan,
        gold_graph=gold_graph,
        training_step=25,
        verbose=True,
    )
    
    print("\n" + "="*50)
    print("测试结果:")
    print(f"  格式分数: {scores['format_score']:.3f}")
    print(f"  语义分数: {scores['semantic_score']:.3f}")
    print(f"  结构分数: {scores['structure_score']:.3f}")
    print(f"  步骤分数: {scores['step_score']:.3f}")
    print(f"  答案分数: {scores['answer_score']:.3f}")
    print(f"  最终奖励: {scores['final_reward']:.3f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_reward_computation()
    else:
        main()
