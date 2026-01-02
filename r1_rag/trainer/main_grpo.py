"""
R1-RAG GRPO Training Entry Point

Implements Group Relative Policy Optimization (GRPO) with:
1. DAG-based process supervision (structure + semantic)
2. Progressive weight annealing
3. Multi-turn search-augmented generation

Architecture:
    Question → Planning DAG → Sub-Goal Execution → Answer
                   ↓                ↓
              R_structure      R_semantic
                   ↓                ↓
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
    """Aggregated metrics for logging."""
    answer_accuracy: float
    format_compliance: float
    semantic_similarity: float  
    structure_match: float
    step_completion: float
    annealing_weight: float


class DAGRewardComputer:
    """Computes DAG-based rewards for R1-RAG training.
    
    Integrates multiple reward signals:
    - Format compliance: Does output follow expected structure?
    - Semantic similarity: Are sub-questions correct? (E5 embedding)
    - Structure match: Is dependency graph correct? (GED)
    - Step completion: Are intermediate answers correct? (F1)
    - Answer correctness: Is final answer correct? (EM)
    
    Uses progressive annealing to balance process vs outcome rewards.
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
        """Initialize DAG reward computer.
        
        Args:
            embedding_model_path: Path to E5 model for semantic scoring
            format_weight: Weight for format compliance reward
            semantic_weight: Weight for semantic similarity reward
            structure_weight: Weight for structural match reward  
            step_weight: Weight for sub-goal completion reward
            node_threshold: Min cosine similarity for node matching
            ged_beta: GED normalization factor
            annealing_steps: Total steps for annealing schedule
            annealing_center: Center point of sigmoid (fraction of total)
            annealing_temp: Temperature for sigmoid steepness
        """
        # Store config
        self.format_weight = format_weight
        self.semantic_weight = semantic_weight
        self.structure_weight = structure_weight
        self.step_weight = step_weight
        self.node_threshold = node_threshold
        self.ged_beta = ged_beta
        self.annealing_steps = annealing_steps
        self.annealing_center = annealing_center
        self.annealing_temp = annealing_temp
        
        # Initialize scorers lazily to avoid import issues
        self._semantic_scorer = None
        self._structure_scorer = None
        self._embedding_model_path = embedding_model_path
        
    @property
    def semantic_scorer(self):
        """Lazy initialization of semantic scorer."""
        if self._semantic_scorer is None:
            from r1_rag.reward import SemanticScorer
            self._semantic_scorer = SemanticScorer(self._embedding_model_path)
        return self._semantic_scorer
    
    @property
    def structure_scorer(self):
        """Lazy initialization of structure scorer."""
        if self._structure_scorer is None:
            from r1_rag.reward import StructureScorer
            self._structure_scorer = StructureScorer()
        return self._structure_scorer
    
    def get_annealing_weight(self, step: int) -> float:
        """Calculate progressive annealing weight.
        
        Uses sigmoid schedule:
            α(t) = 1 / (1 + exp((t - center*T) / k))
        
        Early training (α ≈ 1): Focus on process rewards
        Late training (α → 0): Focus on outcome rewards
        
        Args:
            step: Current training step
            
        Returns:
            Annealing weight in [0, 1]
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
        """Compute comprehensive reward for a single response.
        
        Args:
            response: Model-generated response text
            gold_answer: List of acceptable gold answers
            gold_plan: Golden planning DAG (optional)
            gold_graph: Golden execution graph (optional)
            step: Current training step for annealing
            verbose: Whether to print detailed scores
            
        Returns:
            Dictionary with component scores and final reward
        """
        from r1_rag.reward import DAGRewardEvaluator, RewardConfig
        
        # Create config for this computation
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
    """Manages reward computation for veRL training batches.
    
    This class interfaces with the veRL training framework to:
    1. Decode model responses from token tensors
    2. Extract ground truth and metadata from batch
    3. Compute multi-dimensional rewards
    4. Handle validation vs training modes
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        e5_model_path: str = "intfloat/e5-base-v2",
        format_weight: float = 0.1,
        validation: bool = False,
    ):
        """Initialize reward manager.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding
            num_examine: Number of samples to print per data source
            e5_model_path: Path to E5 embedding model
            format_weight: Weight for format compliance
            validation: If True, only use answer score (no process rewards)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.validation = validation
        
        # Initialize DAG reward computer
        self.reward_computer = DAGRewardComputer(
            embedding_model_path=e5_model_path,
            format_weight=format_weight,
        )
        
        print(f"[RewardManager] Initialized with E5 model: {e5_model_path}")
        print(f"[RewardManager] Mode: {'Validation' if validation else 'Training'}")
    
    def __call__(self, data: DataProto, step: int = 0) -> torch.Tensor:
        """Compute rewards for a batch of samples.
        
        Args:
            data: DataProto batch from veRL containing:
                - batch['prompts']: Prompt token IDs
                - batch['responses']: Response token IDs  
                - batch['attention_mask']: Valid token mask
                - non_tensor_batch['reward_model']: Ground truth info
                - non_tensor_batch['metadata']: Plan and graph annotations
            step: Current training step for annealing
            
        Returns:
            Reward tensor with shape matching responses
        """
        # Check for pre-computed rewards
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']
        
        # Initialize reward tensor
        reward_tensor = torch.zeros_like(
            data.batch['responses'], 
            dtype=torch.float32
        )
        
        # Track printed samples per data source
        examined = {}
        
        for i in range(len(data)):
            item = data[i]
            
            # Get prompt/response boundaries
            prompt_ids = item.batch['prompts']
            prompt_len = prompt_ids.shape[-1]
            
            valid_prompt_len = item.batch['attention_mask'][:prompt_len].sum()
            
            response_ids = item.batch['responses']
            valid_response_len = item.batch['attention_mask'][prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]
            
            # Decode response
            response_text = self.tokenizer.decode(valid_response_ids)
            
            # Get ground truth
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            gold_answer = ground_truth.get('target', [])
            
            # Get metadata (plan and graph annotations)
            metadata = item.non_tensor_batch.get('metadata', {})
            gold_plan = metadata.get('plan', None)
            gold_graph = metadata.get('graph', None)
            
            # Handle numpy arrays in metadata
            if gold_plan:
                gold_plan = {k: [str(x) for x in v] for k, v in gold_plan.items()}
            if gold_graph and isinstance(gold_graph, np.ndarray):
                gold_graph = gold_graph.tolist()
            
            # Compute reward
            scores = self.reward_computer.compute(
                response=response_text,
                gold_answer=gold_answer,
                gold_plan=gold_plan,
                gold_graph=gold_graph,
                step=step,
            )
            
            # Assign reward to last valid token
            if self.validation:
                # Validation: only answer correctness matters
                reward_tensor[i, valid_response_len - 1] = scores['answer_score']
            else:
                # Training: use full reward with annealing
                reward_tensor[i, valid_response_len - 1] = scores['final_reward']
            
            # Print sample examples
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
        """Print a sample response with scores for debugging."""
        print(f"\n{'='*60}")
        print(f"[Sample from {data_source}]")
        print(f"{'='*60}")
        
        # Truncate long responses
        if len(response) > 800:
            response = response[:400] + "\n...[truncated]...\n" + response[-400:]
        print(response)
        
        print(f"\n[Scores]")
        print(f"  Format:    {scores.get('format_score', 0):.3f}")
        print(f"  Semantic:  {scores.get('semantic_score', 0):.3f}")
        print(f"  Structure: {scores.get('structure_score', 0):.3f}")
        print(f"  Step:      {scores.get('step_score', 0):.3f}")
        print(f"  Answer:    {scores.get('answer_score', 0):.3f}")
        print(f"  Final:     {scores.get('final_reward', 0):.3f}")


class GRPOTrainer:
    """High-level trainer class for R1-RAG GRPO training.
    
    Wraps veRL's RayPPOTrainer with R1-RAG specific:
    - DAG-based reward functions
    - Multi-turn generation with search
    - Progressive annealing schedule
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        e5_model_path: str = "intfloat/e5-base-v2",
    ):
        """Initialize GRPO trainer.
        
        Args:
            config: Hydra/OmegaConf configuration
            tokenizer: HuggingFace tokenizer
            e5_model_path: Path to E5 embedding model
        """
        self.config = config
        self.tokenizer = tokenizer
        self.e5_model_path = e5_model_path
        
        # Create reward functions
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
        """Setup Ray workers for distributed training."""
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        
        # Select worker classes based on strategy
        if self.config.actor_rollout_ref.actor.strategy == 'fsdp':
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray import RayWorkerGroup
            ray_worker_group_cls = RayWorkerGroup
            
        elif self.config.actor_rollout_ref.actor.strategy == 'megatron':
            from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise ValueError(f"Unknown strategy: {self.config.actor_rollout_ref.actor.strategy}")
        
        # Setup role-worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }
        
        # Resource pool configuration
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [self.config.trainer.n_gpus_per_node] * self.config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }
        
        # Handle optional reward model
        if self.config.reward_model.enable:
            if self.config.reward_model.strategy == 'fsdp':
                from verl.workers.fsdp_workers import RewardModelWorker
            elif self.config.reward_model.strategy == 'megatron':
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id
        
        # Create trainer
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
        """Run training loop."""
        if self.trainer is None:
            self.setup_workers()
        
        self.trainer.init_workers()
        self.trainer.fit()


# ============== Entry Points ==============

@hydra.main(config_path='../../../configs', config_name='grpo_qwen_3b', version_base=None)
def main(config):
    """Main entry point for GRPO training."""
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
    """Ray remote task for training."""
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.utils import hf_tokenizer
    
    # Print configuration
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    # Download model checkpoint if needed
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)
    
    # Initialize tokenizer
    tokenizer = hf_tokenizer(local_path)
    
    # Get E5 model path
    e5_model_path = getattr(config, 'e5_model_path', 'intfloat/e5-base-v2')
    
    # Create and run trainer
    trainer = GRPOTrainer(
        config=config,
        tokenizer=tokenizer,
        e5_model_path=e5_model_path,
    )
    trainer.train()


def test_reward_computation():
    """Test reward computation with sample data."""
    from r1_rag.reward import DAGRewardEvaluator, RewardConfig
    
    config = RewardConfig()
    evaluator = DAGRewardEvaluator(config)
    
    # Sample response
    response = """
<think> This is a multi-hop question. I need to decompose it. </think>
<plan>
{"Q1": ["Who created the Money in the Bank match?", "#1"], "Q2": ["Which film was directed by #1?", "#2"]}
</plan>

<subPlan>
    <think> Let me search for the creator. </think>
    <search> Money in the Bank ladder match creator </search>
    <information> The Money in the Bank ladder match was created by Chris Jericho. </information>
    <think> Found it - Chris Jericho. </think>
    <subAnswer> #1 = Chris Jericho </subAnswer>
</subPlan>

<subPlan>
    <think> Now I need to find what film Chris Jericho directed. </think>
    <search> Chris Jericho directed film </search>
    <information> Chris Jericho directed "But I'm Chris Jericho!". </information>
    <think> The film is "But I'm Chris Jericho!". </think>
    <subAnswer> #2 = But I'm Chris Jericho! </subAnswer>
</subPlan>

<think> I have the answer now. </think>
<answer> But I'm Chris Jericho! </answer>
    """.strip()
    
    gold_answer = ["But I'm Chris Jericho!"]
    gold_plan = {
        "Q1": ["Who created the Money in the Bank ladder match?", "<A1>"],
        "Q2": ["Which film was directed by <A1>?", "<A2>"],
    }
    gold_graph = [{
        "Q1": {"answer": "Chris Jericho"},
        "Q2": {"answer": "But I'm Chris Jericho!"},
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
    print("Test Results:")
    print(f"  Format Score: {scores['format_score']:.3f}")
    print(f"  Semantic Score: {scores['semantic_score']:.3f}")
    print(f"  Structure Score: {scores['structure_score']:.3f}")
    print(f"  Step Score: {scores['step_score']:.3f}")
    print(f"  Answer Score: {scores['answer_score']:.3f}")
    print(f"  Final Reward: {scores['final_reward']:.3f}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_reward_computation()
    else:
        main()

