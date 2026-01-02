"""
DAG Reward Evaluator for R1-RAG

The core reward computation module that integrates:
1. Semantic similarity (E5 embedding)
2. Structural similarity (Graph Edit Distance)
3. Sub-goal completion (F1 score)
4. Answer correctness (Exact Match)

Combined with progressive weight annealing for balanced training.
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
    """Main reward evaluator for R1-RAG training.
    
    Computes multi-dimensional rewards that guide the model to:
    1. Generate well-structured reasoning plans (structure reward)
    2. Ask semantically correct sub-questions (semantic reward)
    3. Execute sub-goals correctly (step reward)
    4. Arrive at correct final answer (outcome reward)
    
    The progressive annealing strategy ensures:
    - Early training: Focus on learning good planning habits
    - Late training: Focus on getting correct answers
    """
    
    def __init__(self, config: RewardConfig):
        """Initialize the DAG reward evaluator.
        
        Args:
            config: Reward configuration with weights and thresholds
        """
        self.config = config
        self.semantic_scorer = SemanticScorer(config.embedding_model_path)
        self.structure_scorer = StructureScorer()
        self.subgoal_scorer = SubGoalScorer()
        
        # Regex patterns for parsing model output
        self.plan_pattern = re.compile(r'<plan>(.+?)</plan>', re.DOTALL)
        self.subplan_pattern = re.compile(r'<subPlan>(.+?)</subPlan>', re.DOTALL)
        self.subanswer_pattern = re.compile(r'<subAnswer>\s*#(\d+)\s*=\s*(.*?)\s*</subAnswer>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
    # ==================== Answer Extraction ====================
    
    def extract_final_answer(self, response: str) -> Optional[str]:
        """Extract final answer from model response.
        
        Args:
            response: Full model response text
            
        Returns:
            Extracted answer string or None if not found
        """
        matches = list(self.answer_pattern.finditer(response))
        if not matches:
            return None
        # Return last match (in case of multiple attempts)
        return matches[-1].group(1).strip()
    
    def extract_plan(self, response: str) -> Optional[Dict[str, List[str]]]:
        """Extract planning DAG from model response.
        
        Expected format in response:
        <plan>
        {"Q1": ["sub-question 1", "#1"], "Q2": ["sub-question 2 using #1", "#2"]}
        </plan>
        
        Args:
            response: Full model response
            
        Returns:
            Parsed plan dictionary or None if parsing fails
        """
        match = self.plan_pattern.search(response)
        if not match:
            return None
            
        try:
            plan_text = match.group(1).strip()
            # Clean up common formatting issues
            plan_text = plan_text.replace("json", "").replace("```", "")
            plan = json.loads(plan_text)
            
            # Normalize placeholders: #1 -> <A1>
            normalized = {}
            for key, value in plan.items():
                new_value = []
                for item in value:
                    # Replace #N with <AN>
                    normalized_item = re.sub(r"#(\d+)", r"<A\1>", str(item))
                    new_value.append(normalized_item)
                normalized[key] = new_value
            
            return normalized
        except (json.JSONDecodeError, Exception) as e:
            return None
    
    def extract_execution_graph(self, response: str) -> Dict[str, Dict]:
        """Extract execution results from subPlan blocks.
        
        Parses the intermediate answers from each subPlan to build
        the execution graph showing what answer was found for each sub-question.
        
        Args:
            response: Full model response
            
        Returns:
            Execution graph {"Q1": {"answer": "..."}, "Q2": {...}}
        """
        graph = {}
        
        subplans = self.subplan_pattern.findall(response)
        for subplan in subplans:
            # Extract <subAnswer> #N = answer </subAnswer>
            match = self.subanswer_pattern.search(subplan)
            if match:
                q_idx = match.group(1)
                answer = match.group(2).strip()
                graph[f"Q{q_idx}"] = {"answer": answer}
        
        return graph
    
    # ==================== Answer Scoring ====================
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer text for exact match comparison."""
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = "".join(c for c in text if c not in string.punctuation)
        text = " ".join(text.split())
        return text
    
    def exact_match(self, prediction: str, gold_answers: List[str]) -> float:
        """Check if prediction exactly matches any gold answer.
        
        Args:
            prediction: Predicted answer
            gold_answers: List of acceptable gold answers
            
        Returns:
            1.0 if match, 0.0 otherwise
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
    
    # ==================== Format Checking ====================
    
    def check_format(self, response: str) -> float:
        """Check if response follows required format.
        
        Required structure:
        1. <think>...</think> before <plan>
        2. <plan>...</plan> block with valid JSON
        3. One or more <subPlan>...</subPlan> blocks
        4. Each subPlan has <search>, <information>, <subAnswer>
        5. Final <think>...</think> before <answer>
        
        Args:
            response: Model response to check
            
        Returns:
            1.0 if valid format, 0.0 otherwise
        """
        errors = []
        
        # Check plan block with preceding think
        if not re.search(r"<think>.*?</think>\s*<plan>.*?</plan>", response, re.DOTALL):
            errors.append("Missing <plan> block or preceding <think>")
        
        # Check answer block with preceding think
        if not re.search(r"<think>.*?</think>\s*<answer>.*?</answer>", response, re.DOTALL):
            errors.append("Missing <answer> block or preceding <think>")
        
        # Check subPlan blocks
        subplans = re.findall(r"<subPlan>.*?</subPlan>", response, re.DOTALL)
        if not subplans:
            errors.append("No <subPlan> blocks found")
        
        # Validate each subPlan structure
        for i, sp in enumerate(subplans, 1):
            if not re.search(r"<think>.*?</think>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: missing <think>")
            if not re.search(r"<search>.*?</search>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: missing <search>")
            if not re.search(r"<information>.*?</information>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: missing <information>")
            if not re.search(r"<subAnswer>.*?</subAnswer>", sp, re.DOTALL):
                errors.append(f"subPlan {i}: missing <subAnswer>")
        
        return 1.0 if not errors else 0.0
    
    # ==================== Main Reward Computation ====================
    
    def compute_reward(
        self,
        response: str,
        gold_answer: List[str],
        gold_plan: Optional[Dict[str, List[str]]] = None,
        gold_graph: Optional[List[Dict]] = None,
        training_step: int = 0,
        verbose: bool = False
    ) -> Dict[str, float]:
        """Compute comprehensive reward for model response.
        
        Combines multiple reward signals with progressive annealing:
        
        R_total = α(t) * R_process + R_outcome
        
        where:
        - α(t) decreases over training (annealing)
        - R_process = w_f * format + w_s * semantic + w_g * structure + w_step * step
        - R_outcome = answer_score (exact match)
        
        Args:
            response: Model response text
            gold_answer: List of acceptable gold answers
            gold_plan: Golden planning DAG (optional)
            gold_graph: Golden execution graph (optional)
            training_step: Current training step for annealing
            verbose: Whether to print detailed scores
            
        Returns:
            Dictionary containing all component scores and final reward
        """
        # Initialize scores
        scores = {
            "format_score": 0.0,
            "semantic_score": 0.0,
            "structure_score": 0.0,
            "step_score": 0.0,
            "answer_score": 0.0,
            "final_reward": 0.0
        }
        
        # 1. Format compliance
        scores["format_score"] = self.check_format(response)
        
        # 2. Answer correctness (outcome reward)
        pred_answer = self.extract_final_answer(response)
        scores["answer_score"] = self.exact_match(pred_answer, gold_answer)
        
        # 3. Process rewards (only if golden annotations available)
        if gold_plan and gold_graph:
            # Extract predicted plan and execution
            pred_plan = self.extract_plan(response)
            pred_graph = self.extract_execution_graph(response)
            
            if pred_plan:
                # Semantic similarity
                semantic_score, mapping = self.semantic_scorer.compute_semantic_score(
                    pred_plan, gold_plan, threshold=self.config.node_match_threshold
                )
                scores["semantic_score"] = semantic_score
                
                # Structural similarity
                ged, norm_ged, struct_match = self.structure_scorer.compute_structure_score(
                    pred_plan, gold_plan, mapping,
                    alpha=self.config.plan_alpha,
                    beta=self.config.ged_beta
                )
                scores["structure_score"] = struct_match
                scores["ged_raw"] = ged
                scores["ged_normalized"] = norm_ged
                
                # Sub-goal completion
                if pred_graph and gold_graph:
                    gold_graph_dict = gold_graph[0] if isinstance(gold_graph, list) else gold_graph
                    scores["step_score"] = self.subgoal_scorer.compute_step_score(
                        pred_graph, gold_graph_dict, mapping
                    )
        
        # 4. Compute final reward with annealing
        if self.config.validation_mode:
            # Validation: only answer score matters
            scores["final_reward"] = scores["answer_score"]
        else:
            # Training: combine process and outcome rewards
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
        
        # Verbose logging
        if verbose or random.random() < 0.02:  # Log ~2% of samples
            print(f"\n{'='*50}")
            print(f"[DAG Evaluator] Training Step: {training_step}")
            print(f"  Format:    {scores['format_score']:.3f}")
            print(f"  Semantic:  {scores['semantic_score']:.3f}")
            print(f"  Structure: {scores['structure_score']:.3f}")
            print(f"  Step:      {scores['step_score']:.3f}")
            print(f"  Answer:    {scores['answer_score']:.3f}")
            print(f"  Annealing: {scores.get('annealing_weight', 1.0):.3f}")
            print(f"  Final:     {scores['final_reward']:.3f}")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {pred_answer}")
        
        return scores


class RewardManager:
    """Manages reward computation for batch training.
    
    Integrates with veRL training framework to provide
    reward signals during GRPO optimization.
    """
    
    def __init__(
        self,
        tokenizer,
        config: RewardConfig,
        num_examine: int = 1
    ):
        """Initialize reward manager.
        
        Args:
            tokenizer: HuggingFace tokenizer for decoding
            config: Reward configuration
            num_examine: Number of samples to print per data source
        """
        self.tokenizer = tokenizer
        self.evaluator = DAGRewardEvaluator(config)
        self.config = config
        self.num_examine = num_examine
        
    def __call__(self, data, training_step: int = 0):
        """Compute rewards for a batch of samples.
        
        Args:
            data: DataProto batch from veRL
            training_step: Current training step
            
        Returns:
            Reward tensor with shape matching responses
        """
        import torch
        
        reward_tensor = torch.zeros_like(
            data.batch['responses'], dtype=torch.float32
        )
        
        examined = {}
        
        for i in range(len(data)):
            item = data[i]
            
            # Get prompt and response lengths
            prompt_ids = item.batch['prompts']
            prompt_len = prompt_ids.shape[-1]
            
            valid_prompt_len = item.batch['attention_mask'][:prompt_len].sum()
            
            response_ids = item.batch['responses']
            valid_response_len = item.batch['attention_mask'][prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]
            
            # Decode response
            response_text = self.tokenizer.decode(valid_response_ids)
            
            # Get ground truth and metadata
            ground_truth = item.non_tensor_batch['reward_model']['ground_truth']
            gold_answer = ground_truth.get('target', [])
            
            metadata = item.non_tensor_batch.get('metadata', {})
            gold_plan = metadata.get('plan', None)
            gold_graph = metadata.get('graph', None)
            
            # Fix numpy array issues
            if gold_plan:
                gold_plan = {k: [str(x) for x in v] for k, v in gold_plan.items()}
            if gold_graph and isinstance(gold_graph, np.ndarray):
                gold_graph = gold_graph.tolist()
            
            # Compute reward
            scores = self.evaluator.compute_reward(
                response=response_text,
                gold_answer=gold_answer,
                gold_plan=gold_plan,
                gold_graph=gold_graph,
                training_step=training_step
            )
            
            # Assign reward to last token
            reward_tensor[i, valid_response_len - 1] = scores['final_reward']
            
            # Print examples
            data_source = item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in examined:
                examined[data_source] = 0
            if examined[data_source] < self.num_examine:
                examined[data_source] += 1
                print(f"\n[Sample from {data_source}]")
                print(response_text[:500] + "..." if len(response_text) > 500 else response_text)
        
        return reward_tensor

