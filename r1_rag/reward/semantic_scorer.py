"""
Semantic Scorer for R1-RAG

Uses E5 embedding model to compute semantic similarity between:
- Predicted sub-questions and golden sub-questions
- Predicted intermediate answers and golden answers

This provides the "content reward" signal for RL training.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
from sentence_transformers import SentenceTransformer, util


class SemanticScorer:
    """Computes semantic similarity between planning DAGs using embeddings.
    
    The scorer uses a pre-trained E5 model to:
    1. Encode sub-questions into dense vectors
    2. Compute pairwise cosine similarities
    3. Find optimal node mappings between predicted and golden plans
    
    Key insight: Semantic similarity captures whether the model asks
    the "right questions" even if phrasing differs from golden plan.
    """
    
    def __init__(self, model_path: str = "intfloat/e5-base-v2"):
        """Initialize semantic scorer with E5 model.
        
        Args:
            model_path: Path to sentence transformer model
        """
        self.model = SentenceTransformer(model_path)
        self.placeholder_pattern = re.compile(r"<A(\d+)>")
        
    def encode_questions(self, questions: List[str]) -> np.ndarray:
        """Encode questions into dense embeddings.
        
        Args:
            questions: List of question strings
            
        Returns:
            Embedding matrix of shape (N, embedding_dim)
        """
        # Normalize placeholders to generic "entity" for better matching
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
        """Compute pairwise cosine similarity between question sets.
        
        Args:
            pred_questions: Predicted sub-questions
            gold_questions: Golden sub-questions
            
        Returns:
            Similarity matrix of shape (len(gold), len(pred))
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
        """Find optimal node mapping using greedy matching.
        
        Uses cosine similarity to match predicted nodes to golden nodes.
        Only matches above threshold are considered valid.
        
        Args:
            similarity_matrix: Pairwise similarities (gold x pred)
            gold_nodes: Golden node IDs
            pred_nodes: Predicted node IDs  
            threshold: Minimum similarity for valid match
            
        Returns:
            Tuple of (mapping dict, similarity scores for matched nodes)
        """
        mapping = {}
        similarities = []
        used_pred = set()
        
        # Greedy matching: iterate through gold nodes
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
        """Compute overall semantic similarity score between plans.
        
        Args:
            pred_plan: Predicted plan {"Q1": ["question", "<A1>"], ...}
            gold_plan: Golden plan in same format
            threshold: Similarity threshold for matching
            
        Returns:
            Tuple of (average similarity score, node mapping)
        """
        if not pred_plan or not gold_plan:
            return 0.0, {}
        
        # Extract questions from plans
        gold_questions = [v[0] for v in gold_plan.values()]
        pred_questions = [v[0] for v in pred_plan.values()]
        
        gold_nodes = list(gold_plan.keys())
        pred_nodes = list(pred_plan.keys())
        
        # Compute similarity matrix
        sim_matrix = self.compute_similarity_matrix(pred_questions, gold_questions)
        
        # Find optimal mapping
        mapping, similarities = self.find_optimal_mapping(
            sim_matrix, gold_nodes, pred_nodes, threshold
        )
        
        # Average similarity normalized by golden plan size
        if similarities:
            avg_score = sum(similarities) / len(gold_plan)
        else:
            avg_score = 0.0
            
        return avg_score, mapping


class SubGoalScorer:
    """Scores sub-goal completion using token-level F1.
    
    For each matched sub-question, compares the predicted intermediate
    answer with the golden answer using F1 metric.
    """
    
    @staticmethod
    def normalize_answer(text: str) -> str:
        """Normalize answer for comparison."""
        import string
        
        text = text.lower()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove punctuation
        text = "".join(c for c in text if c not in string.punctuation)
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text
    
    @staticmethod
    def token_f1(pred: str, gold: str) -> float:
        """Compute token-level F1 score."""
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
        """Compute sub-goal completion score.
        
        Args:
            pred_graph: Predicted execution results {"Q1": {"answer": "..."}, ...}
            gold_graph: Golden execution results
            mapping: Node mapping from golden to predicted
            
        Returns:
            Average F1 score across matched sub-goals
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

