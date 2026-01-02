"""
Structure Scorer for R1-RAG

Uses Graph Edit Distance (GED) to measure structural similarity
between predicted and golden planning DAGs.

This provides the "structure reward" signal that encourages
proper dependency modeling in multi-hop reasoning.
"""

import re
from typing import Dict, List, Tuple, Optional
import numpy as np

import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher


class StructureScorer:
    """Computes structural similarity between planning DAGs.
    
    The key insight is that multi-hop reasoning requires proper
    dependency structure: Q2 should depend on Q1's answer, etc.
    
    GED captures:
    - Missing/extra nodes (wrong number of sub-questions)
    - Missing/extra edges (wrong dependencies)
    - Incorrect dependency ordering
    """
    
    def __init__(self, placeholder_pattern: str = r"<A(\d+)>"):
        """Initialize structure scorer.
        
        Args:
            placeholder_pattern: Regex pattern for answer placeholders
        """
        self.placeholder_pattern = re.compile(placeholder_pattern)
        
    def plan_to_dag(self, plan: Dict[str, List[str]]) -> nx.DiGraph:
        """Convert planning dict to directed acyclic graph.
        
        Plan format: {"Q1": ["question text", "<A1>"], "Q2": ["text with <A1>", "<A2>"]}
        
        Creates a DAG where:
        - Each node is a sub-question (Q1, Q2, ...)
        - Edges represent dependencies (Q1 → Q2 if Q2 uses <A1>)
        
        Args:
            plan: Planning dictionary
            
        Returns:
            NetworkX DiGraph representing the plan structure
        """
        dag = nx.DiGraph()
        
        for node_id, (question, _) in plan.items():
            # Add node with question as attribute
            dag.add_node(node_id, template=question)
            
            # Find dependencies from placeholders in question
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
        """Compute Graph Edit Distance between DAGs.
        
        GED = |node_diff| + |edge_diff|
        
        Uses the provided node mapping to align the graphs before
        computing differences.
        
        Args:
            pred_dag: Predicted planning DAG
            gold_dag: Golden planning DAG
            mapping: Node mapping from golden to predicted nodes
            beta: Normalization factor for exponential decay
            
        Returns:
            Tuple of (raw GED, normalized GED score in [0,1])
        """
        # Relabel nodes based on mapping
        gold_relabel = {}
        pred_relabel = {}
        common_idx = 1
        
        # Matched nodes get common labels
        for gold_node, pred_node in mapping.items():
            common_label = f"C{common_idx}"
            gold_relabel[gold_node] = common_label
            pred_relabel[pred_node] = common_label
            common_idx += 1
        
        # Unmatched nodes get unique labels
        for idx, node in enumerate(gold_dag.nodes()):
            if node not in gold_relabel:
                gold_relabel[node] = f"Gg{idx}"
                
        for idx, node in enumerate(pred_dag.nodes()):
            if node not in pred_relabel:
                pred_relabel[node] = f"Gp{idx}"
        
        # Apply relabeling
        gold_relabeled = nx.relabel_nodes(gold_dag, gold_relabel)
        pred_relabeled = nx.relabel_nodes(pred_dag, pred_relabel)
        
        # Compute symmetric differences
        node_union = set(gold_relabeled.nodes()) | set(pred_relabeled.nodes())
        node_intersection = set(gold_relabeled.nodes()) & set(pred_relabeled.nodes())
        
        edge_union = set(gold_relabeled.edges()) | set(pred_relabeled.edges())
        edge_intersection = set(gold_relabeled.edges()) & set(pred_relabeled.edges())
        
        # GED = node diff + edge diff
        ged = len(node_union - node_intersection) + len(edge_union - edge_intersection)
        
        # Normalize using exponential decay
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
        """Compute overall structural similarity score.
        
        Combines:
        - Raw GED (for interpretability)
        - Normalized GED score (for reward)
        - Binary structure match (GED == 0)
        
        Args:
            pred_plan: Predicted plan
            gold_plan: Golden plan
            mapping: Node mapping from semantic scorer
            alpha: Weight for semantic vs structural components
            beta: GED normalization factor
            
        Returns:
            Tuple of (raw GED, normalized score, binary match indicator)
        """
        if not pred_plan or not gold_plan:
            return 0, 0.0, 0.0
        
        pred_dag = self.plan_to_dag(pred_plan)
        gold_dag = self.plan_to_dag(gold_plan)
        
        ged, normalized_ged = self.compute_ged(pred_dag, gold_dag, mapping, beta)
        
        # Binary indicator: 1 if structure matches exactly (or close)
        structure_match = 1.0 if ged < 1 else 0.0
        
        return ged, normalized_ged, structure_match
    
    def find_subgraph_isomorphism(
        self,
        pred_dag: nx.DiGraph,
        gold_dag: nx.DiGraph
    ) -> List[str]:
        """Find isomorphic subgraph between predicted and golden DAG.
        
        Useful for partial credit when predicted plan captures a
        subset of the golden reasoning structure.
        
        Args:
            pred_dag: Predicted DAG
            gold_dag: Golden DAG (pattern to match)
            
        Returns:
            List of nodes in pred_dag that form isomorphic subgraph
        """
        matcher = DiGraphMatcher(pred_dag, gold_dag)
        matched_nodes = []
        
        for mapping in matcher.subgraph_isomorphisms_iter():
            matched_nodes.extend(mapping.keys())
        
        return list(set(matched_nodes))

