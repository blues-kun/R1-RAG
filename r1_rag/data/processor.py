"""
Planning Data Processor for R1-RAG

Handles data pipeline:
1. Load raw multi-hop QA datasets
2. Generate or load golden plan annotations
3. Build training data with proper format
4. Export to parquet for veRL training
"""

import os
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd
from tqdm import tqdm

from .prompts import PLANNING_PROMPT_TEMPLATE, ONE_SHOT_EXAMPLE
from .gpt4o_annotator import GPT4oPlanGenerator, AnnotationResult


@dataclass
class TrainingSample:
    """A single training sample for R1-RAG."""
    id: int
    question: str
    golden_answers: List[str]
    data_source: str
    prompt: List[Dict[str, str]]
    ability: str = "multi-hop-reasoning"
    reward_model: Dict = None
    metadata: Dict = None
    extra_info: Dict = None
    
    def __post_init__(self):
        if self.reward_model is None:
            self.reward_model = {
                "ground_truth": {"target": self.golden_answers},
                "style": "rule"
            }
        if self.metadata is None:
            self.metadata = {}
        if self.extra_info is None:
            self.extra_info = {"index": self.id, "split": "train"}
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame."""
        return {
            "id": self.id,
            "question": self.question,
            "golden_answers": self.golden_answers,
            "data_source": self.data_source,
            "prompt": self.prompt,
            "ability": self.ability,
            "reward_model": self.reward_model,
            "metadata": self.metadata,
            "extra_info": self.extra_info
        }


class PlanningDataProcessor:
    """Processes multi-hop QA data for R1-RAG training.
    
    Supports two modes:
    1. Generate new annotations using GPT-4o
    2. Load pre-existing annotations from file
    
    The processor ensures:
    - Consistent prompt formatting
    - Proper metadata structure for reward computation
    - Quality filtering based on answer validation
    """
    
    def __init__(
        self,
        data_source: str = "multi_hop_qa",
        use_one_shot: bool = True
    ):
        """Initialize the data processor.
        
        Args:
            data_source: Name identifier for the data source
            use_one_shot: Whether to include one-shot example in prompts
        """
        self.data_source = data_source
        self.use_one_shot = use_one_shot
        self.samples: List[TrainingSample] = []
        
    def _format_prompt(self, question: str) -> List[Dict[str, str]]:
        """Format question into chat prompt format.
        
        Args:
            question: The raw question text
            
        Returns:
            List with single user message containing formatted prompt
        """
        one_shot = ONE_SHOT_EXAMPLE if self.use_one_shot else ""
        
        prompt_text = PLANNING_PROMPT_TEMPLATE.format(
            one_shot_example=one_shot,
            question=question
        )
        
        return [{"role": "user", "content": prompt_text}]
    
    def load_from_jsonl(
        self,
        file_path: str,
        question_key: str = "question",
        answer_key: str = "golden_answers",
        plan_key: str = "plan",
        graph_key: str = "graph",
        hop_key: str = "hop"
    ) -> int:
        """Load data from JSONL file with pre-existing annotations.
        
        Args:
            file_path: Path to JSONL file
            question_key: Key for question field
            answer_key: Key for golden answers
            plan_key: Key for plan annotation (optional)
            graph_key: Key for graph annotation (optional)
            hop_key: Key for hop count (optional)
            
        Returns:
            Number of samples loaded
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, desc="Loading data")):
                try:
                    data = json.loads(line.strip())
                    
                    question = data.get(question_key, "")
                    answers = data.get(answer_key, [])
                    if isinstance(answers, str):
                        answers = [answers]
                    
                    # Build metadata
                    metadata = {}
                    if plan_key in data:
                        metadata["plan"] = data[plan_key]
                    if graph_key in data:
                        graph = data[graph_key]
                        if not isinstance(graph, list):
                            graph = [graph]
                        metadata["graph"] = graph
                    if hop_key in data:
                        metadata["hop"] = data[hop_key]
                    
                    sample = TrainingSample(
                        id=idx,
                        question=question,
                        golden_answers=answers,
                        data_source=self.data_source,
                        prompt=self._format_prompt(question),
                        metadata=metadata
                    )
                    self.samples.append(sample)
                    
                except Exception as e:
                    print(f"[Warning] Failed to parse line {idx}: {e}")
                    continue
        
        print(f"[DataProcessor] Loaded {len(self.samples)} samples from {file_path}")
        return len(self.samples)
    
    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        question_key: str = "question",
        answer_key: str = "golden_answers"
    ) -> int:
        """Load data from HuggingFace datasets.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            question_key: Key for question field
            answer_key: Key for answers field
            
        Returns:
            Number of samples loaded
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        for idx, item in enumerate(tqdm(dataset, desc="Loading HF data")):
            question = item.get(question_key, "")
            answers = item.get(answer_key, [])
            if isinstance(answers, str):
                answers = [answers]
            
            # Extract any available metadata
            metadata = {}
            for key in ["plan", "graph", "hop", "metadata"]:
                if key in item:
                    metadata[key] = item[key]
            
            sample = TrainingSample(
                id=idx,
                question=question,
                golden_answers=answers,
                data_source=self.data_source,
                prompt=self._format_prompt(question),
                metadata=metadata
            )
            self.samples.append(sample)
        
        print(f"[DataProcessor] Loaded {len(self.samples)} samples from {dataset_name}")
        return len(self.samples)
    
    def generate_annotations(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_samples: Optional[int] = None
    ) -> int:
        """Generate plan annotations using GPT-4o.
        
        Updates samples in-place with generated annotations.
        Filters out samples where annotation generation failed.
        
        Args:
            api_key: OpenAI API key
            model: GPT model to use
            max_samples: Maximum samples to annotate (for testing)
            
        Returns:
            Number of successfully annotated samples
        """
        annotator = GPT4oPlanGenerator(api_key=api_key, model=model)
        
        # Prepare samples for annotation
        to_annotate = self.samples[:max_samples] if max_samples else self.samples
        
        raw_samples = [
            {"question": s.question, "golden_answers": s.golden_answers}
            for s in to_annotate
        ]
        
        # Generate annotations
        results = annotator.generate_batch(raw_samples)
        
        # Update samples with annotations
        valid_samples = []
        for sample, result in zip(to_annotate, results):
            if result.is_valid:
                sample.metadata["plan"] = result.plan
                sample.metadata["graph"] = result.graph
                valid_samples.append(sample)
        
        # Replace samples with valid ones
        if max_samples:
            # Keep un-annotated samples
            self.samples = valid_samples + self.samples[max_samples:]
        else:
            self.samples = valid_samples
        
        print(f"[DataProcessor] Generated {len(valid_samples)} valid annotations")
        return len(valid_samples)
    
    def filter_with_annotations(self) -> int:
        """Filter to keep only samples with plan annotations.
        
        Returns:
            Number of samples remaining
        """
        self.samples = [
            s for s in self.samples
            if s.metadata.get("plan") and s.metadata.get("graph")
        ]
        print(f"[DataProcessor] Filtered to {len(self.samples)} samples with annotations")
        return len(self.samples)
    
    def export_parquet(
        self,
        output_path: str,
        train_ratio: float = 0.9
    ) -> Dict[str, str]:
        """Export processed data to parquet files.
        
        Args:
            output_path: Directory to save parquet files
            train_ratio: Ratio of data for training (rest for validation)
            
        Returns:
            Dict with paths to train and val files
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Convert to DataFrame
        data = [s.to_dict() for s in self.samples]
        df = pd.DataFrame(data)
        
        # Split train/val
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]
        
        # Save
        train_path = os.path.join(output_path, "train.parquet")
        val_path = os.path.join(output_path, "val.parquet")
        
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        
        print(f"[DataProcessor] Exported {len(train_df)} train, {len(val_df)} val samples")
        
        return {"train": train_path, "val": val_path}
    
    def export_jsonl(self, output_path: str) -> str:
        """Export processed data to JSONL format.
        
        Args:
            output_path: Path to output file
            
        Returns:
            Path to output file
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
        
        print(f"[DataProcessor] Exported {len(self.samples)} samples to {output_path}")
        return output_path
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded data.
        
        Returns:
            Dictionary with various statistics
        """
        stats = {
            "total_samples": len(self.samples),
            "with_plan": sum(1 for s in self.samples if s.metadata.get("plan")),
            "with_graph": sum(1 for s in self.samples if s.metadata.get("graph")),
            "data_sources": list(set(s.data_source for s in self.samples))
        }
        
        # Hop distribution
        hop_counts = {}
        for s in self.samples:
            hop = s.metadata.get("hop", "unknown")
            hop_counts[hop] = hop_counts.get(hop, 0) + 1
        stats["hop_distribution"] = hop_counts
        
        return stats

