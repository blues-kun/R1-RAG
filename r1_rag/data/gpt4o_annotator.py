"""
GPT-4o Plan Annotation Generator for R1-RAG

Uses GPT-4o to generate high-quality planning DAG annotations:
1. Decompose multi-hop questions into sub-questions
2. Generate intermediate answers for each sub-question
3. Validate against ground truth for quality filtering

This creates the "golden plans" used for process supervision in RL training.
"""

import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .prompts import GPT4O_PLAN_ANNOTATION_PROMPT


@dataclass
class AnnotationResult:
    """Result of GPT-4o plan annotation."""
    question: str
    gold_answer: List[str]
    plan: Optional[Dict[str, List[str]]]
    graph: Optional[Dict[str, Dict]]
    is_valid: bool
    error_message: Optional[str] = None


class GPT4oPlanGenerator:
    """Generates planning DAG annotations using GPT-4o.
    
    Key design decisions:
    1. Use ground truth to guide and validate annotations
    2. Implement retry logic for robustness
    3. Filter low-quality annotations
    4. Support batch processing with parallelism
    
    The generated plans are used as "golden labels" for:
    - Semantic similarity scoring (E5 embedding)
    - Structural similarity scoring (GED)
    - Sub-goal completion scoring (F1)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        temperature: float = 0.3,
        max_workers: int = 5
    ):
        """Initialize the GPT-4o annotator.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4o recommended)
            max_retries: Maximum retry attempts for API calls
            retry_delay: Delay between retries in seconds
            temperature: Sampling temperature (lower = more deterministic)
            max_workers: Number of parallel workers for batch processing
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_workers = max_workers
        
        # Pattern for extracting JSON from response
        self.json_pattern = re.compile(r'\{[\s\S]*\}')
    
    def _call_gpt4o(self, prompt: str) -> Optional[str]:
        """Make API call to GPT-4o with retry logic.
        
        Args:
            prompt: The prompt to send
            
        Returns:
            Response text or None if all retries fail
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates structured reasoning plans for multi-hop questions."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=2048
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(f"[GPT4o] Failed after {self.max_retries} attempts: {e}")
                    return None
        return None
    
    def _parse_response(self, response: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Parse GPT-4o response to extract plan and graph.
        
        Args:
            response: Raw GPT-4o response
            
        Returns:
            Tuple of (plan dict, graph dict) or (None, None) on failure
        """
        try:
            # Find JSON in response
            match = self.json_pattern.search(response)
            if not match:
                return None, None
            
            data = json.loads(match.group())
            plan = data.get("plan", {})
            graph = data.get("graph", {})
            
            # Validate structure
            if not plan or not graph:
                return None, None
            
            # Normalize placeholders
            normalized_plan = {}
            for key, value in plan.items():
                if isinstance(value, list) and len(value) >= 2:
                    # Convert #N to <AN>
                    question = re.sub(r"#(\d+)", r"<A\1>", str(value[0]))
                    placeholder = re.sub(r"#(\d+)", r"<A\1>", str(value[1]))
                    normalized_plan[key] = [question, placeholder]
            
            return normalized_plan, graph
            
        except (json.JSONDecodeError, Exception) as e:
            return None, None
    
    def _validate_annotation(
        self,
        plan: Dict[str, List[str]],
        graph: Dict[str, Dict],
        gold_answer: List[str]
    ) -> bool:
        """Validate that the annotation leads to correct answer.
        
        Checks:
        1. Plan has at least one sub-question
        2. Graph has answers for all plan questions
        3. Final answer in graph matches gold answer
        
        Args:
            plan: Generated plan
            graph: Generated execution graph
            gold_answer: Ground truth answers
            
        Returns:
            True if annotation is valid
        """
        if not plan or not graph:
            return False
        
        # Check all plan questions have answers
        for q_key in plan.keys():
            if q_key not in graph:
                return False
            if "answer" not in graph[q_key]:
                return False
        
        # Check final answer matches gold (relaxed matching)
        final_q = f"Q{len(plan)}"
        if final_q in graph:
            final_answer = graph[final_q].get("answer", "").lower().strip()
            for gold in gold_answer:
                if gold.lower().strip() in final_answer or final_answer in gold.lower().strip():
                    return True
        
        return False
    
    def generate_annotation(
        self,
        question: str,
        gold_answer: List[str]
    ) -> AnnotationResult:
        """Generate plan annotation for a single question.
        
        Args:
            question: The multi-hop question
            gold_answer: List of acceptable gold answers
            
        Returns:
            AnnotationResult with plan and graph (or error info)
        """
        # Format prompt
        prompt = GPT4O_PLAN_ANNOTATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer[0] if gold_answer else "N/A"
        )
        
        # Call GPT-4o
        response = self._call_gpt4o(prompt)
        if not response:
            return AnnotationResult(
                question=question,
                gold_answer=gold_answer,
                plan=None,
                graph=None,
                is_valid=False,
                error_message="GPT-4o API call failed"
            )
        
        # Parse response
        plan, graph = self._parse_response(response)
        if not plan or not graph:
            return AnnotationResult(
                question=question,
                gold_answer=gold_answer,
                plan=None,
                graph=None,
                is_valid=False,
                error_message="Failed to parse GPT-4o response"
            )
        
        # Validate annotation
        is_valid = self._validate_annotation(plan, graph, gold_answer)
        
        return AnnotationResult(
            question=question,
            gold_answer=gold_answer,
            plan=plan,
            graph=[graph],  # Wrap in list for compatibility
            is_valid=is_valid,
            error_message=None if is_valid else "Annotation validation failed"
        )
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        question_key: str = "question",
        answer_key: str = "golden_answers"
    ) -> List[AnnotationResult]:
        """Generate annotations for a batch of samples.
        
        Uses parallel processing for efficiency.
        
        Args:
            samples: List of sample dicts with question and answers
            question_key: Key for question field
            answer_key: Key for gold answers field
            
        Returns:
            List of AnnotationResults
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for i, sample in enumerate(samples):
                question = sample.get(question_key, "")
                gold_answer = sample.get(answer_key, [])
                
                if isinstance(gold_answer, str):
                    gold_answer = [gold_answer]
                
                future = executor.submit(
                    self.generate_annotation,
                    question,
                    gold_answer
                )
                futures[future] = i
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating annotations"):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append((idx, result))
                except Exception as e:
                    results.append((idx, AnnotationResult(
                        question=samples[idx].get(question_key, ""),
                        gold_answer=samples[idx].get(answer_key, []),
                        plan=None,
                        graph=None,
                        is_valid=False,
                        error_message=str(e)
                    )))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def filter_valid_annotations(
        self,
        results: List[AnnotationResult]
    ) -> List[AnnotationResult]:
        """Filter to keep only valid annotations.
        
        Args:
            results: List of annotation results
            
        Returns:
            Filtered list with only valid annotations
        """
        valid = [r for r in results if r.is_valid]
        print(f"[GPT4o] Valid annotations: {len(valid)}/{len(results)} ({100*len(valid)/len(results):.1f}%)")
        return valid

