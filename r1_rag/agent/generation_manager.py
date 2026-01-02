"""
Generation Manager for R1-RAG

Manages the multi-turn LLM generation loop with search integration:
1. Generate response with reasoning and search queries
2. Execute search operations via retrieval server
3. Inject search results as observations
4. Continue generation until answer or max turns

Core component for iterative retrieval-augmented reasoning in GRPO training.
"""

import re
import torch
import requests
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from verl import DataProto


@dataclass
class GenerationConfig:
    """Configuration for multi-turn generation."""
    max_turns: int = 4              # Maximum search iterations
    max_start_length: int = 2048    # Max length of initial prompt
    max_prompt_length: int = 4096   # Max total context length
    max_response_length: int = 512  # Max tokens per generation
    max_obs_length: int = 600       # Max tokens for search results
    num_gpus: int = 1
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3                   # Number of search results
    no_think_rl: bool = False       # Whether to mask thinking in RL


class TensorHelper:
    """Helper class for tensor operations during generation."""
    
    def __init__(
        self,
        pad_token_id: int,
        max_prompt_length: int,
        max_obs_length: int,
        max_start_length: int,
    ):
        self.pad_token_id = pad_token_id
        self.max_prompt_length = max_prompt_length
        self.max_obs_length = max_obs_length
        self.max_start_length = max_start_length
    
    def concatenate_with_padding(
        self, 
        tensors: List[torch.Tensor],
        pad_to_left: bool = True
    ) -> torch.Tensor:
        """Concatenate tensors and handle padding alignment."""
        concatenated = torch.cat(tensors, dim=1)
        
        mask = concatenated != self.pad_token_id if pad_to_left else concatenated == self.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        return concatenated.gather(1, sorted_indices)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask from input IDs."""
        return (input_ids != self.pad_token_id).long()
    
    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Create position IDs from attention mask."""
        return attention_mask.cumsum(dim=-1) - 1


class LLMGenerationManager:
    """Manages multi-turn generation with search integration.
    
    The generation loop implements:
    1. Model generates <think>...</think><search>query</search>
    2. System executes search, returns <information>results</information>
    3. Model continues with next sub-question
    4. Repeat until <answer>...</answer> or max turns
    
    Key features:
    - Tracks active/completed samples in batch
    - Handles variable-length conversations
    - Creates info_mask to exclude search results from RL gradients
    - Compatible with veRL's DataProto format
    """
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        """Initialize generation manager.
        
        Args:
            tokenizer: HuggingFace tokenizer
            actor_rollout_wg: Worker group for generation
            config: Generation configuration
            is_validation: Whether in validation mode
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        # Initialize tensor helper
        self.tensor_fn = TensorHelper(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
        )
        
        # Action patterns
        self.search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    # ==================== Tokenization ====================
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses,
            add_special_tokens=False,
            return_tensors='pt',
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses(
        self, 
        responses: torch.Tensor
    ) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search or answer operations.
        
        Args:
            responses: Raw response token IDs
            
        Returns:
            Tuple of (processed token IDs, string responses)
        """
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )
        
        # Truncate at action boundaries
        processed_str = []
        for resp in responses_str:
            if '</search>' in resp:
                processed_str.append(resp.split('</search>')[0] + '</search>')
            elif '</answer>' in resp:
                processed_str.append(resp.split('</answer>')[0] + '</answer>')
            else:
                processed_str.append(resp)
        
        processed_ids = self._batch_tokenize(processed_str)
        return processed_ids, processed_str
    
    def _process_observations(self, observations: List[str]) -> torch.Tensor:
        """Process observations (search results) for injection.
        
        Args:
            observations: List of observation strings
            
        Returns:
            Tokenized observations
        """
        obs_ids = self.tokenizer(
            observations,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']
        
        # Truncate if too long
        if obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] Observation too long: {obs_ids.shape[1]} > {self.config.max_obs_length}")
            obs_ids = obs_ids[:, :self.config.max_obs_length]
        
        return obs_ids
    
    # ==================== Action Parsing ====================
    
    def parse_action(self, response: str) -> Tuple[str, str, bool]:
        """Extract action type and content from model response.
        
        Args:
            response: Raw model output
            
        Returns:
            Tuple of (action_type, content, is_valid)
            action_type: "search", "answer", or None
        """
        search_match = self.search_pattern.search(response)
        if search_match:
            return "search", search_match.group(1).strip(), True
        
        answer_match = self.answer_pattern.search(response)
        if answer_match:
            return "answer", answer_match.group(1).strip(), True
        
        return None, "", False
    
    # ==================== Search Execution ====================
    
    def execute_search(self, queries: List[str]) -> List[str]:
        """Execute batch search against retrieval server.
        
        Args:
            queries: List of search queries
            
        Returns:
            List of formatted search results
        """
        if not queries:
            return []
        
        try:
            payload = {
                "queries": queries,
                "topk": self.config.topk,
                "return_scores": True
            }
            response = requests.post(self.config.search_url, json=payload)
            results = response.json().get("result", [])
            
            # Format results
            formatted = []
            for result_list in results:
                text = ""
                for idx, doc in enumerate(result_list):
                    content = doc.get("document", {}).get("contents", "")
                    # Split title and body
                    lines = content.split("\n")
                    title = lines[0] if lines else ""
                    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
                    text += f"Doc {idx+1}(Title: {title}) {body}\n"
                formatted.append(text)
            
            return formatted
            
        except Exception as e:
            print(f"[Search Error] {e}")
            return ["Search failed. Please try a different query."] * len(queries)
    
    # ==================== State Management ====================
    
    def _update_rolling_state(
        self,
        rollings: DataProto,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor
    ) -> DataProto:
        """Update rolling state with new responses and observations.
        
        Args:
            rollings: Current rolling state
            cur_responses: Current response tokens
            next_obs_ids: Next observation tokens
            
        Returns:
            Updated DataProto
        """
        # Concatenate with padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
        # Truncate to max length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings
    
    def _info_masked_concatenate(
        self,
        prompt: torch.Tensor,
        prompt_with_mask: torch.Tensor,
        response: torch.Tensor,
        info: Optional[torch.Tensor] = None,
        pad_to_left: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate tensors with info masking for RL gradients.
        
        Creates two versions:
        1. Full tensor with all content
        2. Masked tensor where info blocks are replaced with pad tokens
        
        This allows excluding search results from RL gradient computation.
        
        Args:
            prompt: Prompt tokens
            prompt_with_mask: Prompt tokens (for masking version)
            response: Response tokens
            info: Information/observation tokens (optional)
            pad_to_left: Whether to pad to left
            
        Returns:
            Tuple of (full tensor, masked tensor)
        """
        pad_id = self.tokenizer.pad_token_id
        
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            # Create info mask (all pad tokens)
            info_mask = torch.full(
                info.size(), 
                pad_id, 
                dtype=info.dtype, 
                device=info.device
            )
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_masked = torch.cat(tensors_with_mask, dim=1)
        
        # Sort by padding position
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        padded = concatenated.gather(1, sorted_indices)
        padded_masked = concatenated_masked.gather(1, sorted_indices)
        
        return padded, padded_masked
    
    def _update_right_side(
        self,
        right_side: Dict,
        cur_responses: torch.Tensor,
        next_obs_ids: Optional[torch.Tensor] = None
    ) -> Dict:
        """Update right side (response) state.
        
        Args:
            right_side: Current right side state
            cur_responses: Current response tokens
            next_obs_ids: Next observation tokens (optional)
            
        Returns:
            Updated right side state
        """
        responses, responses_masked = self._info_masked_concatenate(
            right_side['responses'],
            right_side['responses_with_info_mask'],
            cur_responses,
            next_obs_ids,
            pad_to_left=False
        )
        
        # Truncate to max length
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_masked[:, :max_len]
        }
    
    # ==================== GPU Handling ====================
    
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """Generate with multi-GPU padding handling.
        
        When batch size isn't divisible by num_gpus, pad with first sequence.
        
        Args:
            active_batch: Batch to generate from
            
        Returns:
            Generated outputs
        """
        num_gpus = self.config.num_gpus
        
        # Cast to long
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        # Generate and remove padding
        output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        for k, v in output.batch.items():
            output.batch[k] = v[:batch_size]
        
        return output
    
    # ==================== Main Generation Loop ====================
    
    def run_generation_loop(
        self,
        gen_batch: DataProto
    ) -> Dict[str, Any]:
        """Run the full multi-turn generation loop.
        
        Implements the core R1-RAG generation process:
        1. Initialize state for all samples
        2. Generate responses for active samples
        3. Parse actions (search/answer)
        4. Execute searches and inject results
        5. Repeat until done or max turns
        
        Args:
            gen_batch: Initial DataProto batch
            
        Returns:
            Dictionary with outputs including:
            - prompts: Original prompt tokens
            - responses: Generated response tokens
            - responses_with_info_mask: Responses with info blocks masked
            - attention_mask: Full attention mask
            - statistics: Turn counts, search counts, etc.
        """
        batch_size = gen_batch.batch['input_ids'].shape[0]
        device = gen_batch.batch['input_ids'].device
        
        # Initialize tracking
        active_mask = [True] * batch_size
        original_prompts = gen_batch.batch['input_ids'].clone()
        
        # Initialize right side (response accumulator)
        right_side = {
            'responses': torch.full(
                (batch_size, 1), 
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            ),
            'responses_with_info_mask': torch.full(
                (batch_size, 1),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=device
            )
        }
        
        # Statistics
        turn_counts = [0] * batch_size
        search_counts = [0] * batch_size
        valid_action_counts = [0] * batch_size
        
        # Current rolling state
        rollings = gen_batch
        
        # Main generation loop
        for turn in range(self.config.max_turns):
            if not any(active_mask):
                break
            
            # Generate responses for active samples
            gen_output = self._generate_with_gpu_padding(rollings)
            cur_responses = gen_output.batch['responses']
            
            # Post-process responses
            cur_responses, responses_str = self._postprocess_responses(cur_responses)
            
            # Parse actions and collect search queries
            search_indices = []
            search_queries = []
            observations = [""] * batch_size
            dones = [False] * batch_size
            
            for i, (resp_str, active) in enumerate(zip(responses_str, active_mask)):
                if not active:
                    dones[i] = True
                    continue
                
                action_type, content, is_valid = self.parse_action(resp_str)
                
                if is_valid:
                    valid_action_counts[i] += 1
                
                if action_type == "answer":
                    # Final answer - mark as done
                    dones[i] = True
                elif action_type == "search":
                    # Queue search
                    search_indices.append(i)
                    search_queries.append(content)
                    search_counts[i] += 1
                else:
                    # Invalid action - provide guidance
                    observations[i] = (
                        "\nMy previous action is invalid. "
                        "I should use <search>query</search> to search, "
                        "or <answer>result</answer> to give the final answer. "
                        "Let me try again.\n"
                    )
                
                turn_counts[i] += 1
            
            # Execute searches in batch
            if search_queries:
                results = self.execute_search(search_queries)
                for idx, result in zip(search_indices, results):
                    observations[idx] = f"\n\n<information>{result.strip()}</information>\n\n"
            
            # Process observations
            obs_ids = self._process_observations(observations)
            
            # Update right side (response accumulator)
            right_side = self._update_right_side(
                right_side,
                cur_responses,
                obs_ids if any(not d for d in dones) else None
            )
            
            # Update rolling state for next turn
            rollings = self._update_rolling_state(rollings, cur_responses, obs_ids)
            
            # Update active mask
            active_mask = [not d for d in dones]
        
        # Final generation for any remaining active samples
        if any(active_mask):
            gen_output = self._generate_with_gpu_padding(rollings)
            final_responses = gen_output.batch['responses']
            final_responses, _ = self._postprocess_responses(final_responses)
            
            right_side = self._update_right_side(right_side, final_responses)
        
        # Prepare final output
        responses = right_side['responses']
        responses_masked = right_side['responses_with_info_mask']
        
        # Create full attention mask
        prompt_mask = (original_prompts != self.tokenizer.pad_token_id).long()
        response_mask = (responses != self.tokenizer.pad_token_id).long()
        full_attention_mask = torch.cat([prompt_mask, response_mask], dim=1)
        
        return {
            'prompts': original_prompts,
            'responses': responses,
            'responses_with_info_mask': responses_masked,
            'attention_mask': full_attention_mask,
            'statistics': {
                'turn_counts': turn_counts,
                'search_counts': search_counts,
                'valid_action_counts': valid_action_counts,
                'avg_turns': sum(turn_counts) / len(turn_counts),
                'avg_searches': sum(search_counts) / len(search_counts),
            }
        }
    
    # ==================== Simple Step Interface ====================
    
    def step(
        self,
        responses: List[str],
        active_mask: List[bool]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """Execute one step of the generation loop.
        
        Simplified interface for external control.
        
        Args:
            responses: Model responses for each sample
            active_mask: Which samples are still active
            
        Returns:
            Tuple of (observations, done_flags, valid_action_flags)
        """
        observations = []
        dones = []
        valid_actions = []
        
        search_indices = []
        search_queries = []
        
        for i, (response, active) in enumerate(zip(responses, active_mask)):
            if not active:
                observations.append("")
                dones.append(True)
                valid_actions.append(False)
                continue
            
            action_type, content, is_valid = self.parse_action(response)
            
            if action_type == "answer":
                observations.append("")
                dones.append(True)
                valid_actions.append(True)
            elif action_type == "search":
                search_indices.append(i)
                search_queries.append(content)
                observations.append(None)  # Placeholder
                dones.append(False)
                valid_actions.append(True)
            else:
                observations.append(
                    "\nMy previous action is invalid. "
                    "I should use <search>query</search> to search, "
                    "or <answer>result</answer> to give the final answer. "
                    "Let me try again.\n"
                )
                dones.append(False)
                valid_actions.append(False)
        
        # Execute searches
        if search_queries:
            results = self.execute_search(search_queries)
            for idx, result in zip(search_indices, results):
                observations[idx] = f"\n\n<information>{result.strip()}</information>\n\n"
        
        return observations, dones, valid_actions
