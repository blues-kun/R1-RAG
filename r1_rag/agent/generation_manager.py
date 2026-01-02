"""
R1-RAG 生成管理器

管理带搜索集成的多轮LLM生成循环:
1. 生成包含推理和搜索查询的响应
2. 通过检索服务器执行搜索操作
3. 将搜索结果作为观察注入
4. 继续生成直到得到答案或达到最大轮次

GRPO训练中迭代检索增强推理的核心组件。
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
    """多轮生成配置"""
    max_turns: int = 4              # 最大搜索迭代次数
    max_start_length: int = 2048    # 初始prompt的最大长度
    max_prompt_length: int = 4096   # 总上下文最大长度
    max_response_length: int = 512  # 每次生成的最大token数
    max_obs_length: int = 600       # 搜索结果的最大token数
    num_gpus: int = 1
    search_url: str = "http://127.0.0.1:8000/retrieve"
    topk: int = 3                   # 搜索结果数量
    no_think_rl: bool = False       # 是否在RL中屏蔽思考过程


class TensorHelper:
    """生成过程中的张量操作辅助类"""
    
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
        """拼接张量并处理填充对齐"""
        concatenated = torch.cat(tensors, dim=1)
        
        mask = concatenated != self.pad_token_id if pad_to_left else concatenated == self.pad_token_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        
        return concatenated.gather(1, sorted_indices)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """从input_ids创建attention mask"""
        return (input_ids != self.pad_token_id).long()
    
    def create_position_ids(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """从attention mask创建position ids"""
        return attention_mask.cumsum(dim=-1) - 1


class LLMGenerationManager:
    """管理带搜索集成的多轮生成
    
    生成循环实现:
    1. 模型生成 <think>...</think><search>query</search>
    2. 系统执行搜索，返回 <information>results</information>
    3. 模型继续处理下一个子问题
    4. 重复直到 <answer>...</answer> 或达到最大轮次
    
    核心特性:
    - 跟踪批次中的活跃/完成样本
    - 处理变长对话
    - 创建info_mask以在RL梯度中排除搜索结果
    - 兼容veRL的DataProto格式
    """
    
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        """初始化生成管理器
        
        Args:
            tokenizer: HuggingFace tokenizer
            actor_rollout_wg: 生成的worker group
            config: 生成配置
            is_validation: 是否为验证模式
        """
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        # 初始化张量辅助器
        self.tensor_fn = TensorHelper(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length,
        )
        
        # 动作模式
        self.search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    
    # ==================== 分词处理 ====================
    
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """批量分词响应"""
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
        """处理响应，在搜索或答案操作处停止
        
        Args:
            responses: 原始响应token IDs
            
        Returns:
            (处理后的token IDs, 字符串响应) 元组
        """
        responses_str = self.tokenizer.batch_decode(
            responses,
            skip_special_tokens=True
        )
        
        # 在动作边界处截断
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
        """处理观察（搜索结果）用于注入
        
        Args:
            observations: 观察字符串列表
            
        Returns:
            分词后的观察
        """
        obs_ids = self.tokenizer(
            observations,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']
        
        # 如果太长则截断
        if obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[警告] 观察过长: {obs_ids.shape[1]} > {self.config.max_obs_length}")
            obs_ids = obs_ids[:, :self.config.max_obs_length]
        
        return obs_ids
    
    # ==================== 动作解析 ====================
    
    def parse_action(self, response: str) -> Tuple[str, str, bool]:
        """从模型响应中提取动作类型和内容
        
        Args:
            response: 原始模型输出
            
        Returns:
            (动作类型, 内容, 是否有效) 元组
            动作类型: "search", "answer" 或 None
        """
        search_match = self.search_pattern.search(response)
        if search_match:
            return "search", search_match.group(1).strip(), True
        
        answer_match = self.answer_pattern.search(response)
        if answer_match:
            return "answer", answer_match.group(1).strip(), True
        
        return None, "", False
    
    # ==================== 搜索执行 ====================
    
    def execute_search(self, queries: List[str]) -> List[str]:
        """对检索服务器执行批量搜索
        
        Args:
            queries: 搜索查询列表
            
        Returns:
            格式化的搜索结果列表
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
            
            # 格式化结果
            formatted = []
            for result_list in results:
                text = ""
                for idx, doc in enumerate(result_list):
                    content = doc.get("document", {}).get("contents", "")
                    # 分离标题和正文
                    lines = content.split("\n")
                    title = lines[0] if lines else ""
                    body = "\n".join(lines[1:]) if len(lines) > 1 else ""
                    text += f"文档 {idx+1}(标题: {title}) {body}\n"
                formatted.append(text)
            
            return formatted
            
        except Exception as e:
            print(f"[搜索错误] {e}")
            return ["搜索失败，请尝试不同的查询。"] * len(queries)
    
    # ==================== 状态管理 ====================
    
    def _update_rolling_state(
        self,
        rollings: DataProto,
        cur_responses: torch.Tensor,
        next_obs_ids: torch.Tensor
    ) -> DataProto:
        """用新的响应和观察更新滚动状态
        
        Args:
            rollings: 当前滚动状态
            cur_responses: 当前响应tokens
            next_obs_ids: 下一个观察tokens
            
        Returns:
            更新后的DataProto
        """
        # 带填充拼接
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])
        
        # 创建attention mask和position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)
        
        # 截断到最大长度
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
        """带info masking的张量拼接，用于RL梯度
        
        创建两个版本:
        1. 包含所有内容的完整张量
        2. info块被替换为pad token的masked张量
        
        这允许在RL梯度计算中排除搜索结果。
        
        Args:
            prompt: Prompt tokens
            prompt_with_mask: Prompt tokens（用于masking版本）
            response: Response tokens
            info: Information/observation tokens（可选）
            pad_to_left: 是否左填充
            
        Returns:
            (完整张量, masked张量) 元组
        """
        pad_id = self.tokenizer.pad_token_id
        
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            # 创建info mask（全是pad tokens）
            info_mask = torch.full(
                info.size(), 
                pad_id, 
                dtype=info.dtype, 
                device=info.device
            )
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_masked = torch.cat(tensors_with_mask, dim=1)
        
        # 按填充位置排序
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
        """更新右侧（响应）状态
        
        Args:
            right_side: 当前右侧状态
            cur_responses: 当前响应tokens
            next_obs_ids: 下一个观察tokens（可选）
            
        Returns:
            更新后的右侧状态
        """
        responses, responses_masked = self._info_masked_concatenate(
            right_side['responses'],
            right_side['responses_with_info_mask'],
            cur_responses,
            next_obs_ids,
            pad_to_left=False
        )
        
        # 截断到最大长度
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len],
            'responses_with_info_mask': responses_masked[:, :max_len]
        }
    
    # ==================== GPU处理 ====================
    
    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """带多GPU填充处理的生成
        
        当batch size不能被num_gpus整除时，用第一个序列填充。
        
        Args:
            active_batch: 要生成的批次
            
        Returns:
            生成的输出
        """
        num_gpus = self.config.num_gpus
        
        # 转换为long类型
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # 添加填充序列
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)
        
        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()
        
        # 生成并移除填充
        output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        for k, v in output.batch.items():
            output.batch[k] = v[:batch_size]
        
        return output
    
    # ==================== 主生成循环 ====================
    
    def run_generation_loop(
        self,
        gen_batch: DataProto
    ) -> Dict[str, Any]:
        """运行完整的多轮生成循环
        
        实现R1-RAG核心生成流程:
        1. 为所有样本初始化状态
        2. 为活跃样本生成响应
        3. 解析动作（search/answer）
        4. 执行搜索并注入结果
        5. 重复直到完成或达到最大轮次
        
        Args:
            gen_batch: 初始DataProto批次
            
        Returns:
            包含输出的字典:
            - prompts: 原始prompt tokens
            - responses: 生成的response tokens
            - responses_with_info_mask: info块被masked的响应
            - attention_mask: 完整的attention mask
            - statistics: 轮次计数、搜索计数等
        """
        batch_size = gen_batch.batch['input_ids'].shape[0]
        device = gen_batch.batch['input_ids'].device
        
        # 初始化跟踪
        active_mask = [True] * batch_size
        original_prompts = gen_batch.batch['input_ids'].clone()
        
        # 初始化右侧（响应累加器）
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
        
        # 统计信息
        turn_counts = [0] * batch_size
        search_counts = [0] * batch_size
        valid_action_counts = [0] * batch_size
        
        # 当前滚动状态
        rollings = gen_batch
        
        # 主生成循环
        for turn in range(self.config.max_turns):
            if not any(active_mask):
                break
            
            # 为活跃样本生成响应
            gen_output = self._generate_with_gpu_padding(rollings)
            cur_responses = gen_output.batch['responses']
            
            # 后处理响应
            cur_responses, responses_str = self._postprocess_responses(cur_responses)
            
            # 解析动作并收集搜索查询
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
                    # 最终答案 - 标记完成
                    dones[i] = True
                elif action_type == "search":
                    # 加入搜索队列
                    search_indices.append(i)
                    search_queries.append(content)
                    search_counts[i] += 1
                else:
                    # 无效动作 - 提供指导
                    observations[i] = (
                        "\n我之前的动作无效。"
                        "我应该使用<search>查询</search>来搜索，"
                        "或使用<answer>结果</answer>给出最终答案。"
                        "让我重试。\n"
                    )
                
                turn_counts[i] += 1
            
            # 批量执行搜索
            if search_queries:
                results = self.execute_search(search_queries)
                for idx, result in zip(search_indices, results):
                    observations[idx] = f"\n\n<information>{result.strip()}</information>\n\n"
            
            # 处理观察
            obs_ids = self._process_observations(observations)
            
            # 更新右侧（响应累加器）
            right_side = self._update_right_side(
                right_side,
                cur_responses,
                obs_ids if any(not d for d in dones) else None
            )
            
            # 更新滚动状态用于下一轮
            rollings = self._update_rolling_state(rollings, cur_responses, obs_ids)
            
            # 更新活跃掩码
            active_mask = [not d for d in dones]
        
        # 为剩余活跃样本进行最终生成
        if any(active_mask):
            gen_output = self._generate_with_gpu_padding(rollings)
            final_responses = gen_output.batch['responses']
            final_responses, _ = self._postprocess_responses(final_responses)
            
            right_side = self._update_right_side(right_side, final_responses)
        
        # 准备最终输出
        responses = right_side['responses']
        responses_masked = right_side['responses_with_info_mask']
        
        # 创建完整的attention mask
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
    
    # ==================== 简单步骤接口 ====================
    
    def step(
        self,
        responses: List[str],
        active_mask: List[bool]
    ) -> Tuple[List[str], List[bool], List[bool]]:
        """执行生成循环的一步
        
        用于外部控制的简化接口。
        
        Args:
            responses: 每个样本的模型响应
            active_mask: 哪些样本仍然活跃
            
        Returns:
            (观察, 完成标志, 有效动作标志) 元组
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
                observations.append(None)  # 占位符
                dones.append(False)
                valid_actions.append(True)
            else:
                observations.append(
                    "\n我之前的动作无效。"
                    "我应该使用<search>查询</search>来搜索，"
                    "或使用<answer>结果</answer>给出最终答案。"
                    "让我重试。\n"
                )
                dones.append(False)
                valid_actions.append(False)
        
        # 执行搜索
        if search_queries:
            results = self.execute_search(search_queries)
            for idx, result in zip(search_indices, results):
                observations[idx] = f"\n\n<information>{result.strip()}</information>\n\n"
        
        return observations, dones, valid_actions
