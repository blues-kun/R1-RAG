"""
R1-RAG GPT-4o规划标注生成器

使用GPT-4o生成高质量的规划DAG标注:
1. 将多跳问题分解为子问题
2. 为每个子问题生成中间答案
3. 对照ground truth进行质量过滤验证

创建用于RL训练过程监督的"黄金规划"。
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
    """GPT-4o规划标注的结果"""
    question: str
    gold_answer: List[str]
    plan: Optional[Dict[str, List[str]]]
    graph: Optional[Dict[str, Dict]]
    is_valid: bool
    error_message: Optional[str] = None


class GPT4oPlanGenerator:
    """使用GPT-4o生成规划DAG标注
    
    关键设计决策:
    1. 使用ground truth来引导和验证标注
    2. 实现重试逻辑以提高鲁棒性
    3. 过滤低质量标注
    4. 支持并行批处理
    
    生成的规划用作"黄金标签"用于:
    - 语义相似度评分（E5嵌入）
    - 结构相似度评分（GED）
    - 子目标完成度评分（F1）
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
        """初始化GPT-4o标注器
        
        Args:
            api_key: OpenAI API密钥
            model: 模型名称（推荐gpt-4o）
            max_retries: API调用的最大重试次数
            retry_delay: 重试之间的延迟（秒）
            temperature: 采样温度（越低越确定）
            max_workers: 批处理的并行worker数
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("请安装openai: pip install openai")
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_workers = max_workers
        
        # 从响应中提取JSON的模式
        self.json_pattern = re.compile(r'\{[\s\S]*\}')
    
    def _call_gpt4o(self, prompt: str) -> Optional[str]:
        """带重试逻辑的GPT-4o API调用
        
        Args:
            prompt: 要发送的提示
            
        Returns:
            响应文本，如果所有重试都失败则返回None
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个为多跳问题生成结构化推理规划的助手。"},
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
                    print(f"[GPT4o] {self.max_retries}次尝试后失败: {e}")
                    return None
        return None
    
    def _parse_response(self, response: str) -> Tuple[Optional[Dict], Optional[Dict]]:
        """解析GPT-4o响应以提取plan和graph
        
        Args:
            response: 原始GPT-4o响应
            
        Returns:
            (plan字典, graph字典) 元组，失败时返回(None, None)
        """
        try:
            # 在响应中查找JSON
            match = self.json_pattern.search(response)
            if not match:
                return None, None
            
            data = json.loads(match.group())
            plan = data.get("plan", {})
            graph = data.get("graph", {})
            
            # 验证结构
            if not plan or not graph:
                return None, None
            
            # 规范化占位符
            normalized_plan = {}
            for key, value in plan.items():
                if isinstance(value, list) and len(value) >= 2:
                    # 将 #N 转换为 <AN>
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
        """验证标注是否导向正确答案
        
        检查:
        1. 规划至少有一个子问题
        2. 图中包含所有规划问题的答案
        3. 图中的最终答案与黄金答案匹配
        
        Args:
            plan: 生成的规划
            graph: 生成的执行图
            gold_answer: Ground truth答案
            
        Returns:
            如果标注有效则返回True
        """
        if not plan or not graph:
            return False
        
        # 检查所有规划问题都有答案
        for q_key in plan.keys():
            if q_key not in graph:
                return False
            if "answer" not in graph[q_key]:
                return False
        
        # 检查最终答案是否匹配黄金答案（宽松匹配）
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
        """为单个问题生成规划标注
        
        Args:
            question: 多跳问题
            gold_answer: 可接受的黄金答案列表
            
        Returns:
            包含plan和graph的AnnotationResult（或错误信息）
        """
        # 格式化prompt
        prompt = GPT4O_PLAN_ANNOTATION_PROMPT.format(
            question=question,
            gold_answer=gold_answer[0] if gold_answer else "N/A"
        )
        
        # 调用GPT-4o
        response = self._call_gpt4o(prompt)
        if not response:
            return AnnotationResult(
                question=question,
                gold_answer=gold_answer,
                plan=None,
                graph=None,
                is_valid=False,
                error_message="GPT-4o API调用失败"
            )
        
        # 解析响应
        plan, graph = self._parse_response(response)
        if not plan or not graph:
            return AnnotationResult(
                question=question,
                gold_answer=gold_answer,
                plan=None,
                graph=None,
                is_valid=False,
                error_message="解析GPT-4o响应失败"
            )
        
        # 验证标注
        is_valid = self._validate_annotation(plan, graph, gold_answer)
        
        return AnnotationResult(
            question=question,
            gold_answer=gold_answer,
            plan=plan,
            graph=[graph],  # 包装为列表以兼容
            is_valid=is_valid,
            error_message=None if is_valid else "标注验证失败"
        )
    
    def generate_batch(
        self,
        samples: List[Dict[str, Any]],
        question_key: str = "question",
        answer_key: str = "golden_answers"
    ) -> List[AnnotationResult]:
        """为一批样本生成标注
        
        使用并行处理以提高效率。
        
        Args:
            samples: 包含问题和答案的样本字典列表
            question_key: 问题字段的键
            answer_key: 黄金答案字段的键
            
        Returns:
            AnnotationResult列表
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
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="生成标注"):
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
        
        # 按原始索引排序
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]
    
    def filter_valid_annotations(
        self,
        results: List[AnnotationResult]
    ) -> List[AnnotationResult]:
        """过滤只保留有效标注
        
        Args:
            results: 标注结果列表
            
        Returns:
            只包含有效标注的过滤列表
        """
        valid = [r for r in results if r.is_valid]
        print(f"[GPT4o] 有效标注: {len(valid)}/{len(results)} ({100*len(valid)/len(results):.1f}%)")
        return valid
