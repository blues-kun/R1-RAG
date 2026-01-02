"""
R1-RAG 检索服务器

基于FastAPI的检索服务:
1. 加载预构建的FAISS索引
2. 处理批量检索请求
3. 返回top-k相关文档

针对多轮RAG训练优化。
"""

import os
import json
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


# ============== 请求/响应模型 ==============

class RetrievalRequest(BaseModel):
    """检索端点的请求模型"""
    queries: List[str]
    topk: int = 3
    return_scores: bool = True


class DocumentResult(BaseModel):
    """单个文档结果"""
    docid: str
    contents: str
    score: Optional[float] = None


class RetrievalResponse(BaseModel):
    """检索端点的响应模型"""
    result: List[List[Dict[str, Any]]]
    query_count: int


# ============== 检索服务器 ==============

class RetrievalServer:
    """基于E5的稠密检索服务器
    
    特性:
    - 使用E5模型进行高效批量编码
    - FAISS索引用于快速相似度搜索
    - 从JSONL加载文档语料库
    - 可用时GPU加速
    """
    
    def __init__(
        self,
        index_path: str,
        corpus_path: str,
        model_path: str = "intfloat/e5-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """初始化检索服务器
        
        Args:
            index_path: FAISS索引文件路径
            corpus_path: 文档语料库JSONL路径
            model_path: E5模型标识符或路径
            device: 编码设备（cuda/cpu）
        """
        self.device = device
        self.model_path = model_path
        
        # 加载组件
        print(f"[检索器] 加载E5模型: {model_path}")
        self._load_encoder()
        
        print(f"[检索器] 加载索引: {index_path}")
        self._load_index(index_path)
        
        print(f"[检索器] 加载语料库: {corpus_path}")
        self._load_corpus(corpus_path)
        
        print(f"[检索器] 就绪! 语料库大小: {len(self.corpus)}")
    
    def _load_encoder(self):
        """加载E5 sentence transformer模型"""
        from sentence_transformers import SentenceTransformer
        
        self.encoder = SentenceTransformer(self.model_path, device=self.device)
    
    def _load_index(self, index_path: str):
        """从文件加载FAISS索引"""
        import faiss
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"索引文件未找到: {index_path}")
        
        self.index = faiss.read_index(index_path)
        
        # 如果可用则移到GPU
        if self.device == "cuda":
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                print("[检索器] 索引已移至GPU")
            except Exception as e:
                print(f"[检索器] GPU索引失败，使用CPU: {e}")
    
    def _load_corpus(self, corpus_path: str):
        """从JSONL文件加载文档语料库"""
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(f"语料库文件未找到: {corpus_path}")
        
        self.corpus = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line.strip())
                self.corpus.append(doc)
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """将查询编码为稠密向量
        
        Args:
            queries: 查询字符串列表
            
        Returns:
            形状为 (N, dim) 的查询嵌入
        """
        # E5要求查询添加"query: "前缀
        prefixed = [f"query: {q}" for q in queries]
        embeddings = self.encoder.encode(
            prefixed,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)
    
    def search(
        self,
        queries: List[str],
        topk: int = 3,
        return_scores: bool = True,
    ) -> List[List[Dict[str, Any]]]:
        """搜索相关文档
        
        Args:
            queries: 查询字符串列表
            topk: 每个查询的结果数量
            return_scores: 是否包含相似度分数
            
        Returns:
            结果列表的列表，每个查询一个
        """
        # 编码查询
        query_vectors = self.encode_queries(queries)
        
        # 搜索索引
        scores, indices = self.index.search(query_vectors, topk)
        
        # 格式化结果
        results = []
        for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            for score, doc_idx in zip(query_scores, query_indices):
                if doc_idx < 0 or doc_idx >= len(self.corpus):
                    continue
                    
                doc = self.corpus[doc_idx]
                result = {
                    "docid": doc.get("id", str(doc_idx)),
                    "document": {
                        "contents": doc.get("contents", doc.get("text", "")),
                    }
                }
                if return_scores:
                    result["score"] = float(score)
                    
                query_results.append(result)
            
            results.append(query_results)
        
        return results


# ============== FastAPI应用 ==============

app = FastAPI(
    title="R1-RAG 检索服务器",
    description="用于多跳问答的E5稠密检索",
    version="1.0.0",
)

# 全局服务器实例
_server: Optional[RetrievalServer] = None


def get_server() -> RetrievalServer:
    """获取全局检索服务器实例"""
    if _server is None:
        raise HTTPException(
            status_code=503, 
            detail="检索服务器未初始化"
        )
    return _server


@app.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(request: RetrievalRequest):
    """检索查询的相关文档
    
    Args:
        request: 包含查询和参数的RetrievalRequest
        
    Returns:
        包含搜索结果的RetrievalResponse
    """
    server = get_server()
    
    try:
        results = server.search(
            queries=request.queries,
            topk=request.topk,
            return_scores=request.return_scores,
        )
        
        return RetrievalResponse(
            result=results,
            query_count=len(request.queries),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "corpus_size": len(_server.corpus) if _server else 0}


# ============== CLI入口 ==============

def main():
    """从命令行运行检索服务器"""
    global _server
    
    parser = argparse.ArgumentParser(description="R1-RAG 检索服务器")
    parser.add_argument("--index_path", type=str, required=True, help="FAISS索引路径")
    parser.add_argument("--corpus_path", type=str, required=True, help="语料库JSONL路径")
    parser.add_argument("--model_path", type=str, default="intfloat/e5-base-v2", help="E5模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机")
    parser.add_argument("--port", type=int, default=8000, help="服务器端口")
    
    args = parser.parse_args()
    
    # 初始化服务器
    _server = RetrievalServer(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        model_path=args.model_path,
    )
    
    # 运行FastAPI
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
