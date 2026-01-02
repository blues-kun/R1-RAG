"""
R1-RAG 文档索引器

从文档语料库构建FAISS索引:
1. 从各种格式加载文档
2. 使用E5模型编码
3. 构建并保存FAISS索引

支持带批处理的大规模索引。
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm

import numpy as np
import torch


class DocumentIndexer:
    """从文档语料库构建稠密检索索引
    
    工作流程:
    1. 加载文档（JSONL、JSON或HuggingFace）
    2. 使用E5模型批量编码
    3. 构建FAISS索引（Flat、IVF或HNSW）
    4. 保存索引和语料库映射
    """
    
    def __init__(
        self,
        model_path: str = "intfloat/e5-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ):
        """初始化文档索引器
        
        Args:
            model_path: E5模型路径或标识符
            device: 编码设备
            batch_size: 编码批次大小
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        
        self._encoder = None
        self._embedding_dim = None
    
    @property
    def encoder(self):
        """延迟加载E5编码器"""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_path, device=self.device)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()
        return self._encoder
    
    @property
    def embedding_dim(self) -> int:
        """获取嵌入维度"""
        _ = self.encoder  # 确保已加载
        return self._embedding_dim
    
    def load_corpus_jsonl(self, path: str) -> List[Dict]:
        """从JSONL文件加载语料库
        
        每行预期格式:
        {"id": "doc1", "contents": "文档文本...", "title": "可选标题"}
        
        Args:
            path: JSONL文件路径
            
        Returns:
            文档字典列表
        """
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="加载语料库"):
                doc = json.loads(line.strip())
                documents.append(doc)
        return documents
    
    def load_corpus_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        id_field: str = "id",
    ) -> List[Dict]:
        """从HuggingFace数据集加载语料库
        
        Args:
            dataset_name: HuggingFace数据集标识符
            split: 数据集分片
            text_field: 包含文档文本的字段
            id_field: 包含文档ID的字段
            
        Returns:
            文档字典列表
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        documents = []
        for idx, item in enumerate(tqdm(dataset, desc="加载HF语料库")):
            doc = {
                "id": item.get(id_field, str(idx)),
                "contents": item.get(text_field, ""),
            }
            if "title" in item:
                doc["title"] = item["title"]
            documents.append(doc)
        
        return documents
    
    def encode_documents(
        self,
        documents: List[Dict],
        text_field: str = "contents",
    ) -> np.ndarray:
        """将文档编码为稠密向量
        
        Args:
            documents: 文档字典列表
            text_field: 包含要编码文本的字段
            
        Returns:
            形状为 (N, dim) 的文档嵌入
        """
        # 提取文本，为E5添加passage前缀
        texts = [f"passage: {doc.get(text_field, '')}" for doc in documents]
        
        # 批量编码
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="编码中"):
            batch = texts[i:i + self.batch_size]
            embeddings = self.encoder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def build_flat_index(self, embeddings: np.ndarray):
        """构建flat（精确）FAISS索引
        
        适用于小型语料库（<100K文档）。
        
        Args:
            embeddings: 文档嵌入
            
        Returns:
            FAISS索引
        """
        import faiss
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # 内积（归一化后等于余弦）
        index.add(embeddings)
        
        return index
    
    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """构建IVF索引以加快搜索
        
        适用于中型语料库（100K-10M文档）。
        
        Args:
            embeddings: 文档嵌入
            nlist: 聚类数量
            nprobe: 搜索的聚类数量
            
        Returns:
            FAISS索引
        """
        import faiss
        
        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # 在嵌入上训练
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = nprobe
        
        return index
    
    def build_hnsw_index(
        self,
        embeddings: np.ndarray,
        M: int = 32,
        ef_construction: int = 200,
    ):
        """构建HNSW索引用于快速近似搜索
        
        适用于大型语料库（>10M文档）。
        
        Args:
            embeddings: 文档嵌入
            M: 每个元素的连接数
            ef_construction: 构建时的准确度
            
        Returns:
            FAISS索引
        """
        import faiss
        
        dim = embeddings.shape[1]
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(embeddings)
        
        return index
    
    def build_index(
        self,
        corpus_path: str,
        output_dir: str,
        index_type: str = "flat",
        text_field: str = "contents",
    ) -> Dict[str, str]:
        """构建并保存完整索引
        
        Args:
            corpus_path: 语料库文件路径
            output_dir: 保存索引和语料库的目录
            index_type: 索引类型（flat, ivf, hnsw）
            text_field: 包含文档文本的字段
            
        Returns:
            保存文件的路径
        """
        import faiss
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载语料库
        print(f"[索引器] 从 {corpus_path} 加载语料库")
        if corpus_path.endswith('.jsonl'):
            documents = self.load_corpus_jsonl(corpus_path)
        else:
            documents = self.load_corpus_huggingface(corpus_path)
        
        print(f"[索引器] 加载了 {len(documents)} 个文档")
        
        # 编码文档
        print("[索引器] 编码文档中...")
        embeddings = self.encode_documents(documents, text_field)
        
        # 构建索引
        print(f"[索引器] 构建 {index_type} 索引中...")
        if index_type == "flat":
            index = self.build_flat_index(embeddings)
        elif index_type == "ivf":
            nlist = min(100, len(documents) // 100)
            index = self.build_ivf_index(embeddings, nlist=nlist)
        elif index_type == "hnsw":
            index = self.build_hnsw_index(embeddings)
        else:
            raise ValueError(f"未知索引类型: {index_type}")
        
        # 保存索引
        index_path = os.path.join(output_dir, f"e5_{index_type.capitalize()}.index")
        faiss.write_index(index, index_path)
        print(f"[索引器] 索引已保存到 {index_path}")
        
        # 保存语料库
        corpus_output = os.path.join(output_dir, "corpus.jsonl")
        with open(corpus_output, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"[索引器] 语料库已保存到 {corpus_output}")
        
        return {
            "index": index_path,
            "corpus": corpus_output,
            "num_documents": len(documents),
        }


def main():
    """构建索引的CLI"""
    parser = argparse.ArgumentParser(description="R1-RAG 文档索引器")
    parser.add_argument("--corpus_path", type=str, required=True, help="语料库文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, default="intfloat/e5-base-v2", help="E5模型")
    parser.add_argument("--index_type", type=str, default="flat", choices=["flat", "ivf", "hnsw"])
    parser.add_argument("--batch_size", type=int, default=64)
    
    args = parser.parse_args()
    
    indexer = DocumentIndexer(
        model_path=args.model_path,
        batch_size=args.batch_size,
    )
    
    result = indexer.build_index(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        index_type=args.index_type,
    )
    
    print("\n" + "="*50)
    print("索引构建完成!")
    print(f"  文档数: {result['num_documents']}")
    print(f"  索引: {result['index']}")
    print(f"  语料库: {result['corpus']}")


if __name__ == "__main__":
    main()
