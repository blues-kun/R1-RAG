"""
Document Indexer for R1-RAG

Builds FAISS indices from document corpora:
1. Load documents from various formats
2. Encode with E5 model
3. Build and save FAISS index

Supports large-scale indexing with batching.
"""

import os
import json
import argparse
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm

import numpy as np
import torch


class DocumentIndexer:
    """Builds dense retrieval indices from document corpora.
    
    Workflow:
    1. Load documents (JSONL, JSON, or HuggingFace)
    2. Batch encode with E5 model
    3. Build FAISS index (Flat, IVF, or HNSW)
    4. Save index and corpus mapping
    """
    
    def __init__(
        self,
        model_path: str = "intfloat/e5-base-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 64,
    ):
        """Initialize document indexer.
        
        Args:
            model_path: E5 model path or identifier
            device: Device for encoding
            batch_size: Batch size for encoding
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        
        self._encoder = None
        self._embedding_dim = None
    
    @property
    def encoder(self):
        """Lazy load E5 encoder."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.model_path, device=self.device)
            self._embedding_dim = self._encoder.get_sentence_embedding_dimension()
        return self._encoder
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        _ = self.encoder  # Ensure loaded
        return self._embedding_dim
    
    def load_corpus_jsonl(self, path: str) -> List[Dict]:
        """Load corpus from JSONL file.
        
        Expected format per line:
        {"id": "doc1", "contents": "Document text...", "title": "Optional Title"}
        
        Args:
            path: Path to JSONL file
            
        Returns:
            List of document dicts
        """
        documents = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading corpus"):
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
        """Load corpus from HuggingFace dataset.
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split
            text_field: Field containing document text
            id_field: Field containing document ID
            
        Returns:
            List of document dicts
        """
        from datasets import load_dataset
        
        dataset = load_dataset(dataset_name, split=split)
        
        documents = []
        for idx, item in enumerate(tqdm(dataset, desc="Loading HF corpus")):
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
        """Encode documents into dense vectors.
        
        Args:
            documents: List of document dicts
            text_field: Field containing text to encode
            
        Returns:
            Document embeddings of shape (N, dim)
        """
        # Extract texts with passage prefix for E5
        texts = [f"passage: {doc.get(text_field, '')}" for doc in documents]
        
        # Batch encode
        all_embeddings = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch = texts[i:i + self.batch_size]
            embeddings = self.encoder.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype(np.float32)
    
    def build_flat_index(self, embeddings: np.ndarray):
        """Build flat (exact) FAISS index.
        
        Best for small corpora (<100K documents).
        
        Args:
            embeddings: Document embeddings
            
        Returns:
            FAISS index
        """
        import faiss
        
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product (cosine after normalization)
        index.add(embeddings)
        
        return index
    
    def build_ivf_index(
        self,
        embeddings: np.ndarray,
        nlist: int = 100,
        nprobe: int = 10,
    ):
        """Build IVF index for faster search.
        
        Good for medium corpora (100K-10M documents).
        
        Args:
            embeddings: Document embeddings
            nlist: Number of clusters
            nprobe: Number of clusters to search
            
        Returns:
            FAISS index
        """
        import faiss
        
        dim = embeddings.shape[1]
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train on embeddings
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
        """Build HNSW index for fast approximate search.
        
        Good for large corpora (>10M documents).
        
        Args:
            embeddings: Document embeddings
            M: Number of connections per element
            ef_construction: Construction time accuracy
            
        Returns:
            FAISS index
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
        """Build and save complete index.
        
        Args:
            corpus_path: Path to corpus file
            output_dir: Directory to save index and corpus
            index_type: Index type (flat, ivf, hnsw)
            text_field: Field containing document text
            
        Returns:
            Paths to saved files
        """
        import faiss
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load corpus
        print(f"[Indexer] Loading corpus from {corpus_path}")
        if corpus_path.endswith('.jsonl'):
            documents = self.load_corpus_jsonl(corpus_path)
        else:
            documents = self.load_corpus_huggingface(corpus_path)
        
        print(f"[Indexer] Loaded {len(documents)} documents")
        
        # Encode documents
        print("[Indexer] Encoding documents...")
        embeddings = self.encode_documents(documents, text_field)
        
        # Build index
        print(f"[Indexer] Building {index_type} index...")
        if index_type == "flat":
            index = self.build_flat_index(embeddings)
        elif index_type == "ivf":
            nlist = min(100, len(documents) // 100)
            index = self.build_ivf_index(embeddings, nlist=nlist)
        elif index_type == "hnsw":
            index = self.build_hnsw_index(embeddings)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Save index
        index_path = os.path.join(output_dir, f"e5_{index_type.capitalize()}.index")
        faiss.write_index(index, index_path)
        print(f"[Indexer] Saved index to {index_path}")
        
        # Save corpus
        corpus_output = os.path.join(output_dir, "corpus.jsonl")
        with open(corpus_output, 'w', encoding='utf-8') as f:
            for doc in documents:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        print(f"[Indexer] Saved corpus to {corpus_output}")
        
        return {
            "index": index_path,
            "corpus": corpus_output,
            "num_documents": len(documents),
        }


def main():
    """CLI for building indices."""
    parser = argparse.ArgumentParser(description="R1-RAG Document Indexer")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, default="intfloat/e5-base-v2", help="E5 model")
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
    print("Indexing complete!")
    print(f"  Documents: {result['num_documents']}")
    print(f"  Index: {result['index']}")
    print(f"  Corpus: {result['corpus']}")


if __name__ == "__main__":
    main()

