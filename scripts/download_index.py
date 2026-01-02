#!/usr/bin/env python3
"""
R1-RAG 预构建检索索引下载

下载:
1. E5 FAISS索引（wiki-18语料库）
2. 语料库文档（JSONL格式）

用法:
    python scripts/download_index.py --save_path data/retriever
"""

import os
import argparse
import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

try:
    import requests
except ImportError:
    print("请安装requests: pip install requests")
    exit(1)


# 预构建索引URL（替换为实际托管地址）
INDEX_URLS = {
    "wiki18_e5_flat": {
        "index": "https://huggingface.co/datasets/GenIRAG/search-r1-retriever/resolve/main/e5_Flat.index",
        "corpus": "https://huggingface.co/datasets/GenIRAG/search-r1-retriever/resolve/main/wiki-18.jsonl.gz",
    },
    # 在这里添加更多索引选项
}


def download_file(url: str, dest_path: str, desc: str = None):
    """带进度条下载文件"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def decompress_gzip(gz_path: str, output_path: str):
    """解压gzip文件"""
    print(f"解压 {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"已保存到 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="下载R1-RAG检索索引")
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/retriever",
        help="保存索引和语料库的目录"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="wiki18_e5_flat",
        choices=list(INDEX_URLS.keys()),
        help="要下载的索引"
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="跳过语料库下载（如果已有）"
    )
    
    args = parser.parse_args()
    
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    urls = INDEX_URLS[args.index_name]
    
    print("=" * 60)
    print("R1-RAG 索引下载器")
    print("=" * 60)
    print(f"索引: {args.index_name}")
    print(f"保存路径: {save_path}")
    print()
    
    # 下载索引
    index_path = save_path / "e5_Flat.index"
    if not index_path.exists():
        print("[1/2] 下载FAISS索引...")
        download_file(urls["index"], str(index_path), "索引")
    else:
        print(f"[1/2] 索引已存在: {index_path}")
    
    # 下载语料库
    if not args.skip_corpus:
        corpus_path = save_path / "wiki-18.jsonl"
        if not corpus_path.exists():
            print("[2/2] 下载语料库...")
            gz_path = save_path / "wiki-18.jsonl.gz"
            download_file(urls["corpus"], str(gz_path), "语料库")
            decompress_gzip(str(gz_path), str(corpus_path))
        else:
            print(f"[2/2] 语料库已存在: {corpus_path}")
    
    print()
    print("=" * 60)
    print("下载完成!")
    print()
    print("启动检索服务器:")
    print(f"  python -m r1_rag.retriever.server \\")
    print(f"      --index_path {save_path}/e5_Flat.index \\")
    print(f"      --corpus_path {save_path}/wiki-18.jsonl \\")
    print(f"      --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()
