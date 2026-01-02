#!/usr/bin/env python3
"""
Download Pre-built Retrieval Index for R1-RAG

Downloads:
1. E5 FAISS index (wiki-18 corpus)
2. Corpus documents (JSONL format)

Usage:
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
    print("Please install requests: pip install requests")
    exit(1)


# Pre-built index URLs (replace with actual hosting)
INDEX_URLS = {
    "wiki18_e5_flat": {
        "index": "https://huggingface.co/datasets/GenIRAG/search-r1-retriever/resolve/main/e5_Flat.index",
        "corpus": "https://huggingface.co/datasets/GenIRAG/search-r1-retriever/resolve/main/wiki-18.jsonl.gz",
    },
    # Add more index options here
}


def download_file(url: str, dest_path: str, desc: str = None):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def decompress_gzip(gz_path: str, output_path: str):
    """Decompress gzip file."""
    print(f"Decompressing {gz_path}...")
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(gz_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Download R1-RAG retrieval index")
    parser.add_argument(
        "--save_path",
        type=str,
        default="data/retriever",
        help="Directory to save index and corpus"
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="wiki18_e5_flat",
        choices=list(INDEX_URLS.keys()),
        help="Index to download"
    )
    parser.add_argument(
        "--skip_corpus",
        action="store_true",
        help="Skip corpus download (if already have it)"
    )
    
    args = parser.parse_args()
    
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    urls = INDEX_URLS[args.index_name]
    
    print("=" * 60)
    print("R1-RAG Index Downloader")
    print("=" * 60)
    print(f"Index: {args.index_name}")
    print(f"Save path: {save_path}")
    print()
    
    # Download index
    index_path = save_path / "e5_Flat.index"
    if not index_path.exists():
        print("[1/2] Downloading FAISS index...")
        download_file(urls["index"], str(index_path), "Index")
    else:
        print(f"[1/2] Index already exists: {index_path}")
    
    # Download corpus
    if not args.skip_corpus:
        corpus_path = save_path / "wiki-18.jsonl"
        if not corpus_path.exists():
            print("[2/2] Downloading corpus...")
            gz_path = save_path / "wiki-18.jsonl.gz"
            download_file(urls["corpus"], str(gz_path), "Corpus")
            decompress_gzip(str(gz_path), str(corpus_path))
        else:
            print(f"[2/2] Corpus already exists: {corpus_path}")
    
    print()
    print("=" * 60)
    print("Download complete!")
    print()
    print("To start the retrieval server:")
    print(f"  python -m r1_rag.retriever.server \\")
    print(f"      --index_path {save_path}/e5_Flat.index \\")
    print(f"      --corpus_path {save_path}/wiki-18.jsonl \\")
    print(f"      --port 8000")
    print("=" * 60)


if __name__ == "__main__":
    main()

