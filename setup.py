"""
Setup script for R1-RAG
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="r1_rag",
    version="1.0.0",
    author="Kunbo Xu",
    author_email="blues924@outlook.com",
    description="Reasoning-First RAG with Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/blues-kun/R1-RAG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.4.0",
        "transformers>=4.40.0",
        "sentence-transformers>=2.5.0",
        "networkx>=3.2.0",
        "python-Levenshtein>=0.25.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "requests>=2.31.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "train": [
            "vllm>=0.6.3",
            "ray>=2.10.0",
            "hydra-core>=1.3.0",
            "wandb>=0.16.0",
            "flash-attn>=2.5.0",
        ],
        "retriever": [
            "faiss-gpu>=1.8.0",
            "pyserini>=0.22.0",
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
        ],
        "annotation": [
            "openai>=1.12.0",
        ],
    },
)

