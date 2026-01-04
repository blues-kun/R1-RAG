# R1-RAG: Learning to Plan in Retrieval with GRPO-Optimized Thinking 

<div align="center">

**通过 GRPO 优化的检索增强生成与高效思考学习规划**

</div>

## 📖 概述

R1-RAG 是一个强化学习框架，旨在增强多跳问答中的**规划推理**能力。与传统的被动执行检索的RAG系统不同，R1-RAG 教导语言模型：

1. **规划推理**: 将复杂问题分解为结构化的子目标（DAG）
2. **可靠执行**: 执行协调的检索和推理
3. **从过程中学习**: 在中间步骤上使用密集监督

### 核心创新

- **基于DAG的规划结构**: 子问题之间的显式依赖建模
- **双重奖励机制**: 
  - 通过图编辑距离（GED）的结构奖励
  - 通过E5嵌入相似度的语义奖励
- **渐进式权重退火**: 从过程关注到结果关注的平滑过渡
- **GPT-4o标注流水线**: 自动生成高质量的黄金规划

## 🏗️ Architecture

```
Question → [Planning DAG] → [Sub-Goal Execution] → Answer
              ↓                    ↓
         <plan>               <subPlan>
         Q1 → Q2 → Q3         search → info → subAnswer
              ↓
         [GRPO Training]
              ↓
         R = α(t)·R_process + R_outcome
```

### Reward Design

```python
R_total = α(t) * R_process + R_outcome

where:
  R_process = w_f * format_score      # Format compliance
            + w_s * semantic_score    # E5 embedding similarity  
            + w_g * structure_score   # Graph Edit Distance
            + w_step * step_score     # Sub-goal F1 completion
            
  R_outcome = exact_match(pred, gold)  # Final answer correctness
  
  α(t) = 1 / (1 + exp((t - 0.9T) / 10)) # Progressive annealing
```

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/blues-kun/R1-RAG.git
cd R1-RAG

# Create environment
conda create -n r1rag python=3.12
conda activate r1rag

# Install PyTorch (adjust for your CUDA version)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install R1-RAG
pip install -e .
```

### Optional: Retriever Environment

```bash
# For local retrieval server
conda create -n retriever python=3.12
conda activate retriever

conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install uvicorn fastapi
```

## 🚀 Quick Start

### 1. Prepare Data with Golden Plans

```bash
# Option A: Generate new annotations with GPT-4o
python scripts/prepare_data.py \
    --dataset hotpotqa \
    --generate_annotations \
    --api_key YOUR_OPENAI_KEY \
    --output_dir data/r1_rag

# Option B: Use pre-annotated data
python scripts/prepare_data.py \
    --input_file path/to/annotated_data.jsonl \
    --output_dir data/r1_rag
```

### 2. Start Retrieval Server

```bash
# Download index and corpus
python scripts/download_index.py --save_path data/retriever

# Launch server
conda activate retriever
python -m r1_rag.retriever.server \
    --index_path data/retriever/e5_Flat.index \
    --corpus_path data/retriever/wiki-18.jsonl \
    --port 8000
```

### 3. Train with GRPO

```bash
conda activate r1rag
bash scripts/train_grpo.sh
```

## 📊 Data Format

> **注意**: 本项目的训练数据使用**英文数据集**（HotpotQA, 2WikiMultihopQA, Musique等）。以下示例使用中文仅是为了便于理解数据格式和模型输出结构。实际训练时，所有问题、答案和规划均为英文。

### Training Sample Structure

```python
{
    "question": "《泰坦尼克号》的导演的第一部电影？",
    "golden_answers": ["Piranha II: The Spawning", "食人鱼2"],
    "metadata": {
        "hop": "2hop",
        "plan": {
            "Q1": ["谁执导了《泰坦尼克号》？", "<A1>"],
            "Q2": ["<A1>的第一部电影是什么？", "<A2>"]
        },
        "graph": [{
            "Q1": {"answer": "詹姆斯·卡梅隆"},
            "Q2": {"answer": "食人鱼2"}
        }]
    }
}
```

### Expected Model Output

> **说明**: 以下为中文示例，实际训练和推理使用英文数据。

```xml
<think> 这个问题需要分两步：首先找到《泰坦尼克号》的导演，然后查询该导演的第一部电影。 </think>
<plan>
{"Q1": ["谁执导了《泰坦尼克号》？", "#1"], "Q2": ["#1的第一部电影是什么？", "#2"]}
</plan>

<subPlan>
    <think> 先搜索《泰坦尼克号》的导演信息。 </think>
    <search> 泰坦尼克号 导演 </search>
    <information> 《泰坦尼克号》是1997年上映的史诗级爱情灾难片，由詹姆斯·卡梅隆执导... </information>
    <think> 根据检索结果，导演是詹姆斯·卡梅隆。 </think>
    <subAnswer> #1 = 詹姆斯·卡梅隆 </subAnswer>
</subPlan>

<subPlan>
    <think> 现在需要查询詹姆斯·卡梅隆的导演处女作。 </think>
    <search> 詹姆斯·卡梅隆 第一部电影 导演处女作 </search>
    <information> 詹姆斯·卡梅隆的导演处女作是1982年的《食人鱼2：繁殖》(Piranha II: The Spawning)... </information>
    <think> 他的第一部电影是《食人鱼2》。 </think>
    <subAnswer> #2 = 食人鱼2 </subAnswer>
</subPlan>

<think> 已获取所有子问题的答案，可以给出最终结果。 </think>
<answer> 食人鱼2 </answer>
```




## 🗂️ Project Structure

```
R1_RAG/
├── r1_rag/
│   ├── reward/                 # DAG-based reward computation
│   │   ├── config.py           # Reward configuration
│   │   ├── dag_evaluator.py    # Main reward evaluator
│   │   ├── semantic_scorer.py  # E5 embedding similarity
│   │   └── structure_scorer.py # Graph Edit Distance
│   ├── data/                   # Data processing
│   │   ├── processor.py        # Dataset processing
│   │   ├── gpt4o_annotator.py  # Golden plan generation
│   │   └── prompts.py          # Prompt templates
│   └── agent/                  # Generation loop
│       └── generation_manager.py
├── scripts/
│   ├── train_grpo.sh           # Training script
│   └── prepare_data.py         # Data preparation
├── configs/
│   └── grpo_qwen_3b.yaml       # Training config
└── requirements.txt



## 🙏 Acknowledgements

This project builds upon several excellent open-source works:

- [veRL](https://github.com/volcengine/verl) - RL training framework
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - Reasoning-search interleaving
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - E5 embeddings
- [NetworkX](https://networkx.org/) - Graph algorithms


