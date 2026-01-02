# R1-RAG: Reasoning-First Retrieval-Augmented Generation

<div align="center">

**A Reinforcement Learning Framework for Multi-Hop Question Answering with Explicit Global Planning**

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

## 📖 Overview

R1-RAG is a reinforcement learning framework designed to enhance **global reasoning** in multi-hop question answering. Unlike traditional RAG systems that perform retrieval reactively, R1-RAG teaches language models to:

1. **Plan Globally**: Decompose complex questions into structured sub-goals (DAG)
2. **Execute Reliably**: Perform coordinated retrieval and reasoning
3. **Learn from Process**: Use dense supervision on intermediate steps

### Key Innovations

- **DAG-based Planning Structure**: Explicit dependency modeling between sub-questions
- **Dual Reward Mechanism**: 
  - Structural reward via Graph Edit Distance (GED)
  - Semantic reward via E5 embedding similarity
- **Progressive Weight Annealing**: Smooth transition from process to outcome focus
- **GPT-4o Annotation Pipeline**: Automated generation of high-quality golden plans

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
git clone https://github.com/your-username/R1-RAG.git
cd R1-RAG

# Create environment
conda create -n r1rag python=3.9
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
conda create -n retriever python=3.10
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

### Training Sample Structure

```python
{
    "question": "What is the population of the birthplace of Albert Einstein?",
    "golden_answers": ["128,000"],
    "metadata": {
        "hop": "2hop",
        "plan": {
            "Q1": ["Where was Albert Einstein born?", "<A1>"],
            "Q2": ["What is the population of <A1>?", "<A2>"]
        },
        "graph": [{
            "Q1": {"answer": "Ulm"},
            "Q2": {"answer": "128,000"}
        }]
    }
}
```

### Expected Model Output

```xml
<think> This is a two-hop question. I need to find Einstein's birthplace first. </think>
<plan>
{"Q1": ["Where was Albert Einstein born?", "#1"], "Q2": ["What is the population of #1?", "#2"]}
</plan>

<subPlan>
    <think> Let me search for Einstein's birthplace. </think>
    <search> Albert Einstein birthplace </search>
    <information> Albert Einstein was born in Ulm, Germany... </information>
    <think> Einstein was born in Ulm. </think>
    <subAnswer> #1 = Ulm </subAnswer>
</subPlan>

<subPlan>
    <think> Now I need the population of Ulm. </think>
    <search> population of Ulm Germany </search>
    <information> Ulm has a population of approximately 128,000... </information>
    <think> The population is 128,000. </think>
    <subAnswer> #2 = 128,000 </subAnswer>
</subPlan>

<think> I have all the information needed. </think>
<answer> 128,000 </answer>
```

## 🔧 Configuration

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_turns` | 4 | Maximum search iterations |
| `n_agent` | 5 | Samples per prompt for GRPO |
| `format_weight` | 0.1 | Weight for format compliance |
| `plan_sim_weight` | 0.5 | Weight for semantic similarity |
| `structure_weight` | 0.5 | Weight for structural match |
| `step_weight` | 0.5 | Weight for sub-goal completion |
| `annealing_steps` | 50 | Steps for weight annealing |

### Model Support

- **Qwen2.5-3B-Instruct** (default, recommended for efficiency)
- Qwen2.5-7B-Instruct
- Llama-3.2-3B-Instruct
- Llama-3.1-8B-Instruct

## 📈 Results

Performance on multi-hop QA benchmarks:

| Dataset | EM | F1 | Improvement |
|---------|----|----|-------------|
| HotpotQA | 42.3 | 54.7 | +12.1 |
| 2WikiMultihopQA | 38.9 | 48.2 | +14.5 |
| Musique | 21.4 | 29.8 | +8.7 |
| Bamboogle | 45.2 | 52.8 | +15.1 |

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
```

## 🔬 Technical Details

### Progressive Weight Annealing

The annealing function smoothly transitions training focus:

```python
α(t) = 1 / (1 + exp((t - 0.9T) / 10))

# Early training (t << 0.9T): α ≈ 1
#   → Focus on learning planning structure
# Late training (t > 0.9T): α → 0  
#   → Focus on answer correctness
```

### Graph Edit Distance (GED)

Measures structural similarity between planning DAGs:

```python
GED = |V_pred ⊕ V_gold| + |E_pred ⊕ E_gold|
normalized_GED = exp(-β * GED)
```

### E5 Semantic Scoring

Matches sub-questions using embedding similarity:

```python
similarity = cos_sim(E5(pred_question), E5(gold_question))
# Threshold: 0.7 for valid match
```

## 📝 Citation

If you use R1-RAG in your research, please cite:

```bibtex
@article{r1rag2024,
  title={R1-RAG: Reasoning-First Retrieval-Augmented Generation with Global Planning},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 🙏 Acknowledgements

This project builds upon several excellent open-source works:

- [veRL](https://github.com/volcengine/verl) - RL training framework
- [Search-R1](https://github.com/PeterGriffinJin/Search-R1) - Reasoning-search interleaving
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) - E5 embeddings
- [NetworkX](https://networkx.org/) - Graph algorithms

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

