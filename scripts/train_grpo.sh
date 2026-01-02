#!/bin/bash
# R1-RAG Training Script
# GRPO with DAG-based Process Supervision
# Base Model: Qwen2.5-3B-Instruct

set -e

# ============== Configuration ==============
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR="data/r1_rag"

# Experiment settings
export WANDB_PROJECT="R1-RAG"
export EXPERIMENT_NAME="qwen2.5-3b-instruct-grpo-dag"

# Model settings
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export E5_MODEL_PATH="intfloat/e5-base-v2"

# vLLM backend fix for Qwen
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============== Training ==============
echo "Starting R1-RAG training..."
echo "Model: $BASE_MODEL"
echo "Experiment: $EXPERIMENT_NAME"

PYTHONUNBUFFERED=1 python3 -m r1_rag.trainer.main_grpo \
    data.train_files="[$DATA_DIR/train.parquet]" \
    data.val_files="$DATA_DIR/val.parquet" \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=256 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.max_start_length=2048 \
    data.max_obs_length=600 \
    data.shuffle_train_dataloader=true \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=64 \
    actor_rollout_ref.rollout.n_agent=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=64 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    algorithm.no_think_rl=false \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.total_training_steps=1000 \
    trainer.save_freq=50 \
    trainer.test_freq=25 \
    trainer.val_only=false \
    trainer.val_before_train=true \
    trainer.default_local_dir="checkpoints/$EXPERIMENT_NAME" \
    trainer.default_hdfs_dir=null \
    max_turns=4 \
    +e5_model_path=$E5_MODEL_PATH \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee "logs/${EXPERIMENT_NAME}.log"

echo "Training completed!"

