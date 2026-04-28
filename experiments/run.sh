#!/bin/bash
#SBATCH --job-name=term-definition-extraction
#SBATCH --output=../logs/%x_%A_%a.out
#SBATCH --error=../logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=250G
#SBATCH --gres=gpu:2
#SBATCH --array=0-19  # <-- number of models
#SBATCH --mail-type=END --mail-user=necva.bolucu@csiro.au -A OD-240335


export HOME=/scratch3/bol107
export XDG_CACHE_HOME=/scratch3/bol107/.cache

export HF_HOME=$XDG_CACHE_HOME/huggingface
export TORCH_HOME=$XDG_CACHE_HOME/torch
export VLLM_CACHE_ROOT=$XDG_CACHE_HOME/vllm
export TRITON_CACHE_DIR=$XDG_CACHE_HOME/triton

# IMPORTANT: avoid shared memory issues
export TMPDIR=/scratch3/bol107/tmp/$SLURM_JOB_ID
mkdir -p $TMPDIR
export PYTORCH_TMPDIR=$TMPDIR

# multiprocessing safety
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export VLLM_USE_FLASHINFER=0


MODELS=(
"../../models/Qwen3-14B"
"../../models/Qwen3-32B"
 "../../models/gpt-oss-20b"
"../../models/gpt-oss-120b"
"../../models/Moonlight-16B-A3B-Instruct"
"../../models/DeepSeek-R1-Distill-Llama-70B"
"../../models/DeepSeek-R1-Distill-Llama-8B"
"../../models/DeepSeek-R1-Distill-Qwen-14B"
"../../models/DeepSeek-R1-Distill-Qwen-32B"
"../../models/AM-Thinking-v1"
"../../models/Kimi-Linear-48B-A3B-Instruct"
"../../models/gemma-4-31B-it"
"../../models/gemma-4-E2B-it"
"../../models/gemma-4-E4B-it"
"../../models/Olmo-3-7B-Think"
"../../models/Olmo-3-32B-Think"
"../../models/gemma-3-1b-it"
"../../models/gemma-3-4b-it"
"../../models/gemma-3-12b-it"
"../../models/gemma-3-27b-it"
)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
python run_experiment.py \
    --data   "../dataset/mlonlysum_majority.json" \
    --model  "$MODEL" \
    --output "../results/$(basename $MODEL)_guideline.json" \
    --eval_output "../results/$(basename $MODEL)_eval_guideline.json" \
    --guideline "../guideline.md"
