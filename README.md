# MULDER: A Multilingual Dataset for Term-Definition Extraction in Scientific Literature



## How to run
```bash
    python run_experiments.py \
        --model                  "$MODEL" \
        --data                   "../dataset/all_experiments.json" \
        --arxiv_data             "../dataset/arxiv_experiments.json" \
        --outdir                 "$OUTDIR" \
        --exps                   $EXPS \
        --guideline              "../guideline.md" \
        --gpu_memory_utilization "${GPU_MEM_UTIL:-0.9}" \
        --max_model_len          "${MAX_MODEL_LEN:-32768}"

```
## Results

### EXP1

### EXP2

### EXP3

### EXP4

### EXP5

### EXP6
