### Setup

Key environment dependencies: 

- Python 3.11
- PyTorch 2.7.1 (CUDA 11.8)
- `transformers==4.28.1`
- `accelerate`, `peft`, `datasets`
- `fschat==0.2.20` (FastChat, for conversation templates)
- `openai`, `anthropic` (for proprietary model API calls)
- `vllm` (for efficient batched inference)
- `matplotlib`, `numpy`, `scipy`

### Running the Attack

```bash
# AdvBench
python autodan_hga_eval.py \
    --dataset_path ./data/advbench/harmful_behaviors.csv \
    --save_suffix advbench

# HarmBench (standard)
python autodan_hga_eval.py \
    --dataset_path ./data/harmbench/harmbench_standard_and_targets.csv \
    --save_suffix harmbench_standard


Generated attack strings are saved as JSON files under `results/autodan_hga/`.