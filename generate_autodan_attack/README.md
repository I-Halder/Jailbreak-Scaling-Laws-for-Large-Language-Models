### Setup

```bash
conda env create -f torch26_llm_attack.yaml
conda activate torch26_llm_attack
```

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