# Jailbreak Scaling Laws for Large Language Models: Polynomial-Exponential Crossover

This repository contains the code accompanying our paper `Jailbreak Scaling Laws for Large Language Models: Polynomial-Exponential Crossover`. It covers the full pipeline: generating adversarial attack strings (GCG and AutoDAN), querying victim models with *k*-sample inference-time sampling, judging the responses to compute attack success rates (ASR), and the theoretical spin-glass analysis that explains the observed scaling behavior.

---

## Repository Structure

```
jailbreak-scaling-laws/
├── datasets/                        # Prompt datasets with harm categories
├── generate_gcg_attack/             # Generate GCG adversarial suffix strings
├── generate_autodan_attack/         # Generate AutoDAN adversarial suffix strings
├── generate_score_victim_response/  # Query victim models and compute ASR vs k using a judge model
├── generation_results/              # Output folder for generated model responses
├── lm_eval/                         # lm-evaluation-harness framework (extended)
└── spin_glass_theory_results/       # Theoretical & numerical spin-glass results
```

---

## Datasets

**Path:** `datasets/`

Contains CSV files with harmful prompts drawn from multiple benchmarks, annotated with harm categories.


---

## Generating GCG Attack Strings

**Path:** `generate_gcg_attack/`

Generates adversarial suffixes using the Greedy Coordinate Gradient (GCG) attack on open-source victim models. Built on top of the `[llm-attacks](https://github.com/llm-attacks/llm-attacks)` library. See the README specific to that directory for detailed instructions and usage. 

---

## Generating AutoDAN Attack Strings

**Path:** `generate_autodan_attack/`

Generates adversarial jailbreak prompts using the AutoDAN genetic algorithm (HGA variant). Built on top of the `[AutoDAN](https://github.com/SheltonLiu-N/AutoDAN)` library. See the README specific to that directory for detailed instructions and usage.

---

## Generating Victim Responses and Computing ASR vs *k*

**Path:** `generate_score_victim_response/`

This is the core module of the paper. It queries victim models using *k*-sample generation at inference time (i.e., generating *k* independent responses per prompt), judges each response for jailbreak success, and computes the attack success rate as a function of *k*. We define attack success by whether at least of the k generations were jailbroken. 

### Scripts


| Script                             | Description                                          |
| ---------------------------------- | ---------------------------------------------------- |
| `compute_asr_vs_k_gcg.py`          | GCG attack; open-source models via vLLM              |
| `compute_asr_vs_k_gcg_api.py`      | GCG attack; proprietary models via API (e.g., GPT-4) |
| `compute_asr_vs_k_autodan.py`      | AutoDAN attack; open-source models via vLLM          |
| `compute_asr_vs_k_autodan_api.py`  | AutoDAN attack; proprietary models via API           |
| `compute_asr_vs_k_categorywise.py` | ASR vs *k* broken down by harm category              |
| `prompts_asr_distribution.py`      | Per-prompt ASR distribution and log-log analysis     |


Generated model responses are saved under `generation_results/`.

### Example Usage

**GCG attack, open-source model:**

```bash
python generate_score_victim_response/compute_asr_vs_k_gcg.py \
    --k_min 1 --k_max 6 --k_step 4 \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.9" \
    --output_path ./generation_results/ \
    --tasks advbench \
    --yaml_path "./lm_eval/tasks/advbench/advbench.yaml" \
    --use_judge_model \
    --limit 0.01
```

**AutoDAN attack, open-source model:**

```bash
python generate_score_victim_response/compute_asr_vs_k_autodan.py \
    --k_min 1 --k_max 6 --k_step 4 \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.9" \
    --output_path ./generation_results/ \
    --tasks advbench_high_level_injection \
    --yaml_path "./lm_eval/tasks/advbench/advbench_high_level_injection.yaml" \
    --use_judge_model --limit 0.01 \
    --jailbreak_file "vicuna_0_advbench.json" \
    --jailbreak_prompt_field "goal" \
    --jailbreak_attack_field "final_suffix"
```

**GCG attack, proprietary model (API):**

```bash
python generate_score_victim_response/compute_asr_vs_k_gcg_api.py \
    --k_min 1 --k_max 3 --k_step 4 \
    --model_args "model=gpt-4-turbo" \
    --output_path ./generation_results/ \
    --tasks advbench \
    --yaml_path "./lm_eval/tasks/advbench/advbench.yaml" \
    --use_judge_model \
    --limit 0.01
```

**AutoDAN attack, proprietary model (API):**

```bash
python generate_score_victim_response/compute_asr_vs_k_autodan_api.py \
    --k_min 1 --k_max 6 --k_step 4 \
    --model_args "pretrained=meta-llama/Meta-Llama-3-8B-Instruct,tensor_parallel_size=1,data_parallel_size=1,gpu_memory_utilization=0.9" \
    --output_path ./generation_results/ \
    --tasks advbench_high_level_injection \
    --yaml_path "./lm_eval/tasks/advbench/advbench_high_level_injection.yaml" \
    --use_judge_model --limit 0.01 \
    --jailbreak_file "vicuna_0_advbench.json" \
    --jailbreak_prompt_field "goal" \
    --jailbreak_attack_field "final_suffix"
```

**Prompt category-wise ASR computation:**
Note that in this case, we expect that the victim model responses have already been generated using the codes mentioned above. 

```bash
python generate_score_victim_response/compute_asr_vs_k_categorywise.py \
    --k_min 1 --k_max 10 --k_step 4 \
    --output_path generation_results \
    --tasks advbench_high_level_injection \
    --categories_csv datasets/harmful_behaviors_with_categories_advbench.csv
```

**Per-prompt ASR distribution:**

```bash
python generate_score_victim_response/prompts_asr_distribution.py \
    --root_dir "./generation_results/" \
    --scores_glob "**/*task_advbench_run*/**/judge_scores_k*.json" \
    --bin_width 0.01 \
    --output_png ./per_prompt_analysis/prompts_P_distribution.png \
    --output_loglog_png ./per_prompt_analysis/prompts_logP_distribution.png \
    --output_json ./per_prompt_analysis/prompts_asr_distribution_summary.json
```

### Judging

By default the scripts use a **refusal-string detector** to label responses. Pass `--use_judge_model` to enable LLM-based judging (using a separate judge model such as Mistral-7B-Instruct). Results are saved as `.jsonl` and `.json` files in `generation_results/`.

---

## LM Evaluation Harness

**Path:** `lm_eval/`

An extended version of the `[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)` framework. Custom task configurations for the supported datasets are located in:

Advbench dataset: 

```
lm_eval/tasks/advbench/
    advbench.yaml
    advbench_high_level_injection.yaml
```

Harmbench dataset (standard prompts): 

```
lm_eval/tasks/harmbench_standard/
    harmbench_standard.yaml
    harmbench_standard_high_level_injection.yaml
```

These YAML configs are referenced by the `--yaml_path` argument in the generation/scoring scripts.

---

Each subdirectory contains Python scripts that produce publication-ready figures. A cross-model category-wise plot script is at the top level:

```bash
python plot_jailbreak_results/plot_asr_categorywise.py
```

---

## Spin-Glass Theory Results

**Path:** `spin_glass_theory_results/`

Contains the code for the theoretical analysis and numerical verification based on spin-glass theory, which explains the power-law scaling of ASR with *k*.

### Files


| File                                                                     | Description                                                                                               |
| ------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| `spin-glass-theory_multigpu.py`                                          | Main simulation: finite-*N* spin-glass model with disorder averaging, supports multi-GPU via `torchrun`   |
| `plot_loggap_from_logs.py`                                               | Plots \log(-\log(\Pi_k)) vs \log k numerical results and compares with theoretical results from Theorem 4 |
| `plot_loggap_from_logs_curvefit_constrained_loglogpi.py`                 | Plots \log(-\log(\Pi_k)) vs \log k with curve fitting                                                     |
| `spin-glass-theory-multigpu-N24-disorder1024-m1-curvefit_logs.json`      | Pre-computed simulation logs (N=24, 1024 disorder samples)                                                |


### Example Usage

**Multi-GPU**

```bash
torchrun --nproc_per_node=2 spin-glass-theory_multigpu.py \
  --N 24 --beta 10.0 --j0 1.0 --m_unsafe 1 \
  --h_values 0,0.05,0.1,0.15,0.2 \
  --k_values 1,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64 \
  --n_disorder 1024 --n_sel 8 \
  --pd_B 8 --pd_num_perms 32 \
  --device cuda --threads 1 \
  --max_states_per_rank 20000000 \
  --out ./spin-glass-theory-multigpu-N24-disorder1024-m1-curvefit.png
```

**Plotting simulation logs:**

```bash
python spin_glass_theory_results/plot_loggap_from_logs.py \
    --logs spin_glass_theory_results/spin-glass-theory-multigpu-N24-disorder1024-m1-curvefit_logs.json \
    --out spin_glass_theory_results/figure5A.png \
```

```bash
python spin_glass_theory_results/plot_loggap_from_logs_curvefit_constrained_loglogpi.py \
    --logs spin_glass_theory_results/spin-glass-theory-multigpu-N24-disorder1024-m1-curvefit_logs.json \
    --out spin_glass_theory_results/figure5B.png \
```

---

## Environment Setup

Key dependencies:

- Python 3.10.15
- PyTorch 2.9.0 (CUDA 12.8)
- `transformers==4.57.3`
- `accelerate==1.9.0`, `peft==0.17.0`, `datasets==3.1.0`
- `fschat==0.2.23` 
- `openai==2.31.0`, `anthropic==0.71.0` 
- `vllm==0.11.2`
- `matplotlib==3.10.0`, `numpy==2.2.6`, `scipy==1.15.2`

### API Keys

Set the following environment variables before running API-based scripts:

```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export HF_TOKEN="your-huggingface-token"
export HF_HOME="/path/to/hf/cache"
```


