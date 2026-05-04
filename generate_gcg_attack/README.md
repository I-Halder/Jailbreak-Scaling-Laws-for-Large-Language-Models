### Setup

```bash
cd generate_gcg_attack
pip install -r requirements.txt
pip install -e .
```

### Running the Attack

```bash
python experiments/main.py \
    --config <path_to_config>
```

Example code for generating the attack string: `.experiment//launch_scripts/run_gcg_transfer.sh`

### Evaluating Harmfulness of Generated Responses

```bash
# Evaluate using a judge LLM (e.g., GPT-4)
python experiments/eval_harmfullness.py \
    --input_file <path_to_generated_responses.json> \
    --judge_model gpt-4
```

Additional evaluation scripts:
- `experiments/evaluate_proprietary_gcg.py` — GCG injection evaluation to proprietary models (GPT, Claude)
- `experiments/evaluate_proprietary_suffix.py` — "Sure here's " injection evaluation
- `experiments/evaluate_proprietary_baseline.py` — baseline no injection transfer


Key dependencies:

- Python 3.11
- PyTorch 2.7.1 (CUDA 11.8)
- `transformers==4.28.1`
- `accelerate`, `peft`, `datasets`
- `fschat==0.2.20` (FastChat, for conversation templates)
- `openai`, `anthropic` (for proprietary model API calls)
- `vllm` (for efficient batched inference)
- `matplotlib`, `numpy`, `scipy`


For the GCG attack module specifically:

```bash
cd generate_gcg_attack
pip install -r requirements.txt   # transformers==4.28.1, ml_collections, fschat==0.2.20
pip install -e .
```