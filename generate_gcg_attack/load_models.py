from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "google/gemma-3-27b-it" #lmsys/vicuna-13b-v1.5

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


load_attempts = [
    dict(
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ),
    dict(
        load_in_8bit=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ),
    dict(
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ),
]

last_err = None
for i, kwargs in enumerate(load_attempts, 1):
    try:
        print(f"[load_models] Attempt {i}: {kwargs}")
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        print("[load_models] Model loaded successfully.")
        break
    except Exception as e:
        last_err = e
        print(f"[load_models] Attempt {i} failed: {e}")
        model = None

if model is None:
    raise RuntimeError(
        "Failed to load model with all memory-safe strategies. "
        "Check transformers/bitsandbytes/cuda compatibility."
    ) from last_err
