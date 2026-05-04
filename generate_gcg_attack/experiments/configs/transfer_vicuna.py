import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "DIR/hugging_face_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
        "DIR/hugging_face_cache/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2"
    ]
    config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.model_paths = [
        "DIR/hugging_face_cache/hub/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d",
        "DIR/hugging_face_cache/hub/models--lmsys--vicuna-13b-v1.5/snapshots/c8327bf999adbd2efe2e75f6509fa01436100dc2"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    config.conversation_templates = ["vicuna", "vicuna"]
    config.devices = ["cuda:0", "cuda:1"]
    # config.devices = ["cuda:0"]

    return config
