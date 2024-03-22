from dotenv import load_dotenv
from src.config_utils import Weights, get_gptq_file_list, exllama_kwargs, vllm_kwargs
from src.utils import get_env_var_or_default
from src.inference_engines.vllm_exllama_engine import ExllamaVllmEngine

load_dotenv()

MODEL_NAME = "llama-2-7b"

# Inference weights

exllama_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/default_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_DEFAULT_INFERENCE_WEIGHTS_PATH", None),
    remote_files=get_gptq_file_list("gptq_model-4bit-128g.safetensors"),
)

vllm_weights = Weights(
    local_path=f"models/{MODEL_NAME}/model_artifacts/lora_inference_weights",
    remote_path=get_env_var_or_default("REMOTE_VLLM_INFERENCE_WEIGHTS_PATH", None),
    remote_files=[],
)

# Inference config

TOKENIZER_PATH = f"models/{MODEL_NAME}/model_artifacts/default_inference_weights"
USE_SYSTEM_PROMPT = False

ENGINE = ExllamaVllmEngine
exllama_kw = exllama_kwargs(exllama_weights)
vllm_kw = vllm_kwargs(vllm_weights)

ENGINE_KWARGS = {"exllama_args": exllama_kw, "vllm_args": vllm_kw}
