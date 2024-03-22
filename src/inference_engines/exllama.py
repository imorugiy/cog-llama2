import os
import glob
import torch
import time
from src.config_utils import Weights

from .engine import Engine
from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
from exllama.tokenizer import ExLlamaTokenizer
from exllama.generator import ExLlamaGenerator


torch.cuda._lazy_init()
torch.set_printoptions(precision=10)


def next_logits(
    generator, input_ids, apply_lora=None, last_id_only=True, input_mask=None
):
    n_logits = generator.model.forward(
        input_ids, generator.cache, last_id_only, lora=apply_lora, input_mask=input_mask
    )
    return n_logits


def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret


def begin(generator):
    if generator.cache is None:
        generator.cache = ExLlamaCache(generator.model)
    else:
        generator.cache_current_seq_len = 0
    return generator


class ExllamaEngine(Engine):
    def __init__(self, weights: Weights, fused_attn=True):
        model_directory = self.load_weights(weights)
        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path

        config.max_seq_len = 2 * 2048
        config.max_input_len = 2 * 2048
        config.max_attention_size = 2 * 2048**2
        config.fused_attn = fused_attn

        self.model = model = ExLlama(config)
        tokenizer = ExLlamaTokenizer(tokenizer_path)
        cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, cache)

        warmup_ids = torch.randint(0, 31999, (1, 50)).cuda()
        print("warming up exllama kernels...")
        for i in range(1, 3):
            print(f" -- Warmup pass {i}...")
            begin(generator)
            logits = timer("Warmup", lambda: next_logits(generator, warmup_ids, None))

        self.generator = begin(generator)
