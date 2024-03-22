import os
import sys
import glob
import torch
import time
from src.config_utils import Weights

# exllama_path = os.path.abspath("exllama")
# sys.path.insert(0, exllama_path)

from .engine import Engine
from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler


class ExllamaV2Engine(Engine):
    def __init__(self, weights: Weights, fused_attn=True):
        model_directory = self.load_weights(weights)

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        model = ExLlamaV2(config)
        print("Loading model: " + model_directory)

        cache = ExLlamaV2Cache(model, lazy=True)
        model.load_autosplit(cache)

        tokenizer = ExLlamaV2Tokenizer(config)
        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

        generator.warmup()
        self.generator = generator

    def __call__(
        self,
        prompt: str,
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.8,
        token_repetition_penalty: float = 1.01,
        max_new_tokens: int = 150,
    ):
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.token_repetition_penalty = token_repetition_penalty

        output = self.generator.generate_simple(
            prompt=prompt, settings=settings, num_tokens=max_new_tokens, seed=1234
        )

        return output
