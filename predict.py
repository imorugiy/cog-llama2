import time
import torch
import socket
from typing import Any
from cog import BasePredictor, ConcatenateIterator, File, Input, Path
from cog.types import Path as CogPath
import numpy as np

import config
from config import ENGINE, ENGINE_KWARGS, USE_SYSTEM_PROMPT

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{prompt}} {E_INST}"
if not USE_SYSTEM_PROMPT:
    PROMPT_TEMPLATE = "{prompt}"
PROMPT_TEMPLATE = getattr(config, "PROMPT_TEMPLATE", PROMPT_TEMPLATE)

DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant."""
DEFAULT_SYSTEM_PROMPT = getattr(config, "DEFAULT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)


class Predictor(BasePredictor):
    def setup(self):
        print("Starting setup")

        self.engine = ENGINE(**ENGINE_KWARGS)

    def predict(
        self,
        prompt: str = Input(description="Prompt to send to the model."),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior. Should not be blank.",
            default=DEFAULT_SYSTEM_PROMPT,
        ),
        max_new_tokens: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=128,
        ),
        min_new_tokens: int = Input(
            description="Minimum number of tokens to generate. To disable, set to -1. A word is generally 2-3 tokens.",
            ge=-1,
            default=-1,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.7,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.0,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="When decoding text, samples from the top k most likely tokens; lower to ignore less likely tokens",
            ge=-1,
            default=-1,
        ),
        repetition_penalty: float = Input(
            description="A parameter that controls how repetitive text can be. Lower means more repetitive, while higher means less repetitive. Set to 1.0 to disable.",
            ge=0.0,
            default=1.15,
        ),
        stop_sequences: str = Input(
            description="A comma-separated list of sequences to stop generation at. For example, '<end>,<stop>' will stop generation at the first instance of 'end' or '<stop>'.",
            default=None,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None,
        ),
        debug: bool = Input(
            description="provide debugging output in logs", default=False
        ),
        prompt_template: str = Input(
            description="Template for formatting the prompt",
            default=PROMPT_TEMPLATE,
        ),
    ) -> ConcatenateIterator[str]:

        n_tokens = 0
        st = time.time()

        generated_text = ""
        for decoded_token in self.engine(
            prompt,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            stop_sequences=stop_sequences,
        ):
            n_tokens += 1
            yield decoded_token
            generated_text += decoded_token
            if n_tokens == 1 and debug:
                second_start = time.time()
            if seed is not None:
                torch.manual_seed(seed)
            et = time.time()
            t = et - st
            print(f"hostname: {socket.gethostname()}")
