import gc
from typing import Any, Optional, List

import torch
import os

from .engine import Engine
from .exllama import ExllamaEngine


class ExllamaVllmEngine(Engine):

    def __init__(self, vllm_args: dict, exllama_args: dict) -> None:

        self.engine = ExllamaEngine(**exllama_args)
        self.vllm_args = vllm_args
