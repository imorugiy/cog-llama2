import time

from abc import ABC, abstractmethod
from typing import Any

from src.config_utils import Weights
from src.utils import maybe_download_with_pget


class Engine(ABC):

    def load_weights(self, weights: Weights):
        start = time.time()
        maybe_download_with_pget(
            weights.local_path, weights.remote_path, weights.remote_files
        )
        print(f"downloading weights took {time.time() - start:.3f}s")
        return weights.local_path

    @abstractmethod
    def __call__(self, prompt, **kwargs):
        pass
