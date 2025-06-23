from abc import ABC, abstractmethod

from pydantic import BaseModel


class AbstractDiffusionGenerator(ABC):
    """
    Abstract class for "diffusion generators".

    A diffusion generator is any interface that can generate a target pydantic model based on a prompt.
    """

    @abstractmethod
    def generate(self, prompt: str, generation_config: GenerationConfig, target_pydantic_model: BaseModel) -> BaseModel:
        pass
