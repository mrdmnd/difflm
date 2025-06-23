from pydantic import BaseModel

from generation.abstraction_diffusion_generator import AbstractDiffusionGenerator
from generation.generation_config import GenerationConfig


class MockDiffusionGenerator(AbstractDiffusionGenerator):
    def generate(self, prompt: str, generation_config: GenerationConfig, target_pydantic_model: BaseModel) -> BaseModel:
        return target_pydantic_model.model_validate_json(prompt)
