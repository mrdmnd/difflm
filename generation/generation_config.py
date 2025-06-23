from dataclasses import dataclass
from typing import Literal


@dataclass
class GenerationConfig:
    """
    This class contains all configuration information for a diffusion model generation, given a prompt and a target.
    """

    temperature: float = 0.2
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    mask_id: int = 126336
    min_avg_block_confidence: float = 0.85
    max_steps_per_block: int = 15
    tokens_to_unmask_per_step: int = 1
