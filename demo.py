import json
from typing import Any, Dict, List, Union

import numpy as np
import pydantic


# This would be replaced with your actual diffusion model
class MockDiffusionModel:
    def __init__(self):
        pass

    def generate(self, prompt: str, mask: np.ndarray, template: str) -> str:
        """Simulate diffusion model generating content for masked regions"""
        # In a real implementation, this would call your diffusion model
        # with appropriate conditioning on the prompt and respect for the mask

        # For demo purposes, we'll just insert placeholder text
        result = template
        mask_indices = np.where(mask == 1)[0]

        # Group consecutive indices to find masked regions
        if len(mask_indices) > 0:
            groups = []
            current_group = [mask_indices[0]]

            for i in range(1, len(mask_indices)):
                if mask_indices[i] == mask_indices[i - 1] + 1:
                    current_group.append(mask_indices[i])
                else:
                    groups.append(current_group)
                    current_group = [mask_indices[i]]

            groups.append(current_group)

            # Replace masked regions with generated content
            # In reverse order to avoid index shifting
            for group in reversed(groups):
                start, end = group[0], group[-1] + 1
                # This is where your actual diffusion model would generate text
                generated_text = f"[Generated from '{prompt}' at pos {start}-{end}]"
                result = result[:start] + generated_text + result[end:]

        return result


class StructuredOutputEngine:
    def __init__(self, diffusion_model):
        self.diffusion_model = diffusion_model

    def create_template_and_mask(self, model: pydantic.BaseModel) -> tuple:
        """Create a JSON template with placeholders and corresponding mask"""
        schema = model.model_json_schema()

        def process_schema(schema_part):
            if schema_part.get("type") == "object":
                result = {}
                for prop_name, prop_schema in schema_part.get("properties", {}).items():
                    result[prop_name] = process_schema(prop_schema)
                return result
            elif schema_part.get("type") == "array":
                # For arrays, we'll create a single example item
                return [process_schema(schema_part.get("items", {}))]
            elif schema_part.get("type") in ["string", "number", "integer", "boolean"]:
                # For leaf nodes, insert a placeholder
                return "<MASK>"
            else:
                # Default case
                return "<MASK>"

        template_obj = process_schema(schema)
        template_str = json.dumps(template_obj, indent=2)
        print(template_str)

        # Create mask - 1 for positions to be filled by diffusion model, 0 otherwise
        mask = np.zeros(len(template_str), dtype=np.int8)

        # Mark all "<MASK>" positions in the template
        mask_token = "<MASK>"
        start_pos = 0
        while True:
            pos = template_str.find(mask_token, start_pos)
            if pos == -1:
                break
            mask[pos : pos + len(mask_token)] = 1
            start_pos = pos + len(mask_token)

        return template_str, mask

    def generate(self, prompt: str, schema: Union[Dict, pydantic.BaseModel]) -> Dict[str, Any]:
        """Generate structured output based on prompt and schema"""
        template_str, mask = self.create_template_and_mask(schema)

        # Use diffusion model to fill in the masked regions
        filled_json_str = self.diffusion_model.generate(prompt, mask, template_str)

        # Clean up the output by removing mask tokens if any remain
        filled_json_str = filled_json_str.replace("<MASK>", "")

        # Parse the result back to a Python dictionary
        try:
            result = json.loads(filled_json_str)
            return result
        except json.JSONDecodeError:
            # In a real implementation, you might want to retry or adjust parameters
            print("Warning: Generated output is not valid JSON")
            return {"error": "Failed to generate valid JSON"}


# Example usage with a Pydantic model
class PersonInfo(pydantic.BaseModel):
    name: str
    age: int
    bio: str
    skills: List[str]


# Create the engine with our mock diffusion model
engine = StructuredOutputEngine(MockDiffusionModel())

# Generate structured output
prompt = "Create a profile for a software engineer named Sarah who specializes in machine learning"
result = engine.generate(prompt, PersonInfo)

print(json.dumps(result, indent=2))
