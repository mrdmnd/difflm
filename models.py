from enum import StrEnum

from pydantic import BaseModel


class NoseColor(StrEnum):
    RED = "red"
    GREEN = "green"
    RAINBOW = "rainbow"


class EnchantmentType(StrEnum):
    FIRE = "fire"
    ICE = "ice"
    LIGHTNING = "lightning"


class EquipmentType(StrEnum):
    SWORD = "sword"
    SHIELD = "shield"
    RING = "ring"


class Equipment(BaseModel):
    equipment_type: EquipmentType
    fantasy_name: str
    enchantments: list[EnchantmentType] | None = None


class ClownModel(BaseModel):
    shoe_size: int
    nose_color: NoseColor
    inventory: list[Equipment] | None = None


if __name__ == "__main__":
    from polyfactory.factories import pydantic_factory

    class ClownModelFactory(pydantic_factory.ModelFactory[ClownModel]):
        __model__ = ClownModel
        __randomize_collection_length__ = True
        __max_collection_length__ = 3

    instance = ClownModelFactory.build()
    # print(instance.model_dump_json(indent=2))
    print(instance.model_json_schema())
