# config/registry.py
from typing import Dict
from config.parameters_factory import ParametersFactory

class ParametersRegistry:
    _factories: Dict[str, ParametersFactory] = {}

    @classmethod
    def get_factory(cls, variant: str) -> ParametersFactory:
        if variant not in cls._factories:
            raise ValueError(f"Unknown parameter variant: {variant}")
        return cls._factories[variant]

def register_parameters(name):
    def decorator(cls):
        ParametersRegistry._factories[name] = cls()
        return cls
    return decorator