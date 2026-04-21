# config/parameters_factory.py
from abc import ABC, abstractmethod
from config.parameters import Parameters

class ParametersFactory(ABC):
    @abstractmethod
    def create_parameters(self) -> Parameters:
        """Create and return a Parameters instance"""
        pass