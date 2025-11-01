# models/base_model.py - Demonstrates OOP Inheritance and Polymorphism
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    """Abstract Base Class defining the interface for all prediction models."""
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass