"""
Base evaluator class for all model evaluation classes.

This module provides a common base class that all evaluator classes should inherit from,
ensuring consistent interface and shared functionality across different model evaluation tasks.
"""

import os
import logging
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class ModelEvaluator(ABC):
    """
    Base class for all model evaluators.
    
    This abstract base class defines the common interface and shared functionality
    that all specific evaluator classes should implement.
    
    Attributes:
        model_path (str): Path to the trained model
        tokenizer_path (str): Path to the tokenizer
        device (str): Device to run the model on (e.g., 'cuda', 'cpu')
        model: The loaded model instance
        tokenizer: The loaded tokenizer instance
        vocab_size (int): Size of the vocabulary
    """
    
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cuda', **kwargs):
        """
        Initialize the base evaluator.
        
        Args:
            model_path (str): Path to the trained model
            tokenizer_path (str): Path to the tokenizer
            device (str): Device to run the model on
            **kwargs: Additional keyword arguments for specific evaluators
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        
        # Validate paths
        self._validate_paths()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.vocab_size = None
        
        # Initialize tokenizer and model (and assign their return values)
        self.tokenizer = self._init_tokenizer()
        self.model = self._init_model()
        
        logger.info(f"Initialized {self.__class__.__name__} with model: {model_path}")
    
    def _validate_paths(self):
        """Validate that the provided paths exist."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        # tokenizer_pathがNoneの場合はスキップ（protein_sequenceなど、組み込みトークナイザーを使用する場合）
        if self.tokenizer_path is not None and not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path not found: {self.tokenizer_path}")
    
    @abstractmethod
    def _init_tokenizer(self):
        """
        Initialize the tokenizer.
        
        This method should be implemented by each specific evaluator class
        to load and configure the appropriate tokenizer for the model type.
        """
        pass
    
    @abstractmethod
    def _init_model(self):
        """
        Initialize and load the model.
        
        This method should be implemented by each specific evaluator class
        to load and configure the appropriate model.
        """
        pass
    
    @abstractmethod
    def encode_sequence(self, sequence: str) -> Union[list, torch.Tensor]:
        """
        Encode input sequence to model input format.
        
        Args:
            sequence (str): Input sequence to encode
            
        Returns:
            Encoded representation of the input sequence
        """
        pass
    
    def set_device(self, device: str):
        """
        Change the device for model evaluation.
        
        Args:
            device (str): New device to use (e.g., 'cuda', 'cpu')
        """
        old_device = self.device
        self.device = device
        
        if self.model is not None:
            self.model = self.model.to(device)
            logger.info(f"Moved model from {old_device} to {device}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'model_path': self.model_path,
            'tokenizer_path': self.tokenizer_path,
            'device': self.device,
            'vocab_size': self.vocab_size,
            'model_class': self.__class__.__name__
        }
        
        if hasattr(self.model, 'config'):
            info['model_config'] = vars(self.model.config)
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}(model_path='{self.model_path}', device='{self.device}')"