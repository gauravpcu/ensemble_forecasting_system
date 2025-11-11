"""
Ensemble prediction system combining multiple models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from src.config import env_config


class EnsemblePredictor:
    """Ensemble predictor combining multiple forecasting models"""
    
    def __init__(self, models: Dict, weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor
        
        Args:
            models: Dictionary of model objects
            weights: Dictionary of model weights (must sum to 1.0)
        """
        self.models = models
        self.weights = weights or env_config.ENSEMBLE_WEIGHTS
        self._validate_weights()
        
        if env_config.VERBOSE:
            print("\n" + "="*60)
            print("ENSEMBLE CONFIGURATION")
            print("="*60)
            print(f"Models: {list(self.models.keys())}")
            print(f"Weights: {self.weights}")
            print("="*60 + "\n")
    
    def _validate_weights(self):
        """Validate that weights sum to 1.0"""
        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0, atol=0.01):
            print(f"Warning: Weights sum to {total_weight:.3f}, normalizing...")
            # Normalize weights
            for key in self.weights:
                self.weights[key] /= total_weight
    
    def predict(self, X, y_actual: Optional[np.ndarray] = None) -> Dict:
        """
        Generate ensemble predictions
        
        Args:
            X: Feature matrix
            y_actual: Actual values (optional, for DeepAR mock predictions)
            
        Returns:
            Dictionary with predictions and metadata
        """
        predictions = {}
        errors = {}
        
        print("\n" + "="*60)
        print("GENERATING PREDICTIONS")
        print("="*60)
        
        # Generate predictions from each model
        for model_name, model in self.models.items():
            try:
                if env_config.VERBOSE:
                    print(f"Predicting with {model_name}...")
                
                if model_name == 'deepar':
                    # DeepAR needs special handling
                    pred = model.predict_simple(X, y_actual)
                else:
                    # Standard prediction interface
                    pred = model.predict(X)
                
                predictions[model_name] = pred
                print(f"✓ {model_name}: {len(pred)} predictions generated")
                
            except Exception as e:
                errors[model_name] = str(e)
                print(f"✗ {model_name} failed: {str(e)}")
        
        if not predictions:
            raise Exception("All models failed to generate predictions")
        
        # Combine predictions
        ensemble_pred = self._weighted_ensemble(predictions)
        
        print(f"\n✓ Ensemble prediction completed")
        print("="*60 + "\n")
        
        return {
            'ensemble': ensemble_pred,
            'individual': predictions,
            'errors': errors,
            'weights': self.weights,
            'models_used': list(predictions.keys())
        }
    
    def _weighted_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Combine predictions using weighted average
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Weighted ensemble predictions
        """
        # Get the minimum length (in case predictions have different lengths)
        min_length = min(len(pred) for pred in predictions.values())
        
        # Initialize ensemble prediction
        ensemble_pred = np.zeros(min_length)
        total_weight = 0.0
        
        # Weighted average
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.0)
            if weight > 0:
                ensemble_pred += weight * pred[:min_length]
                total_weight += weight
        
        # Normalize if weights don't sum to 1.0
        if total_weight > 0 and not np.isclose(total_weight, 1.0):
            ensemble_pred /= total_weight
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def predict_with_multiple_configs(self, X, y_actual: Optional[np.ndarray] = None) -> Dict:
        """
        Generate predictions with multiple ensemble configurations
        
        Args:
            X: Feature matrix
            y_actual: Actual values
            
        Returns:
            Dictionary with predictions for each configuration
        """
        results = {}
        
        print("\n" + "="*60)
        print("TESTING MULTIPLE ENSEMBLE CONFIGURATIONS")
        print("="*60)
        
        for config_name, weights in config.ENSEMBLE_CONFIGS.items():
            print(f"\nTesting '{config_name}' configuration: {weights}")
            
            # Temporarily update weights
            original_weights = self.weights.copy()
            self.weights = weights
            
            try:
                result = self.predict(X, y_actual)
                results[config_name] = result
            except Exception as e:
                print(f"✗ Configuration '{config_name}' failed: {str(e)}")
                results[config_name] = {'error': str(e)}
            
            # Restore original weights
            self.weights = original_weights
        
        print("="*60 + "\n")
        return results
    
    def get_model_contributions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate the contribution of each model to the ensemble
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            Dictionary of contribution percentages
        """
        contributions = {}
        total_weight = sum(self.weights.get(name, 0) for name in predictions.keys())
        
        for model_name in predictions.keys():
            weight = self.weights.get(model_name, 0)
            contributions[model_name] = (weight / total_weight * 100) if total_weight > 0 else 0
        
        return contributions


def create_ensemble_predictor(models: Dict, 
                              config_name: str = 'conservative') -> EnsemblePredictor:
    """
    Create an ensemble predictor with a specific configuration
    
    Args:
        models: Dictionary of loaded models
        config_name: Name of ensemble configuration to use
        
    Returns:
        EnsemblePredictor instance
    """
    if config_name in config.ENSEMBLE_CONFIGS:
        weights = config.ENSEMBLE_CONFIGS[config_name]
    else:
        print(f"Warning: Configuration '{config_name}' not found, using default")
        weights = config.ENSEMBLE_WEIGHTS
    
    return EnsemblePredictor(models, weights)
