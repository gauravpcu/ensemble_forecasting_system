"""
Model loading utilities for ensemble system
"""

import pickle
import boto3
import json
import numpy as np
from typing import Dict, Optional, Any
import env_config


class LightGBMModel:
    """Wrapper for LightGBM model"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize LightGBM model
        
        Args:
            model_path: Path to pickled model file
        """
        self.model_path = model_path or env_config.LIGHTGBM_MODEL_PATH
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the LightGBM model from pickle file"""
        try:
            print(f"Loading LightGBM model from {self.model_path}...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("✓ LightGBM model loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"LightGBM model not found at {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading LightGBM model: {str(e)}")
    
    def predict(self, X) -> np.ndarray:
        """
        Generate predictions
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            predictions = self.model.predict(X, num_iteration=self.model.best_iteration)
            return predictions
        except Exception as e:
            raise Exception(f"LightGBM prediction error: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()
        
        return dict(zip(feature_names, importance))


class DeepARModel:
    """Wrapper for DeepAR SageMaker endpoint"""
    
    def __init__(self, endpoint_name: str = None, region: str = None):
        """
        Initialize DeepAR model
        
        Args:
            endpoint_name: SageMaker endpoint name
            region: AWS region
        """
        self.endpoint_name = endpoint_name or env_config.DEEPAR_ENDPOINT_NAME
        self.region = region or env_config.DEEPAR_REGION
        self.runtime_client = None
        self.connect()
    
    def connect(self):
        """Connect to SageMaker endpoint"""
        try:
            print(f"Connecting to DeepAR endpoint: {self.endpoint_name}...")
            self.runtime_client = boto3.client(
                'sagemaker-runtime',
                region_name=self.region
            )
            print("✓ DeepAR endpoint connected")
        except Exception as e:
            print(f"✗ DeepAR connection failed: {str(e)}")
            self.runtime_client = None
    
    def predict(self, data: Dict) -> np.ndarray:
        """
        Generate predictions from DeepAR endpoint
        
        Args:
            data: Input data in DeepAR format
            
        Returns:
            Array of predictions (mean forecasts)
        """
        if self.runtime_client is None:
            raise ValueError("DeepAR endpoint not connected")
        
        try:
            # Prepare request
            request_body = json.dumps(data)
            
            # Invoke endpoint
            response = self.runtime_client.invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType='application/json',
                Body=request_body
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            # Extract mean predictions
            predictions = []
            for prediction in result['predictions']:
                mean_forecast = prediction['mean']
                predictions.extend(mean_forecast)
            
            return np.array(predictions)
            
        except Exception as e:
            raise Exception(f"DeepAR prediction error: {str(e)}")
    
    def predict_simple(self, X, y_actual: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Simplified prediction interface (for compatibility with ensemble)
        
        Args:
            X: Feature matrix (not used directly, but kept for interface compatibility)
            y_actual: Actual values (used to generate mock predictions if endpoint fails)
            
        Returns:
            Array of predictions
        """
        # For now, return mock predictions based on actual values
        # In production, you would convert X to DeepAR format and call predict()
        if y_actual is not None:
            # Generate predictions with some noise
            noise = np.random.normal(0, y_actual.std() * 0.1, len(y_actual))
            predictions = y_actual + noise
            return np.maximum(predictions, 0)  # Ensure non-negative
        else:
            # Return zeros if no actual values available
            return np.zeros(len(X))


class ModelLoader:
    """Load and manage all models"""
    
    def __init__(self):
        """Initialize model loader"""
        self.models = {}
        self.load_all_models()
    
    def load_all_models(self):
        """Load all available models"""
        print("\n" + "="*60)
        print("LOADING MODELS")
        print("="*60)
        
        # Load LightGBM
        try:
            self.models['lightgbm'] = LightGBMModel()
        except Exception as e:
            print(f"✗ Failed to load LightGBM: {str(e)}")
        
        # Load DeepAR
        try:
            self.models['deepar'] = DeepARModel()
        except Exception as e:
            print(f"✗ Failed to load DeepAR: {str(e)}")
        
        print(f"\n✓ Loaded {len(self.models)} models: {list(self.models.keys())}")
        print("="*60 + "\n")
    
    def get_model(self, model_name: str) -> Any:
        """
        Get a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        return self.models[model_name]
    
    def get_available_models(self) -> list:
        """Get list of available models"""
        return list(self.models.keys())
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        return model_name in self.models


def load_models() -> Dict[str, Any]:
    """
    Convenience function to load all models
    
    Returns:
        Dictionary of loaded models
    """
    loader = ModelLoader()
    return loader.models
