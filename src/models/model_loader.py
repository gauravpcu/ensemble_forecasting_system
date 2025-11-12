"""
Model loading utilities for ensemble system
"""

import pickle
import boto3
import json
import numpy as np
from typing import Dict, Optional, Any
from src.config import env_config


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
            
            # Try to use SSO profile if available
            import os
            profile_name = os.environ.get('AWS_PROFILE', 'AWSAdministratorAccess-236357498302')
            
            try:
                # Try with SSO profile
                session = boto3.Session(profile_name=profile_name)
                self.runtime_client = session.client(
                    'sagemaker-runtime',
                    region_name=self.region
                )
                print(f"✓ DeepAR endpoint connected (using profile: {profile_name})")
            except Exception as profile_error:
                # Fall back to default credentials
                print(f"⚠️  SSO profile not available, using default credentials")
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
    
    def predict_simple(self, X, context_data: Optional[Dict] = None) -> np.ndarray:
        """
        Simplified prediction interface (for compatibility with ensemble)
        
        Args:
            X: Feature matrix with item_id and timestamp columns
            context_data: Dictionary with historical data for each item_id
                         Format: {item_id: {'start': timestamp, 'target': [values]}}
            
        Returns:
            Array of predictions
        """
        if self.runtime_client is None:
            print("⚠️  DeepAR endpoint not available, returning zeros")
            return np.zeros(len(X))
        
        try:
            # Prepare DeepAR format data
            deepar_data = self._prepare_deepar_format(X, context_data)
            
            if not deepar_data or not deepar_data.get('instances'):
                print("⚠️  No valid data for DeepAR, returning zeros")
                return np.zeros(len(X))
            
            # Batch predictions to avoid request size limits
            # SageMaker has a 5MB request limit, so batch to ~100 items at a time
            batch_size = 100
            instances = deepar_data['instances']
            all_predictions = []
            
            print(f"   Making {len(instances)} predictions in batches of {batch_size}...")
            
            for i in range(0, len(instances), batch_size):
                batch_instances = instances[i:i+batch_size]
                batch_request = {
                    "instances": batch_instances,
                    "configuration": deepar_data['configuration']
                }
                
                # Make predictions for this batch
                batch_predictions = self.predict(batch_request)
                all_predictions.extend(batch_predictions)
            
            predictions = np.array(all_predictions)
            
            # Ensure we have the right number of predictions
            if len(predictions) != len(X):
                print(f"⚠️  DeepAR returned {len(predictions)} predictions, expected {len(X)}")
                # Pad or truncate to match expected length
                if len(predictions) < len(X):
                    predictions = np.pad(predictions, (0, len(X) - len(predictions)), 'edge')
                else:
                    predictions = predictions[:len(X)]
            
            return predictions
            
        except Exception as e:
            print(f"⚠️  DeepAR prediction failed: {str(e)}, returning zeros")
            return np.zeros(len(X))
    
    def _prepare_deepar_format(self, X, context_data: Optional[Dict] = None) -> Dict:
        """
        Prepare data in DeepAR format
        
        DeepAR expects:
        {
            "instances": [
                {
                    "start": "2023-01-01 00:00:00",
                    "target": [1.0, 2.0, 3.0, ...],
                    "cat": [0]  # optional categorical features
                },
                ...
            ],
            "configuration": {
                "num_samples": 100,
                "output_types": ["mean", "quantiles"],
                "quantiles": ["0.1", "0.5", "0.9"]
            }
        }
        
        Args:
            X: Feature matrix (pandas DataFrame expected)
            context_data: Historical data for each item
            
        Returns:
            Dictionary in DeepAR format
        """
        import pandas as pd
        
        if context_data is None:
            # If no context data provided, we can't make meaningful predictions
            return {}
        
        instances = []
        
        # Group by item_id to create time series for each item
        if isinstance(X, pd.DataFrame) and 'item_id' in X.columns:
            for item_id in X['item_id'].unique():
                if item_id in context_data:
                    item_context = context_data[item_id]
                    
                    # Create instance for this item
                    instance = {
                        "start": item_context.get('start', '2023-01-01 00:00:00'),
                        "target": item_context.get('target', [])
                    }
                    
                    # Add categorical features if available
                    if 'cat' in item_context:
                        instance['cat'] = item_context['cat']
                    
                    # Add dynamic features if available
                    if 'dynamic_feat' in item_context:
                        instance['dynamic_feat'] = item_context['dynamic_feat']
                    
                    instances.append(instance)
        
        if not instances:
            return {}
        
        # Prepare request
        request = {
            "instances": instances,
            "configuration": {
                "num_samples": 100,
                "output_types": ["mean"],
                "quantiles": ["0.5"]
            }
        }
        
        return request


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
