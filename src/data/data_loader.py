"""
Data loading and preparation utilities
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
import json
from src.config import env_config


class DataLoader:
    """Load and prepare data for ensemble predictions"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize data loader
        
        Args:
            config_path: Path to feature configuration JSON
        """
        self.config_path = config_path or env_config.FEATURE_CONFIG_PATH
        self.feature_config = self._load_feature_config()
        
    def _load_feature_config(self) -> Dict:
        """Load feature configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Feature config not found at {self.config_path}")
            return {}
    
    def load_data(self, dataset: str = "test", sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from specified dataset
        
        Args:
            dataset: Which dataset to load ("test", "val", or "both")
            sample_size: Number of samples to load (None = all)
            
        Returns:
            DataFrame with data
        """
        if dataset == "test":
            data_path = env_config.TEST_DATA_DIR + "/test_data.csv"
            print(f"Loading TEST data from {data_path}...")
        elif dataset == "val":
            data_path = env_config.TEST_DATA_DIR + "/val_data.csv"
            print(f"Loading VALIDATION data from {data_path}...")
        elif dataset == "both":
            print(f"Loading BOTH validation and test data...")
            val_df = pd.read_csv(env_config.TEST_DATA_DIR + "/val_data.csv")
            test_df = pd.read_csv(env_config.TEST_DATA_DIR + "/test_data.csv")
            
            # Convert timestamps
            if 'timestamp' in val_df.columns:
                val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])
            if 'timestamp' in test_df.columns:
                test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
            
            # Combine datasets
            df = pd.concat([val_df, test_df], ignore_index=True)
            print(f"Combined {len(val_df)} validation + {len(test_df)} test = {len(df)} total records")
            
            if sample_size and sample_size < len(df):
                print(f"Sampling {sample_size} records from {len(df)} total")
                df = df.sample(n=sample_size, random_state=env_config.DEFAULT_SAMPLE_RANDOM_STATE)
            
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            return df
        else:
            raise ValueError(f"Invalid dataset: {dataset}. Must be 'test', 'val', or 'both'")
        
        df = pd.read_csv(data_path)
        
        if sample_size and sample_size < len(df):
            print(f"Sampling {sample_size} records from {len(df)} total")
            df = df.sample(n=sample_size, random_state=env_config.DEFAULT_SAMPLE_RANDOM_STATE)
        
        # Convert timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        print(f"Loaded {len(df)} records")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df
    
    def load_test_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load test data (backward compatibility)
        
        Args:
            sample_size: Number of samples to load (None = all)
            
        Returns:
            DataFrame with test data
        """
        return self.load_data("test", sample_size)
    
    def prepare_lightgbm_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for LightGBM model
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features DataFrame, target array)
        """
        df = df.copy()
        
        # Load expected features from config
        with open(env_config.FEATURE_CONFIG_PATH, 'r') as f:
            feature_config = json.load(f)
        
        expected_features = feature_config['features']
        
        # Ensure categorical encodings exist with correct names
        if 'CustomerID_encoded' not in df.columns:
            if 'customer_encoded' in df.columns:
                df['CustomerID_encoded'] = df['customer_encoded']
            else:
                df['CustomerID_encoded'] = pd.Categorical(df['CustomerID']).codes
                
        if 'FacilityID_encoded' not in df.columns:
            if 'facility_encoded' in df.columns:
                df['FacilityID_encoded'] = df['facility_encoded']
            else:
                df['FacilityID_encoded'] = pd.Categorical(df['FacilityID']).codes
                
        if 'ProductID_encoded' not in df.columns:
            if 'product_encoded' in df.columns:
                df['ProductID_encoded'] = df['product_encoded']
            else:
                df['ProductID_encoded'] = pd.Categorical(df['ProductID']).codes
        
        # Add missing features that the model expects
        self._add_missing_features(df)
        
        # Select features in the exact order expected by the model
        available_features = [f for f in expected_features if f in df.columns]
        
        if len(available_features) != len(expected_features):
            missing = set(expected_features) - set(available_features)
            print(f"Warning: Missing features: {missing}")
            print(f"Available features: {len(available_features)}/{len(expected_features)}")
        
        X = df[available_features].copy()
        
        # Handle missing values
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(0)
        
        # Get target if available
        y = df['target_value'].values if 'target_value' in df.columns else None
        
        if env_config.VERBOSE:
            print(f"Prepared LightGBM features: {X.shape}")
            print(f"Features: {len(available_features)}")
        
        return X, y
    
    def _add_missing_features(self, df: pd.DataFrame):
        """Add missing features that the LightGBM model expects"""
        
        # Add year feature if missing
        if 'year' not in df.columns and 'timestamp' in df.columns:
            df['year'] = pd.to_datetime(df['timestamp']).dt.year
        
        # Add longer rolling windows if missing
        if 'rolling_mean_90d' not in df.columns and 'target_value' in df.columns:
            if 'item_id' in df.columns:
                df['rolling_mean_90d'] = df.groupby('item_id')['target_value'].transform(
                    lambda x: x.rolling(window=min(90, len(x)), min_periods=1).mean()
                )
            else:
                df['rolling_mean_90d'] = df['target_value'].rolling(window=90, min_periods=1).mean()
        
        if 'rolling_mean_365d' not in df.columns and 'target_value' in df.columns:
            if 'item_id' in df.columns:
                df['rolling_mean_365d'] = df.groupby('item_id')['target_value'].transform(
                    lambda x: x.rolling(window=min(365, len(x)), min_periods=1).mean()
                )
            else:
                df['rolling_mean_365d'] = df['target_value'].rolling(window=365, min_periods=1).mean()
        
        # Add longer lag features if missing
        if 'lag_365' not in df.columns and 'target_value' in df.columns:
            if 'item_id' in df.columns:
                df['lag_365'] = df.groupby('item_id')['target_value'].shift(365)
                df['lag_365'] = df.groupby('item_id')['lag_365'].transform(
                    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
                )
            else:
                df['lag_365'] = df['target_value'].shift(365).fillna(0)
        
        # Fill any remaining NaN values for the new features
        for col in ['rolling_mean_90d', 'rolling_mean_365d', 'lag_365', 'year']:
            if col in df.columns:
                df[col] = df[col].fillna(0)
    
    def prepare_deepar_input(self, df: pd.DataFrame, 
                            prediction_length: int = 14) -> Dict:
        """
        Prepare input for DeepAR model
        
        Args:
            df: Input DataFrame
            prediction_length: Number of time steps to predict
            
        Returns:
            Dictionary with DeepAR input format
        """
        # Group by item_id
        grouped = df.groupby('item_id')
        
        instances = []
        for item_id, group in grouped:
            group = group.sort_values('timestamp')
            
            # Get time series data
            target = group['target_value'].values.tolist()
            start = group['timestamp'].min()
            
            # Dynamic features (if available)
            dynamic_feat = []
            if 'day_of_week' in group.columns and 'month' in group.columns:
                dynamic_feat = [
                    group['day_of_week'].values.tolist(),
                    group['month'].values.tolist()
                ]
            
            instance = {
                'start': start.strftime('%Y-%m-%d %H:%M:%S'),
                'target': target
            }
            
            if dynamic_feat:
                instance['dynamic_feat'] = dynamic_feat
            
            instances.append(instance)
        
        if env_config.VERBOSE:
            print(f"Prepared DeepAR input: {len(instances)} time series")
        
        return {'instances': instances, 'configuration': {'num_samples': 100}}
    
    def get_data_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get statistics about the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_records': len(df),
            'unique_items': df['item_id'].nunique() if 'item_id' in df.columns else 0,
            'date_range': {
                'start': str(df['timestamp'].min()) if 'timestamp' in df.columns else None,
                'end': str(df['timestamp'].max()) if 'timestamp' in df.columns else None
            }
        }
        
        if 'target_value' in df.columns:
            stats['target_statistics'] = {
                'mean': float(df['target_value'].mean()),
                'std': float(df['target_value'].std()),
                'min': float(df['target_value'].min()),
                'max': float(df['target_value'].max()),
                'median': float(df['target_value'].median())
            }
        
        return stats


def load_and_prepare_data(dataset: str = None, sample_size: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Convenience function to load and prepare all data
    
    Args:
        dataset: Which dataset to load ("test", "val", "both", or None for default)
        sample_size: Number of samples to load
        
    Returns:
        Tuple of (original_df, features_df, target_array)
    """
    loader = DataLoader()
    
    # Use default dataset if not specified
    if dataset is None:
        dataset = "test"
    
    # Load data
    df = loader.load_data(dataset, sample_size)
    
    # Prepare features
    X, y = loader.prepare_lightgbm_features(df)
    
    # Print statistics
    if env_config.VERBOSE:
        stats = loader.get_data_statistics(df)
        print("\nData Statistics:")
        print(f"  Dataset: {dataset.upper()}")
        print(f"  Total records: {stats['total_records']:,}")
        print(f"  Unique items: {stats['unique_items']:,}")
        if 'target_statistics' in stats:
            print(f"  Target mean: {stats['target_statistics']['mean']:.2f}")
            print(f"  Target std: {stats['target_statistics']['std']:.2f}")
    
    return df, X, y
