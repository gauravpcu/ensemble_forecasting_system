#!/usr/bin/env python3
"""
Unified Prediction Generator
=============================
Core prediction engine that handles all forecasting scenarios.

This class orchestrates the entire prediction pipeline:
1. Load historical data
2. Engineer features (rolling averages, lags, seasonality)
3. Load ML models (LightGBM + DeepAR)
4. Generate ensemble predictions
5. Apply customer calibrations
6. Return predictions with metadata

Key Features:
- Flexible data sources (raw CSV or preprocessed data)
- Customer filtering (single, multiple, or all)
- Date-based predictions (single date or date range)
- Configurable context window (default 90 days)
- Automatic feature engineering
- Ensemble model predictions (95% LightGBM + 5% DeepAR)
- Customer-specific calibrations
- Prediction date tracking

Architecture:
    Raw Data → Feature Engineering → Model Loading → Prediction → Calibration → Output
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Import configuration settings
from src.config.env_config import (
    SOURCE_DATA_FILE,           # Path to source data CSV
    TEST_DATA_DIR,              # Test data directory
    CLASSIFICATION_THRESHOLD,   # Threshold for reorder classification (default: 4 units)
    ROLLING_WINDOW_SHORT,       # Short rolling window (default: 7 days)
    ROLLING_WINDOW_LONG,        # Long rolling window (default: 30 days)
    ROLLING_WINDOW_MIN_PERIODS  # Minimum periods for rolling calculations
)
import os

# Import model and data utilities
from src.models.model_loader import load_models
from src.models.ensemble_predictor import EnsemblePredictor
from src.data.data_loader import DataLoader


class PredictionGenerator:
    """
    Unified prediction generator for all scenarios
    
    This class handles the complete prediction workflow from data loading
    to final predictions with calibrations.
    
    Workflow:
        1. Initialize with parameters (customers, dates, data source)
        2. Call generate() to run the pipeline
        3. Returns DataFrame with predictions
    
    Examples:
        # Example 1: Predict for specific customers and date
        gen = PredictionGenerator(
            customers=['scionhealth', 'mercy'],  # Which customers
            target_date='2025-10-21',            # What date to predict for
            context_days=90                       # Use 90 days of history
        )
        predictions = gen.generate()
        
        # Example 2: Predict for all customers using preprocessed test data
        gen = PredictionGenerator(
            source_data='test/data/test_data.csv',  # Use test data
            use_preprocessed=True                    # Data already has features
        )
        predictions = gen.generate()
        
        # Example 3: Predict for date range
        gen = PredictionGenerator(
            customers=['scionhealth'],
            start_date='2025-10-20',  # Start of range
            end_date='2025-10-25',    # End of range
            context_days=90
        )
        predictions = gen.generate()
    
    Attributes:
        customers: List of customer IDs to predict for
        target_date: Single date to predict for
        context_days: Number of days of historical data to use
        models: Loaded ML models (LightGBM + DeepAR)
        predictor: Ensemble predictor instance
        predictions: Final predictions DataFrame
    """
    
    def __init__(
        self,
        customers: Optional[List[str]] = None,
        target_date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        context_days: int = 90,
        source_data: str = None,
        use_preprocessed: bool = False,
        classification_threshold: Optional[float] = None,
        verbose: bool = True
    ):
        """
        Initialize prediction generator with configuration
        
        This sets up all parameters for the prediction pipeline.
        No data is loaded or processed until generate() is called.
        
        Args:
            customers: List of customer IDs to predict for (None = all customers)
                      Examples: ['scionhealth'], ['scionhealth', 'mercy']
            
            target_date: Single date to predict for (YYYY-MM-DD format)
                        Example: '2025-11-15'
                        If None, uses current date
            
            start_date: Start date for date range predictions (YYYY-MM-DD)
                       Used with end_date for multi-day predictions
            
            end_date: End date for date range predictions (YYYY-MM-DD)
                     Used with start_date for multi-day predictions
            
            context_days: Number of days of historical data to use (default: 90)
                         More days = better seasonal patterns
                         Fewer days = more responsive to recent changes
                         Recommended: 60-90 days
            
            source_data: Path to source data file (overrides env config)
                        Can be raw order history or preprocessed features
            
            use_preprocessed: If True, assumes data already has features engineered
                             If False, will engineer features from raw data
                             Use True for test data, False for production
            
            classification_threshold: Threshold for reorder classification (default: 4 units)
                                     Items with predicted value >= threshold get "ORDER" recommendation
            
            verbose: If True, prints progress messages during generation
        
        Raises:
            ValueError: If parameters are invalid (e.g., start_date without end_date)
        """
        
        # ====================================================================
        # Store configuration parameters
        # ====================================================================
        self.customers = customers
        
        # Convert date strings to datetime objects for easier manipulation
        self.target_date = pd.to_datetime(target_date) if target_date else None
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # Context window: how many days of history to use
        self.context_days = context_days
        
        # Data source: use provided path or default from config
        self.source_data = source_data or SOURCE_DATA_FILE
        
        # Preprocessing flag: True if data already has features
        self.use_preprocessed = use_preprocessed
        
        # Classification threshold: items >= this value get "ORDER" recommendation
        self.classification_threshold = classification_threshold or CLASSIFICATION_THRESHOLD
        
        # Verbose flag: controls progress message printing
        self.verbose = verbose
        
        # ====================================================================
        # Initialize model placeholders (loaded later in load_models())
        # ====================================================================
        self.models = None      # Will hold LightGBM and DeepAR models
        self.predictor = None   # Will hold EnsemblePredictor instance
        
        # ====================================================================
        # Initialize data storage (populated during generate())
        # ====================================================================
        self.raw_data = None        # Raw order history data
        self.processed_data = None  # Data with engineered features
        self.predictions = None     # Final predictions DataFrame
        
        # ====================================================================
        # Validate parameters before proceeding
        # ====================================================================
        self._validate_parameters()
    
    def _validate_parameters(self):
        """
        Validate input parameters for logical consistency
        
        Checks:
        1. Can't specify both target_date and date range
        2. Date range must have both start and end
        3. Start date must be before end date
        
        Raises:
            ValueError: If parameters are invalid
        """
        # Check for conflicting date specifications
        if self.target_date and (self.start_date or self.end_date):
            raise ValueError("Cannot specify both target_date and date range (start_date/end_date)")
        
        # Check for incomplete date range
        if (self.start_date and not self.end_date) or (self.end_date and not self.start_date):
            raise ValueError("Must specify both start_date and end_date for date range")
        
        # Check for invalid date order
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")
    
    def _print(self, message: str):
        """
        Print message if verbose mode is enabled
        
        This is a helper method to control console output.
        Only prints if self.verbose is True.
        
        Args:
            message: Message to print
        """
        if self.verbose:
            print(message)
    
    def load_data(self):
        """
        Load and filter source data
        
        This is Step 1 of the prediction pipeline.
        Loads historical order data from CSV and filters by customers if specified.
        
        Process:
        1. Load data from CSV file
           - If use_preprocessed=True: Data already has features
           - If use_preprocessed=False: Raw order history
        2. Parse dates (if raw data)
        3. Filter by customers (if specified)
        
        Returns:
            self: For method chaining
        """
        self._print(f"\n[1/5] Loading data from {self.source_data}...")
        
        if self.use_preprocessed:
            # Load preprocessed data (already has features engineered)
            # Used for testing with test_data.csv
            self.raw_data = pd.read_csv(self.source_data)
            self._print(f"   Loaded {len(self.raw_data):,} preprocessed records")
        else:
            # Load raw order history from production data
            # Needs feature engineering in prepare_features()
            self.raw_data = pd.read_csv(self.source_data, low_memory=False)
            self.raw_data['CreateDate'] = pd.to_datetime(self.raw_data['CreateDate'])
            self._print(f"   Loaded {len(self.raw_data):,} records")
        
        # Filter by customers if specified
        # If self.customers is None, keeps all customers
        if self.customers:
            self.raw_data = self.raw_data[self.raw_data['CustomerID'].isin(self.customers)]
            self._print(f"   Filtered to {len(self.raw_data):,} records for {len(self.customers)} customers")
        
        return self
    
    def prepare_features(self):
        """Prepare features for prediction"""
        if self.use_preprocessed:
            self._print("\n[2/5] Using preprocessed features...")
            self.processed_data = self.raw_data.copy()
            self._print(f"   Features ready: {len(self.processed_data):,} records")
            return self
        
        self._print("\n[2/5] Preparing features...")
        
        # ====================================================================
        # Determine date range for context (BEFORE prediction period)
        # ====================================================================
        # CRITICAL: Context period must END before prediction START date
        # to prevent data leakage (model seeing the future it's predicting)
        #
        # Example:
        #   Prediction: Oct 26 - Nov 8 (14 days)
        #   Context: Jul 28 - Oct 25 (90 days BEFORE Oct 26)
        #   No overlap = No data leakage
        #
        if self.target_date:
            # Single target date: context ends day before
            max_date = self.target_date - timedelta(days=1)
            min_date = max_date - timedelta(days=self.context_days - 1)
        elif self.start_date:
            # Date range: context ends day before start_date
            max_date = self.start_date - timedelta(days=1)
            min_date = max_date - timedelta(days=self.context_days - 1)
        else:
            # No date specified: use most recent data
            max_date = self.raw_data['CreateDate'].max()
            min_date = max_date - timedelta(days=self.context_days)
        
        self._print(f"   Context period: {min_date.date()} to {max_date.date()}")
        self._print(f"   (90 days BEFORE prediction start - no data leakage)")
        
        # Filter for context period
        df_context = self.raw_data[
            (self.raw_data['CreateDate'] >= min_date) & 
            (self.raw_data['CreateDate'] <= max_date)
        ].copy()
        
        self._print(f"   Context data: {len(df_context):,} records")
        
        # Create item_id
        df_context['item_id'] = (
            df_context['CustomerID'].astype(str) + '_' + 
            df_context['FacilityID'].astype(str) + '_' + 
            df_context['ProductID'].astype(str)
        )
        
        # Aggregate by item and date
        self._print("   Aggregating by item and date...")
        agg_df = df_context.groupby(['item_id', 'CreateDate']).agg({
            'OrderUnits': 'sum',
            'OrderID': 'count',
            'Price': ['mean', 'std'],
            'CustomerID': 'first',
            'FacilityID': 'first',
            'ProductID': 'first',
            'MainProductID': 'first',
            'ProductName': 'first',
            'CategoryName': 'first',
            'DepartmentID': 'first',
            'VendorID': 'first',
            'VendorName': 'first',
            'UserID': 'first',
            'PortalID': 'first'
        }).reset_index()
        
        # Flatten columns
        agg_df.columns = [
            'item_id', 'timestamp', 'target_value', 'order_count', 
            'price_mean', 'price_std', 'CustomerID', 'FacilityID', 
            'ProductID', 'MainProductID', 'ProductName', 'CategoryName',
            'DepartmentID', 'VendorID', 'VendorName', 'UserID', 'PortalID'
        ]
        
        self._print(f"   Aggregated to {len(agg_df):,} item-date combinations")
        
        # Engineer features
        self._print("   Engineering features...")
        agg_df = self._engineer_features(agg_df)
        
        # Get latest data for each item (for prediction)
        self.processed_data = agg_df.groupby('item_id').last().reset_index()
        self._print(f"   Features ready: {len(self.processed_data):,} unique items")
        
        return self
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer all features for prediction"""
        # Sort by item and timestamp
        df = df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)
        
        # Time-based features
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
        df['is_quarter_end'] = df['timestamp'].dt.is_quarter_end.astype(int)
        
        # Vendor reliability
        vendor_counts = df.groupby('VendorID')['item_id'].count().to_dict()
        df['vendor_reliability'] = df['VendorID'].map(vendor_counts).fillna(0)
        
        # Price volatility
        df['price_volatility'] = df['price_std'].fillna(0) / (df['price_mean'] + 1e-6)
        df['order_frequency'] = df['order_count']
        
        # Rolling features
        df['rolling_mean_7d'] = df.groupby('item_id')['target_value'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
        )
        df['rolling_std_7d'] = df.groupby('item_id')['target_value'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
        )
        df['rolling_mean_30d'] = df.groupby('item_id')['target_value'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW_LONG, min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
        )
        df['rolling_std_30d'] = df.groupby('item_id')['target_value'].transform(
            lambda x: x.rolling(window=ROLLING_WINDOW_LONG, min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
        )
        
        # Lag features
        df['lag_7'] = df.groupby('item_id')['target_value'].shift(7)
        df['lag_14'] = df.groupby('item_id')['target_value'].shift(14)
        df['lag_30'] = df.groupby('item_id')['target_value'].shift(30)
        
        # Fill lag NaNs
        df['lag_7'] = df.groupby('item_id')['lag_7'].transform(
            lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
        )
        df['lag_14'] = df.groupby('item_id')['lag_14'].transform(
            lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
        )
        df['lag_30'] = df.groupby('item_id')['lag_30'].transform(
            lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
        )
        
        # Seasonal features
        df['seasonal_trend'] = df.groupby('item_id')['target_value'].transform('mean')
        df['seasonal_seasonal'] = (
            df.groupby(['item_id', 'day_of_week'])['target_value'].transform('mean') - 
            df['seasonal_trend']
        )
        
        # Encode categorical variables
        df['customer_encoded'] = pd.Categorical(df['CustomerID']).codes
        df['facility_encoded'] = pd.Categorical(df['FacilityID']).codes
        df['product_encoded'] = pd.Categorical(df['ProductID']).codes
        
        return df
    
    def load_models(self):
        """Load prediction models"""
        self._print("\n[3/5] Loading models...")
        self.models = load_models()
        self.predictor = EnsemblePredictor(models=self.models)
        return self
    
    def generate_predictions(self):
        """Generate predictions using ensemble model"""
        self._print("\n[4/5] Generating predictions...")
        
        # Prepare features for LightGBM using DataLoader
        loader = DataLoader()
        X_features, _ = loader.prepare_lightgbm_features(self.processed_data)
        self._print(f"   Prepared LightGBM features: {X_features.shape}")
        
        # Make predictions
        predictions = self.predictor.predict(X_features)
        
        # Add prediction metadata
        prediction_date = self.target_date if self.target_date else datetime.now()
        self.processed_data['prediction_date'] = prediction_date
        self.processed_data['prediction_generated_at'] = datetime.now()
        
        # Add predictions to data
        self.processed_data['predicted_value'] = predictions['ensemble']
        if 'lightgbm' in predictions:
            self.processed_data['lightgbm_prediction'] = predictions['lightgbm']
        if 'deepar' in predictions:
            self.processed_data['deepar_prediction'] = predictions['deepar']
        
        # Apply classification
        self.processed_data['predicted_reorder'] = (
            self.processed_data['predicted_value'] >= self.classification_threshold
        ).astype(int)
        self.processed_data['reorder_recommendation'] = self.processed_data['predicted_reorder'].map({
            1: 'ORDER', 
            0: 'NO ORDER'
        })
        
        # Sort by customer and predicted value
        self.processed_data = self.processed_data.sort_values(
            ['CustomerID', 'predicted_value'], 
            ascending=[True, False]
        )
        
        self.predictions = self.processed_data
        self._print(f"   ✓ Generated {len(self.predictions):,} predictions")
        self._print(f"   ✓ Prediction date: {prediction_date}")
        
        return self
    
    def save_predictions(self, output_path: str = None):
        """Save predictions to CSV"""
        if output_path is None:
            # Generate default filename
            if self.target_date:
                date_str = self.target_date.strftime('%Y%m%d')
            elif self.end_date:
                date_str = self.end_date.strftime('%Y%m%d')
            else:
                date_str = datetime.now().strftime('%Y%m%d')
            
            output_path = f"test/data/predictions_{date_str}.csv"
        
        self._print(f"\n[5/5] Saving predictions to {output_path}...")
        
        # Reorder columns to put important info first
        important_cols = [
            'prediction_date', 'prediction_generated_at',
            'CustomerID', 'FacilityID', 'ProductID', 'ProductName',
            'predicted_value', 'predicted_reorder', 'reorder_recommendation'
        ]
        
        # Get remaining columns
        remaining_cols = [col for col in self.predictions.columns if col not in important_cols]
        
        # Reorder: important columns first, then the rest
        ordered_cols = [col for col in important_cols if col in self.predictions.columns] + remaining_cols
        predictions_ordered = self.predictions[ordered_cols]
        
        predictions_ordered.to_csv(output_path, index=False)
        self._print(f"   ✓ Saved {len(self.predictions):,} predictions")
        
        return output_path
    
    def generate(self, save: bool = True, output_path: str = None) -> pd.DataFrame:
        """
        Run complete prediction pipeline
        
        Args:
            save: Whether to save predictions to file
            output_path: Custom output path (optional)
            
        Returns:
            DataFrame with predictions
        """
        self.load_data()
        self.prepare_features()
        self.load_models()
        self.generate_predictions()
        
        if save:
            self.save_predictions(output_path)
        
        return self.predictions
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary statistics by customer"""
        if self.predictions is None:
            raise ValueError("No predictions generated yet. Call generate() first.")
        
        summary = []
        customers = self.predictions['CustomerID'].unique()
        
        for customer in customers:
            customer_data = self.predictions[self.predictions['CustomerID'] == customer]
            summary.append({
                'Customer': customer,
                'Items': len(customer_data),
                'Items_to_Order': customer_data['predicted_reorder'].sum(),
                'Total_Units': customer_data['predicted_value'].sum(),
                'Avg_per_Item': customer_data['predicted_value'].mean(),
                'Median_per_Item': customer_data['predicted_value'].median(),
                'Max_Prediction': customer_data['predicted_value'].max(),
                'Facilities': customer_data['FacilityID'].nunique()
            })
        
        return pd.DataFrame(summary).sort_values('Total_Units', ascending=False)
    
    def print_summary(self):
        """Print summary statistics"""
        if self.predictions is None:
            raise ValueError("No predictions generated yet. Call generate() first.")
        
        summary_df = self.get_summary()
        
        print("\n" + "=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"\n{row['Customer'].upper()}:")
            print(f"   Total Items:              {row['Items']:,}")
            print(f"   Items to Order (≥{self.classification_threshold}):    {row['Items_to_Order']:,}")
            print(f"   Total Predicted Units:    {row['Total_Units']:,.0f}")
            print(f"   Average per Item:         {row['Avg_per_Item']:.2f} units")
            print(f"   Median per Item:          {row['Median_per_Item']:.2f} units")
            print(f"   Max Prediction:           {row['Max_Prediction']:.2f} units")
            print(f"   Facilities:               {row['Facilities']}")
        
        print("\n" + "=" * 80)
    
    def get_top_items(self, customer: str, n: int = 10) -> pd.DataFrame:
        """
        Get top N items by predicted volume for a customer
        
        Args:
            customer: Customer ID
            n: Number of items to return
            
        Returns:
            DataFrame with top items
        """
        if self.predictions is None:
            raise ValueError("No predictions generated yet. Call generate() first.")
        
        customer_data = self.predictions[self.predictions['CustomerID'] == customer]
        
        columns = ['item_id', 'ProductName', 'FacilityID', 'predicted_value', 'reorder_recommendation']
        
        # Add model-specific predictions if available
        if 'lightgbm_prediction' in customer_data.columns:
            columns.append('lightgbm_prediction')
        if 'deepar_prediction' in customer_data.columns:
            columns.append('deepar_prediction')
        
        return customer_data.nlargest(n, 'predicted_value')[columns]


# Convenience functions for common use cases

def predict_for_date(
    customers: List[str],
    date: str,
    context_days: int = 90,
    save: bool = True
) -> pd.DataFrame:
    """
    Generate predictions for specific customers and date
    
    Args:
        customers: List of customer IDs
        date: Target date (YYYY-MM-DD)
        context_days: Days of historical context
        save: Save to file
        
    Returns:
        DataFrame with predictions
    """
    generator = PredictionGenerator(
        customers=customers,
        target_date=date,
        context_days=context_days
    )
    return generator.generate(save=save)


def predict_from_test_data(
    test_data_path: str = 'test/data/test_data.csv',
    customers: Optional[List[str]] = None,
    save: bool = True
) -> pd.DataFrame:
    """
    Generate predictions from preprocessed test data
    
    Args:
        test_data_path: Path to test data file
        customers: Optional list of customers to filter
        save: Save to file
        
    Returns:
        DataFrame with predictions
    """
    generator = PredictionGenerator(
        customers=customers,
        source_data=test_data_path,
        use_preprocessed=True
    )
    return generator.generate(save=save)


if __name__ == "__main__":
    """Example usage"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python prediction_generator.py <date> [customer1,customer2,...]")
        print("\nExample:")
        print("  python prediction_generator.py 2025-10-21 scionhealth,mercy")
        sys.exit(1)
    
    target_date = sys.argv[1]
    customers = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    
    # Generate predictions
    generator = PredictionGenerator(
        customers=customers,
        target_date=target_date,
        context_days=90
    )
    
    predictions = generator.generate()
    generator.print_summary()
    
    # Show top items for each customer
    if customers:
        for customer in customers:
            print(f"\nTop 10 items for {customer.upper()}:")
            top_items = generator.get_top_items(customer, n=10)
            print(top_items.to_string(index=False))
