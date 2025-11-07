"""
Run predictions on test data and compare with validation data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from model_loader import load_models
from ensemble_predictor import EnsemblePredictor
from data_loader import DataLoader
import env_config

print("=" * 80)
print("FORECASTING PIPELINE - TEST DATA PREDICTIONS")
print("=" * 80)

# Step 1: Load test and validation data
print("\n[1/5] Loading data...")
test_df = pd.read_csv('test/data/test_data.csv')
val_df = pd.read_csv('test/data/val_data.csv')

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

print(f"   Test data: {len(test_df):,} records ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")
print(f"   Validation data: {len(val_df):,} records ({val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()})")
print(f"   Unique items in test: {test_df['item_id'].nunique():,}")
print(f"   Unique items in validation: {val_df['item_id'].nunique():,}")

# Step 2: Initialize predictor
print("\n[2/5] Initializing ensemble predictor...")
try:
    models = load_models()
    predictor = EnsemblePredictor(models)
    print(f"   ✓ Ensemble predictor initialized")
    print(f"   Models loaded: {list(models.keys())}")
    print(f"   Weights: LightGBM={env_config.ENSEMBLE_WEIGHTS['lightgbm']}, DeepAR={env_config.ENSEMBLE_WEIGHTS['deepar']}")
except Exception as e:
    print(f"   ✗ Error initializing predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Prepare features
print("\n[3/5] Preparing features...")
loader = DataLoader()

# Get features for test data
X_test, y_test = loader.prepare_lightgbm_features(test_df)
print(f"   ✓ Test features prepared: {X_test.shape}")

# Step 4: Make predictions
print("\n[4/5] Making predictions...")
print("   This may take a few minutes for large datasets...")

try:
    # Make predictions on test data
    predictions = predictor.predict(X_test, y_test)
    
    # Add predictions to test dataframe
    test_df['predicted_value'] = predictions['ensemble']
    
    # Add individual model predictions if available
    if 'individual' in predictions:
        for model_name, model_preds in predictions['individual'].items():
            test_df[f'{model_name}_prediction'] = model_preds
    
    print(f"   ✓ Predictions completed: {len(predictions['ensemble']):,} forecasts")
    print(f"   Prediction range: [{predictions['ensemble'].min():.2f}, {predictions['ensemble'].max():.2f}]")
    print(f"   Prediction mean: {predictions['ensemble'].mean():.2f}")
    
except Exception as e:
    print(f"   ✗ Error making predictions: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Compare with validation data
print("\n[5/5] Comparing with validation data...")

# Find common items between test predictions and validation actuals
common_items = set(test_df['item_id']) & set(val_df['item_id'])
print(f"   Common items: {len(common_items):,}")

if len(common_items) == 0:
    print("   ⚠️  No common items found between test and validation data")
    print("   This is expected if items don't repeat across periods")
else:
    # For common items, compare predictions vs actuals
    test_common = test_df[test_df['item_id'].isin(common_items)].copy()
    val_common = val_df[val_df['item_id'].isin(common_items)].copy()
    
    # Aggregate by item for comparison
    test_agg = test_common.groupby('item_id').agg({
        'predicted_value': 'mean',
        'target_value': 'mean'
    }).reset_index()
    
    val_agg = val_common.groupby('item_id').agg({
        'target_value': 'mean'
    }).reset_index()
    val_agg.columns = ['item_id', 'actual_value']
    
    # Merge predictions with actuals
    comparison = test_agg.merge(val_agg, on='item_id', how='inner')
    
    # Calculate metrics
    mae = np.abs(comparison['predicted_value'] - comparison['actual_value']).mean()
    rmse = np.sqrt(((comparison['predicted_value'] - comparison['actual_value']) ** 2).mean())
    mape = (np.abs((comparison['actual_value'] - comparison['predicted_value']) / (comparison['actual_value'] + 1e-6)) * 100).mean()
    
    print(f"\n   Metrics for {len(comparison):,} common items:")
    print(f"   MAE:  {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")

# Save predictions
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_file = 'test/data/predictions.csv'
test_df.to_csv(output_file, index=False)
print(f"\n✓ Predictions saved: {output_file}")
print(f"  Columns: {', '.join(test_df.columns)}")

# Create summary report
summary_file = 'test/data/prediction_summary.csv'
summary_df = test_df.groupby('item_id').agg({
    'target_value': ['mean', 'sum', 'count'],
    'predicted_value': ['mean', 'sum']
}).reset_index()
summary_df.columns = ['item_id', 'actual_mean', 'actual_sum', 'days', 'predicted_mean', 'predicted_sum']
summary_df['error'] = summary_df['predicted_mean'] - summary_df['actual_mean']
summary_df['abs_error'] = np.abs(summary_df['error'])
summary_df['pct_error'] = (summary_df['error'] / (summary_df['actual_mean'] + 1e-6)) * 100

summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Summary saved: {summary_file}")

# Overall statistics
print("\n" + "=" * 80)
print("PREDICTION STATISTICS")
print("=" * 80)

print(f"\nTest Data Actuals:")
print(f"  Mean: {test_df['target_value'].mean():.2f}")
print(f"  Median: {test_df['target_value'].median():.2f}")
print(f"  Std: {test_df['target_value'].std():.2f}")

print(f"\nPredictions:")
print(f"  Mean: {test_df['predicted_value'].mean():.2f}")
print(f"  Median: {test_df['predicted_value'].median():.2f}")
print(f"  Std: {test_df['predicted_value'].std():.2f}")

print(f"\nOverall Metrics (Test vs Predictions):")
mae_overall = np.abs(test_df['target_value'] - test_df['predicted_value']).mean()
rmse_overall = np.sqrt(((test_df['target_value'] - test_df['predicted_value']) ** 2).mean())
mape_overall = (np.abs((test_df['target_value'] - test_df['predicted_value']) / (test_df['target_value'] + 1e-6)) * 100).mean()

print(f"  MAE:  {mae_overall:.2f}")
print(f"  RMSE: {rmse_overall:.2f}")
print(f"  MAPE: {mape_overall:.2f}%")
print(f"  Accuracy: {100 - mape_overall:.2f}%")

print("\n" + "=" * 80)
print("COMPLETE!")
print("=" * 80)
print(f"\nNext steps:")
print(f"  1. Review predictions: test/data/predictions.csv")
print(f"  2. Analyze summary: test/data/prediction_summary.csv")
print(f"  3. Compare with validation data for forward-looking accuracy")
