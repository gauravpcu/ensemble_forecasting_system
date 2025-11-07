"""
Compare test predictions with validation actuals for forward-looking accuracy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

print("=" * 80)
print("VALIDATION COMPARISON - FORWARD-LOOKING ACCURACY")
print("=" * 80)

# Load data
print("\n[1/4] Loading data...")

# Try to load predictions file first, fall back to test_data.csv
try:
    test_df = pd.read_csv('test/data/predictions.csv')
    print("   ✓ Loaded predictions file")
except FileNotFoundError:
    test_df = pd.read_csv('test/data/test_data.csv')
    print("   ✓ Loaded test data file")

val_df = pd.read_csv('test/data/val_data.csv')

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

print(f"   Test data: {len(test_df):,} records")
print(f"   Validation data: {len(val_df):,} records")

# Check if predictions exist
if 'predicted_value' not in test_df.columns:
    print("\n❌ No predictions found in test data!")
    print("   Please run 'run_predictions.py' first to generate predictions.")
    exit(1)

print(f"   ✓ Predictions found in test data")

# Analyze prediction patterns
print("\n[2/4] Analyzing prediction patterns...")

# Get last prediction for each item from test data
test_last = test_df.sort_values('timestamp').groupby('item_id').last().reset_index()
test_last = test_last[['item_id', 'predicted_value', 'rolling_mean_7d', 'rolling_mean_30d']]
test_last.columns = ['item_id', 'predicted_next', 'test_rolling_7d', 'test_rolling_30d']

print(f"   Items with predictions: {len(test_last):,}")

# Get first actual values from validation data
val_first = val_df.sort_values('timestamp').groupby('item_id').first().reset_index()
val_first = val_first[['item_id', 'target_value', 'timestamp']]
val_first.columns = ['item_id', 'actual_next', 'val_timestamp']

print(f"   Items in validation: {len(val_first):,}")

# Merge predictions with actuals
print("\n[3/4] Matching predictions with actuals...")
comparison = test_last.merge(val_first, on='item_id', how='inner')

print(f"   Matched items: {len(comparison):,}")

if len(comparison) == 0:
    print("\n⚠️  No matching items found!")
    print("   This means items in test period don't appear in validation period.")
    print("   This is common for one-time orders or new items.")
    
    # Show overlap statistics
    test_items = set(test_df['item_id'])
    val_items = set(val_df['item_id'])
    
    print(f"\n   Test-only items: {len(test_items - val_items):,}")
    print(f"   Validation-only items: {len(val_items - test_items):,}")
    print(f"   Common items: {len(test_items & val_items):,}")
    
    # Alternative: Compare aggregate patterns
    print("\n[4/4] Comparing aggregate patterns instead...")
    
    test_daily = test_df.groupby('timestamp').agg({
        'target_value': 'sum',
        'predicted_value': 'sum'
    }).reset_index()
    
    val_daily = val_df.groupby('timestamp').agg({
        'target_value': 'sum'
    }).reset_index()
    
    print(f"\n   Daily Aggregates:")
    print(f"   Test period avg actual: {test_daily['target_value'].mean():,.0f} units/day")
    print(f"   Test period avg predicted: {test_daily['predicted_value'].mean():,.0f} units/day")
    print(f"   Validation period avg actual: {val_daily['target_value'].mean():,.0f} units/day")
    
    # Use test predictions to estimate validation
    test_pred_avg = test_daily['predicted_value'].mean()
    val_actual_avg = val_daily['target_value'].mean()
    
    error = test_pred_avg - val_actual_avg
    pct_error = (error / val_actual_avg) * 100
    
    print(f"\n   Forward-looking aggregate accuracy:")
    print(f"   Predicted daily avg: {test_pred_avg:,.0f}")
    print(f"   Actual daily avg: {val_actual_avg:,.0f}")
    print(f"   Error: {error:,.0f} ({pct_error:+.1f}%)")
    
else:
    # Calculate metrics for matched items
    comparison['error'] = comparison['predicted_next'] - comparison['actual_next']
    comparison['abs_error'] = np.abs(comparison['error'])
    comparison['pct_error'] = (comparison['abs_error'] / (comparison['actual_next'] + 1e-6)) * 100
    
    # Metrics
    mae = comparison['abs_error'].mean()
    rmse = np.sqrt((comparison['error'] ** 2).mean())
    mape = comparison['pct_error'].mean()
    
    print(f"\n[4/4] Forward-looking accuracy metrics:")
    print(f"   MAE:  {mae:.2f} units")
    print(f"   RMSE: {rmse:.2f} units")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Accuracy: {100 - mape:.2f}%")
    
    # Distribution analysis
    print(f"\n   Error distribution:")
    print(f"   Mean error: {comparison['error'].mean():.2f}")
    print(f"   Median error: {comparison['error'].median():.2f}")
    print(f"   Std error: {comparison['error'].std():.2f}")
    
    # Accuracy by volume
    comparison['volume_category'] = pd.cut(
        comparison['actual_next'],
        bins=[0, 5, 20, 100, float('inf')],
        labels=['Low (0-5)', 'Medium (5-20)', 'High (20-100)', 'Very High (100+)']
    )
    
    print(f"\n   Accuracy by volume category:")
    for cat in ['Low (0-5)', 'Medium (5-20)', 'High (20-100)', 'Very High (100+)']:
        cat_data = comparison[comparison['volume_category'] == cat]
        if len(cat_data) > 0:
            cat_mape = cat_data['pct_error'].mean()
            cat_mae = cat_data['abs_error'].mean()
            print(f"   {cat:20s}: {len(cat_data):6,} items, MAE={cat_mae:6.2f}, MAPE={cat_mape:6.2f}%")
    
    # Save comparison
    output_file = 'test/data/validation_comparison.csv'
    comparison.to_csv(output_file, index=False)
    print(f"\n✓ Comparison saved: {output_file}")
    
    # Top errors
    print(f"\n   Top 10 items by absolute error:")
    top_errors = comparison.nlargest(10, 'abs_error')[['item_id', 'predicted_next', 'actual_next', 'abs_error']]
    for idx, row in top_errors.iterrows():
        print(f"   {row['item_id']:40s} Pred: {row['predicted_next']:6.1f}, Actual: {row['actual_next']:6.1f}, Error: {row['abs_error']:6.1f}")

print("\n" + "=" * 80)
print("VALIDATION COMPARISON COMPLETE")
print("=" * 80)
