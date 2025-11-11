#!/usr/bin/env python3
"""
Generate truly forward-looking predictions
Uses ONLY data up to test period end date to predict validation period
NO data leakage - validation data is never seen during prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
# Add parent directory to path for src imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.env_config import SOURCE_DATA_FILE, CLASSIFICATION_THRESHOLD
from src.core.prediction_generator import PredictionGenerator

print("=" * 80)
print("TRULY FORWARD-LOOKING PREDICTIONS")
print("=" * 80)
print("\nThis script generates predictions WITHOUT seeing validation data")
print("Ensures no data leakage for accurate performance measurement")
print("=" * 80)

# Configuration
TEST_END_DATE = "2025-10-21"  # Last date of test period
VALIDATION_START_DATE = "2025-10-22"  # First date of validation period
VALIDATION_END_DATE = "2025-11-03"  # Last date of validation period
CONTEXT_DAYS = 90

print(f"\nConfiguration:")
print(f"  Test period ends:     {TEST_END_DATE}")
print(f"  Validation period:    {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")
print(f"  Context days:         {CONTEXT_DAYS}")
print(f"  Data cutoff:          {TEST_END_DATE} (no data after this date used)")

# Step 1: Generate predictions for validation period
# Using ONLY data up to TEST_END_DATE
print(f"\n[1/3] Generating forward-looking predictions...")
print(f"   Using data up to {TEST_END_DATE} only")
print(f"   Predicting for {VALIDATION_START_DATE} to {VALIDATION_END_DATE}")

# Load historical data up to test end date
print(f"\n   Loading historical data...")
df = pd.read_csv(SOURCE_DATA_FILE, low_memory=False)
df['CreateDate'] = pd.to_datetime(df['CreateDate'])

# CRITICAL: Only use data up to test end date
test_end = pd.to_datetime(TEST_END_DATE)
df = df[df['CreateDate'] <= test_end]
print(f"   Loaded {len(df):,} records (up to {TEST_END_DATE})")

# Calculate context period
context_start = test_end - timedelta(days=CONTEXT_DAYS - 1)
df_context = df[df['CreateDate'] >= context_start]
print(f"   Context period: {context_start.date()} to {test_end.date()}")
print(f"   Context records: {len(df_context):,}")

# Create item_id and aggregate
print(f"\n   Aggregating by item and date...")
df_context['item_id'] = (
    df_context['CustomerID'].astype(str) + '_' + 
    df_context['FacilityID'].astype(str) + '_' + 
    df_context['ProductID'].astype(str)
)

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

agg_df.columns = [
    'item_id', 'timestamp', 'target_value', 'order_count', 
    'price_mean', 'price_std', 'CustomerID', 'FacilityID', 
    'ProductID', 'MainProductID', 'ProductName', 'CategoryName',
    'DepartmentID', 'VendorID', 'VendorName', 'UserID', 'PortalID'
]

print(f"   Aggregated to {len(agg_df):,} item-date combinations")
print(f"   Unique items: {agg_df['item_id'].nunique():,}")

# Engineer features using ONLY historical data
print(f"\n   Engineering features from historical data only...")
from src.data.data_loader import DataLoader

# Sort and prepare
agg_df = agg_df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

# Time features
agg_df['day_of_week'] = agg_df['timestamp'].dt.dayofweek
agg_df['day_of_month'] = agg_df['timestamp'].dt.day
agg_df['month'] = agg_df['timestamp'].dt.month
agg_df['quarter'] = agg_df['timestamp'].dt.quarter
agg_df['is_month_end'] = agg_df['timestamp'].dt.is_month_end.astype(int)
agg_df['is_quarter_end'] = agg_df['timestamp'].dt.is_quarter_end.astype(int)

# Vendor reliability
vendor_counts = agg_df.groupby('VendorID')['item_id'].count().to_dict()
agg_df['vendor_reliability'] = agg_df['VendorID'].map(vendor_counts).fillna(0)

# Price volatility
agg_df['price_volatility'] = agg_df['price_std'].fillna(0) / (agg_df['price_mean'] + 1e-6)
agg_df['order_frequency'] = agg_df['order_count']

# Rolling features
agg_df['rolling_mean_7d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)
agg_df['rolling_std_7d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=7, min_periods=1).std().fillna(0)
)
agg_df['rolling_mean_30d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
)
agg_df['rolling_std_30d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=30, min_periods=1).std().fillna(0)
)

# Lag features
agg_df['lag_7'] = agg_df.groupby('item_id')['target_value'].shift(7)
agg_df['lag_14'] = agg_df.groupby('item_id')['target_value'].shift(14)
agg_df['lag_30'] = agg_df.groupby('item_id')['target_value'].shift(30)

# Fill lags
agg_df['lag_7'] = agg_df.groupby('item_id')['lag_7'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)
agg_df['lag_14'] = agg_df.groupby('item_id')['lag_14'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)
agg_df['lag_30'] = agg_df.groupby('item_id')['lag_30'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)

# Seasonal features
agg_df['seasonal_trend'] = agg_df.groupby('item_id')['target_value'].transform('mean')
agg_df['seasonal_seasonal'] = (
    agg_df.groupby(['item_id', 'day_of_week'])['target_value'].transform('mean') - 
    agg_df['seasonal_trend']
)

# Encode categoricals
agg_df['customer_encoded'] = pd.Categorical(agg_df['CustomerID']).codes
agg_df['facility_encoded'] = pd.Categorical(agg_df['FacilityID']).codes
agg_df['product_encoded'] = pd.Categorical(agg_df['ProductID']).codes

# Get latest state for each item (as of TEST_END_DATE)
latest_data = agg_df.groupby('item_id').last().reset_index()
print(f"   Features ready for {len(latest_data):,} items")

# Generate predictions
print(f"\n   Loading models and generating predictions...")
from src.models.model_loader import load_models
from src.models.ensemble_predictor import EnsemblePredictor

models = load_models()
predictor = EnsemblePredictor(models=models)

# Prepare features
loader = DataLoader()
X_features, _ = loader.prepare_lightgbm_features(latest_data)

# Predict
predictions = predictor.predict(X_features)
latest_data['predicted_value'] = predictions['ensemble']
latest_data['predicted_reorder'] = (latest_data['predicted_value'] >= CLASSIFICATION_THRESHOLD).astype(int)

print(f"   ✓ Generated {len(latest_data):,} forward-looking predictions")

# Step 2: Load validation actuals
print(f"\n[2/3] Loading validation actuals...")
val_df = pd.read_csv('test/data/val_data.csv')
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

# Aggregate validation by item
val_agg = val_df.groupby('item_id').agg({
    'target_value': 'sum',
    'CustomerID': 'first'
}).reset_index()

print(f"   Validation items: {len(val_agg):,}")

# Step 3: Compare predictions with actuals
print(f"\n[3/3] Comparing predictions with actuals...")

# Merge
comparison = latest_data[['item_id', 'CustomerID', 'predicted_value', 'predicted_reorder']].merge(
    val_agg[['item_id', 'target_value']],
    on='item_id',
    how='inner'
)

print(f"   Matched items: {len(comparison):,}")

# Calculate metrics
comparison['actual_reorder'] = (comparison['target_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
comparison['error'] = comparison['predicted_value'] - comparison['target_value']
comparison['abs_error'] = abs(comparison['error'])
comparison['pct_error'] = (comparison['abs_error'] / (comparison['target_value'] + 1e-6)) * 100

# Overall metrics
mae = comparison['abs_error'].mean()
rmse = np.sqrt((comparison['error'] ** 2).mean())
mape = comparison['pct_error'].mean()

print(f"\n   Overall Metrics:")
print(f"   MAE:  {mae:.2f} units")
print(f"   RMSE: {rmse:.2f} units")
print(f"   MAPE: {mape:.2f}%")

# Classification metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

accuracy = accuracy_score(comparison['actual_reorder'], comparison['predicted_reorder'])
precision = precision_score(comparison['actual_reorder'], comparison['predicted_reorder'], zero_division=0)
recall = recall_score(comparison['actual_reorder'], comparison['predicted_reorder'], zero_division=0)
f1 = f1_score(comparison['actual_reorder'], comparison['predicted_reorder'], zero_division=0)

print(f"\n   Classification Metrics:")
print(f"   Accuracy:  {accuracy:.3f} ({accuracy*100:.1f}%)")
print(f"   Precision: {precision:.3f} ({precision*100:.1f}%)")
print(f"   Recall:    {recall:.3f} ({recall*100:.1f}%)")
print(f"   F1 Score:  {f1:.3f}")

# Confusion matrix
tn, fp, fn, tp = confusion_matrix(comparison['actual_reorder'], comparison['predicted_reorder']).ravel()

print(f"\n   Confusion Matrix:")
print(f"                    Predicted: No    Predicted: Yes")
print(f"   Actual: No       {tn:>13,}    {fp:>14,}")
print(f"   Actual: Yes      {fn:>13,}    {tp:>14,}")

# Save results
output_file = 'test/data/forward_predictions.csv'
comparison.to_csv(output_file, index=False)
print(f"\n✓ Saved forward-looking predictions: {output_file}")

# Customer-specific analysis
print(f"\n" + "=" * 80)
print("CUSTOMER-SPECIFIC RESULTS")
print("=" * 80)

for customer in ['scionhealth', 'mercy']:
    cust_data = comparison[comparison['CustomerID'] == customer]
    
    if len(cust_data) == 0:
        continue
    
    print(f"\n{customer.upper()}:")
    print(f"   Items:                {len(cust_data):,}")
    print(f"   Total Actual:         {cust_data['target_value'].sum():,.0f} units")
    print(f"   Total Predicted:      {cust_data['predicted_value'].sum():,.0f} units")
    vol_err = ((cust_data['predicted_value'].sum() - cust_data['target_value'].sum()) / cust_data['target_value'].sum()) * 100
    print(f"   Volume Error:         {vol_err:+.1f}%")
    print(f"   MAE:                  {cust_data['abs_error'].mean():.2f} units")
    
    # Classification
    cust_acc = accuracy_score(cust_data['actual_reorder'], cust_data['predicted_reorder'])
    cust_prec = precision_score(cust_data['actual_reorder'], cust_data['predicted_reorder'], zero_division=0)
    cust_rec = recall_score(cust_data['actual_reorder'], cust_data['predicted_reorder'], zero_division=0)
    cust_f1 = f1_score(cust_data['actual_reorder'], cust_data['predicted_reorder'], zero_division=0)
    
    print(f"   Accuracy:             {cust_acc:.3f} ({cust_acc*100:.1f}%)")
    print(f"   Precision:            {cust_prec:.3f} ({cust_prec*100:.1f}%)")
    print(f"   Recall:               {cust_rec:.3f} ({cust_rec*100:.1f}%)")
    print(f"   F1 Score:             {cust_f1:.3f}")
    
    # Confusion matrix
    if len(cust_data) > 0:
        try:
            ctn, cfp, cfn, ctp = confusion_matrix(cust_data['actual_reorder'], cust_data['predicted_reorder']).ravel()
            print(f"   True Positives:       {ctp:,}")
            print(f"   False Positives:      {cfp:,}")
            print(f"   False Negatives:      {cfn:,}")
            print(f"   True Negatives:       {ctn:,}")
        except:
            pass

print(f"\n" + "=" * 80)
print("FORWARD-LOOKING PREDICTION COMPLETE")
print("=" * 80)
print(f"\n✓ These metrics represent TRUE forward-looking performance")
print(f"✓ No validation data was used during prediction")
print(f"✓ Model only saw data up to {TEST_END_DATE}")
print(f"✓ Results saved to: {output_file}")
