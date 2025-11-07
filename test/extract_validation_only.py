#!/usr/bin/env python3
"""
Extract ONLY validation data from new source file
Keeps existing test data, extracts new validation period
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings('ignore')
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import (
    SOURCE_DATA_FILE, TEST_DATA_DIR, VALIDATION_DAYS, 
    ROLLING_WINDOW_SHORT, ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_MIN_PERIODS, ROLLING_MAX_WINDOW_SIZE
)

# Configuration
SOURCE_FILE = SOURCE_DATA_FILE
OUTPUT_DIR = TEST_DATA_DIR + "/"

print("=" * 80)
print("VALIDATION DATA EXTRACTION (NEW SOURCE)")
print("=" * 80)
print(f"Source: {SOURCE_FILE}")
print(f"Extracting validation data only (keeping existing test data)")
print("=" * 80)

# Step 1: Load new source data
print("\n[1/6] Loading NEW order history data...")
df = pd.read_csv(SOURCE_FILE)
print(f"   Loaded {len(df):,} records from new source")

# Step 2: Parse dates and filter for validation period
print(f"\n[2/6] Filtering for validation period ({VALIDATION_DAYS} days)...")
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
max_date = df['CreateDate'].max()
validation_start = max_date - timedelta(days=VALIDATION_DAYS-1)  # Last N days for validation

print(f"   Max date in new data: {max_date.date()}")
print(f"   Validation period: {validation_start.date()} to {max_date.date()}")

# Filter for validation period only
df_validation = df[df['CreateDate'] >= validation_start].copy()
print(f"   Filtered to {len(df_validation):,} validation records")

if len(df_validation) == 0:
    print("❌ No validation data found in the specified period!")
    sys.exit(1)

# Step 3: Create item_id and aggregate
print("\n[3/6] Aggregating validation data by item and date...")
df_validation['item_id'] = (df_validation['CustomerID'].astype(str) + '_' + 
                           df_validation['FacilityID'].astype(str) + '_' + 
                           df_validation['ProductID'].astype(str))

# Aggregate by item_id and date
agg_df = df_validation.groupby(['item_id', 'CreateDate']).agg({
    'PortalID': 'first',
    'CustomerID': 'first', 
    'FacilityID': 'first',
    'OrderID': 'first',
    'ProductID': 'first',
    'MainProductID': 'first',
    'ProductName': 'first',
    'CategoryName': 'first',
    'DepartmentID': 'first',
    'OrderUnits': 'sum',
    'Price': 'mean',
    'VendorID': 'first',
    'VendorName': 'first',
    'UserID': 'first'
}).reset_index()

# Rename columns to match expected format
agg_df = agg_df.rename(columns={
    'CreateDate': 'timestamp',
    'OrderUnits': 'target_value',
    'Price': 'price_mean'
})

print(f"   Aggregated to {len(agg_df):,} item-date combinations")
print(f"   Unique items: {agg_df['item_id'].nunique():,}")

# Step 4: Engineer features for validation data
print("\n[4/6] Engineering features for validation data...")

# Add order count (always 1 for aggregated data)
agg_df['order_count'] = 1

# Rolling and lag features (grouped by item)
print("   Calculating rolling statistics and lags...")
agg_df['rolling_mean_7d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
)
agg_df['rolling_std_7d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
)
agg_df['rolling_mean_30d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=min(ROLLING_WINDOW_LONG, len(x)), min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
)
agg_df['rolling_std_30d'] = agg_df.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=min(ROLLING_WINDOW_LONG, len(x)), min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
)

# Lag features
agg_df['lag_7'] = agg_df.groupby('item_id')['target_value'].shift(7)
agg_df['lag_14'] = agg_df.groupby('item_id')['target_value'].shift(14)
agg_df['lag_30'] = agg_df.groupby('item_id')['target_value'].shift(30)

# Fill lag NaNs with first value per item
agg_df['lag_7'] = agg_df.groupby('item_id')['lag_7'].transform(lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0))
agg_df['lag_14'] = agg_df.groupby('item_id')['lag_14'].transform(lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0))
agg_df['lag_30'] = agg_df.groupby('item_id')['lag_30'].transform(lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0))

# Simplified seasonal features (mean-based trend and day-of-week seasonality)
print("   Calculating seasonal features...")
agg_df['seasonal_trend'] = agg_df.groupby('item_id')['target_value'].transform('mean')
agg_df['seasonal_seasonal'] = 0  # Simplified for validation data

# Calendar features
agg_df['day_of_week'] = agg_df['timestamp'].dt.dayofweek
agg_df['day_of_month'] = agg_df['timestamp'].dt.day
agg_df['month'] = agg_df['timestamp'].dt.month
agg_df['quarter'] = agg_df['timestamp'].dt.quarter
agg_df['is_month_end'] = agg_df['timestamp'].dt.is_month_end.astype(int)
agg_df['is_quarter_end'] = agg_df['timestamp'].dt.is_quarter_end.astype(int)

# Business features
agg_df['price_volatility'] = agg_df.groupby('item_id')['price_mean'].transform('std').fillna(0)
agg_df['order_frequency'] = agg_df['order_count']
agg_df['vendor_reliability'] = 1  # Simplified

print(f"   Features calculated for {len(agg_df):,} records")

# Step 5: Encoding categorical variables
print("\n[5/6] Encoding categorical variables...")

# Create categorical encodings
agg_df['customer_encoded'] = pd.Categorical(agg_df['CustomerID']).codes
agg_df['facility_encoded'] = pd.Categorical(agg_df['FacilityID']).codes  
agg_df['product_encoded'] = pd.Categorical(agg_df['ProductID']).codes

# Step 6: Save validation data
print("\n[6/6] Saving validation data...")

# Select final columns (matching test data format)
final_columns = [
    'PortalID', 'CustomerID', 'FacilityID', 'OrderID', 'ProductID', 'MainProductID',
    'ProductName', 'CategoryName', 'DepartmentID', 'timestamp', 'target_value',
    'price_mean', 'VendorID', 'VendorName', 'UserID', 'order_count', 'item_id',
    'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d', 'rolling_std_30d',
    'lag_7', 'lag_14', 'lag_30', 'seasonal_trend', 'seasonal_seasonal',
    'day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_end', 'is_quarter_end',
    'price_volatility', 'order_frequency', 'vendor_reliability',
    'customer_encoded', 'facility_encoded', 'product_encoded'
]

result_df = agg_df[final_columns].copy()

# Save validation data
val_file = OUTPUT_DIR + "val_data.csv"
result_df.to_csv(val_file, index=False)

print("=" * 80)
print("SAVING FILES")
print("=" * 80)

print(f"✓ Validation data saved: {val_file}")
print(f"  - {len(result_df):,} records")
print(f"  - {result_df['item_id'].nunique():,} unique items")
print(f"  - Date range: {result_df['timestamp'].min()} to {result_df['timestamp'].max()}")

print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print("Validation Data:")
print(f"  Target mean: {result_df['target_value'].mean():.2f}")
print(f"  Target std: {result_df['target_value'].std():.2f}")
print(f"  Target median: {result_df['target_value'].median():.2f}")
print(f"  Target range: [{result_df['target_value'].min()}, {result_df['target_value'].max()}]")

print("=" * 80)
print("VALIDATION EXTRACTION COMPLETE!")
print("=" * 80)
print("✓ Existing test data preserved")
print("✓ New validation data extracted")
print("✓ Ready for predictions with existing test data")