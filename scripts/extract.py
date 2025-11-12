"""
Data Extraction Script

Generates context and validation data files needed for testing and predictions.

What it does:
  1. Extracts order history from SOURCE_DATA_FILE
  2. Splits into context period (for features) and validation period (for testing)
  3. Engineers features on context data only (prevents data leakage)
  4. Saves validation data as actuals only (no features)

Output files:
  - test_data.csv: Context data with engineered features
  - val_data.csv: Validation data with actuals only

Usage:
    python scripts/extract.py

Configuration (.env):
    SOURCE_DATA_FILE - Path to order history CSV
    TEST_DATA_DIR - Output directory for extracted files
    TOTAL_EXTRACTION_DAYS - Total days to extract (default: 90)
    VALIDATION_DAYS - Days for validation period (default: 14)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.env_config import (
    SOURCE_DATA_FILE, TEST_DATA_DIR, VALIDATION_DAYS, TEST_DAYS, 
    TOTAL_EXTRACTION_DAYS, ROLLING_WINDOW_SHORT, ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_MIN_PERIODS, ROLLING_MAX_WINDOW_SIZE
)

# Configuration
SOURCE_FILE = SOURCE_DATA_FILE
OUTPUT_DIR = TEST_DATA_DIR + "/"

print("=" * 80)
print("DATA EXTRACTION - Generate Context & Validation Files")
print("=" * 80)
print(f"\nConfiguration:")
print(f"  Total extraction days: {TOTAL_EXTRACTION_DAYS}")
print(f"  Validation days: {VALIDATION_DAYS}")
print(f"  Test days: {TEST_DAYS}")
print(f"  Output directory: {TEST_DATA_DIR}")
print("=" * 80)

# Step 1: Load data
print("\n[1/7] Loading order history data...")
df = pd.read_csv(SOURCE_FILE)
print(f"   Loaded {len(df):,} records")

# Step 2: Parse dates and filter
print("\n[2/7] Filtering for extraction period...")
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
max_date = df['CreateDate'].max()

# Calculate dates correctly:
# - Validation period: last VALIDATION_DAYS days
# - Context period: TOTAL_EXTRACTION_DAYS days BEFORE validation period
validation_start = max_date - timedelta(days=VALIDATION_DAYS-1)
context_start = validation_start - timedelta(days=TOTAL_EXTRACTION_DAYS)
context_end = validation_start - timedelta(days=1)

# Total extraction includes both context and validation
min_date = context_start
df_filtered = df[df['CreateDate'] >= min_date].copy()

print(f"   Full date range: {min_date.date()} to {max_date.date()}")
print(f"   Total days: {(max_date - min_date).days + 1}")
print(f"   Filtered to {len(df_filtered):,} records")

# Step 3: Determine split date FIRST (before any feature engineering)
print("\n[3/7] Determining split date...")
split_date = context_end
context_days = (context_end - context_start).days + 1
validation_days = (max_date - validation_start).days + 1

print(f"   Split date: {split_date.date()}")
print(f"   Context period: {context_start.date()} to {context_end.date()} ({context_days} days)")
print(f"   Validation period: {validation_start.date()} to {max_date.date()} ({validation_days} days)")

# Step 4: Split data BEFORE feature engineering (CRITICAL FIX)
print("\n[4/7] Splitting data BEFORE feature engineering...")
df_test = df_filtered[df_filtered['CreateDate'] <= split_date].copy()
df_val = df_filtered[df_filtered['CreateDate'] > split_date].copy()
print(f"   Test records: {len(df_test):,}")
print(f"   Validation records: {len(df_val):,}")

# Step 5: Process TEST data with full feature engineering
print("\n[5/7] Processing TEST data with feature engineering...")

# Create item_id and aggregate TEST data
df_test['item_id'] = (df_test['CustomerID'].astype(str) + '_' + 
                      df_test['FacilityID'].astype(str) + '_' + 
                      df_test['ProductID'].astype(str))

agg_test = df_test.groupby(['item_id', 'CreateDate']).agg({
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

agg_test.columns = [
    'item_id', 'timestamp', 'target_value', 'order_count', 
    'price_mean', 'price_std', 'CustomerID', 'FacilityID', 
    'ProductID', 'MainProductID', 'ProductName', 'CategoryName',
    'DepartmentID', 'VendorID', 'VendorName', 'UserID', 'PortalID'
]

print(f"   Aggregated to {len(agg_test):,} item-date combinations")

# Engineer features on TEST data only
print("   Engineering features on TEST data...")
agg_test = agg_test.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

# Time-based features
agg_test['day_of_week'] = agg_test['timestamp'].dt.dayofweek
agg_test['day_of_month'] = agg_test['timestamp'].dt.day
agg_test['month'] = agg_test['timestamp'].dt.month
agg_test['quarter'] = agg_test['timestamp'].dt.quarter
agg_test['is_month_end'] = agg_test['timestamp'].dt.is_month_end.astype(int)
agg_test['is_quarter_end'] = agg_test['timestamp'].dt.is_quarter_end.astype(int)

# Vendor reliability
vendor_counts = df_test.groupby('VendorID')['OrderID'].count().to_dict()
agg_test['vendor_reliability'] = agg_test['VendorID'].map(vendor_counts).fillna(0)

# Price volatility
agg_test['price_volatility'] = agg_test['price_std'].fillna(0) / (agg_test['price_mean'] + 1e-6)
agg_test['order_frequency'] = agg_test['order_count']

# Rolling features (calculated on TEST data only)
agg_test['rolling_mean_7d'] = agg_test.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
)
agg_test['rolling_std_7d'] = agg_test.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=ROLLING_WINDOW_SHORT, min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
)
agg_test['rolling_mean_30d'] = agg_test.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=min(ROLLING_WINDOW_LONG, len(x)), min_periods=ROLLING_WINDOW_MIN_PERIODS).mean()
)
agg_test['rolling_std_30d'] = agg_test.groupby('item_id')['target_value'].transform(
    lambda x: x.rolling(window=min(ROLLING_WINDOW_LONG, len(x)), min_periods=ROLLING_WINDOW_MIN_PERIODS).std().fillna(0)
)

# Lag features
agg_test['lag_7'] = agg_test.groupby('item_id')['target_value'].shift(7)
agg_test['lag_14'] = agg_test.groupby('item_id')['target_value'].shift(14)
agg_test['lag_30'] = agg_test.groupby('item_id')['target_value'].shift(30)

# Fill lag NaNs
agg_test['lag_7'] = agg_test.groupby('item_id')['lag_7'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)
agg_test['lag_14'] = agg_test.groupby('item_id')['lag_14'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)
agg_test['lag_30'] = agg_test.groupby('item_id')['lag_30'].transform(
    lambda x: x.fillna(x.iloc[0] if len(x) > 0 else 0)
)

# Seasonal features
agg_test['seasonal_trend'] = agg_test.groupby('item_id')['target_value'].transform('mean')
agg_test['seasonal_seasonal'] = (
    agg_test.groupby(['item_id', 'day_of_week'])['target_value'].transform('mean') - 
    agg_test['seasonal_trend']
)

# Encode categorical variables
agg_test['customer_encoded'] = pd.Categorical(agg_test['CustomerID']).codes
agg_test['facility_encoded'] = pd.Categorical(agg_test['FacilityID']).codes
agg_test['product_encoded'] = pd.Categorical(agg_test['ProductID']).codes

print(f"   ✓ Test data processed: {len(agg_test):,} records with features")

# Step 6: Process VALIDATION data (ACTUALS ONLY - NO FEATURES)
print("\n[6/7] Processing VALIDATION data (actuals only, NO features)...")

# Create item_id and aggregate VALIDATION data
df_val['item_id'] = (df_val['CustomerID'].astype(str) + '_' + 
                     df_val['FacilityID'].astype(str) + '_' + 
                     df_val['ProductID'].astype(str))

agg_val = df_val.groupby(['item_id', 'CreateDate']).agg({
    'OrderUnits': 'sum',
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

agg_val.columns = [
    'item_id', 'timestamp', 'target_value', 'CustomerID', 'FacilityID', 
    'ProductID', 'MainProductID', 'ProductName', 'CategoryName',
    'DepartmentID', 'VendorID', 'VendorName', 'UserID', 'PortalID'
]

print(f"   ✓ Validation data processed: {len(agg_val):,} records (ACTUALS ONLY)")
print(f"   ✓ NO FEATURES in validation data (prevents data leakage)")

# Step 7: Save files
print("\n[7/7] Saving files...")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save test data with features
test_output = f"{OUTPUT_DIR}test_data.csv"
agg_test.to_csv(test_output, index=False)
print(f"\n✓ Test data saved: {test_output}")
print(f"  - {len(agg_test):,} records")
print(f"  - {agg_test['item_id'].nunique():,} unique items")
print(f"  - Date range: {agg_test['timestamp'].min().date()} to {agg_test['timestamp'].max().date()}")
print(f"  - Columns: {len(agg_test.columns)} (includes features)")

# Save validation data WITHOUT features
val_output = f"{OUTPUT_DIR}val_data.csv"
agg_val.to_csv(val_output, index=False)
print(f"\n✓ Validation data saved: {val_output}")
print(f"  - {len(agg_val):,} records")
print(f"  - {agg_val['item_id'].nunique():,} unique items")
print(f"  - Date range: {agg_val['timestamp'].min().date()} to {agg_val['timestamp'].max().date()}")
print(f"  - Columns: {len(agg_val.columns)} (ACTUALS ONLY - NO FEATURES)")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

print(f"\nContext Data (test_data.csv):")
print(f"  Records:       {len(agg_test):,}")
print(f"  Unique items:  {agg_test['item_id'].nunique():,}")
print(f"  Target mean:   {agg_test['target_value'].mean():.2f}")
print(f"  Target median: {agg_test['target_value'].median():.2f}")
print(f"  Target range:  [{agg_test['target_value'].min():.0f}, {agg_test['target_value'].max():.0f}]")

print(f"\nValidation Data (val_data.csv):")
print(f"  Records:       {len(agg_val):,}")
print(f"  Unique items:  {agg_val['item_id'].nunique():,}")
print(f"  Target mean:   {agg_val['target_value'].mean():.2f}")
print(f"  Target median: {agg_val['target_value'].median():.2f}")
print(f"  Target range:  [{agg_val['target_value'].min():.0f}, {agg_val['target_value'].max():.0f}]")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE!")
print("=" * 80)
print("\n✓ Context data (test_data.csv) - Features engineered from context period")
print("✓ Validation data (val_data.csv) - Actuals only, no features")
print("✓ No data leakage - Validation features not calculated")
print("✓ Ready for predictions and testing")
print("\n" + "=" * 80)
