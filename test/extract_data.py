"""
Extract and process order history data for forecasting
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
    SOURCE_DATA_FILE, TEST_DATA_DIR, VALIDATION_DAYS, TEST_DAYS, 
    TOTAL_EXTRACTION_DAYS, ROLLING_WINDOW_SHORT, ROLLING_WINDOW_LONG,
    ROLLING_WINDOW_MIN_PERIODS, ROLLING_MAX_WINDOW_SIZE
)

# Configuration
SOURCE_FILE = SOURCE_DATA_FILE
OUTPUT_DIR = TEST_DATA_DIR + "/"

print("=" * 80)
print("ORDER HISTORY DATA EXTRACTION")
print("=" * 80)

# Step 1: Load data
print("\n[1/6] Loading order history data...")
df = pd.read_csv(SOURCE_FILE)
print(f"   Loaded {len(df):,} records")

# Step 2: Parse dates and filter
print("\n[2/6] Filtering for last 28 days...")
df['CreateDate'] = pd.to_datetime(df['CreateDate'])
max_date = df['CreateDate'].max()
min_date = max_date - timedelta(days=TOTAL_EXTRACTION_DAYS-1)  # Total extraction days
df_filtered = df[df['CreateDate'] >= min_date].copy()
print(f"   Date range: {min_date.date()} to {max_date.date()}")
print(f"   Filtered to {len(df_filtered):,} records")

# Step 3: Create item_id and aggregate
print("\n[3/6] Aggregating by item and date...")
df_filtered['item_id'] = (df_filtered['CustomerID'].astype(str) + '_' + 
                          df_filtered['FacilityID'].astype(str) + '_' + 
                          df_filtered['ProductID'].astype(str))

# Aggregate by item_id and date
agg_df = df_filtered.groupby(['item_id', 'CreateDate']).agg({
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

# Flatten column names
agg_df.columns = ['item_id', 'timestamp', 'target_value', 'order_count', 
                  'price_mean', 'price_std', 'CustomerID', 'FacilityID', 
                  'ProductID', 'MainProductID', 'ProductName', 'CategoryName',
                  'DepartmentID', 'VendorID', 'VendorName', 'UserID', 'PortalID']

print(f"   Aggregated to {len(agg_df):,} item-date combinations")
print(f"   Unique items: {agg_df['item_id'].nunique():,}")

# Step 4: Calculate features (vectorized approach)
print("\n[4/6] Engineering features...")

# Sort by item and timestamp
agg_df = agg_df.sort_values(['item_id', 'timestamp']).reset_index(drop=True)

# Time-based features (vectorized)
agg_df['day_of_week'] = agg_df['timestamp'].dt.dayofweek
agg_df['day_of_month'] = agg_df['timestamp'].dt.day
agg_df['month'] = agg_df['timestamp'].dt.month
agg_df['quarter'] = agg_df['timestamp'].dt.quarter
agg_df['is_month_end'] = agg_df['timestamp'].dt.is_month_end.astype(int)
agg_df['is_quarter_end'] = agg_df['timestamp'].dt.is_quarter_end.astype(int)

# Calculate vendor reliability
vendor_reliability = df_filtered.groupby('VendorID')['OrderID'].count().to_dict()
agg_df['vendor_reliability'] = agg_df['VendorID'].map(vendor_reliability).fillna(0)

# Price volatility
agg_df['price_volatility'] = agg_df['price_std'].fillna(0) / (agg_df['price_mean'] + 1e-6)
agg_df['order_frequency'] = agg_df['order_count']

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
agg_df['seasonal_seasonal'] = agg_df.groupby(['item_id', 'day_of_week'])['target_value'].transform('mean') - agg_df['seasonal_trend']

result_df = agg_df
print(f"   Features calculated for {len(result_df):,} records")

# Step 5: Encode categorical variables
print("\n[5/6] Encoding categorical variables...")
result_df['customer_encoded'] = pd.Categorical(result_df['CustomerID']).codes
result_df['facility_encoded'] = pd.Categorical(result_df['FacilityID']).codes
result_df['product_encoded'] = pd.Categorical(result_df['ProductID']).codes

# Step 6: Split into test and validation sets
print("\n[6/6] Splitting into test and validation sets...")
split_date = max_date - timedelta(days=VALIDATION_DAYS-1)  # Last N days for validation

val_df = result_df[result_df['timestamp'] > split_date].copy()
test_df = result_df[result_df['timestamp'] <= split_date].copy()

print(f"   Test data: {len(test_df):,} records ({test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()})")
print(f"   Validation data: {len(val_df):,} records ({val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()})")

# Reorder columns to match expected format
column_order = [
    'PortalID', 'CustomerID', 'FacilityID', 'OrderID', 'ProductID', 'MainProductID',
    'ProductName', 'CategoryName', 'DepartmentID', 'timestamp', 'target_value',
    'price_mean', 'VendorID', 'VendorName', 'UserID', 'order_count', 'item_id',
    'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d', 'rolling_std_30d',
    'lag_7', 'lag_14', 'lag_30', 'seasonal_trend', 'seasonal_seasonal',
    'day_of_week', 'day_of_month', 'month', 'quarter', 'is_month_end', 'is_quarter_end',
    'price_volatility', 'order_frequency', 'vendor_reliability',
    'customer_encoded', 'facility_encoded', 'product_encoded'
]

# Add OrderID placeholder (use order_count as proxy)
test_df['OrderID'] = test_df['order_count']
val_df['OrderID'] = val_df['order_count']

# Rename price_mean back to Price for compatibility
test_df['Price'] = test_df['price_mean']
val_df['Price'] = val_df['price_mean']

# Select and reorder columns
available_cols = [col for col in column_order if col in test_df.columns]
test_df = test_df[available_cols]
val_df = val_df[available_cols]

# Save to CSV
print("\n" + "=" * 80)
print("SAVING FILES")
print("=" * 80)

# Create output directory if it doesn't exist
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

test_output = f"{OUTPUT_DIR}test_data.csv"
val_output = f"{OUTPUT_DIR}val_data.csv"

test_df.to_csv(test_output, index=False)
val_df.to_csv(val_output, index=False)

print(f"\n✓ Test data saved: {test_output}")
print(f"  - {len(test_df):,} records")
print(f"  - {test_df['item_id'].nunique():,} unique items")
print(f"  - Date range: {test_df['timestamp'].min().date()} to {test_df['timestamp'].max().date()}")

print(f"\n✓ Validation data saved: {val_output}")
print(f"  - {len(val_df):,} records")
print(f"  - {val_df['item_id'].nunique():,} unique items")
print(f"  - Date range: {val_df['timestamp'].min().date()} to {val_df['timestamp'].max().date()}")

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nTest Data:")
print(f"  Target mean: {test_df['target_value'].mean():.2f}")
print(f"  Target std: {test_df['target_value'].std():.2f}")
print(f"  Target median: {test_df['target_value'].median():.2f}")
print(f"  Target range: [{test_df['target_value'].min():.0f}, {test_df['target_value'].max():.0f}]")

print(f"\nValidation Data:")
print(f"  Target mean: {val_df['target_value'].mean():.2f}")
print(f"  Target std: {val_df['target_value'].std():.2f}")
print(f"  Target median: {val_df['target_value'].median():.2f}")
print(f"  Target range: [{val_df['target_value'].min():.0f}, {val_df['target_value'].max():.0f}]")

print("\n" + "=" * 80)
print("EXTRACTION COMPLETE!")
print("=" * 80)
