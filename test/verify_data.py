"""
Verify extracted data quality and structure
"""

import pandas as pd
import os

def verify_dataset(filepath, dataset_name):
    """Verify a dataset file"""
    print(f"\n{'='*80}")
    print(f"VERIFYING {dataset_name.upper()}")
    print(f"{'='*80}")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return False
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n✓ File loaded: {filepath}")
    print(f"  Records: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    
    # Check required columns
    required_cols = [
        'item_id', 'timestamp', 'target_value',
        'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d', 'rolling_std_30d',
        'lag_7', 'lag_14', 'lag_30',
        'seasonal_trend', 'seasonal_seasonal',
        'day_of_week', 'day_of_month', 'month', 'quarter',
        'is_month_end', 'is_quarter_end',
        'price_volatility', 'order_frequency', 'vendor_reliability',
        'customer_encoded', 'facility_encoded', 'product_encoded'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\n❌ Missing columns: {missing_cols}")
        return False
    else:
        print(f"\n✓ All required columns present ({len(required_cols)} features)")
    
    # Check for nulls
    null_counts = df[required_cols].isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n⚠️  Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count:,} nulls ({count/len(df)*100:.1f}%)")
    else:
        print(f"\n✓ No null values in feature columns")
    
    # Parse timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Date range
    print(f"\n✓ Date range:")
    print(f"    Start: {df['timestamp'].min()}")
    print(f"    End: {df['timestamp'].max()}")
    print(f"    Days: {(df['timestamp'].max() - df['timestamp'].min()).days + 1}")
    
    # Unique items
    print(f"\n✓ Unique items: {df['item_id'].nunique():,}")
    
    # Target statistics
    print(f"\n✓ Target value statistics:")
    print(f"    Mean: {df['target_value'].mean():.2f}")
    print(f"    Median: {df['target_value'].median():.2f}")
    print(f"    Std: {df['target_value'].std():.2f}")
    print(f"    Min: {df['target_value'].min():.0f}")
    print(f"    Max: {df['target_value'].max():.0f}")
    print(f"    25th percentile: {df['target_value'].quantile(0.25):.2f}")
    print(f"    75th percentile: {df['target_value'].quantile(0.75):.2f}")
    
    # Feature ranges
    print(f"\n✓ Feature ranges:")
    print(f"    Rolling mean 7d: [{df['rolling_mean_7d'].min():.2f}, {df['rolling_mean_7d'].max():.2f}]")
    print(f"    Rolling std 7d: [{df['rolling_std_7d'].min():.2f}, {df['rolling_std_7d'].max():.2f}]")
    print(f"    Lag 7: [{df['lag_7'].min():.2f}, {df['lag_7'].max():.2f}]")
    
    # Sample records
    print(f"\n✓ Sample records (first 3):")
    sample_cols = ['item_id', 'timestamp', 'target_value', 'rolling_mean_7d', 'day_of_week']
    print(df[sample_cols].head(3).to_string(index=False))
    
    return True

if __name__ == "__main__":
    print("="*80)
    print("DATA VERIFICATION")
    print("="*80)
    
    test_file = "test/data/test_data.csv"
    val_file = "test/data/val_data.csv"
    
    test_ok = verify_dataset(test_file, "Test Data")
    val_ok = verify_dataset(val_file, "Validation Data")
    
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    print(f"\nTest Data: {'✓ PASS' if test_ok else '❌ FAIL'}")
    print(f"Validation Data: {'✓ PASS' if val_ok else '❌ FAIL'}")
    
    if test_ok and val_ok:
        print(f"\n✓ All datasets verified successfully!")
    else:
        print(f"\n❌ Some datasets failed verification")
