"""
Quick statistics viewer for extracted data
"""

import pandas as pd

def show_stats(filepath, name):
    """Show quick statistics for a dataset"""
    print(f"\n{'='*80}")
    print(f"{name.upper()}")
    print(f"{'='*80}")
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"\n📊 Overview:")
    print(f"   Records: {len(df):,}")
    print(f"   Unique items: {df['item_id'].nunique():,}")
    print(f"   Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days + 1}")
    
    print(f"\n📦 Order Statistics:")
    print(f"   Total units: {df['target_value'].sum():,.0f}")
    print(f"   Avg units/day: {df['target_value'].mean():.2f}")
    print(f"   Median units/day: {df['target_value'].median():.2f}")
    print(f"   Max single order: {df['target_value'].max():,.0f}")
    
    print(f"\n🏢 Top 10 Customers by Volume:")
    top_customers = df.groupby('CustomerID')['target_value'].sum().sort_values(ascending=False).head(10)
    for i, (customer, volume) in enumerate(top_customers.items(), 1):
        print(f"   {i:2d}. {customer:30s} {volume:>10,.0f} units")
    
    print(f"\n📦 Top 10 Products by Volume:")
    top_products = df.groupby('ProductName')['target_value'].sum().sort_values(ascending=False).head(10)
    for i, (product, volume) in enumerate(top_products.items(), 1):
        product_name = str(product)[:50] if pd.notna(product) else "Unknown"
        print(f"   {i:2d}. {product_name:50s} {volume:>10,.0f} units")
    
    print(f"\n🏭 Top 5 Vendors by Order Count:")
    top_vendors = df.groupby('VendorName')['order_count'].sum().sort_values(ascending=False).head(5)
    for i, (vendor, count) in enumerate(top_vendors.items(), 1):
        vendor_name = str(vendor)[:40] if pd.notna(vendor) else "Unknown"
        print(f"   {i}. {vendor_name:40s} {count:>8,.0f} orders")
    
    print(f"\n📅 Daily Order Pattern:")
    daily = df.groupby('day_of_week')['target_value'].agg(['sum', 'mean', 'count'])
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for day_num, day_name in enumerate(days):
        if day_num in daily.index:
            row = daily.loc[day_num]
            print(f"   {day_name}: {row['sum']:>10,.0f} units ({row['mean']:>6.2f} avg, {row['count']:>6,.0f} records)")

if __name__ == "__main__":
    print("="*80)
    print("EXTRACTED DATA STATISTICS")
    print("="*80)
    
    show_stats('test/data/test_data.csv', 'Test Data (Oct 1-15)')
    show_stats('test/data/val_data.csv', 'Validation Data (Oct 16-28)')
    
    print(f"\n{'='*80}")
    print("COMPLETE")
    print(f"{'='*80}\n")
