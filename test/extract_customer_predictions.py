#!/usr/bin/env python3
"""
Extract predictions for specific customers and dates
"""

import pandas as pd
import sys

def extract_predictions(customers, date):
    """Extract predictions for specific customers and date"""
    
    print("=" * 80)
    print(f"EXTRACTING PREDICTIONS FOR {date}")
    print("=" * 80)
    
    # Load predictions
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        print(f"✓ Loaded predictions: {len(predictions_df):,} records")
    except FileNotFoundError:
        print("❌ Predictions file not found")
        return
    
    # Convert timestamp to date
    predictions_df['date'] = pd.to_datetime(predictions_df['timestamp']).dt.date
    target_date = pd.to_datetime(date).date()
    
    # Filter by customers and date
    filtered_df = predictions_df[
        (predictions_df['CustomerID'].isin(customers)) & 
        (predictions_df['date'] == target_date)
    ].copy()
    
    print(f"✓ Found {len(filtered_df):,} records for {', '.join(customers)} on {date}")
    
    if len(filtered_df) == 0:
        print("\n❌ No records found. Available dates:")
        print(predictions_df['date'].unique()[:10])
        return
    
    # Select relevant columns
    output_columns = [
        'CustomerID', 'FacilityID', 'ProductID', 'ProductName', 
        'CategoryName', 'VendorName', 'timestamp', 'date',
        'target_value', 'predicted_value', 
        'lightgbm_prediction', 'deepar_prediction'
    ]
    
    # Keep only columns that exist
    output_columns = [col for col in output_columns if col in filtered_df.columns]
    output_df = filtered_df[output_columns].copy()
    
    # Sort by customer, facility, predicted value
    output_df = output_df.sort_values(['CustomerID', 'FacilityID', 'predicted_value'], ascending=[True, True, False])
    
    # Save to CSV
    output_file = f'test/data/predictions_{date.replace("/", "-")}_{"_".join(customers)}.csv'
    output_df.to_csv(output_file, index=False)
    print(f"✓ Saved to: {output_file}")
    
    # Show summary statistics
    print(f"\n📊 SUMMARY BY CUSTOMER:")
    print("-" * 80)
    
    for customer in customers:
        customer_df = output_df[output_df['CustomerID'] == customer]
        
        if len(customer_df) == 0:
            print(f"\n{customer.upper()}: No records found")
            continue
        
        print(f"\n{customer.upper()}:")
        print(f"  Records:           {len(customer_df):,}")
        print(f"  Facilities:        {customer_df['FacilityID'].nunique()}")
        print(f"  Products:          {customer_df['ProductID'].nunique()}")
        print(f"  Total Actual:      {customer_df['target_value'].sum():,.0f} units")
        print(f"  Total Predicted:   {customer_df['predicted_value'].sum():,.0f} units")
        print(f"  Avg Actual:        {customer_df['target_value'].mean():.2f} units")
        print(f"  Avg Predicted:     {customer_df['predicted_value'].mean():.2f} units")
        print(f"  Max Predicted:     {customer_df['predicted_value'].max():.2f} units")
        
        # Top 10 predictions
        print(f"\n  Top 10 Predicted Items:")
        top_10 = customer_df.nlargest(10, 'predicted_value')[['FacilityID', 'ProductID', 'ProductName', 'target_value', 'predicted_value']]
        for idx, row in top_10.iterrows():
            product_name = row['ProductName'][:40] if pd.notna(row['ProductName']) else 'N/A'
            print(f"    Facility {row['FacilityID']:<4} | Product {row['ProductID']:<8} | {product_name:<40} | Actual: {row['target_value']:>6.0f} | Pred: {row['predicted_value']:>8.2f}")
    
    print("\n" + "=" * 80)
    print(f"✅ COMPLETE! File saved: {output_file}")
    print("=" * 80)
    
    return output_df

if __name__ == "__main__":
    # Default: ScionHealth and Mercy on 10/15
    customers = ['scionhealth', 'mercy']
    date = '2025-10-15'
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        date = sys.argv[1]
    if len(sys.argv) > 2:
        customers = sys.argv[2].split(',')
    
    extract_predictions(customers, date)
