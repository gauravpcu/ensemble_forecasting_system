#!/usr/bin/env python3
"""
Extract all predictions for specific customers across entire test period
"""

import pandas as pd
import sys

def extract_all_facilities(customers):
    """Extract all predictions for specific customers"""
    
    print("=" * 80)
    print(f"EXTRACTING ALL PREDICTIONS FOR: {', '.join(customers).upper()}")
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
    
    # Filter by customers
    filtered_df = predictions_df[predictions_df['CustomerID'].isin(customers)].copy()
    
    print(f"✓ Found {len(filtered_df):,} records for {', '.join(customers)}")
    
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
    
    # Sort by customer, facility, date, predicted value
    output_df = output_df.sort_values(['CustomerID', 'FacilityID', 'date', 'predicted_value'], 
                                      ascending=[True, True, True, False])
    
    # Save to CSV
    output_file = f'test/data/predictions_all_{"_".join(customers)}.csv'
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
        print(f"  Total Records:     {len(customer_df):,}")
        print(f"  Date Range:        {customer_df['date'].min()} to {customer_df['date'].max()}")
        print(f"  Facilities:        {customer_df['FacilityID'].nunique()}")
        print(f"  Products:          {customer_df['ProductID'].nunique()}")
        print(f"  Total Actual:      {customer_df['target_value'].sum():,.0f} units")
        print(f"  Total Predicted:   {customer_df['predicted_value'].sum():,.0f} units")
        print(f"  Avg Actual:        {customer_df['target_value'].mean():.2f} units")
        print(f"  Avg Predicted:     {customer_df['predicted_value'].mean():.2f} units")
        
        # Facility breakdown
        facility_summary = customer_df.groupby('FacilityID').agg({
            'ProductID': 'count',
            'target_value': 'sum',
            'predicted_value': 'sum'
        }).reset_index()
        facility_summary.columns = ['FacilityID', 'Records', 'Actual', 'Predicted']
        facility_summary = facility_summary.sort_values('Actual', ascending=False)
        
        print(f"\n  Top 10 Facilities by Volume:")
        print(f"  {'Facility':<10} {'Records':<8} {'Actual':<10} {'Predicted':<10} {'Error%':<8}")
        print(f"  {'-'*50}")
        
        for _, row in facility_summary.head(10).iterrows():
            error_pct = ((row['Predicted'] - row['Actual']) / row['Actual'] * 100) if row['Actual'] > 0 else 0
            print(f"  {row['FacilityID']:<10} {row['Records']:<8} {row['Actual']:<10.0f} {row['Predicted']:<10.0f} {error_pct:<+8.1f}")
    
    print("\n" + "=" * 80)
    print(f"✅ COMPLETE! File saved: {output_file}")
    print(f"File size: {len(output_df):,} records")
    print("=" * 80)
    
    return output_df

if __name__ == "__main__":
    # Default: ScionHealth and Mercy
    customers = ['scionhealth', 'mercy']
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        customers = sys.argv[1].split(',')
    
    extract_all_facilities(customers)
