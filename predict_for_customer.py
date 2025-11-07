"""
Predict items for a specific customer and facility for the next 14 days
Simple interface: Provide customer_id, facility_id, and start_date
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_loader import DataLoader
from model_loader import load_models
from ensemble_predictor import EnsemblePredictor


def predict_for_customer(customer_id, facility_id, start_date, output_file=None):
    """
    Predict items for a specific customer and facility for next 14 days
    
    Args:
        customer_id: Customer ID (e.g., 'scionhealth')
        facility_id: Facility ID (e.g., '287')
        start_date: Start date for predictions (e.g., '2025-11-05')
        output_file: Optional output file path
        
    Returns:
        DataFrame with predictions for all products
    """
    
    print("\n" + "="*80)
    print("CUSTOMER-SPECIFIC DEMAND FORECAST")
    print("="*80)
    print(f"Customer ID: {customer_id}")
    print(f"Facility ID: {facility_id}")
    print(f"Forecast Start Date: {start_date}")
    print(f"Forecast Period: 14 days")
    print("="*80 + "\n")
    
    # Step 1: Load historical data for this customer/facility
    print("Step 1: Loading historical data...")
    loader = DataLoader()
    
    # Load recent data (we'll use test data as it's most recent)
    df = loader.load_data('test', sample_size=None)  # Load all test data
    
    # Filter for this customer and facility
    customer_data = df[
        (df['CustomerID'] == customer_id) & 
        (df['FacilityID'].astype(str) == str(facility_id))
    ].copy()
    
    if len(customer_data) == 0:
        print(f"❌ No data found for Customer '{customer_id}' and Facility '{facility_id}'")
        print("\nTrying to find similar entries...")
        
        # Show available customers
        print(f"\nAvailable customers (showing first 10):")
        print(df['CustomerID'].unique()[:10])
        
        # Show available facilities for this customer
        if customer_id in df['CustomerID'].values:
            facilities = df[df['CustomerID'] == customer_id]['FacilityID'].unique()
            print(f"\nAvailable facilities for '{customer_id}':")
            print(facilities[:10])
        
        return None
    
    print(f"✓ Found {len(customer_data)} historical records")
    print(f"  Unique products: {customer_data['ProductID'].nunique()}")
    print(f"  Date range: {customer_data['timestamp'].min()} to {customer_data['timestamp'].max()}")
    
    # Step 2: Prepare features for prediction
    print("\nStep 2: Preparing features for prediction...")
    X, y_true = loader.prepare_lightgbm_features(customer_data)
    
    # Step 3: Load models and generate predictions
    print("\nStep 3: Generating predictions...")
    models = load_models()
    ensemble = EnsemblePredictor(models)
    predictions = ensemble.predict(X, y_true)
    
    # Step 4: Create forecast output
    print("\nStep 4: Creating forecast report...")
    
    # Add predictions to dataframe
    customer_data['predicted_demand_14days'] = predictions['ensemble']
    customer_data['actual_recent_demand'] = y_true
    
    # Calculate order recommendations
    customer_data['recommended_order_qty'] = customer_data['predicted_demand_14days'].apply(
        lambda x: int(np.ceil(x * 1.15)) if x >= 20 else 
                  int(np.ceil(x * 1.20)) if x >= 5 else 
                  int(np.ceil(x * 1.50))
    )
    
    customer_data['priority'] = customer_data['predicted_demand_14days'].apply(
        lambda x: 'High' if x >= 20 else 'Medium' if x >= 5 else 'Low'
    )
    
    # Calculate forecast dates
    forecast_start = pd.to_datetime(start_date)
    forecast_end = forecast_start + timedelta(days=14)
    
    customer_data['forecast_start_date'] = forecast_start
    customer_data['forecast_end_date'] = forecast_end
    customer_data['forecast_period_days'] = 14
    
    # Create clean output
    forecast_output = customer_data[[
        'CustomerID',
        'FacilityID',
        'ProductID',
        'predicted_demand_14days',
        'recommended_order_qty',
        'priority',
        'price_mean',
        'forecast_start_date',
        'forecast_end_date',
        'forecast_period_days'
    ]].copy()
    
    # Calculate order value
    forecast_output['unit_price'] = forecast_output['price_mean']
    forecast_output['total_order_value'] = (
        forecast_output['recommended_order_qty'] * 
        forecast_output['unit_price']
    )
    
    # Sort by priority and predicted demand
    priority_order = {'High': 1, 'Medium': 2, 'Low': 3}
    forecast_output['priority_rank'] = forecast_output['priority'].map(priority_order)
    forecast_output = forecast_output.sort_values(
        ['priority_rank', 'predicted_demand_14days'], 
        ascending=[True, False]
    )
    forecast_output = forecast_output.drop('priority_rank', axis=1)
    
    # Round numbers for readability
    forecast_output['predicted_demand_14days'] = forecast_output['predicted_demand_14days'].round(2)
    forecast_output['unit_price'] = forecast_output['unit_price'].round(2)
    forecast_output['total_order_value'] = forecast_output['total_order_value'].round(2)
    
    # Step 5: Generate summary
    print("\n" + "="*80)
    print("FORECAST SUMMARY")
    print("="*80)
    
    summary = {
        'customer_id': customer_id,
        'facility_id': facility_id,
        'forecast_start': str(forecast_start.date()),
        'forecast_end': str(forecast_end.date()),
        'total_products': len(forecast_output),
        'high_priority': len(forecast_output[forecast_output['priority'] == 'High']),
        'medium_priority': len(forecast_output[forecast_output['priority'] == 'Medium']),
        'low_priority': len(forecast_output[forecast_output['priority'] == 'Low']),
        'total_predicted_demand': forecast_output['predicted_demand_14days'].sum(),
        'total_recommended_order': forecast_output['recommended_order_qty'].sum(),
        'total_order_value': forecast_output['total_order_value'].sum()
    }
    
    print(f"\n📅 Forecast Period: {summary['forecast_start']} to {summary['forecast_end']}")
    print(f"📦 Total Products: {summary['total_products']}")
    print(f"📊 Total Predicted Demand: {summary['total_predicted_demand']:.0f} units")
    print(f"🛒 Total Recommended Order: {summary['total_recommended_order']} units")
    print(f"💰 Total Order Value: ${summary['total_order_value']:,.2f}")
    
    print(f"\n🔴 High Priority: {summary['high_priority']} products")
    print(f"🟡 Medium Priority: {summary['medium_priority']} products")
    print(f"🟢 Low Priority: {summary['low_priority']} products")
    
    # Show top 10 products
    print(f"\n📋 Top 10 Products by Predicted Demand:")
    print("-"*80)
    top_10 = forecast_output.head(10)
    for idx, row in top_10.iterrows():
        print(f"  {row['ProductID']}: {row['predicted_demand_14days']:.1f} units → "
              f"Order {row['recommended_order_qty']} units "
              f"({row['priority']} priority, ${row['total_order_value']:.2f})")
    
    # Step 6: Save output
    if output_file is None:
        output_file = f"local/results/forecast_{customer_id}_{facility_id}_{start_date}.csv"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    forecast_output.to_csv(output_file, index=False)
    
    print(f"\n✅ Forecast saved to: {output_file}")
    
    # Save summary
    summary_file = output_file.replace('.csv', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("DEMAND FORECAST SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Customer ID: {summary['customer_id']}\n")
        f.write(f"Facility ID: {summary['facility_id']}\n")
        f.write(f"Forecast Period: {summary['forecast_start']} to {summary['forecast_end']}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("SUMMARY\n")
        f.write("-"*60 + "\n")
        f.write(f"Total Products: {summary['total_products']}\n")
        f.write(f"Total Predicted Demand: {summary['total_predicted_demand']:.0f} units\n")
        f.write(f"Total Recommended Order: {summary['total_recommended_order']} units\n")
        f.write(f"Total Order Value: ${summary['total_order_value']:,.2f}\n\n")
        f.write("BY PRIORITY\n")
        f.write("-"*60 + "\n")
        f.write(f"High Priority: {summary['high_priority']} products\n")
        f.write(f"Medium Priority: {summary['medium_priority']} products\n")
        f.write(f"Low Priority: {summary['low_priority']} products\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    print("\n" + "="*80)
    print("✅ FORECAST COMPLETED SUCCESSFULLY")
    print("="*80 + "\n")
    
    return forecast_output, summary


def main():
    """Main entry point with command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Predict items for a specific customer and facility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict for customer 'scionhealth' at facility '287' starting 2025-11-05
  python3 predict_for_customer.py --customer scionhealth --facility 287 --date 2025-11-05
  
  # Predict for customer 'piedmont' at facility '522' starting today
  python3 predict_for_customer.py --customer piedmont --facility 522 --date 2025-11-05
        """
    )
    
    parser.add_argument('--customer', type=str, required=True,
                       help='Customer ID (e.g., scionhealth, piedmont)')
    parser.add_argument('--facility', type=str, required=True,
                       help='Facility ID (e.g., 287, 522)')
    parser.add_argument('--date', type=str, required=True,
                       help='Start date for forecast (YYYY-MM-DD, e.g., 2025-11-05)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Validate date format
    try:
        datetime.strptime(args.date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("   Use YYYY-MM-DD format (e.g., 2025-11-05)")
        sys.exit(1)
    
    # Generate forecast
    result = predict_for_customer(
        customer_id=args.customer,
        facility_id=args.facility,
        start_date=args.date,
        output_file=args.output
    )
    
    if result is None:
        sys.exit(1)
    
    return result


if __name__ == '__main__':
    main()
