#!/usr/bin/env python3
"""
Universal Test Runner
Handles all testing scenarios with parameters
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

from src.config import env_config
from src.core.prediction_generator import PredictionGenerator


def run_quick_test(customer=None):
    """Quick test using existing test data"""
    print("=" * 80)
    print("QUICK TEST - Using Preprocessed Data")
    print("=" * 80)
    
    test_data = os.path.join(env_config.TEST_DATA_DIR, 'test_data.csv')
    
    generator = PredictionGenerator(
        customers=[customer] if customer else None,
        source_data=test_data,
        use_preprocessed=True,
        verbose=True
    )
    
    predictions = generator.generate(save=False)
    
    print("\n" + "=" * 80)
    print("QUICK TEST RESULTS")
    print("=" * 80)
    print(f"Total Predictions:    {len(predictions):,}")
    print(f"Items to Order:       {predictions['predicted_reorder'].sum():,}")
    print(f"Average Prediction:   {predictions['predicted_value'].mean():.2f} units")
    print("=" * 80)
    
    return predictions


def run_full_test(customer=None):
    """Full test with validation comparison"""
    print("=" * 80)
    print("FULL TEST - With Validation")
    print("=" * 80)
    
    # Load test and validation data
    test_data = os.path.join(env_config.TEST_DATA_DIR, 'test_data.csv')
    val_data = os.path.join(env_config.TEST_DATA_DIR, 'val_data.csv')
    
    # Generate predictions
    generator = PredictionGenerator(
        customers=[customer] if customer else None,
        source_data=test_data,
        use_preprocessed=True,
        verbose=True
    )
    
    predictions = generator.generate(save=False)
    
    # Load validation actuals
    print("\nLoading validation data...")
    val_df = pd.read_csv(val_data)
    
    # Merge predictions with actuals
    comparison = predictions.merge(
        val_df[['item_id', 'target_value']],
        on='item_id',
        how='inner',
        suffixes=('_pred', '_actual')
    )
    
    print(f"Matched {len(comparison):,} items for validation")
    
    # Calculate metrics
    y_true = comparison['target_value_actual']
    y_pred = comparison['predicted_value']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # Classification metrics
    y_true_binary = (y_true >= env_config.CLASSIFICATION_THRESHOLD).astype(int)
    y_pred_binary = (y_pred >= env_config.CLASSIFICATION_THRESHOLD).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Print results
    print("\n" + "=" * 80)
    print("FULL TEST RESULTS")
    print("=" * 80)
    print(f"\nRegression Metrics:")
    print(f"  MAE:              {mae:.2f} units")
    print(f"  RMSE:             {rmse:.2f} units")
    print(f"  MAPE:             {mape:.1f}%")
    
    print(f"\nClassification Metrics (threshold={env_config.CLASSIFICATION_THRESHOLD}):")
    print(f"  Precision:        {precision:.1%}")
    print(f"  Recall:           {recall:.1%}")
    print(f"  F1 Score:         {f1:.1%}")
    
    print(f"\nData Summary:")
    print(f"  Total Items:      {len(predictions):,}")
    print(f"  Validated Items:  {len(comparison):,}")
    print(f"  Items to Order:   {predictions['predicted_reorder'].sum():,}")
    print("=" * 80)
    
    return predictions, comparison


def run_customer_test(customer):
    """Test specific customer with detailed analysis"""
    print("=" * 80)
    print(f"CUSTOMER TEST - {customer.upper()}")
    print("=" * 80)
    
    test_data = os.path.join(env_config.TEST_DATA_DIR, 'test_data.csv')
    val_data = os.path.join(env_config.TEST_DATA_DIR, 'val_data.csv')
    
    # Generate predictions
    generator = PredictionGenerator(
        customers=[customer],
        source_data=test_data,
        use_preprocessed=True,
        verbose=True
    )
    
    predictions = generator.generate(save=False)
    
    # Load validation
    val_df = pd.read_csv(val_data)
    val_df = val_df[val_df['CustomerID'] == customer]
    
    # Merge
    comparison = predictions.merge(
        val_df[['item_id', 'target_value']],
        on='item_id',
        how='inner',
        suffixes=('_pred', '_actual')
    )
    
    # Calculate metrics
    if len(comparison) > 0:
        y_true = comparison['target_value_actual']
        y_pred = comparison['predicted_value']
        
        mae = mean_absolute_error(y_true, y_pred)
        
        y_true_binary = (y_true >= env_config.CLASSIFICATION_THRESHOLD).astype(int)
        y_pred_binary = (y_pred >= env_config.CLASSIFICATION_THRESHOLD).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    else:
        mae = precision = recall = 0
    
    # Print results
    print("\n" + "=" * 80)
    print(f"CUSTOMER TEST RESULTS - {customer.upper()}")
    print("=" * 80)
    print(f"Total Items:          {len(predictions):,}")
    print(f"Validated Items:      {len(comparison):,}")
    print(f"Items to Order:       {predictions['predicted_reorder'].sum():,}")
    print(f"MAE:                  {mae:.2f} units")
    print(f"Precision:            {precision:.1%}")
    print(f"Recall:               {recall:.1%}")
    
    # Top items
    print(f"\nTop 10 Items by Predicted Volume:")
    top_10 = predictions.nlargest(10, 'predicted_value')[
        ['ProductName', 'predicted_value', 'reorder_recommendation']
    ]
    for idx, row in top_10.iterrows():
        product_name = str(row['ProductName'])[:50]
        print(f"  {product_name:<50} {row['predicted_value']:>8.1f} units  [{row['reorder_recommendation']}]")
    
    print("=" * 80)
    
    return predictions, comparison


def main():
    parser = argparse.ArgumentParser(
        description='Universal Test Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_test.py --quick                    # Quick test
  python run_test.py --full                     # Full test with validation
  python run_test.py --customer scionhealth     # Customer-specific test
  python run_test.py --quick --customer mercy   # Quick test for one customer
        """
    )
    
    parser.add_argument('--quick', action='store_true', help='Quick test (no validation)')
    parser.add_argument('--full', action='store_true', help='Full test with validation')
    parser.add_argument('--customer', type=str, help='Test specific customer')
    
    args = parser.parse_args()
    
    # Default to quick test if no mode specified
    if not args.quick and not args.full:
        args.quick = True
    
    try:
        if args.customer:
            # Customer-specific test
            run_customer_test(args.customer)
        elif args.full:
            # Full test
            run_full_test()
        else:
            # Quick test
            run_quick_test()
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
