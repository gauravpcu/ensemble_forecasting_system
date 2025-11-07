#!/usr/bin/env python3
"""
Apply Customer Calibration to Predictions
Takes existing predictions and applies customer-specific calibration factors
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import CUSTOMER_CALIBRATION, CLASSIFICATION_THRESHOLD, get_safety_multiplier

def apply_calibration_to_predictions():
    """Apply customer calibration to existing predictions"""
    
    print("=" * 80)
    print("APPLYING CUSTOMER CALIBRATION TO PREDICTIONS")
    print("=" * 80)
    
    # Load predictions
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        print(f"✓ Loaded predictions: {len(predictions_df):,} records")
    except FileNotFoundError:
        print("❌ Run predictions first")
        return
    
    # Apply customer-specific calibration
    print(f"\n🔧 Applying calibration factors:")
    print(f"Number of customer calibration factors: {len(CUSTOMER_CALIBRATION)}")
    
    # Create calibrated predictions
    predictions_df['calibrated_prediction'] = predictions_df['predicted_value'].copy()
    
    # Step 1: Apply customer-specific calibration
    customer_calibration_applied = 0
    for customer, factor in CUSTOMER_CALIBRATION.items():
        mask = predictions_df['CustomerID'] == customer
        count = mask.sum()
        if count > 0:
            predictions_df.loc[mask, 'calibrated_prediction'] *= factor
            print(f"  {customer}: {factor:.3f}x applied to {count:,} records")
            customer_calibration_applied += count
    
    print(f"\n✓ Customer calibration applied to {customer_calibration_applied:,} records ({customer_calibration_applied/len(predictions_df)*100:.1f}%)")
    
    # Step 2: Apply volume-based safety multipliers
    print(f"\n🔧 Applying volume-based safety multipliers:")
    
    # Apply safety multipliers based on predicted volume
    volume_calibration_applied = 0
    for idx, row in predictions_df.iterrows():
        safety_multiplier = get_safety_multiplier(row['calibrated_prediction'])
        if safety_multiplier != 1.0:  # Only apply if different from 1.0
            predictions_df.at[idx, 'calibrated_prediction'] *= safety_multiplier
            volume_calibration_applied += 1
    
    # Count by volume category for reporting
    low_volume = (predictions_df['predicted_value'] < 5).sum()
    medium_volume = ((predictions_df['predicted_value'] >= 5) & (predictions_df['predicted_value'] < 20)).sum()
    high_volume = ((predictions_df['predicted_value'] >= 20) & (predictions_df['predicted_value'] < 100)).sum()
    very_high_volume = (predictions_df['predicted_value'] >= 100).sum()
    
    print(f"  Low volume (0-5):      1.8x applied to {low_volume:,} records")
    print(f"  Medium volume (5-20):  1.3x applied to {medium_volume:,} records") 
    print(f"  High volume (20-100):  1.1x applied to {high_volume:,} records")
    print(f"  Very high (100+):      1.0x applied to {very_high_volume:,} records")
    
    print(f"\n✓ Volume-based calibration applied to {volume_calibration_applied:,} records ({volume_calibration_applied/len(predictions_df)*100:.1f}%)")
    
    # Show calibration impact before updating
    original_mean = predictions_df['predicted_value'].mean()
    calibrated_mean = predictions_df['calibrated_prediction'].mean()
    
    # Update the main prediction column
    predictions_df['predicted_value'] = predictions_df['calibrated_prediction']
    predictions_df = predictions_df.drop('calibrated_prediction', axis=1)
    
    # Save calibrated predictions
    predictions_df.to_csv('test/data/predictions.csv', index=False)
    print(f"✓ Saved calibrated predictions to test/data/predictions.csv")
    
    # Show improvement summary
    print(f"\n📊 CALIBRATION IMPACT:")
    print(f"Original prediction mean:   {original_mean:.2f}")
    print(f"Calibrated prediction mean: {calibrated_mean:.2f}")
    print(f"Change:                     {((calibrated_mean - original_mean) / original_mean * 100):+.1f}%")
    print(f"Classification threshold:   {CLASSIFICATION_THRESHOLD}")
    
    return predictions_df

def analyze_calibrated_performance():
    """Analyze performance after calibration"""
    
    print(f"\n" + "=" * 80)
    print("CALIBRATED PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Load validation data
    try:
        val_df = pd.read_csv('test/data/val_data.csv')
        predictions_df = pd.read_csv('test/data/predictions.csv')
        print(f"✓ Loaded validation data: {len(val_df):,} records")
        print(f"✓ Loaded calibrated predictions: {len(predictions_df):,} records")
    except FileNotFoundError:
        print("❌ Missing data files")
        return
    
    # Create item_id for matching
    predictions_df['item_id'] = (predictions_df['CustomerID'].astype(str) + '_' + 
                                predictions_df['FacilityID'].astype(str) + '_' + 
                                predictions_df['ProductID'].astype(str))
    val_df['item_id'] = (val_df['CustomerID'].astype(str) + '_' + 
                        val_df['FacilityID'].astype(str) + '_' + 
                        val_df['ProductID'].astype(str))
    
    # Find common items
    common_items = set(predictions_df['item_id']) & set(val_df['item_id'])
    print(f"✓ Common items for comparison: {len(common_items):,}")
    
    # Aggregate by item
    pred_agg = predictions_df[predictions_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'predicted_value': 'sum',
        'CustomerID': 'first'
    }).reset_index()
    
    val_agg = val_df[val_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'target_value': 'sum'
    }).reset_index()
    
    # Merge for comparison
    comparison_df = pred_agg.merge(val_agg, on='item_id', how='inner')
    
    # Calculate metrics
    mae = np.mean(np.abs(comparison_df['predicted_value'] - comparison_df['target_value']))
    
    # Binary classification
    actual_binary = (comparison_df['target_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
    predicted_binary = (comparison_df['predicted_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
    
    accuracy = (actual_binary == predicted_binary).mean()
    precision = ((predicted_binary == 1) & (actual_binary == 1)).sum() / (predicted_binary == 1).sum()
    recall = ((predicted_binary == 1) & (actual_binary == 1)).sum() / (actual_binary == 1).sum()
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Volume analysis
    total_actual = comparison_df['target_value'].sum()
    total_predicted = comparison_df['predicted_value'].sum()
    volume_error = ((total_predicted - total_actual) / total_actual) * 100
    
    print(f"\n📊 CALIBRATED PERFORMANCE METRICS:")
    print(f"MAE:           {mae:.2f} units")
    print(f"Accuracy:      {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Precision:     {precision:.3f} ({precision*100:.1f}%)")
    print(f"Recall:        {recall:.3f} ({recall*100:.1f}%)")
    print(f"F1 Score:      {f1:.3f}")
    print(f"Volume Error:  {volume_error:+.1f}%")
    
    # Customer-level analysis
    customer_analysis = comparison_df.groupby('CustomerID').agg({
        'target_value': 'sum',
        'predicted_value': 'sum'
    }).reset_index()
    
    customer_analysis['volume_error_pct'] = ((customer_analysis['predicted_value'] - customer_analysis['target_value']) / customer_analysis['target_value']) * 100
    customer_analysis = customer_analysis.sort_values('target_value', ascending=False)
    
    print(f"\n📋 TOP 10 CUSTOMERS AFTER CALIBRATION:")
    print(f"{'Customer':<15} {'Actual':<8} {'Predicted':<10} {'Error%':<8}")
    print("-" * 45)
    
    for _, row in customer_analysis.head(10).iterrows():
        print(f"{row['CustomerID']:<15} {row['target_value']:<8.0f} {row['predicted_value']:<10.0f} {row['volume_error_pct']:<+8.1f}")
    
    return {
        'mae': mae,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'volume_error': volume_error
    }

if __name__ == "__main__":
    apply_calibration_to_predictions()
    analyze_calibrated_performance()