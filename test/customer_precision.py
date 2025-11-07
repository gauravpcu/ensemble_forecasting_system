#!/usr/bin/env python3
"""
Calculate Precision, Recall, and F1 Score for Specific Customers
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import CLASSIFICATION_THRESHOLD

def calculate_customer_precision(customer_id=None):
    """Calculate precision/recall metrics for a specific customer or all customers"""
    
    print("=" * 80)
    print("CUSTOMER-SPECIFIC PRECISION/RECALL ANALYSIS")
    print("=" * 80)
    
    # Load data
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        val_df = pd.read_csv('test/data/val_data.csv')
        print(f"✓ Loaded predictions: {len(predictions_df):,} records")
        print(f"✓ Loaded validation: {len(val_df):,} records")
    except FileNotFoundError:
        print("❌ Missing data files. Run predictions first.")
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
    print(f"✓ Common items: {len(common_items):,}")
    print(f"✓ Classification threshold: {CLASSIFICATION_THRESHOLD} units\n")
    
    # Aggregate by item
    pred_agg = predictions_df[predictions_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'predicted_value': 'sum',
        'CustomerID': 'first'
    }).reset_index()
    
    val_agg = val_df[val_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'target_value': 'sum'
    }).reset_index()
    
    comparison_df = pred_agg.merge(val_agg, on='item_id', how='inner')
    
    # If specific customer requested
    if customer_id:
        customer_df = comparison_df[comparison_df['CustomerID'] == customer_id].copy()
        
        if len(customer_df) == 0:
            print(f"❌ No data found for customer: {customer_id}")
            return
        
        print("=" * 80)
        print(f"CUSTOMER: {customer_id.upper()}")
        print("=" * 80)
        
        # Binary classification
        actual_binary = (customer_df['target_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
        predicted_binary = (customer_df['predicted_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
        
        # Calculate metrics
        precision = precision_score(actual_binary, predicted_binary, zero_division=0)
        recall = recall_score(actual_binary, predicted_binary, zero_division=0)
        f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
        accuracy = (actual_binary == predicted_binary).mean()
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(actual_binary, predicted_binary).ravel()
        
        # Volume metrics
        total_actual = customer_df['target_value'].sum()
        total_predicted = customer_df['predicted_value'].sum()
        volume_error = ((total_predicted - total_actual) / total_actual) * 100
        mae = np.abs(customer_df['predicted_value'] - customer_df['target_value']).mean()
        
        print(f"\n📊 CLASSIFICATION METRICS:")
        print(f"   Total Items:     {len(customer_df):,}")
        print(f"   Accuracy:        {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"   Precision:       {precision:.3f} ({precision*100:.1f}%)")
        print(f"   Recall:          {recall:.3f} ({recall*100:.1f}%)")
        print(f"   F1 Score:        {f1:.3f}")
        
        print(f"\n📋 CONFUSION MATRIX:")
        print(f"                    Predicted: No    Predicted: Yes")
        print(f"   Actual: No       {tn:>13,}    {fp:>14,}")
        print(f"   Actual: Yes      {fn:>13,}    {tp:>14,}")
        
        print(f"\n🎯 BUSINESS INTERPRETATION:")
        print(f"   • True Positives:  {tp:,} (correctly predicted orders)")
        print(f"   • False Positives: {fp:,} (over-ordering - predicted but not needed)")
        print(f"   • False Negatives: {fn:,} (stockouts - missed predictions)")
        print(f"   • True Negatives:  {tn:,} (correctly predicted no order)")
        
        print(f"\n📈 QUANTITY METRICS:")
        print(f"   Total Actual:    {total_actual:,.0f} units")
        print(f"   Total Predicted: {total_predicted:,.0f} units")
        print(f"   Volume Error:    {volume_error:+.1f}%")
        print(f"   MAE:             {mae:.2f} units")
        
        # Items actually ordered
        items_ordered = (actual_binary == 1).sum()
        items_predicted = (predicted_binary == 1).sum()
        
        print(f"\n📦 ORDER STATISTICS:")
        print(f"   Items actually ordered (≥{CLASSIFICATION_THRESHOLD}):  {items_ordered:,}")
        print(f"   Items predicted to order:  {items_predicted:,}")
        print(f"   Correctly predicted:       {tp:,} ({tp/items_ordered*100:.1f}% of actual)")
        print(f"   Missed orders:             {fn:,} ({fn/items_ordered*100:.1f}% of actual)")
        
        return {
            'customer': customer_id,
            'items': len(customer_df),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'volume_error': volume_error,
            'mae': mae
        }
    
    else:
        # Calculate for all customers
        print("=" * 80)
        print("ALL CUSTOMERS PRECISION/RECALL ANALYSIS")
        print("=" * 80)
        
        customers = comparison_df['CustomerID'].unique()
        results = []
        
        for cust in customers:
            customer_df = comparison_df[comparison_df['CustomerID'] == cust].copy()
            
            if len(customer_df) < 10:  # Skip customers with too few items
                continue
            
            # Binary classification
            actual_binary = (customer_df['target_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
            predicted_binary = (customer_df['predicted_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
            
            # Calculate metrics
            precision = precision_score(actual_binary, predicted_binary, zero_division=0)
            recall = recall_score(actual_binary, predicted_binary, zero_division=0)
            f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
            
            # Volume metrics
            total_actual = customer_df['target_value'].sum()
            total_predicted = customer_df['predicted_value'].sum()
            volume_error = ((total_predicted - total_actual) / total_actual) * 100
            
            results.append({
                'customer': cust,
                'items': len(customer_df),
                'actual_volume': total_actual,
                'predicted_volume': total_predicted,
                'volume_error': volume_error,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('actual_volume', ascending=False)
        
        print(f"\n📊 TOP 20 CUSTOMERS BY VOLUME:")
        print("-" * 110)
        print(f"{'Customer':<20} {'Items':<7} {'Actual':<10} {'Predicted':<10} {'Vol Err%':<9} {'Precision':<10} {'Recall':<8} {'F1':<6}")
        print("-" * 110)
        
        for _, row in results_df.head(20).iterrows():
            print(f"{row['customer']:<20} {row['items']:<7} {row['actual_volume']:<10.0f} {row['predicted_volume']:<10.0f} "
                  f"{row['volume_error']:<+9.1f} {row['precision']:<10.3f} {row['recall']:<8.3f} {row['f1']:<6.3f}")
        
        print("\n" + "=" * 110)
        print(f"OVERALL AVERAGES (across {len(results_df)} customers):")
        print(f"   Average Precision: {results_df['precision'].mean():.3f}")
        print(f"   Average Recall:    {results_df['recall'].mean():.3f}")
        print(f"   Average F1 Score:  {results_df['f1'].mean():.3f}")
        print(f"   Average Vol Error: {results_df['volume_error'].mean():+.1f}%")
        
        # Save results
        results_df.to_csv('test/data/customer_precision_analysis.csv', index=False)
        print(f"\n✓ Saved detailed results to: test/data/customer_precision_analysis.csv")
        
        return results_df

if __name__ == "__main__":
    # Check if customer specified
    if len(sys.argv) > 1:
        customer = sys.argv[1]
        calculate_customer_precision(customer)
    else:
        # Calculate for ScionHealth by default, then show all
        print("\n" + "=" * 80)
        print("DETAILED ANALYSIS FOR SCIONHEALTH")
        print("=" * 80 + "\n")
        calculate_customer_precision('scionhealth')
        
        print("\n\n")
        calculate_customer_precision()
