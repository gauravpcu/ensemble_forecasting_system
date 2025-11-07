#!/usr/bin/env python3
"""
Dual Prediction Analysis: Product-Level + Quantity-Level Metrics
Analyzes both binary predictions (will product be ordered?) and quantity predictions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import CLASSIFICATION_THRESHOLD

def calculate_binary_metrics(actual_binary, predicted_binary):
    """Calculate binary classification metrics"""
    accuracy = accuracy_score(actual_binary, predicted_binary)
    precision = precision_score(actual_binary, predicted_binary, zero_division=0)
    recall = recall_score(actual_binary, predicted_binary, zero_division=0)
    f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
    cm = confusion_matrix(actual_binary, predicted_binary)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def calculate_quantity_metrics(actual_qty, predicted_qty):
    """Calculate quantity regression metrics"""
    mae = mean_absolute_error(actual_qty, predicted_qty)
    rmse = np.sqrt(mean_squared_error(actual_qty, predicted_qty))
    
    # MAPE for non-zero actuals
    non_zero_mask = actual_qty > 0
    if np.sum(non_zero_mask) > 0:
        mape = np.mean(np.abs((actual_qty[non_zero_mask] - predicted_qty[non_zero_mask]) / actual_qty[non_zero_mask])) * 100
    else:
        mape = 0
    
    # R-squared
    ss_res = np.sum((actual_qty - predicted_qty) ** 2)
    ss_tot = np.sum((actual_qty - np.mean(actual_qty)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def analyze_product_predictions():
    """Main analysis function"""
    
    print("=" * 80)
    print("DUAL PREDICTION ANALYSIS")
    print("=" * 80)
    print("Analyzing both:")
    print("1. Product-Level: Will this product be ordered? (Binary)")
    print("2. Quantity-Level: How much will be ordered? (Regression)")
    print()
    
    # Load data
    print("[1/6] Loading prediction and validation data...")
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        validation_df = pd.read_csv('test/data/val_data.csv')
        print(f"   Predictions: {len(predictions_df):,} records")
        print(f"   Validation: {len(validation_df):,} records")
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create item_id for matching
    predictions_df['item_id'] = (predictions_df['CustomerID'].astype(str) + '_' + 
                                predictions_df['FacilityID'].astype(str) + '_' + 
                                predictions_df['ProductID'].astype(str))
    validation_df['item_id'] = (validation_df['CustomerID'].astype(str) + '_' + 
                               validation_df['FacilityID'].astype(str) + '_' + 
                               validation_df['ProductID'].astype(str))
    
    print(f"   Unique items in predictions: {predictions_df['item_id'].nunique():,}")
    print(f"   Unique items in validation: {validation_df['item_id'].nunique():,}")
    
    # Find common items
    print("\n[2/6] Finding common items between prediction and validation periods...")
    common_items = set(predictions_df['item_id']) & set(validation_df['item_id'])
    print(f"   Common items: {len(common_items):,}")
    
    if len(common_items) == 0:
        print("❌ No common items found. Cannot perform comparison.")
        return
    
    # Aggregate predictions and actuals by item
    print("\n[3/6] Aggregating data by item...")
    pred_agg = predictions_df[predictions_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'predicted_value': 'sum',
        'CustomerID': 'first',
        'FacilityID': 'first', 
        'ProductID': 'first',
        'ProductName': 'first'
    }).reset_index()
    
    val_agg = validation_df[validation_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'target_value': 'sum'
    }).reset_index()
    
    # Merge predictions and actuals
    comparison_df = pred_agg.merge(val_agg, on='item_id', how='inner')
    print(f"   Final comparison dataset: {len(comparison_df):,} items")
    
    # Create binary classifications
    print(f"\n[4/6] Creating binary classifications (threshold: {CLASSIFICATION_THRESHOLD} units)...")
    comparison_df['actual_binary'] = (comparison_df['target_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
    comparison_df['predicted_binary'] = (comparison_df['predicted_value'] >= CLASSIFICATION_THRESHOLD).astype(int)
    
    print(f"   Items actually ordered (≥{CLASSIFICATION_THRESHOLD} units): {comparison_df['actual_binary'].sum():,}")
    print(f"   Items predicted to be ordered: {comparison_df['predicted_binary'].sum():,}")
    
    # Calculate binary metrics
    print("\n[5/6] Calculating metrics...")
    print("\n" + "=" * 60)
    print("PRODUCT-LEVEL ANALYSIS (Binary Classification)")
    print("=" * 60)
    print("Question: Will this product be ordered?")
    print(f"Threshold: ≥{CLASSIFICATION_THRESHOLD} units = 'Will be ordered'")
    print()
    
    binary_metrics = calculate_binary_metrics(
        comparison_df['actual_binary'].values,
        comparison_df['predicted_binary'].values
    )
    
    print(f"📊 BINARY CLASSIFICATION METRICS:")
    print(f"   Accuracy:  {binary_metrics['accuracy']:.3f} ({binary_metrics['accuracy']*100:.1f}%)")
    print(f"   Precision: {binary_metrics['precision']:.3f} ({binary_metrics['precision']*100:.1f}%)")
    print(f"   Recall:    {binary_metrics['recall']:.3f} ({binary_metrics['recall']*100:.1f}%)")
    print(f"   F1 Score:  {binary_metrics['f1_score']:.3f}")
    print()
    
    print("📋 CONFUSION MATRIX:")
    cm = binary_metrics['confusion_matrix']
    print(f"                    Predicted: No    Predicted: Yes")
    print(f"   Actual: No       {cm[0,0]:8,}    {cm[0,1]:8,}")
    print(f"   Actual: Yes      {cm[1,0]:8,}    {cm[1,1]:8,}")
    print()
    
    print("🎯 BUSINESS INTERPRETATION:")
    print(f"   • Correctly identified products: {binary_metrics['accuracy']*100:.1f}%")
    print(f"   • Of predicted orders, {binary_metrics['precision']*100:.1f}% were correct (low false alarms)")
    print(f"   • Of actual orders, {binary_metrics['recall']*100:.1f}% were predicted (coverage)")
    print(f"   • False positives (over-ordering): {cm[0,1]:,} products")
    print(f"   • False negatives (stockouts): {cm[1,0]:,} products")
    
    # Calculate quantity metrics
    print("\n" + "=" * 60)
    print("QUANTITY-LEVEL ANALYSIS (Regression)")
    print("=" * 60)
    print("Question: How much of each product will be ordered?")
    print()
    
    quantity_metrics = calculate_quantity_metrics(
        comparison_df['target_value'].values,
        comparison_df['predicted_value'].values
    )
    
    print(f"📊 QUANTITY REGRESSION METRICS:")
    print(f"   MAE:   {quantity_metrics['mae']:.2f} units")
    print(f"   RMSE:  {quantity_metrics['rmse']:.2f} units") 
    print(f"   MAPE:  {quantity_metrics['mape']:.1f}%")
    print(f"   R²:    {quantity_metrics['r2']:.3f}")
    print()
    
    print("📈 VOLUME ANALYSIS:")
    total_actual = comparison_df['target_value'].sum()
    total_predicted = comparison_df['predicted_value'].sum()
    volume_error = ((total_predicted - total_actual) / total_actual) * 100
    
    print(f"   Total Actual:    {total_actual:,} units")
    print(f"   Total Predicted: {total_predicted:,} units")
    print(f"   Volume Error:    {volume_error:+.1f}%")
    
    # Customer-level analysis
    print("\n[6/6] Customer-level breakdown...")
    print("\n" + "=" * 60)
    print("CUSTOMER-LEVEL PERFORMANCE")
    print("=" * 60)
    
    customer_analysis = comparison_df.groupby('CustomerID').agg({
        'target_value': 'sum',
        'predicted_value': 'sum',
        'actual_binary': 'sum',
        'predicted_binary': 'sum',
        'item_id': 'count'
    }).reset_index()
    
    customer_analysis['volume_error_pct'] = ((customer_analysis['predicted_value'] - customer_analysis['target_value']) / customer_analysis['target_value']) * 100
    customer_analysis = customer_analysis.sort_values('target_value', ascending=False)
    
    print("\nTop 15 Customers by Volume:")
    print(f"{'Customer':<20} {'Items':<6} {'Actual':<8} {'Predicted':<10} {'Vol Err%':<8} {'Act Ord':<7} {'Pred Ord':<8}")
    print("-" * 75)
    
    for _, row in customer_analysis.head(15).iterrows():
        print(f"{row['CustomerID']:<20} {row['item_id']:<6} {row['target_value']:<8.0f} {row['predicted_value']:<10.0f} {row['volume_error_pct']:<+8.1f} {row['actual_binary']:<7} {row['predicted_binary']:<8}")
    
    # Save detailed results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save item-level comparison
    comparison_df['volume_error'] = comparison_df['predicted_value'] - comparison_df['target_value']
    comparison_df['volume_error_pct'] = (comparison_df['volume_error'] / comparison_df['target_value']) * 100
    comparison_df['abs_error'] = np.abs(comparison_df['volume_error'])
    
    comparison_df.to_csv('test/data/dual_analysis_items.csv', index=False)
    print("✓ Item-level analysis: test/data/dual_analysis_items.csv")
    
    # Save customer-level analysis
    customer_analysis.to_csv('test/data/dual_analysis_customers.csv', index=False)
    print("✓ Customer-level analysis: test/data/dual_analysis_customers.csv")
    
    # Save summary report
    with open('test/data/dual_analysis_summary.txt', 'w') as f:
        f.write("DUAL PREDICTION ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Items Analyzed: {len(comparison_df):,}\n")
        f.write(f"Classification Threshold: {CLASSIFICATION_THRESHOLD} units\n\n")
        
        f.write("PRODUCT-LEVEL METRICS (Binary)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Accuracy:  {binary_metrics['accuracy']:.3f}\n")
        f.write(f"Precision: {binary_metrics['precision']:.3f}\n")
        f.write(f"Recall:    {binary_metrics['recall']:.3f}\n")
        f.write(f"F1 Score:  {binary_metrics['f1_score']:.3f}\n\n")
        
        f.write("QUANTITY-LEVEL METRICS (Regression)\n")
        f.write("-" * 35 + "\n")
        f.write(f"MAE:   {quantity_metrics['mae']:.2f} units\n")
        f.write(f"RMSE:  {quantity_metrics['rmse']:.2f} units\n")
        f.write(f"MAPE:  {quantity_metrics['mape']:.1f}%\n")
        f.write(f"R²:    {quantity_metrics['r2']:.3f}\n\n")
        
        f.write("VOLUME ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write(f"Total Actual:    {total_actual:,} units\n")
        f.write(f"Total Predicted: {total_predicted:,} units\n")
        f.write(f"Volume Error:    {volume_error:+.1f}%\n")
    
    print("✓ Summary report: test/data/dual_analysis_summary.txt")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nKey Insights:")
    print(f"• Product Accuracy: {binary_metrics['accuracy']*100:.1f}% of products correctly classified")
    print(f"• Quantity Accuracy: {quantity_metrics['mae']:.1f} units average error")
    print(f"• Business Impact: {binary_metrics['precision']*100:.1f}% precision, {binary_metrics['recall']*100:.1f}% recall")
    print(f"• Volume Impact: {volume_error:+.1f}% total volume error")
    
    return {
        'binary_metrics': binary_metrics,
        'quantity_metrics': quantity_metrics,
        'volume_error': volume_error,
        'items_analyzed': len(comparison_df)
    }

if __name__ == "__main__":
    analyze_product_predictions()