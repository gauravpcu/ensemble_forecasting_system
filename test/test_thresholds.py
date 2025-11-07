#!/usr/bin/env python3
"""
Test Different Classification Thresholds
Compares performance across different threshold values to find optimal setting
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_threshold_performance():
    """Test different classification thresholds"""
    
    print("=" * 80)
    print("CLASSIFICATION THRESHOLD OPTIMIZATION")
    print("=" * 80)
    
    # Load calibrated predictions and validation data
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        val_df = pd.read_csv('test/data/val_data.csv')
        print(f"✓ Loaded predictions: {len(predictions_df):,} records")
        print(f"✓ Loaded validation: {len(val_df):,} records")
    except FileNotFoundError:
        print("❌ Missing data files. Run predictions and calibration first.")
        return
    
    # Create item_id for matching
    predictions_df['item_id'] = (predictions_df['CustomerID'].astype(str) + '_' + 
                                predictions_df['FacilityID'].astype(str) + '_' + 
                                predictions_df['ProductID'].astype(str))
    val_df['item_id'] = (val_df['CustomerID'].astype(str) + '_' + 
                        val_df['FacilityID'].astype(str) + '_' + 
                        val_df['ProductID'].astype(str))
    
    # Find common items and aggregate
    common_items = set(predictions_df['item_id']) & set(val_df['item_id'])
    print(f"✓ Common items: {len(common_items):,}")
    
    pred_agg = predictions_df[predictions_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'predicted_value': 'sum',
        'CustomerID': 'first'
    }).reset_index()
    
    val_agg = val_df[val_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'target_value': 'sum'
    }).reset_index()
    
    comparison_df = pred_agg.merge(val_agg, on='item_id', how='inner')
    print(f"✓ Comparison dataset: {len(comparison_df):,} items")
    
    # Test different thresholds
    thresholds = [4, 5, 6, 7, 8, 9, 10, 12, 15]
    results = []
    
    print(f"\n📊 THRESHOLD PERFORMANCE COMPARISON:")
    print("-" * 90)
    print(f"{'Threshold':<9} {'Accuracy':<8} {'Precision':<9} {'Recall':<7} {'F1':<6} {'FP':<6} {'FN':<6} {'Volume Err%':<11}")
    print("-" * 90)
    
    for threshold in thresholds:
        # Create binary classifications
        actual_binary = (comparison_df['target_value'] >= threshold).astype(int)
        predicted_binary = (comparison_df['predicted_value'] >= threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(actual_binary, predicted_binary)
        precision = precision_score(actual_binary, predicted_binary, zero_division=0)
        recall = recall_score(actual_binary, predicted_binary, zero_division=0)
        f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
        
        # Count false positives and negatives
        fp = ((predicted_binary == 1) & (actual_binary == 0)).sum()
        fn = ((predicted_binary == 0) & (actual_binary == 1)).sum()
        
        # Volume error for items above threshold
        above_threshold_actual = comparison_df[actual_binary == 1]['target_value'].sum()
        above_threshold_predicted = comparison_df[predicted_binary == 1]['predicted_value'].sum()
        
        if above_threshold_actual > 0:
            volume_error = ((above_threshold_predicted - above_threshold_actual) / above_threshold_actual) * 100
        else:
            volume_error = 0
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'false_positives': fp,
            'false_negatives': fn,
            'volume_error': volume_error
        })
        
        print(f"{threshold:<9} {accuracy:<8.3f} {precision:<9.3f} {recall:<7.3f} {f1:<6.3f} {fp:<6} {fn:<6} {volume_error:<+11.1f}")
    
    # Find best threshold by different criteria
    results_df = pd.DataFrame(results)
    
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_precision = results_df.loc[results_df['precision'].idxmax()]
    best_recall = results_df.loc[results_df['recall'].idxmax()]
    best_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_volume = results_df.loc[results_df['volume_error'].abs().idxmin()]
    
    print("-" * 90)
    print(f"\n🏆 BEST THRESHOLDS BY CRITERIA:")
    print(f"Best Accuracy:  {best_accuracy['threshold']} (Accuracy: {best_accuracy['accuracy']:.3f})")
    print(f"Best Precision: {best_precision['threshold']} (Precision: {best_precision['precision']:.3f})")
    print(f"Best Recall:    {best_recall['threshold']} (Recall: {best_recall['recall']:.3f})")
    print(f"Best F1 Score:  {best_f1['threshold']} (F1: {best_f1['f1']:.3f})")
    print(f"Best Volume:    {best_volume['threshold']} (Volume Error: {best_volume['volume_error']:+.1f}%)")
    
    # Recommend optimal threshold
    print(f"\n🎯 RECOMMENDATIONS:")
    
    # Balance precision and recall
    balanced_scores = results_df['precision'] * results_df['recall']
    best_balanced = results_df.loc[balanced_scores.idxmax()]
    
    # Business-focused (minimize stockouts while controlling over-ordering)
    business_scores = results_df['recall'] * 0.7 + results_df['precision'] * 0.3  # Weight recall higher
    best_business = results_df.loc[business_scores.idxmax()]
    
    print(f"\n📋 THRESHOLD RECOMMENDATIONS:")
    print(f"1. **Balanced Performance:** Threshold = {best_balanced['threshold']}")
    print(f"   Accuracy: {best_balanced['accuracy']:.3f}, Precision: {best_balanced['precision']:.3f}, Recall: {best_balanced['recall']:.3f}, F1: {best_balanced['f1']:.3f}")
    
    print(f"\n2. **Business Optimized:** Threshold = {best_business['threshold']} (prioritizes recall)")
    print(f"   Accuracy: {best_business['accuracy']:.3f}, Precision: {best_business['precision']:.3f}, Recall: {best_business['recall']:.3f}, F1: {best_business['f1']:.3f}")
    
    print(f"\n3. **High Precision:** Threshold = {best_precision['threshold']} (minimizes false alarms)")
    print(f"   Accuracy: {best_precision['accuracy']:.3f}, Precision: {best_precision['precision']:.3f}, Recall: {best_precision['recall']:.3f}, F1: {best_precision['f1']:.3f}")
    
    # Current vs recommended
    current_threshold = 7
    current_result = results_df[results_df['threshold'] == current_threshold].iloc[0]
    
    print(f"\n📊 CURRENT vs RECOMMENDED:")
    print(f"Current (Threshold={current_threshold}):")
    print(f"  Accuracy: {current_result['accuracy']:.3f}, Precision: {current_result['precision']:.3f}, Recall: {current_result['recall']:.3f}, F1: {current_result['f1']:.3f}")
    
    if best_f1['threshold'] != current_threshold:
        improvement_f1 = best_f1['f1'] - current_result['f1']
        improvement_precision = best_f1['precision'] - current_result['precision']
        improvement_recall = best_f1['recall'] - current_result['recall']
        
        print(f"\nRecommended (Threshold={best_f1['threshold']}):")
        print(f"  Accuracy: {best_f1['accuracy']:.3f}, Precision: {best_f1['precision']:.3f}, Recall: {best_f1['recall']:.3f}, F1: {best_f1['f1']:.3f}")
        print(f"  Improvements: F1 {improvement_f1:+.3f}, Precision {improvement_precision:+.3f}, Recall {improvement_recall:+.3f}")
    
    return results_df, best_f1['threshold']

def apply_optimal_threshold(optimal_threshold):
    """Apply the optimal threshold to configuration"""
    
    print(f"\n🔧 APPLYING OPTIMAL THRESHOLD: {optimal_threshold}")
    
    # Update .env file
    with open('.env', 'r') as f:
        content = f.read()
    
    # Replace threshold
    import re
    content = re.sub(r'CLASSIFICATION_THRESHOLD=\d+', f'CLASSIFICATION_THRESHOLD={optimal_threshold}', content)
    
    with open('.env', 'w') as f:
        f.write(content)
    
    print(f"✓ Updated CLASSIFICATION_THRESHOLD to {optimal_threshold} in .env")
    
    return optimal_threshold

if __name__ == "__main__":
    results_df, optimal_threshold = test_threshold_performance()
    
    print(f"\n" + "=" * 80)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 80)
    
    print(f"\n🎯 SUMMARY:")
    print(f"• Tested {len(results_df)} different thresholds")
    print(f"• Current threshold: 7")
    print(f"• Optimal threshold: {optimal_threshold}")
    print(f"• Expected F1 improvement: {results_df[results_df['threshold']==optimal_threshold]['f1'].iloc[0] - results_df[results_df['threshold']==7]['f1'].iloc[0]:+.3f}")
    
    # Ask user if they want to apply the optimal threshold
    print(f"\n❓ Apply optimal threshold {optimal_threshold}? (y/N): ", end="")
    response = input().strip().lower()
    
    if response == 'y':
        apply_optimal_threshold(int(optimal_threshold))
        print(f"✅ Threshold updated! Run test again to see improvements.")
    else:
        print(f"ℹ️  Keeping current threshold. You can manually update with:")
        print(f"   python3 configure.py --set CLASSIFICATION_THRESHOLD {int(optimal_threshold)}")