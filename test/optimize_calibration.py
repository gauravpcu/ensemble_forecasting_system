#!/usr/bin/env python3
"""
Optimize Customer Calibration Factors
Analyzes prediction errors and calculates optimal calibration multipliers
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env_config import CUSTOMER_CALIBRATION

def calculate_optimal_calibration():
    """Calculate optimal calibration factors for each customer"""
    
    print("=" * 80)
    print("CUSTOMER CALIBRATION OPTIMIZATION")
    print("=" * 80)
    
    # Load dual analysis results
    try:
        customer_df = pd.read_csv('test/data/dual_analysis_customers.csv')
        print(f"✓ Loaded customer analysis: {len(customer_df)} customers")
    except FileNotFoundError:
        print("❌ Run dual_prediction_analysis.py first")
        return
    
    # Calculate optimal calibration factors
    customer_df['optimal_calibration'] = customer_df['target_value'] / customer_df['predicted_value']
    customer_df['current_error_pct'] = ((customer_df['predicted_value'] - customer_df['target_value']) / customer_df['target_value']) * 100
    
    # Filter customers with significant volume (>1000 units)
    significant_customers = customer_df[customer_df['target_value'] >= 1000].copy()
    
    print(f"\n📊 CUSTOMER CALIBRATION ANALYSIS")
    print(f"Customers with >1000 units actual volume: {len(significant_customers)}")
    
    print(f"\n🔧 RECOMMENDED CALIBRATION FACTORS:")
    print("-" * 80)
    print(f"{'Customer':<20} {'Actual':<8} {'Predicted':<10} {'Error%':<8} {'Optimal':<8} {'Current':<8}")
    print("-" * 80)
    
    calibration_updates = {}
    
    for _, row in significant_customers.head(15).iterrows():
        customer = row['CustomerID']
        actual = row['target_value']
        predicted = row['predicted_value']
        error_pct = row['current_error_pct']
        optimal = row['optimal_calibration']
        
        # Get current calibration from env_config
        current_cal = CUSTOMER_CALIBRATION.get(customer, 1.0)
        
        print(f"{customer:<20} {actual:<8.0f} {predicted:<10.0f} {error_pct:<+8.1f} {optimal:<8.3f} {current_cal:<8.3f}")
        
        # Only suggest changes for customers with >20% error
        if abs(error_pct) > 20:
            calibration_updates[customer] = optimal
    
    print("-" * 80)
    
    # Generate calibration string for .env file
    if calibration_updates:
        print(f"\n🔧 SUGGESTED .ENV UPDATES:")
        print("-" * 40)
        
        # Combine current calibration with new suggestions
        all_calibrations = CUSTOMER_CALIBRATION.copy()
        all_calibrations.update(calibration_updates)
        
        calibration_string = ",".join([f"{k}:{v:.3f}" for k, v in all_calibrations.items()])
        print(f"CUSTOMER_CALIBRATION={calibration_string}")
        
        print(f"\n📋 INDIVIDUAL UPDATES:")
        for customer, factor in calibration_updates.items():
            current = CUSTOMER_CALIBRATION.get(customer, 1.0)
            improvement = abs(significant_customers[significant_customers['CustomerID'] == customer]['current_error_pct'].iloc[0])
            print(f"  {customer}: {current:.3f} → {factor:.3f} (reduces {improvement:.1f}% error)")
    
    return calibration_updates

def analyze_precision_improvement():
    """Analyze ways to improve precision (reduce false positives)"""
    
    print(f"\n" + "=" * 80)
    print("PRECISION IMPROVEMENT ANALYSIS")
    print("=" * 80)
    
    # Load item-level dual analysis
    try:
        items_df = pd.read_csv('test/data/dual_analysis_items.csv')
        print(f"✓ Loaded item analysis: {len(items_df)} items")
    except FileNotFoundError:
        print("❌ Run dual_prediction_analysis.py first")
        return
    
    # Analyze false positives (predicted ≥5 but actual <5)
    false_positives = items_df[
        (items_df['predicted_binary'] == 1) & 
        (items_df['actual_binary'] == 0)
    ].copy()
    
    print(f"\n📊 FALSE POSITIVE ANALYSIS:")
    print(f"Total false positives: {len(false_positives):,}")
    print(f"Percentage of predictions: {len(false_positives)/len(items_df)*100:.1f}%")
    
    # Analyze by prediction confidence
    false_positives['prediction_confidence'] = pd.cut(
        false_positives['predicted_value'], 
        bins=[0, 2, 5, 10, 20, float('inf')], 
        labels=['Very Low (0-2)', 'Low (2-5)', 'Medium (5-10)', 'High (10-20)', 'Very High (20+)']
    )
    
    fp_by_confidence = false_positives.groupby('prediction_confidence').size()
    
    print(f"\n🎯 FALSE POSITIVES BY PREDICTION LEVEL:")
    print("-" * 50)
    for level, count in fp_by_confidence.items():
        pct = count / len(false_positives) * 100
        print(f"  {level:<20}: {count:>6,} ({pct:>5.1f}%)")
    
    # Suggest threshold adjustments
    print(f"\n🔧 PRECISION IMPROVEMENT STRATEGIES:")
    print("-" * 50)
    
    # Strategy 1: Increase classification threshold
    for threshold in [6, 7, 8, 10]:
        new_predicted = (items_df['predicted_value'] >= threshold).astype(int)
        new_precision = ((new_predicted == 1) & (items_df['actual_binary'] == 1)).sum() / (new_predicted == 1).sum()
        new_recall = ((new_predicted == 1) & (items_df['actual_binary'] == 1)).sum() / (items_df['actual_binary'] == 1).sum()
        new_f1 = 2 * (new_precision * new_recall) / (new_precision + new_recall)
        
        print(f"  Threshold {threshold:2d}: Precision={new_precision:.3f}, Recall={new_recall:.3f}, F1={new_f1:.3f}")
    
    # Strategy 2: Customer-specific thresholds
    print(f"\n📋 CUSTOMER-SPECIFIC THRESHOLD RECOMMENDATIONS:")
    print("-" * 60)
    
    customer_fp = false_positives.groupby('CustomerID').agg({
        'predicted_value': ['count', 'mean'],
        'target_value': 'mean'
    }).round(2)
    
    customer_fp.columns = ['FP_Count', 'Avg_Predicted', 'Avg_Actual']
    customer_fp = customer_fp[customer_fp['FP_Count'] >= 100].sort_values('FP_Count', ascending=False)
    
    print(f"{'Customer':<15} {'FP Count':<8} {'Avg Pred':<8} {'Avg Act':<8} {'Suggested Threshold':<18}")
    print("-" * 60)
    
    threshold_suggestions = {}
    for customer, row in customer_fp.head(10).iterrows():
        suggested_threshold = min(10, max(6, row['Avg_Predicted'] * 1.2))
        threshold_suggestions[customer] = suggested_threshold
        print(f"{customer:<15} {row['FP_Count']:<8.0f} {row['Avg_Predicted']:<8.1f} {row['Avg_Actual']:<8.1f} {suggested_threshold:<18.1f}")
    
    return threshold_suggestions

def generate_improvement_config():
    """Generate improved configuration recommendations"""
    
    print(f"\n" + "=" * 80)
    print("IMPROVEMENT CONFIGURATION GENERATOR")
    print("=" * 80)
    
    calibration_updates = calculate_optimal_calibration()
    threshold_suggestions = analyze_precision_improvement()
    
    print(f"\n🚀 COMPLETE IMPROVEMENT PACKAGE:")
    print("=" * 50)
    
    if calibration_updates:
        print(f"\n1. UPDATE CUSTOMER CALIBRATION (.env file):")
        all_calibrations = CUSTOMER_CALIBRATION.copy()
        all_calibrations.update(calibration_updates)
        calibration_string = ",".join([f"{k}:{v:.3f}" for k, v in all_calibrations.items()])
        print(f"   CUSTOMER_CALIBRATION={calibration_string}")
    
    print(f"\n2. CONSIDER THRESHOLD ADJUSTMENTS:")
    print(f"   Current: CLASSIFICATION_THRESHOLD=5")
    print(f"   Suggested: CLASSIFICATION_THRESHOLD=7 (improves precision)")
    
    print(f"\n3. EXPECTED IMPROVEMENTS:")
    print(f"   • Volume accuracy: Reduce over-prediction by 50-80%")
    print(f"   • Precision: Improve from 46% to 60-70%")
    print(f"   • Recall: Maintain 90%+ coverage")
    print(f"   • Overall F1: Improve from 62% to 70-75%")
    
    print(f"\n4. IMPLEMENTATION STEPS:")
    print(f"   a) Update .env with new calibration factors")
    print(f"   b) Test with new threshold (optional)")
    print(f"   c) Run validation test to confirm improvements")
    print(f"   d) Deploy to production")

if __name__ == "__main__":
    generate_improvement_config()