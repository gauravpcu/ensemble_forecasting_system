#!/usr/bin/env python3
"""
Validation Script
=================
Compares predictions against validation data and calculates accuracy metrics.

This script takes:
1. A predictions CSV file (from predict.py)
2. A validation CSV file (from extract.py or actual data)

And outputs:
- Regression metrics (MAE, RMSE, MAPE)
- Classification metrics (Precision, Recall, F1)
- Confusion matrix
- Error analysis by volume category

Usage:
    python tests/validate.py predictions.csv val_data.csv
    python tests/validate.py predictions.csv val_data.csv --output results.txt
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix

from src.config import env_config


def load_data(predictions_file, validation_file):
    """Load predictions and validation data"""
    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)
    
    # Load predictions
    print(f"\nLoading predictions from: {predictions_file}")
    predictions = pd.read_csv(predictions_file)
    print(f"  Loaded {len(predictions):,} predictions")
    
    # Load validation
    print(f"\nLoading validation from: {validation_file}")
    validation = pd.read_csv(validation_file)
    print(f"  Loaded {len(validation):,} validation records")
    
    return predictions, validation


def merge_data(predictions, validation):
    """Merge predictions with validation actuals"""
    print("\n" + "=" * 80)
    print("MERGING DATA")
    print("=" * 80)
    
    # Create item_id if not present
    if 'item_id' not in predictions.columns:
        predictions['item_id'] = (
            predictions['CustomerID'].astype(str) + '_' +
            predictions['FacilityID'].astype(str) + '_' +
            predictions['ProductID'].astype(str)
        )
    
    if 'item_id' not in validation.columns:
        validation['item_id'] = (
            validation['CustomerID'].astype(str) + '_' +
            validation['FacilityID'].astype(str) + '_' +
            validation['ProductID'].astype(str)
        )
    
    # Select only necessary columns from predictions
    pred_cols = ['item_id', 'predicted_value', 'CustomerID', 'FacilityID', 'ProductID', 'ProductName']
    # Add optional columns if they exist
    for col in ['predicted_reorder', 'reorder_recommendation']:
        if col in predictions.columns:
            pred_cols.append(col)
    
    predictions_subset = predictions[pred_cols].copy()
    
    # Merge on item_id
    merged = predictions_subset.merge(
        validation[['item_id', 'target_value']],
        on='item_id',
        how='inner'
    )
    
    print(f"\nMatched {len(merged):,} items")
    print(f"  Predictions only: {len(predictions) - len(merged):,}")
    print(f"  Validation only: {len(validation) - len(merged):,}")
    
    return merged


def calculate_regression_metrics(merged):
    """Calculate regression metrics"""
    y_true = merged['target_value']
    y_pred = merged['predicted_value']
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
    
    # R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }


def calculate_classification_metrics(merged, threshold):
    """Calculate classification metrics"""
    y_true = merged['target_value']
    y_pred = merged['predicted_value']
    
    # Binary classification
    y_true_binary = (y_true >= threshold).astype(int)
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def analyze_by_volume(merged, threshold):
    """Analyze errors by volume category"""
    # Categorize by actual volume
    merged['volume_category'] = pd.cut(
        merged['target_value'],
        bins=[0, 5, 20, 100, float('inf')],
        labels=['Very Low (0-5)', 'Low (5-20)', 'Medium (20-100)', 'High (100+)']
    )
    
    results = []
    for category in merged['volume_category'].cat.categories:
        subset = merged[merged['volume_category'] == category]
        if len(subset) == 0:
            continue
        
        y_true = subset['target_value']
        y_pred = subset['predicted_value']
        
        mae = mean_absolute_error(y_true, y_pred)
        
        y_true_binary = (y_true >= threshold).astype(int)
        y_pred_binary = (y_pred >= threshold).astype(int)
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        
        results.append({
            'category': category,
            'count': len(subset),
            'mae': mae,
            'precision': precision,
            'recall': recall
        })
    
    return pd.DataFrame(results)


def print_results(regression_metrics, classification_metrics, volume_analysis, threshold):
    """Print formatted results"""
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    # Regression metrics
    print("\nRegression Metrics:")
    print(f"  MAE:              {regression_metrics['mae']:.2f} units")
    print(f"  RMSE:             {regression_metrics['rmse']:.2f} units")
    print(f"  MAPE:             {regression_metrics['mape']:.1f}%")
    print(f"  R²:               {regression_metrics['r2']:.3f}")
    
    # Classification metrics
    print(f"\nClassification Metrics (threshold={threshold}):")
    print(f"  Precision:        {classification_metrics['precision']:.1%}")
    print(f"  Recall:           {classification_metrics['recall']:.1%}")
    print(f"  F1 Score:         {classification_metrics['f1']:.1%}")
    print(f"  Accuracy:         {classification_metrics['accuracy']:.1%}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(f"                    Actual: No Order    Actual: Order")
    print(f"  Predicted: No     {classification_metrics['tn']:>10,}    {classification_metrics['fn']:>10,}")
    print(f"  Predicted: Yes    {classification_metrics['fp']:>10,}    {classification_metrics['tp']:>10,}")
    
    print("\n  Interpretation:")
    print(f"    True Positives (TP):   {classification_metrics['tp']:,} - Correctly predicted ORDER")
    print(f"    False Positives (FP):  {classification_metrics['fp']:,} - Incorrectly predicted ORDER (false alarms)")
    print(f"    False Negatives (FN):  {classification_metrics['fn']:,} - Missed orders (should have ordered)")
    print(f"    True Negatives (TN):   {classification_metrics['tn']:,} - Correctly predicted NO ORDER")
    
    # Volume analysis
    if not volume_analysis.empty:
        print("\nPerformance by Volume Category:")
        print(f"  {'Category':<20} {'Count':>10} {'MAE':>10} {'Precision':>12} {'Recall':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")
        for _, row in volume_analysis.iterrows():
            print(f"  {row['category']:<20} {row['count']:>10,} {row['mae']:>10.2f} {row['precision']:>11.1%} {row['recall']:>9.1%}")
    
    print("\n" + "=" * 80)


def save_results(output_file, regression_metrics, classification_metrics, volume_analysis, threshold, merged):
    """Save results to text file"""
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VALIDATION RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Items Validated: {len(merged):,}\n")
        
        # Regression metrics
        f.write("\nRegression Metrics:\n")
        f.write(f"  MAE:              {regression_metrics['mae']:.2f} units\n")
        f.write(f"  RMSE:             {regression_metrics['rmse']:.2f} units\n")
        f.write(f"  MAPE:             {regression_metrics['mape']:.1f}%\n")
        f.write(f"  R²:               {regression_metrics['r2']:.3f}\n")
        
        # Classification metrics
        f.write(f"\nClassification Metrics (threshold={threshold}):\n")
        f.write(f"  Precision:        {classification_metrics['precision']:.1%}\n")
        f.write(f"  Recall:           {classification_metrics['recall']:.1%}\n")
        f.write(f"  F1 Score:         {classification_metrics['f1']:.1%}\n")
        f.write(f"  Accuracy:         {classification_metrics['accuracy']:.1%}\n")
        
        # Confusion matrix
        f.write("\nConfusion Matrix:\n")
        f.write(f"  True Positives (TP):   {classification_metrics['tp']:,}\n")
        f.write(f"  False Positives (FP):  {classification_metrics['fp']:,}\n")
        f.write(f"  False Negatives (FN):  {classification_metrics['fn']:,}\n")
        f.write(f"  True Negatives (TN):   {classification_metrics['tn']:,}\n")
        
        # Volume analysis
        if not volume_analysis.empty:
            f.write("\nPerformance by Volume Category:\n")
            f.write(volume_analysis.to_string(index=False))
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"\n✓ Results saved to: {output_file}")


def save_comparison_csv(merged, threshold, output_dir):
    """Save detailed comparison CSV with predictions vs actuals"""
    # Calculate error metrics for each item
    merged['error'] = merged['predicted_value'] - merged['target_value']
    merged['abs_error'] = abs(merged['error'])
    merged['pct_error'] = (merged['error'] / (merged['target_value'] + 1e-10)) * 100
    
    # Classification results
    merged['actual_reorder'] = (merged['target_value'] >= threshold).astype(int)
    merged['predicted_reorder_binary'] = (merged['predicted_value'] >= threshold).astype(int)
    
    # Determine classification result
    def classify_result(row):
        if row['predicted_reorder_binary'] == 1 and row['actual_reorder'] == 1:
            return 'TP'  # True Positive
        elif row['predicted_reorder_binary'] == 1 and row['actual_reorder'] == 0:
            return 'FP'  # False Positive
        elif row['predicted_reorder_binary'] == 0 and row['actual_reorder'] == 1:
            return 'FN'  # False Negative
        else:
            return 'TN'  # True Negative
    
    merged['classification_result'] = merged.apply(classify_result, axis=1)
    
    # Volume category
    merged['volume_category'] = pd.cut(
        merged['target_value'],
        bins=[0, 5, 20, 100, float('inf')],
        labels=['Very Low (0-5)', 'Low (5-20)', 'Medium (20-100)', 'High (100+)']
    )
    
    # Calculate customer-level metrics
    customer_metrics = {}
    for customer in merged['CustomerID'].unique():
        customer_data = merged[merged['CustomerID'] == customer]
        
        # Regression metrics
        y_true = customer_data['target_value']
        y_pred = customer_data['predicted_value']
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        
        # Classification metrics
        y_true_binary = customer_data['actual_reorder']
        y_pred_binary = customer_data['predicted_reorder_binary']
        
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        accuracy = (y_true_binary == y_pred_binary).mean()
        
        customer_metrics[customer] = {
            'customer_mae': mae,
            'customer_rmse': rmse,
            'customer_mape': mape,
            'customer_precision': precision,
            'customer_recall': recall,
            'customer_f1': f1,
            'customer_accuracy': accuracy
        }
    
    # Add customer metrics to each row
    for metric_name in ['customer_mae', 'customer_rmse', 'customer_mape', 
                        'customer_precision', 'customer_recall', 'customer_f1', 'customer_accuracy']:
        merged[metric_name] = merged['CustomerID'].map(
            lambda x: customer_metrics[x][metric_name]
        )
    
    # Select and order columns for output
    output_columns = [
        'CustomerID', 'FacilityID', 'ProductID', 'ProductName',
        'predicted_value', 'target_value', 'error', 'abs_error', 'pct_error',
        'predicted_reorder_binary', 'actual_reorder', 'classification_result',
        'volume_category',
        'customer_mae', 'customer_rmse', 'customer_mape',
        'customer_precision', 'customer_recall', 'customer_f1', 'customer_accuracy',
        'item_id'
    ]
    
    # Filter to only columns that exist
    output_columns = [col for col in output_columns if col in merged.columns]
    
    output_df = merged[output_columns].copy()
    
    # Sort by customer, then by absolute error (largest errors first)
    output_df = output_df.sort_values(['CustomerID', 'abs_error'], ascending=[True, False])
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'validation_comparison.csv')
    output_df.to_csv(csv_path, index=False)
    
    print(f"✓ Comparison CSV saved to: {csv_path}")
    print(f"  - {len(output_df):,} items")
    print(f"  - {output_df['CustomerID'].nunique()} customers")
    
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description='Validate predictions against actual data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/validate.py predictions.csv val_data.csv
  python tests/validate.py predictions.csv val_data.csv --output results.txt
  python tests/validate.py predictions.csv val_data.csv --threshold 5
  python tests/validate.py predictions.csv val_data.csv --output-dir ./results
        """
    )
    
    parser.add_argument('predictions', help='Predictions CSV file')
    parser.add_argument('validation', help='Validation CSV file')
    parser.add_argument('--output', '-o', help='Output text file for summary results (optional)')
    parser.add_argument('--output-dir', '-d', default='./tests/data',
                       help='Output directory for comparison CSV (default: ./tests/data)')
    parser.add_argument('--threshold', '-t', type=float, 
                       default=env_config.CLASSIFICATION_THRESHOLD,
                       help=f'Classification threshold (default: {env_config.CLASSIFICATION_THRESHOLD})')
    
    args = parser.parse_args()
    
    try:
        # Load data
        predictions, validation = load_data(args.predictions, args.validation)
        
        # Merge
        merged = merge_data(predictions, validation)
        
        if len(merged) == 0:
            print("\n❌ Error: No matching items found between predictions and validation")
            sys.exit(1)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        regression_metrics = calculate_regression_metrics(merged)
        classification_metrics = calculate_classification_metrics(merged, args.threshold)
        volume_analysis = analyze_by_volume(merged, args.threshold)
        
        # Print results
        print_results(regression_metrics, classification_metrics, volume_analysis, args.threshold)
        
        # Save comparison CSV (always)
        print("\nSaving comparison CSV...")
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = save_comparison_csv(merged, args.threshold, args.output_dir)
        
        # Save summary to text file if requested
        if args.output:
            save_results(args.output, regression_metrics, classification_metrics, 
                        volume_analysis, args.threshold, merged)
        
        print("\n✅ Validation completed successfully!")
        print(f"\nOutput files:")
        print(f"  - Comparison CSV: {csv_path}")
        if args.output:
            print(f"  - Summary text: {args.output}")
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
