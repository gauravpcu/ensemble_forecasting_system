"""
Calculate accuracy, precision, and recall for each customer-facility combination
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("CUSTOMER-FACILITY ACCURACY ANALYSIS")
print("=" * 80)

# Load predictions
print("\n[1/4] Loading data...")
predictions_df = pd.read_csv('test/data/predictions.csv')
val_df = pd.read_csv('test/data/val_data.csv')

predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
val_df['timestamp'] = pd.to_datetime(val_df['timestamp'])

print(f"   Predictions: {len(predictions_df):,} records")
print(f"   Validation: {len(val_df):,} records")

# Create customer-facility key
predictions_df['customer_facility'] = (predictions_df['CustomerID'].astype(str) + '_' + 
                                       predictions_df['FacilityID'].astype(str))
val_df['customer_facility'] = (val_df['CustomerID'].astype(str) + '_' + 
                               val_df['FacilityID'].astype(str))

print(f"   Unique customer-facility combinations in test: {predictions_df['customer_facility'].nunique():,}")
print(f"   Unique customer-facility combinations in validation: {val_df['customer_facility'].nunique():,}")

# Calculate metrics for each customer-facility combination
print("\n[2/4] Calculating metrics for each customer-facility...")

def calculate_classification_metrics(actual, predicted, threshold=5):
    """
    Calculate precision, recall, F1 for binary classification
    Classify as 'needs_order' if value >= threshold
    """
    actual_binary = (actual >= threshold).astype(int)
    predicted_binary = (predicted >= threshold).astype(int)
    
    if len(np.unique(actual_binary)) < 2 or len(np.unique(predicted_binary)) < 2:
        # Not enough classes for precision/recall
        return {
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'true_positives': np.nan,
            'false_positives': np.nan,
            'true_negatives': np.nan,
            'false_negatives': np.nan
        }
    
    try:
        precision = precision_score(actual_binary, predicted_binary, zero_division=0)
        recall = recall_score(actual_binary, predicted_binary, zero_division=0)
        f1 = f1_score(actual_binary, predicted_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(actual_binary, predicted_binary).ravel()
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    except:
        return {
            'precision': np.nan,
            'recall': np.nan,
            'f1_score': np.nan,
            'true_positives': np.nan,
            'false_positives': np.nan,
            'true_negatives': np.nan,
            'false_negatives': np.nan
        }

def calculate_metrics_for_group(group):
    """Calculate all metrics for a customer-facility group"""
    actual = group['target_value'].values
    predicted = group['predicted_value'].values
    
    # Regression metrics
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    
    # MAPE (only for non-zero actuals)
    non_zero = actual > 0
    if non_zero.sum() > 0:
        mape = (np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero]) * 100).mean()
    else:
        mape = np.nan
    
    # R-squared
    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    # Volume metrics
    total_actual = actual.sum()
    total_predicted = predicted.sum()
    volume_error = total_predicted - total_actual
    volume_error_pct = (volume_error / total_actual * 100) if total_actual > 0 else np.nan
    
    # Classification metrics (threshold = 5 units)
    classification_metrics = calculate_classification_metrics(actual, predicted, threshold=5)
    
    return pd.Series({
        'CustomerID': group['CustomerID'].iloc[0],
        'FacilityID': group['FacilityID'].iloc[0],
        'records': len(group),
        'unique_products': group['ProductID'].nunique(),
        'total_actual': total_actual,
        'total_predicted': total_predicted,
        'volume_error': volume_error,
        'volume_error_pct': volume_error_pct,
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2,
        'precision': classification_metrics['precision'],
        'recall': classification_metrics['recall'],
        'f1_score': classification_metrics['f1_score'],
        'true_positives': classification_metrics['true_positives'],
        'false_positives': classification_metrics['false_positives'],
        'true_negatives': classification_metrics['true_negatives'],
        'false_negatives': classification_metrics['false_negatives'],
        'mean_actual': actual.mean(),
        'median_actual': np.median(actual),
        'std_actual': actual.std(),
        'mean_predicted': predicted.mean(),
        'median_predicted': np.median(predicted),
        'std_predicted': predicted.std()
    })

# Calculate metrics for each customer-facility
customer_facility_metrics = predictions_df.groupby('customer_facility').apply(calculate_metrics_for_group).reset_index(drop=True)

print(f"   Calculated metrics for {len(customer_facility_metrics):,} customer-facility combinations")

# Sort by volume
customer_facility_metrics = customer_facility_metrics.sort_values('total_actual', ascending=False)

# Add accuracy score (100 - MAPE)
customer_facility_metrics['accuracy_pct'] = 100 - customer_facility_metrics['mape']

# Add performance category
def categorize_performance(row):
    if pd.isna(row['mape']):
        return 'Unknown'
    elif row['mape'] < 30:
        return 'Excellent'
    elif row['mape'] < 50:
        return 'Good'
    elif row['mape'] < 75:
        return 'Fair'
    else:
        return 'Poor'

customer_facility_metrics['performance'] = customer_facility_metrics.apply(categorize_performance, axis=1)

# Save detailed results
print("\n[3/4] Saving results...")
output_file = 'test/data/customer_facility_metrics.csv'
customer_facility_metrics.to_csv(output_file, index=False)
print(f"   ✓ Saved: {output_file}")

# Create summary statistics
print("\n[4/4] Generating summary statistics...")
print("-" * 80)

print(f"\nOverall Statistics:")
print(f"   Total customer-facility combinations: {len(customer_facility_metrics):,}")
print(f"   Total records analyzed: {customer_facility_metrics['records'].sum():,}")
print(f"   Total actual volume: {customer_facility_metrics['total_actual'].sum():,.0f} units")
print(f"   Total predicted volume: {customer_facility_metrics['total_predicted'].sum():,.0f} units")

print(f"\nPerformance Distribution:")
perf_dist = customer_facility_metrics['performance'].value_counts()
for perf, count in perf_dist.items():
    pct = count / len(customer_facility_metrics) * 100
    print(f"   {perf:15s}: {count:5,} ({pct:5.1f}%)")

print(f"\nAverage Metrics (across all customer-facilities):")
print(f"   MAE:       {customer_facility_metrics['mae'].mean():.2f} units")
print(f"   RMSE:      {customer_facility_metrics['rmse'].mean():.2f} units")
print(f"   MAPE:      {customer_facility_metrics['mape'].mean():.2f}%")
print(f"   R²:        {customer_facility_metrics['r2'].mean():.3f}")
print(f"   Precision: {customer_facility_metrics['precision'].mean():.3f}")
print(f"   Recall:    {customer_facility_metrics['recall'].mean():.3f}")
print(f"   F1 Score:  {customer_facility_metrics['f1_score'].mean():.3f}")

# Top 20 by volume
print("\n" + "=" * 80)
print("TOP 20 CUSTOMER-FACILITY COMBINATIONS BY VOLUME")
print("=" * 80)
print(f"\n{'Customer':<25} {'Facility':<10} {'Actual':>10} {'Predicted':>10} {'MAE':>8} {'MAPE':>8} {'Precision':>10} {'Recall':>10}")
print("-" * 120)

top_20 = customer_facility_metrics.head(20)
for idx, row in top_20.iterrows():
    print(f"{row['CustomerID']:<25} {str(row['FacilityID']):<10} "
          f"{row['total_actual']:>10,.0f} {row['total_predicted']:>10,.0f} "
          f"{row['mae']:>8.2f} {row['mape']:>7.1f}% "
          f"{row['precision']:>10.3f} {row['recall']:>10.3f}")

# Best performers (by MAPE, min 100 units actual)
print("\n" + "=" * 80)
print("TOP 20 BEST PERFORMERS (Min 100 units actual volume)")
print("=" * 80)

best_performers = customer_facility_metrics[customer_facility_metrics['total_actual'] >= 100].nsmallest(20, 'mape')
print(f"\n{'Customer':<25} {'Facility':<10} {'Actual':>10} {'MAPE':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 110)

for idx, row in best_performers.iterrows():
    print(f"{row['CustomerID']:<25} {str(row['FacilityID']):<10} "
          f"{row['total_actual']:>10,.0f} {row['mape']:>7.1f}% "
          f"{row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1_score']:>10.3f}")

# Worst performers (by MAPE, min 100 units actual)
print("\n" + "=" * 80)
print("TOP 20 WORST PERFORMERS (Min 100 units actual volume)")
print("=" * 80)

worst_performers = customer_facility_metrics[customer_facility_metrics['total_actual'] >= 100].nlargest(20, 'mape')
print(f"\n{'Customer':<25} {'Facility':<10} {'Actual':>10} {'MAPE':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print("-" * 110)

for idx, row in worst_performers.iterrows():
    print(f"{row['CustomerID']:<25} {str(row['FacilityID']):<10} "
          f"{row['total_actual']:>10,.0f} {row['mape']:>7.1f}% "
          f"{row['precision']:>10.3f} {row['recall']:>10.3f} {row['f1_score']:>10.3f}")

# Precision/Recall analysis
print("\n" + "=" * 80)
print("PRECISION & RECALL ANALYSIS")
print("=" * 80)
print("\nClassification threshold: 5 units (items needing reorder)")
print("Precision: Of items predicted to need reorder, how many actually did?")
print("Recall: Of items that actually needed reorder, how many did we predict?")

# Filter out NaN values for precision/recall stats
valid_metrics = customer_facility_metrics.dropna(subset=['precision', 'recall'])

print(f"\n{'Metric':<20} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
print("-" * 70)
print(f"{'Precision':<20} {valid_metrics['precision'].mean():>10.3f} "
      f"{valid_metrics['precision'].median():>10.3f} "
      f"{valid_metrics['precision'].min():>10.3f} "
      f"{valid_metrics['precision'].max():>10.3f}")
print(f"{'Recall':<20} {valid_metrics['recall'].mean():>10.3f} "
      f"{valid_metrics['recall'].median():>10.3f} "
      f"{valid_metrics['recall'].min():>10.3f} "
      f"{valid_metrics['recall'].max():>10.3f}")
print(f"{'F1 Score':<20} {valid_metrics['f1_score'].mean():>10.3f} "
      f"{valid_metrics['f1_score'].median():>10.3f} "
      f"{valid_metrics['f1_score'].min():>10.3f} "
      f"{valid_metrics['f1_score'].max():>10.3f}")

# Confusion matrix totals
print(f"\nConfusion Matrix (Aggregated across all customer-facilities):")
total_tp = customer_facility_metrics['true_positives'].sum()
total_fp = customer_facility_metrics['false_positives'].sum()
total_tn = customer_facility_metrics['true_negatives'].sum()
total_fn = customer_facility_metrics['false_negatives'].sum()

print(f"\n                    Predicted: No Reorder    Predicted: Reorder")
print(f"Actual: No Reorder  {total_tn:>15,.0f}     {total_fp:>15,.0f}")
print(f"Actual: Reorder     {total_fn:>15,.0f}     {total_tp:>15,.0f}")

if total_tp + total_fp > 0:
    overall_precision = total_tp / (total_tp + total_fp)
else:
    overall_precision = 0

if total_tp + total_fn > 0:
    overall_recall = total_tp / (total_tp + total_fn)
else:
    overall_recall = 0

if overall_precision + overall_recall > 0:
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
else:
    overall_f1 = 0

print(f"\nOverall Classification Metrics:")
print(f"   Precision: {overall_precision:.3f}")
print(f"   Recall:    {overall_recall:.3f}")
print(f"   F1 Score:  {overall_f1:.3f}")

# Save summary report
print("\n" + "=" * 80)
print("SAVING SUMMARY REPORT")
print("=" * 80)

summary_file = 'test/data/customer_facility_summary.txt'
with open(summary_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("CUSTOMER-FACILITY ACCURACY ANALYSIS SUMMARY\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total customer-facility combinations: {len(customer_facility_metrics):,}\n")
    f.write(f"Total records analyzed: {customer_facility_metrics['records'].sum():,}\n")
    f.write(f"Total actual volume: {customer_facility_metrics['total_actual'].sum():,.0f} units\n")
    f.write(f"Total predicted volume: {customer_facility_metrics['total_predicted'].sum():,.0f} units\n\n")
    
    f.write("PERFORMANCE DISTRIBUTION\n")
    f.write("-" * 80 + "\n")
    for perf, count in perf_dist.items():
        pct = count / len(customer_facility_metrics) * 100
        f.write(f"{perf:15s}: {count:5,} ({pct:5.1f}%)\n")
    
    f.write("\nAVERAGE METRICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"MAE:       {customer_facility_metrics['mae'].mean():.2f} units\n")
    f.write(f"RMSE:      {customer_facility_metrics['rmse'].mean():.2f} units\n")
    f.write(f"MAPE:      {customer_facility_metrics['mape'].mean():.2f}%\n")
    f.write(f"R²:        {customer_facility_metrics['r2'].mean():.3f}\n")
    f.write(f"Precision: {customer_facility_metrics['precision'].mean():.3f}\n")
    f.write(f"Recall:    {customer_facility_metrics['recall'].mean():.3f}\n")
    f.write(f"F1 Score:  {customer_facility_metrics['f1_score'].mean():.3f}\n\n")
    
    f.write("CLASSIFICATION METRICS (Threshold: 5 units)\n")
    f.write("-" * 80 + "\n")
    f.write(f"Overall Precision: {overall_precision:.3f}\n")
    f.write(f"Overall Recall:    {overall_recall:.3f}\n")
    f.write(f"Overall F1 Score:  {overall_f1:.3f}\n\n")
    
    f.write("TOP 10 PERFORMERS (by MAPE, min 100 units)\n")
    f.write("-" * 80 + "\n")
    for idx, row in best_performers.head(10).iterrows():
        f.write(f"{row['CustomerID']:<25} {str(row['FacilityID']):<10} "
                f"MAPE: {row['mape']:>6.1f}% "
                f"Precision: {row['precision']:.3f} "
                f"Recall: {row['recall']:.3f}\n")

print(f"\n✓ Summary saved: {summary_file}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
print(f"\nGenerated files:")
print(f"  • test/data/customer_facility_metrics.csv - Detailed metrics for each combination")
print(f"  • test/data/customer_facility_summary.txt - Summary report")
print(f"\nKey findings:")
print(f"  • {len(customer_facility_metrics):,} customer-facility combinations analyzed")
print(f"  • Average MAPE: {customer_facility_metrics['mape'].mean():.1f}%")
print(f"  • Average Precision: {customer_facility_metrics['precision'].mean():.3f}")
print(f"  • Average Recall: {customer_facility_metrics['recall'].mean():.3f}")
print(f"  • Overall F1 Score: {overall_f1:.3f}")
