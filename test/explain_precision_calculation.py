#!/usr/bin/env python3
"""
Explain and demonstrate how precision is calculated
Shows the difference between item-level and aggregate precision
"""

import pandas as pd
import numpy as np

def explain_precision():
    """Explain precision calculation with examples"""
    
    print("=" * 80)
    print("PRECISION CALCULATION EXPLANATION")
    print("=" * 80)
    
    print("\n📚 CURRENT CALCULATION METHOD:")
    print("-" * 80)
    print("""
The current precision calculation is ITEM-LEVEL (product-specific):

For each unique item (Customer + Facility + Product combination):
1. Aggregate predicted quantity across all dates
2. Aggregate actual quantity across all dates  
3. Binary classification: Is quantity >= threshold?
4. Compare predicted vs actual for THAT SPECIFIC ITEM

Example from your question:
ScionHealth Facility 101:
  - Item P1: Predicted 2, Actual 5 → Both ≥4 → TRUE POSITIVE ✓
  - Item P2: Predicted 5, Actual 5 → Both ≥4 → TRUE POSITIVE ✓
  - Item P3: Predicted 5, Actual 0 → Pred≥4, Actual<4 → FALSE POSITIVE ✗

Precision = TP / (TP + FP) = 2 / (2 + 1) = 66.7%

This is PRODUCT-SPECIFIC precision, not just facility-level counting.
""")
    
    print("\n🔍 LET'S VERIFY WITH ACTUAL DATA:")
    print("-" * 80)
    
    # Load data
    try:
        predictions_df = pd.read_csv('test/data/predictions.csv')
        val_df = pd.read_csv('test/data/val_data.csv')
    except FileNotFoundError:
        print("❌ Missing data files")
        return
    
    # Create item_id
    predictions_df['item_id'] = (predictions_df['CustomerID'].astype(str) + '_' + 
                                predictions_df['FacilityID'].astype(str) + '_' + 
                                predictions_df['ProductID'].astype(str))
    val_df['item_id'] = (val_df['CustomerID'].astype(str) + '_' + 
                        val_df['FacilityID'].astype(str) + '_' + 
                        val_df['ProductID'].astype(str))
    
    # Find common items
    common_items = set(predictions_df['item_id']) & set(val_df['item_id'])
    
    # Aggregate by item
    pred_agg = predictions_df[predictions_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'predicted_value': 'sum',
        'CustomerID': 'first',
        'FacilityID': 'first',
        'ProductID': 'first'
    }).reset_index()
    
    val_agg = val_df[val_df['item_id'].isin(common_items)].groupby('item_id').agg({
        'target_value': 'sum'
    }).reset_index()
    
    comparison_df = pred_agg.merge(val_agg, on='item_id', how='inner')
    
    # Example: ScionHealth Facility 101
    threshold = 4
    example = comparison_df[(comparison_df['CustomerID'] == 'scionhealth') & 
                           (comparison_df['FacilityID'] == 101)].copy()
    
    if len(example) > 0:
        print(f"\n📋 EXAMPLE: ScionHealth Facility 101")
        print(f"Total unique products: {len(example)}")
        print(f"Threshold: {threshold} units\n")
        
        # Binary classification
        example['actual_binary'] = (example['target_value'] >= threshold).astype(int)
        example['predicted_binary'] = (example['predicted_value'] >= threshold).astype(int)
        
        # Classify each item
        example['classification'] = 'TN'
        example.loc[(example['predicted_binary'] == 1) & (example['actual_binary'] == 1), 'classification'] = 'TP'
        example.loc[(example['predicted_binary'] == 1) & (example['actual_binary'] == 0), 'classification'] = 'FP'
        example.loc[(example['predicted_binary'] == 0) & (example['actual_binary'] == 1), 'classification'] = 'FN'
        
        # Count
        tp = (example['classification'] == 'TP').sum()
        fp = (example['classification'] == 'FP').sum()
        fn = (example['classification'] == 'FN').sum()
        tn = (example['classification'] == 'TN').sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"Classification Results:")
        print(f"  True Positives (TP):  {tp} items (predicted ≥{threshold} AND actual ≥{threshold})")
        print(f"  False Positives (FP): {fp} items (predicted ≥{threshold} BUT actual <{threshold})")
        print(f"  False Negatives (FN): {fn} items (predicted <{threshold} BUT actual ≥{threshold})")
        print(f"  True Negatives (TN):  {tn} items (predicted <{threshold} AND actual <{threshold})")
        
        print(f"\n  Precision: {precision:.3f} ({precision*100:.1f}%)")
        print(f"  Recall:    {recall:.3f} ({recall*100:.1f}%)")
        
        # Show some examples
        print(f"\n  Sample True Positives (correctly predicted orders):")
        tp_examples = example[example['classification'] == 'TP'].head(5)
        for _, row in tp_examples.iterrows():
            print(f"    Product {row['ProductID']}: Predicted {row['predicted_value']:.1f}, Actual {row['target_value']:.0f} ✓")
        
        if fp > 0:
            print(f"\n  Sample False Positives (over-predicted):")
            fp_examples = example[example['classification'] == 'FP'].head(5)
            for _, row in fp_examples.iterrows():
                print(f"    Product {row['ProductID']}: Predicted {row['predicted_value']:.1f}, Actual {row['target_value']:.0f} ✗")
        
        if fn > 0:
            print(f"\n  Sample False Negatives (under-predicted/missed):")
            fn_examples = example[example['classification'] == 'FN'].head(5)
            for _, row in fn_examples.iterrows():
                print(f"    Product {row['ProductID']}: Predicted {row['predicted_value']:.1f}, Actual {row['target_value']:.0f} ✗")
    
    print("\n" + "=" * 80)
    print("KEY TAKEAWAY:")
    print("=" * 80)
    print("""
✓ Precision IS product-specific (item-level)
✓ Each unique Customer+Facility+Product is evaluated separately
✓ We compare predicted vs actual for EACH SPECIFIC PRODUCT
✓ NOT just counting facilities or aggregate volumes

This means:
- If we predict Product A needs reorder but it doesn't → FALSE POSITIVE
- If we predict Product B needs reorder and it does → TRUE POSITIVE
- Precision = (Correct product predictions) / (All products we predicted)
""")

if __name__ == "__main__":
    explain_precision()
