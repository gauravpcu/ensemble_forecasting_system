# Ensemble Forecasting System - Quick Reference

## 🚀 Quick Start

```bash
# Run full test pipeline
python3 test/run_full_test.py

# View results
cat test/data/ANALYSIS_SUMMARY.txt
```

---

## ⚙️ Common Configuration Changes

```bash
# Adjust classification threshold (4-10)
python3 configure.py --set CLASSIFICATION_THRESHOLD 4

# Adjust volume safety multipliers
python3 configure.py --set SAFETY_MULTIPLIER_LOW 1.8
python3 configure.py --set SAFETY_MULTIPLIER_MEDIUM 1.3
python3 configure.py --set SAFETY_MULTIPLIER_HIGH 1.1

# Update customer calibration
python3 configure.py --set CUSTOMER_CALIBRATION "scionhealth:0.206,mercy:0.362"
```

---

## 📊 Key Output Files

| File | What It Shows | Use For |
|------|---------------|---------|
| `predictions.csv` | All predictions with actuals | Detailed analysis |
| `dual_analysis_items.csv` | Product-level precision (78K items) | Which products correct/incorrect |
| `product_precision_scionhealth_mercy.csv` | ScionHealth & Mercy precision (16K items) | Customer-specific analysis |
| `customer_analysis.csv` | Customer-level metrics | Identify calibration needs |
| `customer_facility_metrics.csv` | Facility-level metrics | Facility performance |
| `ANALYSIS_SUMMARY.txt` | Executive summary | Quick overview |

---

## 📈 Key Metrics Explained

### Classification Metrics

| Metric | Formula | Good Value | Meaning |
|--------|---------|------------|---------|
| **Precision** | TP/(TP+FP) | >80% | When we say "order", how often are we right? |
| **Recall** | TP/(TP+FN) | >90% | Of items that need ordering, how many do we catch? |
| **F1 Score** | 2×(P×R)/(P+R) | >80% | Balanced precision and recall |

### Regression Metrics

| Metric | Good Value | Meaning |
|--------|------------|---------|
| **MAE** | <5 units | Average error in units |
| **MAPE** | <50% | Average percentage error |
| **Volume Error** | 20-50% | Total over/under-prediction |

---

## 🎯 Classification Labels

| Label | Meaning | Business Impact |
|-------|---------|-----------------|
| **TP** (True Positive) | Predicted order ✓, Actually ordered ✓ | Correct prediction |
| **FP** (False Positive) | Predicted order ✓, Not ordered ✗ | Over-ordering |
| **FN** (False Negative) | Didn't predict ✗, Actually ordered ✓ | Stockout risk |
| **TN** (True Negative) | Didn't predict ✓, Not ordered ✓ | Correct no-order |

---

## 🔧 Common Tasks

### Optimize Threshold
```bash
python3 test/test_thresholds.py
# Follow prompts to apply optimal threshold
```

### Apply Calibration
```bash
python3 test/apply_calibration.py
```

### Analyze Specific Customer
```bash
python3 test/customer_precision.py scionhealth
```

### Extract Customer Predictions
```bash
# All dates
python3 test/extract_all_facilities.py

# Specific date
python3 test/extract_customer_predictions.py 2025-10-15
```

---

## 🎛️ Parameter Impact Guide

### To Reduce Over-Ordering (High Volume Error)
- ✓ Decrease `SAFETY_MULTIPLIER_LOW/MEDIUM/HIGH`
- ✓ Increase `CLASSIFICATION_THRESHOLD`
- ✓ Adjust `CUSTOMER_CALIBRATION` (lower multipliers)

### To Reduce Stockouts (Low Recall)
- ✓ Decrease `CLASSIFICATION_THRESHOLD`
- ✓ Increase `SAFETY_MULTIPLIER_LOW/MEDIUM/HIGH`
- ✓ Adjust `CUSTOMER_CALIBRATION` (higher multipliers)

### To Reduce False Alarms (Low Precision)
- ✓ Increase `CLASSIFICATION_THRESHOLD`
- ✓ Decrease `SAFETY_MULTIPLIER_LOW/MEDIUM/HIGH`

---

## 📋 Typical Performance Targets

| Metric | Target | Current (ScionHealth) |
|--------|--------|----------------------|
| Precision | >80% | 82.2% ✓ |
| Recall | >90% | 92.4% ✓ |
| F1 Score | >80% | 87.0% ✓ |
| MAE | <10 units | 24.92 units |
| Volume Error | 20-50% | +57.8% |

---

## 🔍 Quick Diagnostics

### Check Overall Performance
```bash
cat test/data/ANALYSIS_SUMMARY.txt
```

### Check Customer Performance
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('test/data/customer_analysis.csv')
print(df.sort_values('total_actual', ascending=False).head(10))
"
```

### Check Product-Level Issues
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('test/data/product_precision_scionhealth_mercy.csv')
print(f'TP: {len(df[df[\"classification\"]==\"TP\"])}')
print(f'FP: {len(df[df[\"classification\"]==\"FP\"])}')
print(f'FN: {len(df[df[\"classification\"]==\"FN\"])}')
"
```

---

## 🚨 Troubleshooting

| Problem | Quick Fix |
|---------|-----------|
| Volume Error >150% | Reduce safety multipliers by 0.2-0.3 |
| Recall <80% | Decrease threshold by 1-2 units |
| Precision <50% | Increase threshold by 1-2 units |
| Customer over-predicting | Lower customer calibration factor |
| Customer under-predicting | Raise customer calibration factor |

---

## 📞 File Locations

All output files are in: `test/data/`

All configuration in: `.env`

Full documentation: `USER_GUIDE.md`

---

**Quick Tip:** After any configuration change, always run:
```bash
python3 test/apply_calibration.py
python3 test/dual_prediction_analysis.py
```
to see the impact!
