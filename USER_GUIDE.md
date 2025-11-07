# Ensemble Forecasting System - User Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Configuration Parameters](#configuration-parameters)
3. [Running the System](#running-the-system)
4. [Output Files Reference](#output-files-reference)
5. [Understanding Metrics](#understanding-metrics)
6. [Common Workflows](#common-workflows)

---

## System Overview

The Ensemble Forecasting System predicts inventory reorder needs for healthcare supply chain management by combining:
- **LightGBM (95%):** Structured feature-based predictions
- **DeepAR (5%):** Time series pattern recognition

### Key Features
- Extended 90-day context for seasonal pattern detection
- Customer-specific calibration factors
- Volume-based safety multipliers
- Product-level precision tracking
- Comprehensive performance metrics

---

## Configuration Parameters

All parameters are configured in the `.env` file. Use the `configure.py` script to update them:

```bash
python3 configure.py --set PARAMETER_NAME value
```

### 1. Data Extraction Parameters

#### `TOTAL_EXTRACTION_DAYS` (Default: 90)
**What it does:** Total days of historical data to extract for analysis

**Impact:**
- Higher values (60-90): Better seasonal pattern detection, more stable predictions
- Lower values (14-30): Faster processing, captures recent trends only

**Recommended:** 90 days for optimal balance

**Example:**
```bash
python3 configure.py --set TOTAL_EXTRACTION_DAYS 90
```

---

#### `VALIDATION_DAYS` (Default: 14)
**What it does:** Number of most recent days used for forward-looking validation

**Impact:**
- This is your "test period" - data the model hasn't seen
- Used to measure real-world prediction accuracy

**Recommended:** 14 days (2 weeks)

---

#### `TEST_DAYS` (Default: 14)
**What it does:** Number of days before validation period used for feature engineering

**Impact:**
- Provides context for validation predictions
- Not used for accuracy measurement

**Recommended:** 14 days

---

### 2. Classification Parameters

#### `CLASSIFICATION_THRESHOLD` (Default: 4)
**What it does:** Minimum units to classify an item as "needs reorder"

**Impact:**
- **Lower (3-5):** More sensitive, catches more orders, higher recall, more false alarms
- **Higher (7-10):** More conservative, fewer false alarms, higher precision, may miss orders

**Business Meaning:**
- If predicted ≥ threshold → "Order this product"
- If predicted < threshold → "Don't order this product"

**Example:**
```bash
# More sensitive (catch more orders)
python3 configure.py --set CLASSIFICATION_THRESHOLD 4

# More conservative (fewer false alarms)
python3 configure.py --set CLASSIFICATION_THRESHOLD 8
```

**Affects:**
- Precision, Recall, F1 Score
- False Positive and False Negative counts

---

### 3. Ensemble Weights

#### `LIGHTGBM_WEIGHT` (Default: 0.95)
#### `DEEPAR_WEIGHT` (Default: 0.05)
**What they do:** Control how much each model contributes to final prediction

**Impact:**
- LightGBM: Better for structured features, stable patterns
- DeepAR: Better for time series, seasonality

**Must sum to 1.0**

**Example:**
```bash
# Increase DeepAR influence
python3 configure.py --set LIGHTGBM_WEIGHT 0.90
python3 configure.py --set DEEPAR_WEIGHT 0.10
```

---

### 4. Volume-Based Safety Multipliers

#### `SAFETY_MULTIPLIER_LOW` (Default: 1.8)
**What it does:** Multiplier applied to low-volume predictions (0-5 units)

**Impact:**
- Higher values: More conservative, reduces stockout risk for low-volume items
- Lower values: Reduces over-ordering

**Example:**
```bash
python3 configure.py --set SAFETY_MULTIPLIER_LOW 1.8
```

---

#### `SAFETY_MULTIPLIER_MEDIUM` (Default: 1.3)
**What it does:** Multiplier applied to medium-volume predictions (5-20 units)

---

#### `SAFETY_MULTIPLIER_HIGH` (Default: 1.1)
**What it does:** Multiplier applied to high-volume predictions (20-100 units)

---

#### `VOLUME_LOW_THRESHOLD` (Default: 5)
#### `VOLUME_MEDIUM_THRESHOLD` (Default: 20)
#### `VOLUME_HIGH_THRESHOLD` (Default: 100)
**What they do:** Define boundaries for volume categories

**Example:**
```bash
# Adjust volume categories
python3 configure.py --set VOLUME_LOW_THRESHOLD 5
python3 configure.py --set VOLUME_MEDIUM_THRESHOLD 20
python3 configure.py --set VOLUME_HIGH_THRESHOLD 100
```

---

### 5. Customer Calibration

#### `CUSTOMER_CALIBRATION` (Default: {})
**What it does:** Customer-specific multipliers to correct systematic over/under-prediction

**Format:** `customer1:multiplier1,customer2:multiplier2`

**Impact:**
- Multiplier < 1.0: Reduces predictions (for customers we over-predict)
- Multiplier > 1.0: Increases predictions (for customers we under-predict)

**Current Calibration:**
```
scionhealth:0.206    # Reduce by 79.4%
mercy:0.362          # Reduce by 63.8%
ibji:1.575           # Increase by 57.5%
```

**Example:**
```bash
python3 configure.py --set CUSTOMER_CALIBRATION "scionhealth:0.206,mercy:0.362,ibji:1.575"
```

---

### 6. AWS Configuration

#### `DEEPAR_ENDPOINT_NAME` (Required)
**What it does:** SageMaker endpoint name for DeepAR model

**Example:**
```bash
python3 configure.py --set DEEPAR_ENDPOINT_NAME "hybrent-nov"
```

---

#### `DEEPAR_REGION` (Default: us-west-2)
**What it does:** AWS region where DeepAR endpoint is deployed

---

### 7. Testing Parameters

#### `FOCUSED_TEST_CUSTOMERS` (Default: scionhealth,mercy)
**What it does:** Comma-separated list of customers to focus analysis on

**Example:**
```bash
python3 configure.py --set FOCUSED_TEST_CUSTOMERS "scionhealth,mercy,ibji"
```

---

## Running the System

### Full Test Pipeline

Run the complete test pipeline:

```bash
python3 test/run_full_test.py
```

**What it does:**
1. Extracts data from order history
2. Generates predictions using ensemble model
3. Compares with validation data
4. Analyzes overall results
5. Calculates customer-facility accuracy
6. Performs dual prediction analysis

**Duration:** ~5-10 minutes

---

### Individual Scripts

#### 1. Extract Data Only
```bash
python3 test/extract_data.py
```
**Generates:**
- `test/data/test_data.csv`
- `test/data/val_data.csv`

---

#### 2. Generate Predictions Only
```bash
python3 test/run_predictions.py
```
**Generates:**
- `test/data/predictions.csv`
- `test/data/prediction_summary.csv`

---

#### 3. Apply Calibration
```bash
python3 test/apply_calibration.py
```
**What it does:**
- Applies customer-specific calibration
- Applies volume-based safety multipliers
- Updates predictions.csv with calibrated values

---

#### 4. Test Different Thresholds
```bash
python3 test/test_thresholds.py
```
**What it does:**
- Tests thresholds from 4 to 15
- Shows precision, recall, F1 for each
- Recommends optimal threshold

---

#### 5. Calculate Customer Precision
```bash
# For specific customer
python3 test/customer_precision.py scionhealth

# For all customers
python3 test/customer_precision.py
```
**Generates:**
- `test/data/customer_precision_analysis.csv`

---

#### 6. Extract Customer Predictions
```bash
# Specific date
python3 test/extract_customer_predictions.py 2025-10-15

# All dates
python3 test/extract_all_facilities.py
```
**Generates:**
- `test/data/predictions_2025-10-15_scionhealth_mercy.csv`
- `test/data/predictions_all_scionhealth_mercy.csv`

---

## Output Files Reference

### Core Prediction Files

#### `test/data/predictions.csv`
**Size:** ~220 MB  
**Records:** 797,752  
**Description:** Complete predictions with actuals for test period

**Key Columns:**
- `CustomerID`, `FacilityID`, `ProductID`, `ProductName`
- `timestamp`, `date`
- `target_value` - Actual quantity ordered
- `predicted_value` - Calibrated prediction
- `lightgbm_prediction` - Raw LightGBM prediction
- `deepar_prediction` - Raw DeepAR prediction

**Use for:** Detailed analysis of individual predictions

---

#### `test/data/val_data.csv`
**Size:** ~29 MB  
**Records:** 119,664  
**Description:** Validation period data (forward-looking)

**Key Columns:**
- `CustomerID`, `FacilityID`, `ProductID`
- `timestamp`, `target_value`
- All engineered features

**Use for:** Comparing predictions against actual future orders

---

### Analysis Files

#### `test/data/dual_analysis_items.csv`
**Size:** ~12 MB  
**Records:** 78,228 unique items  
**Description:** Product-level precision analysis

**Key Columns:**
- `item_id` - Unique Customer_Facility_Product identifier
- `CustomerID`, `FacilityID`, `ProductID`, `ProductName`
- `target_value` - Actual total quantity
- `predicted_value` - Predicted total quantity
- `actual_binary` - 1 if actual ≥ threshold, 0 otherwise
- `predicted_binary` - 1 if predicted ≥ threshold, 0 otherwise
- `volume_error` - Predicted - Actual
- `volume_error_pct` - (Predicted - Actual) / Actual * 100
- `abs_error` - Absolute error

**Use for:** Understanding which specific products were correctly/incorrectly predicted

**Classification:**
- **TP (True Positive):** predicted_binary=1 AND actual_binary=1 (Correct order prediction)
- **FP (False Positive):** predicted_binary=1 AND actual_binary=0 (Over-predicted)
- **FN (False Negatives):** predicted_binary=0 AND actual_binary=1 (Missed order)
- **TN (True Negative):** predicted_binary=0 AND actual_binary=0 (Correct no-order prediction)

---

#### `test/data/product_precision_scionhealth_mercy.csv`
**Size:** ~2 MB  
**Records:** 15,757 products  
**Description:** Product-level precision for ScionHealth and Mercy only

**Additional Column:**
- `classification` - TP, FP, FN, or TN label

**Use for:** Detailed precision analysis for specific customers

---

#### `test/data/customer_analysis.csv`
**Size:** ~50 KB  
**Records:** 378 customers  
**Description:** Customer-level performance summary

**Key Columns:**
- `CustomerID`
- `total_actual` - Total units actually ordered
- `total_predicted` - Total units predicted
- `volume_error_pct` - Overall volume error percentage
- `mae` - Mean Absolute Error
- `mape` - Mean Absolute Percentage Error

**Use for:** Identifying which customers need calibration adjustments

---

#### `test/data/customer_facility_metrics.csv`
**Size:** ~1.7 MB  
**Records:** 5,824 customer-facility combinations  
**Description:** Detailed metrics for each customer-facility pair

**Key Columns:**
- `CustomerID`, `FacilityID`
- `records` - Number of predictions
- `total_actual`, `total_predicted`
- `mae`, `rmse`, `mape`, `r2`
- `precision`, `recall`, `f1_score`
- `true_positives`, `false_positives`, `false_negatives`, `true_negatives`

**Use for:** Facility-level performance analysis

---

#### `test/data/validation_comparison.csv`
**Size:** ~10 MB  
**Records:** 78,228 items  
**Description:** Forward-looking validation comparison

**Key Columns:**
- `item_id`
- `predicted_value` - From test period
- `actual_value` - From validation period
- `error`, `abs_error`, `pct_error`

**Use for:** Measuring real-world prediction accuracy

---

#### `test/data/customer_precision_analysis.csv`
**Size:** ~30 KB  
**Records:** 307 customers  
**Description:** Precision/recall metrics by customer

**Key Columns:**
- `customer`
- `items` - Number of unique products
- `actual_volume`, `predicted_volume`
- `volume_error` - Percentage error
- `precision`, `recall`, `f1` - Classification metrics

**Use for:** Customer-level precision analysis

---

### Summary Reports

#### `test/data/ANALYSIS_SUMMARY.txt`
**Description:** Executive summary of test results

**Contains:**
- Overall accuracy metrics (MAE, RMSE, MAPE)
- Volume accuracy
- Top customers by volume
- Performance by volume category

---

#### `test/data/customer_facility_summary.txt`
**Description:** Customer-facility performance summary

**Contains:**
- Overall statistics
- Performance distribution (Excellent/Good/Fair/Poor)
- Top performers and worst performers
- Precision/recall analysis

---

#### `test/data/dual_analysis_summary.txt`
**Description:** Product-level and quantity-level analysis summary

**Contains:**
- Binary classification metrics
- Confusion matrix
- Quantity regression metrics
- Customer-level breakdown

---

### Extracted Prediction Files

#### `test/data/predictions_all_scionhealth_mercy.csv`
**Size:** ~50 MB  
**Records:** 220,052  
**Description:** All predictions for ScionHealth and Mercy across entire test period

**Use for:** Complete historical prediction analysis

---

#### `test/data/predictions_2025-10-15_scionhealth_mercy.csv`
**Size:** ~500 KB  
**Records:** 2,841  
**Description:** Predictions for specific date

**Use for:** Daily prediction review

---

## Understanding Metrics

### Classification Metrics

#### Precision
**Formula:** TP / (TP + FP)

**Meaning:** Of all items we predicted need reorder, what percentage actually needed it?

**Business Impact:**
- High precision (>80%): Few false alarms, efficient ordering
- Low precision (<50%): Many false alarms, over-ordering

**Example:** Precision = 82.2% means when we say "order this product," we're correct 82.2% of the time

---

#### Recall
**Formula:** TP / (TP + FN)

**Meaning:** Of all items that actually needed reorder, what percentage did we predict?

**Business Impact:**
- High recall (>90%): Few stockouts, good coverage
- Low recall (<70%): Many stockouts, missed orders

**Example:** Recall = 92.4% means we catch 92.4% of all products that actually need reordering

---

#### F1 Score
**Formula:** 2 × (Precision × Recall) / (Precision + Recall)

**Meaning:** Harmonic mean of precision and recall, balanced metric

**Business Impact:**
- High F1 (>80%): Good balance between precision and recall
- Low F1 (<50%): Poor overall classification performance

---

#### Accuracy
**Formula:** (TP + TN) / (TP + TN + FP + FN)

**Meaning:** Overall percentage of correct classifications

**Note:** Can be misleading with imbalanced data

---

### Regression Metrics

#### MAE (Mean Absolute Error)
**Formula:** Average of |Predicted - Actual|

**Meaning:** Average error in units

**Business Impact:**
- Lower is better
- MAE = 3.10 means average error of 3.1 units per item

**Interpretation:**
- <5 units: Excellent
- 5-10 units: Good
- 10-20 units: Fair
- >20 units: Needs improvement

---

#### RMSE (Root Mean Squared Error)
**Formula:** Square root of average squared errors

**Meaning:** Penalizes large errors more than MAE

**Business Impact:**
- Lower is better
- More sensitive to outliers than MAE

---

#### MAPE (Mean Absolute Percentage Error)
**Formula:** Average of |Predicted - Actual| / Actual × 100

**Meaning:** Average percentage error

**Business Impact:**
- Lower is better
- MAPE = 48% means average 48% error relative to actual

**Interpretation:**
- <30%: Excellent
- 30-50%: Good
- 50-75%: Fair
- >75%: Poor

---

#### R² (R-Squared)
**Formula:** 1 - (Sum of squared residuals / Total sum of squares)

**Meaning:** How well predictions fit actual values

**Range:** -∞ to 1.0

**Interpretation:**
- 1.0: Perfect fit
- 0.5-0.9: Good fit
- 0.0-0.5: Moderate fit
- <0.0: Model worse than predicting mean

---

### Volume Metrics

#### Volume Error
**Formula:** (Total Predicted - Total Actual) / Total Actual × 100

**Meaning:** Overall over/under-prediction percentage

**Business Impact:**
- Positive: Over-ordering (safety buffer but higher costs)
- Negative: Under-ordering (stockout risk)

**Interpretation:**
- 0-20%: Excellent balance
- 20-50%: Good (acceptable safety buffer)
- 50-100%: Fair (significant over-ordering)
- >100%: Poor (excessive over-ordering)

---

## Common Workflows

### Workflow 1: Initial System Setup

```bash
# 1. Configure data source
python3 configure.py --set SOURCE_DATA_FILE "/path/to/order_history.csv"

# 2. Configure DeepAR endpoint
python3 configure.py --set DEEPAR_ENDPOINT_NAME "your-endpoint"
python3 configure.py --set DEEPAR_REGION "us-east-1"

# 3. Run full test
python3 test/run_full_test.py

# 4. Review results
cat test/data/ANALYSIS_SUMMARY.txt
```

---

### Workflow 2: Optimize Classification Threshold

```bash
# 1. Test different thresholds
python3 test/test_thresholds.py

# 2. Review recommendations and apply optimal threshold
# (Script will prompt to apply automatically)

# 3. Re-run analysis to verify improvement
python3 test/dual_prediction_analysis.py
```

---

### Workflow 3: Adjust Customer Calibration

```bash
# 1. Identify customers with high volume error
python3 -c "
import pandas as pd
df = pd.read_csv('test/data/customer_analysis.csv')
print(df.sort_values('volume_error_pct', ascending=False).head(10))
"

# 2. Calculate calibration factors
python3 test/optimize_calibration.py

# 3. Apply calibration
python3 test/apply_calibration.py

# 4. Verify improvement
python3 test/dual_prediction_analysis.py
```

---

### Workflow 4: Adjust Volume-Based Safety Multipliers

```bash
# 1. Check current volume error
cat test/data/dual_analysis_summary.txt | grep "Volume Error"

# 2. Reduce multipliers if over-predicting
python3 configure.py --set SAFETY_MULTIPLIER_LOW 1.8
python3 configure.py --set SAFETY_MULTIPLIER_MEDIUM 1.3
python3 configure.py --set SAFETY_MULTIPLIER_HIGH 1.1

# 3. Apply calibration
python3 test/apply_calibration.py

# 4. Verify improvement
python3 test/dual_prediction_analysis.py
```

---

### Workflow 5: Analyze Specific Customer

```bash
# 1. Calculate precision for customer
python3 test/customer_precision.py scionhealth

# 2. Extract all predictions for customer
python3 test/extract_all_facilities.py

# 3. Review product-level details
python3 -c "
import pandas as pd
df = pd.read_csv('test/data/product_precision_scionhealth_mercy.csv')
scion = df[df['CustomerID'] == 'scionhealth']
print(f'Total products: {len(scion)}')
print(f'True Positives: {len(scion[scion[\"classification\"]==\"TP\"])}')
print(f'False Positives: {len(scion[scion[\"classification\"]==\"FP\"])}')
print(f'False Negatives: {len(scion[scion[\"classification\"]==\"FN\"])}')
"
```

---

### Workflow 6: Daily Prediction Review

```bash
# Extract predictions for specific date
python3 test/extract_customer_predictions.py 2025-10-15

# Review in Excel or CSV viewer
open test/data/predictions_2025-10-15_scionhealth_mercy.csv
```

---

## Troubleshooting

### Issue: High Volume Error (>150%)

**Solution:**
1. Reduce safety multipliers
2. Adjust customer calibration factors
3. Increase classification threshold

---

### Issue: Low Recall (<80%)

**Solution:**
1. Decrease classification threshold
2. Increase safety multipliers for low-volume items
3. Review customer calibration (may be too aggressive)

---

### Issue: Low Precision (<50%)

**Solution:**
1. Increase classification threshold
2. Reduce safety multipliers
3. Review false positives in product_precision file

---

### Issue: Poor Performance for Specific Customer

**Solution:**
1. Calculate customer-specific precision
2. Adjust customer calibration factor
3. Review facility-level metrics
4. Consider customer-specific threshold

---

## Best Practices

1. **Always run full test after configuration changes** to verify impact
2. **Monitor both precision and recall** - don't optimize one at expense of the other
3. **Review product-level details** for false positives and false negatives
4. **Adjust calibration gradually** - small changes (±10-20%) at a time
5. **Document configuration changes** and their impact
6. **Regular retraining** - monthly model updates recommended
7. **Customer feedback** - incorporate business user input into calibration

---

## Support

For questions or issues:
1. Review this guide
2. Check steering files in `.kiro/steering/`
3. Review configuration documentation in `CONFIGURATION.md`
4. Examine test output files for detailed metrics

---

**Last Updated:** November 2025  
**Version:** 2.0
