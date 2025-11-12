# Complete Guide - Ensemble Forecasting System

## Table of Contents

1. [Data Extraction](#data-extraction)
2. [Prediction](#prediction)
3. [Validation](#validation)
4. [Configuration](#configuration)
5. [Training](#training)
6. [Understanding Metrics](#understanding-metrics)
7. [Troubleshooting](#troubleshooting)

---

## Data Extraction

### Generate Context & Validation Files

```bash
python scripts/extract.py
```

**Purpose**: Extracts and processes order history data to create:
- `test_data.csv` - Context data with engineered features
- `val_data.csv` - Validation data with actuals only (no features)

**Process**:
1. Loads order history from SOURCE_DATA_FILE
2. Filters to extraction period (TOTAL_EXTRACTION_DAYS)
3. Splits data BEFORE feature engineering (prevents data leakage)
4. Engineers features on context data only
5. Saves validation data as actuals only
6. Outputs files to TEST_DATA_DIR

**Configuration** (.env):
- `SOURCE_DATA_FILE`: Path to order history CSV
- `TEST_DATA_DIR`: Output directory for extracted files
- `TOTAL_EXTRACTION_DAYS`: Total days to extract (default: 90)
- `VALIDATION_DAYS`: Days for validation period (default: 14)
- `TEST_DAYS`: Days for test period (default: 14)

**Output Files**:
- `test_data.csv`: Context data with features (for predictions)
- `val_data.csv`: Validation data with actuals (for accuracy testing)

**When to Run**:
- Before running tests for the first time
- When you have new order history data
- When you change extraction parameters

**Using Extracted Data**:
Once you've run `extract.py`, you can use the generated `test_data.csv` for faster predictions:
```bash
python scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth
```

This is much faster because:
- Features are already engineered
- Data is already filtered to the context period
- No need to process the full order history

---

## Prediction

### Basic Usage

```bash
# One customer
python scripts/predict.py scionhealth

# Specific date and days
python scripts/predict.py --date 2025-10-26 --customers scionhealth --days 14

# Multiple customers
python scripts/predict.py --date 2025-10-26 --customers scionhealth,mercy --days 14

# Use test_data.csv (faster, preprocessed features)
python scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# Save to file
python scripts/predict.py scionhealth predictions.csv
```

### Parameters

- **--date**: Start date (YYYY-MM-DD), default: today
- **--customers**: Comma-separated customer IDs
- **--days**: Number of days to predict (default: 14)
- **--output**: Output file path (optional)
- **--use-test-data**: Use test_data.csv if available (faster, preprocessed features)

### How It Works

**Date Calculation Example** (predict Oct 26 for 14 days):
```
Context Period:     Jul 28 - Oct 25 (90 days)
Prediction Period:  Oct 26 - Nov 8  (14 days)
No Overlap:         Context ends before prediction starts ✓
```

**Process**:
1. Load 90 days of historical data (context period)
2. Engineer features (rolling averages, lags, seasonality)
3. Load LightGBM and DeepAR models
4. Generate ensemble predictions (95% LightGBM + 5% DeepAR)
5. Apply customer calibrations
6. Classify as ORDER (≥4 units) or NO ORDER (<4 units)

### Output Format

```csv
prediction_date,prediction_generated_at,CustomerID,FacilityID,ProductID,ProductName,predicted_value,predicted_reorder,reorder_recommendation
2025-11-15,2025-11-11 14:30:00,scionhealth,287,12345,Surgical Gloves,25.5,1,ORDER
```

**Key Columns**:
- `prediction_date`: Date prediction is for
- `prediction_generated_at`: When prediction was made
- `predicted_value`: Predicted quantity (units)
- `predicted_reorder`: 1 = order, 0 = don't order
- `reorder_recommendation`: "ORDER" or "NO ORDER"

---

## Configuration

### Environment Configuration

```bash
# Show current config
python scripts/config.py --show

# Update settings
python scripts/config.py --set SOURCE_DATA_FILE /path/to/data.csv
python scripts/config.py --set BATCH_SIZE 2000
python scripts/config.py --set CLASSIFICATION_THRESHOLD 5
```

### Customer Calibrations

```bash
# List all calibrations
python scripts/config.py --calibrations

# Show customer details
python scripts/config.py --customer scionhealth

# Update calibration
python scripts/config.py --update mercy 0.85 --status verified

# Update with metrics
python scripts/config.py --update scionhealth 1.05 \
  --status verified \
  --precision 0.92 \
  --recall 0.51 \
  --mae 4.2 \
  --notes "Optimized for low stockouts"
```

### Key Parameters (.env)

**Data**:
- `SOURCE_DATA_FILE`: Path to order history CSV
- `TEST_DATA_DIR`: Test data directory

**Models**:
- `LIGHTGBM_MODEL_PATH`: LightGBM model file
- `DEEPAR_ENDPOINT_NAME`: SageMaker endpoint
- `DEEPAR_REGION`: AWS region

**Ensemble**:
- `LIGHTGBM_WEIGHT`: 0.95 (95%)
- `DEEPAR_WEIGHT`: 0.05 (5%)

**Classification**:
- `CLASSIFICATION_THRESHOLD`: 4 units (default)

**Context**:
- `CONTEXT_DAYS`: 90 days (default)

---

## Testing

### Quick Test (No Validation)

```bash
python tests/run_test.py --quick
python tests/run_test.py --quick --date 2025-10-26 --days 14
```
- Uses preprocessed test data
- Fast (~30 seconds)
- Shows basic metrics

### Full Test (With Validation)

```bash
python tests/run_test.py --full
python tests/run_test.py --full --date 2025-10-26 --customer scionhealth
```
- Compares predictions to validation data
- Calculates MAE, Precision, Recall
- Slower (~2 minutes)

### Customer-Specific Test

```bash
python tests/run_test.py --customer scionhealth
python tests/run_test.py --customer scionhealth --date 2025-10-26 --days 14
```
- Tests one customer
- Shows top 10 items
- Detailed metrics

### Test Parameters

- **--quick**: Quick test without validation
- **--full**: Full test with validation comparison
- **--customer**: Test specific customer
- **--date**: Start date for predictions (YYYY-MM-DD)
- **--days**: Number of days to predict (default: 14)

---

## Validation

### Validate Predictions Against Actuals

The validation script compares predictions against actual data and calculates accuracy metrics.

```bash
# Basic validation
python tests/validate.py predictions.csv val_data.csv

# Save results to file
python tests/validate.py predictions.csv val_data.csv --output results.txt

# Use custom threshold
python tests/validate.py predictions.csv val_data.csv --threshold 5
```

### What It Does

1. **Loads** predictions and validation data
2. **Merges** on item_id
3. **Calculates** regression metrics (MAE, RMSE, MAPE, R²)
4. **Calculates** classification metrics (Precision, Recall, F1, Accuracy)
5. **Shows** confusion matrix
6. **Analyzes** performance by volume category

### Output Metrics

**Regression Metrics**:
- MAE: Mean Absolute Error (average error in units)
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error
- R²: Coefficient of determination

**Classification Metrics**:
- Precision: % correct when predicting ORDER
- Recall: % of actual orders caught
- F1 Score: Balance of precision and recall
- Accuracy: Overall correctness

**Confusion Matrix**:
- True Positives (TP): Correctly predicted ORDER
- False Positives (FP): Incorrectly predicted ORDER (false alarms)
- False Negatives (FN): Missed orders
- True Negatives (TN): Correctly predicted NO ORDER

**Volume Analysis**:
- Performance breakdown by volume category
- Very Low (0-5), Low (5-20), Medium (20-100), High (100+)

---

## Training

### Train New Model

```bash
python scripts/train.py
```

**Process**:
1. Loads data from SOURCE_DATA_FILE
2. Engineers features
3. Trains LightGBM model
4. Saves to model/lightgbm_model.pkl
5. Saves feature config to model/feature_config.json

### Training Configuration

**Data Processing**:
- `TRAINING_DATA_PATH`: Training data path (uses SOURCE_DATA_FILE if empty)
- `TRAINING_CHUNK_SIZE`: 100000 rows per chunk
- `TRAINING_MAX_CHUNKS`: 100 chunks max
- `TRAINING_TEST_DAYS`: 14 days held out for testing
- `TRAINING_VALIDATION_SPLIT`: 0.1 (10% validation)

**LightGBM Hyperparameters**:
- `LGBM_NUM_LEAVES`: 63
- `LGBM_LEARNING_RATE`: 0.03
- `LGBM_FEATURE_FRACTION`: 0.8
- `LGBM_BAGGING_FRACTION`: 0.8
- `LGBM_NUM_BOOST_ROUND`: 2000
- `LGBM_EARLY_STOPPING_ROUNDS`: 100

### Hyperparameter Tuning

**To improve accuracy**:
```bash
python scripts/config.py --set LGBM_NUM_LEAVES 127
python scripts/config.py --set LGBM_MAX_DEPTH 10
python scripts/config.py --set LGBM_LEARNING_RATE 0.01
python scripts/config.py --set LGBM_NUM_BOOST_ROUND 5000
```

**To prevent overfitting**:
```bash
python scripts/config.py --set LGBM_REG_ALPHA 0.5
python scripts/config.py --set LGBM_REG_LAMBDA 0.5
python scripts/config.py --set LGBM_MIN_CHILD_SAMPLES 100
```

---

## Understanding Metrics

### Precision

**Formula**: `Precision = TP / (TP + FP)`

**Meaning**: "When we predict ORDER, how often are we correct?"

**Example** (92.5% precision):
```
Items we predicted to ORDER: 4,881
  ├─ True Positives (TP):   4,513 ✅ (actually needed ordering)
  └─ False Positives (FP):    368 ❌ (didn't need ordering)

Precision = 4,513 / 4,881 = 92.5%
```

**Interpretation**: When model says "ORDER", it's correct 92.5% of the time. Only 7.5% are false alarms.

### Recall

**Formula**: `Recall = TP / (TP + FN)`

**Meaning**: "Of all items that actually needed ordering, how many did we catch?"

**Example** (50.9% recall):
```
Items that ACTUALLY needed ordering: 8,858
  ├─ True Positives (TP):   4,513 ✅ (we predicted correctly)
  └─ False Negatives (FN):  4,345 ❌ (we missed these)

Recall = 4,513 / 8,858 = 50.9%
```

**Interpretation**: Model catches 51% of items that need ordering. It misses 49%.

### F1 Score

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Meaning**: Harmonic mean of precision and recall (balance metric)

**Example**:
```
F1 = 2 × (0.925 × 0.509) / (0.925 + 0.509) = 0.657
```

### MAE (Mean Absolute Error)

**Formula**: `MAE = average(|predicted - actual|)`

**Meaning**: Average error in units

**Example**:
```
Item 1: Predicted 50, Actual 45 → Error = 5
Item 2: Predicted 30, Actual 35 → Error = 5
Item 3: Predicted 20, Actual 18 → Error = 2

MAE = (5 + 5 + 2) / 3 = 4 units
```

### Confusion Matrix

```
                        ACTUAL
                    Needs Order    Doesn't Need
                    (Actual = 1)   (Actual = 0)
                    ─────────────────────────────
PREDICTION         │
Says "ORDER"       │  TRUE          FALSE
(Predicted = 1)    │  POSITIVE      POSITIVE
                   │  (TP) ✅       (FP) ❌
                   │
Says "DON'T ORDER" │  FALSE         TRUE
(Predicted = 0)    │  NEGATIVE      NEGATIVE
                   │  (FN) ❌       (TN) ✅
```

### Target Metrics

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| Precision | ≥60% | ≥80% | ≥90% |
| Recall | ≥50% | ≥70% | ≥90% |
| F1 Score | ≥0.60 | ≥0.75 | ≥0.85 |
| MAE | <10 | <6 | <4 |

---

## Troubleshooting

### Poor Accuracy

**Check context length**:
```bash
python scripts/config.py --show | grep CONTEXT_DAYS
# Should be 90 for best accuracy
```

**Review customer calibration**:
```bash
python scripts/config.py --customer scionhealth
python tests/run_test.py --customer scionhealth
```

**Adjust calibration**:
```bash
python scripts/config.py --update scionhealth 1.05
```

### Over-Predicting

**Reduce calibration**:
```bash
python scripts/config.py --update mercy 0.85
```

**Increase threshold**:
```bash
python scripts/config.py --set CLASSIFICATION_THRESHOLD 5
```

### Under-Predicting

**Increase calibration**:
```bash
python scripts/config.py --update scionhealth 1.05
```

**Decrease threshold**:
```bash
python scripts/config.py --set CLASSIFICATION_THRESHOLD 3
```

### Import Errors

**Verify imports**:
```bash
python3 -c "from src.config import env_config; print('✓ Works')"
```

**Check Python path**:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Model Not Found

**Check model exists**:
```bash
ls -lh model/lightgbm_model.pkl
```

**Retrain if needed**:
```bash
python scripts/train.py
```

### Data File Not Found

**Check configuration**:
```bash
python scripts/config.py --show | grep SOURCE_DATA_FILE
```

**Update path**:
```bash
python scripts/config.py --set SOURCE_DATA_FILE /path/to/data.csv
```

---

## Common Workflows

### Daily Predictions

```bash
# Generate predictions for today + next 14 days
python scripts/predict.py predictions_$(date +%Y%m%d).csv
```

### Weekly Planning

```bash
# Predict next 7 days for key customers
python scripts/predict.py --days 7 --customers scionhealth,mercy weekly_forecast.csv
```

### Monthly Planning

```bash
# Predict next 30 days
python scripts/predict.py --days 30 --customers scionhealth monthly_forecast.csv
```

### Calibration Tuning

```bash
# 1. Generate predictions
python scripts/predict.py scionhealth test_predictions.csv

# 2. Run test
python tests/run_test.py --customer scionhealth

# 3. Update calibration based on results
python scripts/config.py --update scionhealth 1.05 --status testing

# 4. Test again
python tests/run_test.py --customer scionhealth

# 5. Mark as verified
python scripts/config.py --update scionhealth 1.05 --status verified
```

---

## Best Practices

1. **Always specify date** for production predictions
2. **Use 90 days context** for best accuracy
3. **Test calibrations** before marking as verified
4. **Save predictions** to file for audit trail
5. **Monitor accuracy** regularly with tests
6. **Update calibrations** based on actual performance

---

## Summary

**Extract**: `scripts/extract.py` - Generate context & validation files
**Predict**: `scripts/predict.py` - Generate predictions
**Validate**: `tests/validate.py` - Validate predictions vs actuals
**Configure**: `scripts/config.py` - Manage configuration
**Train**: `scripts/train.py` - Train models

All scripts support `--help` for detailed usage information.
