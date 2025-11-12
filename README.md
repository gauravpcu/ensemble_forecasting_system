# Ensemble Forecasting System

Healthcare supply chain inventory prediction using LightGBM (95%) + DeepAR (5%) ensemble.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Extraction](#data-extraction)
3. [Prediction](#prediction)
4. [Validation](#validation)
5. [Configuration](#configuration)
6. [Training](#training)
7. [Understanding Metrics](#understanding-metrics)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths

# 1. Extract data (generates test_data.csv and val_data.csv)
python3 scripts/extract.py

# 2. Generate predictions (saves to tests/data/predictions.csv by default)
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# 3. Validate predictions
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv
```

## Common Commands

```bash
# Data Extraction
python3 scripts/extract.py                                       # Generate context & validation files

# Predictions (saves to tests/data/predictions.csv by default)
python3 scripts/predict.py scionhealth                           # One customer
python3 scripts/predict.py --date 2025-11-15 --customers scionhealth,mercy --days 14
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth
python3 scripts/predict.py scionhealth --output custom.csv       # Custom output file

# Validation
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv --output results.txt

# Configuration
python3 scripts/config.py --show                                 # Show config
python3 scripts/config.py --calibrations                         # List calibrations
python3 scripts/config.py --update mercy 0.85 --status verified  # Update calibration

# Training
python3 scripts/train.py                                         # Train new model
```

## Project Structure

```
├── src/                    # Source code
│   ├── core/              # prediction_generator.py
│   ├── models/            # model_loader.py, ensemble_predictor.py
│   ├── data/              # data_loader.py, evaluator.py
│   ├── config/            # env_config.py, config_defaults.py
│   ├── calibration/       # calibration_manager.py
│   └── utils/             # visualizer.py
├── scripts/               # Executable scripts
│   ├── extract.py         # Generate context & validation files
│   ├── predict.py         # Generate predictions
│   ├── train.py           # Train models
│   └── config.py          # Configuration management
├── tests/                 # Testing
│   ├── validate.py        # Validate predictions vs actuals
│   └── data/              # Test data (test_data.csv, val_data.csv)
├── config/                # customer_calibrations.json
└── model/                 # Trained models
```

## Key Features

- **90-Day Context**: Uses 90 days of history for predictions
- **Ensemble Model**: LightGBM (95%) + DeepAR (5%)
- **Customer Calibration**: Adjust predictions per customer
- **Multi-Day Forecasting**: Predict 1-30 days ahead
- **No Data Leakage**: Context ends before prediction starts

---

## Data Extraction

### Generate Context & Validation Files

```bash
python3 scripts/extract.py
```

**Purpose**: Extracts and processes order history data to create:
- `test_data.csv` - Context data with engineered features (90 days)
- `val_data.csv` - Validation data with actuals only (14 days)

**Process**:
1. Loads order history from SOURCE_DATA_FILE
2. Splits data BEFORE feature engineering (prevents data leakage)
3. Engineers features on context data only
4. Saves validation data as actuals only
5. Outputs files to TEST_DATA_DIR

**Configuration** (.env):
- `SOURCE_DATA_FILE`: Path to order history CSV
- `TEST_DATA_DIR`: Output directory (default: ./tests/data)
- `TOTAL_EXTRACTION_DAYS`: Context days (default: 90)
- `VALIDATION_DAYS`: Validation days (default: 14)

**Output Files**:
- `tests/data/test_data.csv`: Context data with features
- `tests/data/val_data.csv`: Validation data with actuals

**When to Run**:
- Before generating predictions for the first time
- When you have new order history data
- When you change extraction parameters

---

## Prediction

### Basic Usage

```bash
# One customer
python3 scripts/predict.py scionhealth

# Specific date and days
python3 scripts/predict.py --date 2025-10-26 --customers scionhealth --days 14

# Multiple customers
python3 scripts/predict.py --date 2025-10-26 --customers scionhealth,mercy --days 14

# Use test_data.csv (faster, preprocessed features)
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# Custom output file (default is tests/data/predictions.csv)
python3 scripts/predict.py scionhealth --output custom_predictions.csv
```

### Parameters

- `--date`: Start date (YYYY-MM-DD), default: today
- `--customers`: Comma-separated customer IDs
- `--days`: Number of days to predict (default: 14)
- `--output`: Output file path (optional)
- `--use-test-data`: Use test_data.csv if available (faster)

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

## Validation

### Validate Predictions Against Actuals

```bash
# Basic validation (creates validation_comparison.csv automatically)
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv

# Save summary to text file
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv --output results.txt

# Use custom threshold
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv --threshold 5

# Custom output directory for comparison CSV
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv --output-dir ./results
```

### Typical Workflow

```bash
# 1. Extract data
python3 scripts/extract.py

# 2. Generate predictions (saves to tests/data/predictions.csv by default)
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# 3. Validate predictions
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv --output results.txt
```

### What It Does

1. Loads predictions and validation data
2. Merges on item_id
3. Calculates regression metrics (MAE, RMSE, MAPE, R²)
4. Calculates classification metrics (Precision, Recall, F1, Accuracy)
5. Shows confusion matrix
6. Analyzes performance by volume category
7. **Creates validation_comparison.csv** with detailed item-by-item comparison

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
- Very Low (0-5), Low (5-20), Medium (20-100), High (100+)

### Output Files

**validation_comparison.csv** (always created):
- Item-by-item comparison of predictions vs actuals
- **Item-level columns**: CustomerID, FacilityID, ProductID, ProductName
- **Prediction vs Actual**: predicted_value, target_value, error, abs_error, pct_error
- **Classification**: predicted_reorder_binary, actual_reorder, classification_result (TP/FP/FN/TN)
- **Volume category**: Very Low (0-5), Low (5-20), Medium (20-100), High (100+)
- **Customer-level metrics** (repeated for each row):
  - customer_mae, customer_rmse, customer_mape
  - customer_precision, customer_recall, customer_f1, customer_accuracy
- Sorted by customer, then by absolute error (largest errors first)

**Summary text file** (optional with --output):
- Regression metrics
- Classification metrics
- Confusion matrix
- Volume analysis

### Example Output

```
Regression Metrics:
  MAE:              10.39 units
  RMSE:             46.33 units
  MAPE:             65.8%
  R²:               0.416

Classification Metrics (threshold=4):
  Precision:        95.6%
  Recall:           92.8%
  F1 Score:         94.2%
  Accuracy:         91.3%

Confusion Matrix:
  True Positives (TP):   403,414 - Correctly predicted ORDER
  False Positives (FP):  18,410 - Incorrectly predicted ORDER
  False Negatives (FN):  31,436 - Missed orders
  True Negatives (TN):   119,410 - Correctly predicted NO ORDER

Performance by Volume Category:
  Very Low (0-5):    72.2% precision, 69.0% recall
  Low (5-20):        100.0% precision, 95.8% recall
  Medium (20-100):   100.0% precision, 99.8% recall
  High (100+):       100.0% precision, 99.1% recall
```

---

## Configuration

### Environment Configuration

```bash
# Show current config
python3 scripts/config.py --show

# Update settings
python3 scripts/config.py --set SOURCE_DATA_FILE /path/to/data.csv
python3 scripts/config.py --set CLASSIFICATION_THRESHOLD 5
```

### Customer Calibrations

```bash
# List all calibrations
python3 scripts/config.py --calibrations

# Show customer details
python3 scripts/config.py --customer scionhealth

# Update calibration
python3 scripts/config.py --update mercy 0.85 --status verified

# Update with metrics
python3 scripts/config.py --update scionhealth 1.05 \
  --status verified \
  --precision 0.92 \
  --recall 0.51 \
  --mae 4.2 \
  --notes "Optimized for low stockouts"
```

### Key Parameters (.env)

**Data**:
- `SOURCE_DATA_FILE`: Path to order history CSV
- `TEST_DATA_DIR`: Test data directory (default: ./tests/data)

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

## Training

### Train New Model

```bash
python3 scripts/train.py
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
python3 scripts/config.py --set LGBM_NUM_LEAVES 127
python3 scripts/config.py --set LGBM_MAX_DEPTH 10
python3 scripts/config.py --set LGBM_LEARNING_RATE 0.01
python3 scripts/config.py --set LGBM_NUM_BOOST_ROUND 5000
```

**To prevent overfitting**:
```bash
python3 scripts/config.py --set LGBM_REG_ALPHA 0.5
python3 scripts/config.py --set LGBM_REG_LAMBDA 0.5
python3 scripts/config.py --set LGBM_MIN_CHILD_SAMPLES 100
```

---

## Understanding Metrics

### Precision

**Formula**: `Precision = TP / (TP + FP)`

**Meaning**: "When we predict ORDER, how often are we correct?"

**Example** (95.6% precision):
```
Items we predicted to ORDER: 4,881
  ├─ True Positives (TP):   4,513 ✅ (actually needed ordering)
  └─ False Positives (FP):    368 ❌ (didn't need ordering)

Precision = 4,513 / 4,881 = 92.5%
```

### Recall

**Formula**: `Recall = TP / (TP + FN)`

**Meaning**: "Of all items that actually needed ordering, how many did we catch?"

**Example** (92.8% recall):
```
Items that ACTUALLY needed ordering: 8,858
  ├─ True Positives (TP):   4,513 ✅ (we predicted correctly)
  └─ False Negatives (FN):  4,345 ❌ (we missed these)

Recall = 4,513 / 8,858 = 50.9%
```

### F1 Score

**Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`

**Meaning**: Harmonic mean of precision and recall (balance metric)

### MAE (Mean Absolute Error)

**Formula**: `MAE = average(|predicted - actual|)`

**Meaning**: Average error in units

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
python3 scripts/config.py --show | grep CONTEXT_DAYS
# Should be 90 for best accuracy
```

**Review customer calibration**:
```bash
python3 scripts/config.py --customer scionhealth
```

**Adjust calibration**:
```bash
python3 scripts/config.py --update scionhealth 1.05
```

### Over-Predicting

**Reduce calibration**:
```bash
python3 scripts/config.py --update mercy 0.85
```

**Increase threshold**:
```bash
python3 scripts/config.py --set CLASSIFICATION_THRESHOLD 5
```

### Under-Predicting

**Increase calibration**:
```bash
python3 scripts/config.py --update scionhealth 1.05
```

**Decrease threshold**:
```bash
python3 scripts/config.py --set CLASSIFICATION_THRESHOLD 3
```

### Import Errors

**Verify imports**:
```bash
python3 -c "from src.config import env_config; print('✓ Works')"
```

### Model Not Found

**Check model exists**:
```bash
ls -lh model/lightgbm_model.pkl
```

**Retrain if needed**:
```bash
python3 scripts/train.py
```

### Data File Not Found

**Check configuration**:
```bash
python3 scripts/config.py --show | grep SOURCE_DATA_FILE
```

**Update path**:
```bash
python3 scripts/config.py --set SOURCE_DATA_FILE /path/to/data.csv
```

---

## Common Workflows

### Daily Predictions

```bash
# Default output (tests/data/predictions.csv)
python3 scripts/predict.py

# Custom output with date
python3 scripts/predict.py --output predictions_$(date +%Y%m%d).csv
```

### Weekly Planning

```bash
python3 scripts/predict.py --days 7 --customers scionhealth,mercy --output weekly_forecast.csv
```

### Monthly Planning

```bash
python3 scripts/predict.py --days 30 --customers scionhealth --output monthly_forecast.csv
```

### Calibration Tuning

```bash
# 1. Generate predictions (saves to tests/data/predictions.csv)
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# 2. Validate predictions
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv

# 3. Update calibration based on results
python3 scripts/config.py --update scionhealth 1.05 --status testing

# 4. Test again
python3 scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth
python3 tests/validate.py tests/data/predictions.csv tests/data/val_data.csv

# 5. Mark as verified
python3 scripts/config.py --update scionhealth 1.05 --status verified
```

---

## Performance

- **MAE**: 10-12 units (with 90-day context)
- **Precision**: >95% (very low false alarm rate)
- **Recall**: >92% (catches most orders)
- **Best for**: All volume categories

## Customer Calibrations

- **ScionHealth**: 1.05x (reduce stockouts)
- **Mercy**: 0.85x (reduce over-ordering)
- **IBJI**: 1.575x (high demand)

---

## Help

```bash
python3 scripts/predict.py --help
python3 scripts/config.py --help
python3 tests/validate.py --help
```

## License

Proprietary - Internal use only
