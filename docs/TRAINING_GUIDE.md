# Model Training Guide

Guide for training the LightGBM model for the Ensemble Forecasting System.

---

## Overview

The `train_model.py` script trains a new LightGBM model using historical order data. All training parameters are configurable via the `.env` file.

---

## Quick Start

### 1. Configure Training Data

Edit `.env`:
```bash
# Set your training data path
SOURCE_DATA_FILE=/path/to/order_history.csv

# Or use separate training data
TRAINING_DATA_PATH=/path/to/training_data.csv
```

### 2. Run Training

```bash
python3 train_model.py
```

### 3. Output

The script generates:
- `model/lightgbm_model.pkl` - Trained model
- `model/feature_config.json` - Feature configuration

---

## Configuration Parameters

### Data Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRAINING_DATA_PATH` | (empty) | Training data path (uses SOURCE_DATA_FILE if empty) |
| `TRAINING_CHUNK_SIZE` | 100000 | Rows per chunk for large files |
| `TRAINING_MAX_CHUNKS` | 100 | Maximum chunks to process |
| `TRAINING_TEST_DAYS` | 14 | Days to hold out for testing |
| `TRAINING_VALIDATION_SPLIT` | 0.1 | Validation split (10%) |

### Model Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_OUTPUT_PATH` | ./model/lightgbm_model.pkl | Where to save trained model |
| `CONFIG_OUTPUT_PATH` | ./model/feature_config.json | Where to save feature config |

### LightGBM Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LGBM_NUM_LEAVES` | 63 | Maximum tree leaves |
| `LGBM_LEARNING_RATE` | 0.03 | Learning rate |
| `LGBM_FEATURE_FRACTION` | 0.8 | Feature sampling ratio |
| `LGBM_BAGGING_FRACTION` | 0.8 | Data sampling ratio |
| `LGBM_BAGGING_FREQ` | 5 | Bagging frequency |
| `LGBM_MIN_CHILD_SAMPLES` | 50 | Minimum samples per leaf |
| `LGBM_REG_ALPHA` | 0.1 | L1 regularization |
| `LGBM_REG_LAMBDA` | 0.1 | L2 regularization |
| `LGBM_MAX_DEPTH` | 8 | Maximum tree depth |
| `LGBM_NUM_BOOST_ROUND` | 2000 | Maximum boosting rounds |
| `LGBM_EARLY_STOPPING_ROUNDS` | 100 | Early stopping patience |

---

## Training Process

### Step 1: Data Loading
- Loads data in chunks for memory efficiency
- Processes up to `TRAINING_MAX_CHUNKS` chunks
- Converts dates and filters invalid records

### Step 2: Feature Engineering
- Aggregates to daily level per customer/facility/product
- Creates rolling statistics (7d, 30d, 90d, 365d)
- Generates lag features (7, 14, 30, 365 days)
- Adds calendar features (day, month, quarter, etc.)
- Computes business features (seasonality, frequency, etc.)

### Step 3: Train/Test Split
- Holds out last `TRAINING_TEST_DAYS` for testing
- Splits training data into train/validation
- Validation split controlled by `TRAINING_VALIDATION_SPLIT`

### Step 4: Model Training
- Trains LightGBM with configured hyperparameters
- Uses early stopping to prevent overfitting
- Monitors validation MAE

### Step 5: Evaluation
- Tests on held-out test set
- Reports MAE, RMSE, R²
- Shows feature importance

### Step 6: Save Model
- Saves trained model to `MODEL_OUTPUT_PATH`
- Saves feature config to `CONFIG_OUTPUT_PATH`

---

## Hyperparameter Tuning

### To Improve Accuracy

**Increase model complexity:**
```bash
python3 configure.py --set LGBM_NUM_LEAVES 127
python3 configure.py --set LGBM_MAX_DEPTH 10
python3 configure.py --set LGBM_NUM_BOOST_ROUND 3000
```

**Decrease learning rate (train longer):**
```bash
python3 configure.py --set LGBM_LEARNING_RATE 0.01
python3 configure.py --set LGBM_NUM_BOOST_ROUND 5000
```

### To Prevent Overfitting

**Increase regularization:**
```bash
python3 configure.py --set LGBM_REG_ALPHA 0.5
python3 configure.py --set LGBM_REG_LAMBDA 0.5
python3 configure.py --set LGBM_MIN_CHILD_SAMPLES 100
```

**Reduce model complexity:**
```bash
python3 configure.py --set LGBM_NUM_LEAVES 31
python3 configure.py --set LGBM_MAX_DEPTH 6
```

### To Speed Up Training

**Reduce data:**
```bash
python3 configure.py --set TRAINING_MAX_CHUNKS 50
```

**Increase learning rate:**
```bash
python3 configure.py --set LGBM_LEARNING_RATE 0.05
python3 configure.py --set LGBM_NUM_BOOST_ROUND 1000
```

---

## Features Created

The training script creates 25 features:

### Rolling Statistics (6 features)
- `rolling_mean_7d`, `rolling_std_7d`
- `rolling_mean_30d`, `rolling_std_30d`
- `rolling_mean_90d`, `rolling_mean_365d`

### Lag Features (4 features)
- `lag_7`, `lag_14`, `lag_30`, `lag_365`

### Calendar Features (7 features)
- `day_of_week`, `day_of_month`, `month`, `quarter`, `year`
- `is_month_end`, `is_quarter_end`

### Business Features (5 features)
- `seasonal_trend`, `seasonal_seasonal`
- `price_volatility`, `order_frequency`, `vendor_reliability`

### Entity Encodings (3 features)
- `CustomerID_encoded`, `FacilityID_encoded`, `ProductID_encoded`

---

## Monitoring Training

### During Training

Watch for:
- **Validation MAE decreasing** - Model is learning
- **Early stopping triggered** - Model converged
- **Train/valid gap** - If large, model is overfitting

### After Training

Check test metrics:
- **MAE < 10** - Excellent
- **MAE 10-20** - Good
- **MAE 20-50** - Fair
- **MAE > 50** - Needs improvement

Check R²:
- **R² > 0.8** - Excellent fit
- **R² 0.5-0.8** - Good fit
- **R² 0.2-0.5** - Moderate fit
- **R² < 0.2** - Poor fit

---

## Troubleshooting

### Issue: Out of Memory

**Solution:**
```bash
python3 configure.py --set TRAINING_CHUNK_SIZE 50000
python3 configure.py --set TRAINING_MAX_CHUNKS 50
```

### Issue: Training Too Slow

**Solution:**
```bash
python3 configure.py --set LGBM_LEARNING_RATE 0.05
python3 configure.py --set LGBM_NUM_BOOST_ROUND 1000
```

### Issue: Overfitting (train MAE << valid MAE)

**Solution:**
```bash
python3 configure.py --set LGBM_REG_ALPHA 0.5
python3 configure.py --set LGBM_REG_LAMBDA 0.5
python3 configure.py --set LGBM_MIN_CHILD_SAMPLES 100
python3 configure.py --set LGBM_FEATURE_FRACTION 0.6
```

### Issue: Underfitting (both train and valid MAE high)

**Solution:**
```bash
python3 configure.py --set LGBM_NUM_LEAVES 127
python3 configure.py --set LGBM_MAX_DEPTH 10
python3 configure.py --set LGBM_LEARNING_RATE 0.01
python3 configure.py --set LGBM_NUM_BOOST_ROUND 5000
```

---

## Best Practices

1. **Use Recent Data** - Last 6-12 months for best results
2. **Monitor Validation** - Watch for overfitting
3. **Test Thoroughly** - Run full test pipeline after training
4. **Version Models** - Keep track of model versions and performance
5. **Document Changes** - Note hyperparameter changes and results

---

## Example Workflow

```bash
# 1. Set training data
python3 configure.py --set SOURCE_DATA_FILE "/path/to/order_history.csv"

# 2. Train model
python3 train_model.py

# 3. Test new model
python3 test/run_full_test.py

# 4. Compare with previous model
# Review test/data/ANALYSIS_SUMMARY.txt

# 5. If better, keep new model
# If worse, revert or tune hyperparameters
```

---

## Advanced: Custom Training

To customize training beyond configuration:

1. Copy `train_model.py` to `train_model_custom.py`
2. Modify feature engineering or model parameters
3. Run custom training script
4. Compare results with baseline

---

**Last Updated:** November 2025  
**Version:** 1.0
