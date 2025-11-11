# Ensemble Forecasting System - Usage Guide

## Quick Start

### 1. Generate Predictions for a Customer
```bash
python predict_for_customer.py scionhealth
python predict_for_customer.py mercy
```

### 2. Manage Customer Calibrations
```bash
# List all calibrations
python manage_calibrations.py list

# Show customer details
python manage_calibrations.py show scionhealth

# Update calibration
python manage_calibrations.py update mercy 0.18 --status testing
```

### 3. Run Full Test Pipeline
```bash
python test/run_full_test.py
```

## Core Files

### Configuration
- **`.env`** - Environment variables and default settings
- **`config/customer_calibrations.json`** - Customer-specific calibrations with metrics
- **`config_defaults.py`** - Default configuration values
- **`env_config.py`** - Configuration loader

### Calibration Management
- **`calibration_manager.py`** - Python module for accessing calibrations
- **`manage_calibrations.py`** - CLI tool for managing calibrations

### Prediction
- **`predict_for_customer.py`** - Generate predictions for a customer
- **`prediction_generator.py`** - Core prediction engine
- **`ensemble_predictor.py`** - Ensemble model (LightGBM + DeepAR)

### Data Processing
- **`data_loader.py`** - Load and prepare data
- **`model_loader.py`** - Load trained models

### Evaluation
- **`evaluator.py`** - Calculate performance metrics
- **`visualizer.py`** - Generate visualizations

### Training
- **`train_model.py`** - Train LightGBM model

## Test Files

### Essential Tests
- **`test/run_full_test.py`** - Complete test pipeline
- **`test/run_predictions.py`** - Generate predictions
- **`test/customer_precision.py`** - Calculate customer-specific metrics

### Analysis
- **`test/analyze_results.py`** - Analyze prediction results
- **`test/customer_facility_analysis.py`** - Customer/facility breakdown
- **`test/quick_stats.py`** - Quick statistics

### Data Extraction
- **`test/extract_data.py`** - Extract test/validation data
- **`test/generate_forward_predictions.py`** - Forward-looking predictions
- **`test/verify_data.py`** - Verify data integrity

## Customer Calibration System

### Current Status

| Customer | Calibration | Status | Precision | Recall | Production Ready |
|----------|-------------|--------|-----------|--------|------------------|
| **ScionHealth** | 0.206 | ✅ Verified | 92.5% | 50.9% | **YES** |
| **Mercy** | 0.18 | ⚠️ Testing | 45.9% | 28.5% | **NO** (Pilot only) |
| Others | Various | ⚠️ Not Verified | N/A | N/A | **NO** |

### Calibration Workflow

1. **Test Customer**
   ```bash
   python predict_for_customer.py <customer_id>
   ```

2. **Analyze Results**
   ```bash
   python test/customer_precision.py <customer_id>
   ```

3. **Update Calibration**
   ```bash
   python manage_calibrations.py update <customer_id> <value> \
     --precision <p> --recall <r> --mae <m> \
     --status <status> --notes "<notes>"
   ```

4. **Verify Production Readiness**
   ```python
   from calibration_manager import is_production_ready
   ready, message = is_production_ready('scionhealth')
   ```

## Configuration Hierarchy

1. **Customer-Specific** (Highest Priority)
   - `config/customer_calibrations.json`
   - Per-customer calibration multipliers
   - Performance metrics and status

2. **Environment Variables**
   - `.env` file
   - Global settings and defaults

3. **Code Defaults**
   - `config_defaults.py`
   - Fallback values

## Common Tasks

### Add New Customer
```bash
# 1. Generate predictions
python predict_for_customer.py newcustomer

# 2. Analyze results
python test/customer_precision.py newcustomer

# 3. Add calibration
python manage_calibrations.py update newcustomer 1.0 \
  --status testing \
  --notes "Initial calibration"
```

### Optimize Calibration
```bash
# Test different values
python manage_calibrations.py update mercy 0.18 --status testing
python predict_for_customer.py mercy
python test/customer_precision.py mercy

# Compare results and adjust
python manage_calibrations.py update mercy 0.15 --status testing
python predict_for_customer.py mercy
python test/customer_precision.py mercy
```

### Production Deployment
```python
from calibration_manager import get_manager

manager = get_manager()

# Check readiness
ready, message = manager.is_production_ready('scionhealth')
if ready:
    calibration = manager.get_calibration('scionhealth')
    # Deploy with this calibration
else:
    print(f"Not ready: {message}")
```

## Performance Targets

### Production Ready Criteria
- **Precision:** ≥ 60% (preferably 80%+)
- **Recall:** ≥ 50%
- **F1 Score:** ≥ 0.60
- **MAE:** < 30 units
- **Status:** "verified"

### Current Performance
- **ScionHealth:** ✅ Exceeds all targets (92.5% precision)
- **Mercy:** ⚠️ Below targets (45.9% precision) - needs work

## File Structure

```
ensemble_forecasting_system/
├── config/
│   └── customer_calibrations.json    # Customer calibrations
├── model/
│   ├── lightgbm_model.pkl           # Trained model
│   └── feature_config.json          # Feature configuration
├── test/
│   ├── data/                        # Test data and results
│   ├── run_full_test.py            # Main test script
│   ├── customer_precision.py       # Customer metrics
│   └── ...                         # Other test utilities
├── .env                            # Environment configuration
├── calibration_manager.py          # Calibration module
├── manage_calibrations.py          # Calibration CLI
├── predict_for_customer.py         # Prediction CLI
├── prediction_generator.py         # Prediction engine
└── ensemble_predictor.py           # Ensemble model
```

## Best Practices

1. **Always test before deployment**
   - Generate predictions
   - Verify against validation data
   - Check production readiness

2. **Document calibration changes**
   - Use `--notes` parameter
   - Include reasoning and test results
   - Update recommendations

3. **Monitor performance**
   - Track precision, recall, MAE
   - Update metrics after each test
   - Adjust calibrations as needed

4. **Use version control**
   - Commit calibration changes
   - Document in commit messages
   - Track performance over time

## Troubleshooting

### Low Precision
- Reduce calibration multiplier
- Increase classification threshold
- Improve feature engineering

### Low Recall
- Increase calibration multiplier
- Decrease classification threshold
- Add safety stock

### High Volume Error
- Adjust calibration multiplier
- Check for data quality issues
- Review feature engineering

## Support

For questions or issues:
1. Check this guide
2. Review `config/customer_calibrations.json` for customer-specific notes
3. Run `python manage_calibrations.py show <customer_id>` for details
