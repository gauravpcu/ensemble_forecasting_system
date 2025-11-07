# File Reference Guide

Complete reference for all files in the Ensemble Forecasting System.

---

## Table of Contents
1. [Core System Files](#core-system-files)
2. [Configuration Files](#configuration-files)
3. [Documentation Files](#documentation-files)
4. [Test Scripts](#test-scripts)
5. [Data Files](#data-files)
6. [Model Files](#model-files)
7. [IDE Configuration](#ide-configuration)

---

## Core System Files

### `ensemble_predictor.py`
**Purpose:** Main ensemble prediction engine

**What it does:**
- Combines LightGBM (95%) and DeepAR (5%) predictions
- Implements weighted ensemble logic
- Manages model loading and prediction generation
- Provides unified prediction interface

**Key Functions:**
- `EnsemblePredictor.__init__()` - Initialize with models and weights
- `EnsemblePredictor.predict()` - Generate ensemble predictions
- `create_ensemble_predictor()` - Factory function for creating predictor

**Used by:** All prediction scripts

**When to modify:** When changing ensemble logic or adding new models

---

### `data_loader.py`
**Purpose:** Data loading and feature engineering

**What it does:**
- Loads order history data from CSV
- Engineers features (rolling stats, lags, seasonal)
- Handles data preprocessing and cleaning
- Prepares data for model input

**Key Functions:**
- `DataLoader.load_data()` - Load and preprocess data
- `DataLoader.engineer_features()` - Create predictive features
- `DataLoader.split_data()` - Split into train/test/validation

**Used by:** All scripts that need to process order history

**When to modify:** When adding new features or changing data processing logic

---

### `model_loader.py`
**Purpose:** Model loading utilities

**What it does:**
- Loads LightGBM model from pickle file
- Connects to DeepAR SageMaker endpoint
- Validates model availability
- Provides unified model loading interface

**Key Functions:**
- `load_models()` - Load all models
- `load_lightgbm_model()` - Load LightGBM from file
- `load_deepar_model()` - Connect to DeepAR endpoint

**Used by:** Prediction scripts

**When to modify:** When adding new models or changing model storage

---

### `evaluator.py`
**Purpose:** Performance evaluation and metrics calculation

**What it does:**
- Calculates regression metrics (MAE, RMSE, MAPE, R²)
- Calculates classification metrics (Precision, Recall, F1)
- Generates confusion matrices
- Provides performance analysis

**Key Functions:**
- `Evaluator.evaluate_predictions()` - Calculate all metrics
- `Evaluator.calculate_regression_metrics()` - MAE, RMSE, etc.
- `Evaluator.calculate_classification_metrics()` - Precision, Recall, F1

**Used by:** Analysis and testing scripts

**When to modify:** When adding new metrics or evaluation methods

---

### `visualizer.py`
**Purpose:** Visualization and plotting utilities

**What it does:**
- Creates prediction vs actual plots
- Generates error distribution charts
- Visualizes performance by category
- Produces comparison charts

**Key Functions:**
- `Visualizer.plot_predictions()` - Prediction scatter plots
- `Visualizer.plot_errors()` - Error distribution
- `Visualizer.plot_by_category()` - Category-wise performance

**Used by:** Analysis scripts (when GENERATE_PLOTS=true)

**When to modify:** When adding new visualizations

---

### `predict_for_customer.py`
**Purpose:** Generate predictions for specific customer

**What it does:**
- Loads customer-specific data
- Generates predictions using ensemble
- Saves customer-specific results
- Provides customer-focused output

**Usage:**
```bash
python3 predict_for_customer.py --customer scionhealth
```

**When to use:** For customer-specific prediction runs

---

## Configuration Files

### `.env`
**Purpose:** Active environment configuration

**What it contains:**
- All system parameters (80+ settings)
- Data paths and model locations
- AWS credentials and endpoints
- Customer calibration factors
- Safety multipliers and thresholds

**How to modify:**
```bash
# Use configure.py script
python3 configure.py --set PARAMETER_NAME value

# Or edit directly (be careful with format)
nano .env
```

**Critical settings:**
- `SOURCE_DATA_FILE` - Order history CSV path
- `DEEPAR_ENDPOINT_NAME` - SageMaker endpoint
- `CLASSIFICATION_THRESHOLD` - Reorder threshold
- `CUSTOMER_CALIBRATION` - Customer-specific factors

**When to modify:** When adjusting system behavior or performance

---

### `.env.example`
**Purpose:** Template for new installations

**What it contains:**
- All configuration parameters with default values
- Comments explaining each parameter
- Example values for reference

**Usage:**
```bash
# Copy to create your own configuration
cp .env.example .env
# Then edit .env with your values
```

**When to modify:** When adding new configuration parameters

---

### `config_defaults.py`
**Purpose:** Default values for all configuration parameters

**What it contains:**
- Python constants for all defaults
- Fallback values when .env not set
- Required configuration validation lists

**Key Constants:**
- `DEFAULT_CLASSIFICATION_THRESHOLD = 5`
- `DEFAULT_LIGHTGBM_WEIGHT = 0.95`
- `DEFAULT_SAFETY_MULTIPLIER_LOW = 2.0`

**When to modify:** When changing system defaults

---

### `env_config.py`
**Purpose:** Configuration management and loading

**What it does:**
- Loads .env file
- Parses configuration values
- Provides type conversion
- Validates required settings
- Provides helper functions

**Key Functions:**
- `get_env()` - Get environment variable with default
- `get_customer_calibration()` - Get customer-specific factor
- `get_safety_multiplier()` - Get volume-based multiplier
- `validate_config()` - Validate configuration

**Used by:** All scripts that need configuration

**When to modify:** When adding new configuration logic

---

### `configure.py`
**Purpose:** Command-line configuration management

**What it does:**
- Updates .env file parameters
- Validates parameter values
- Provides easy configuration interface
- Shows current configuration

**Usage:**
```bash
# Set a parameter
python3 configure.py --set CLASSIFICATION_THRESHOLD 4

# Show current config
python3 configure.py --show

# Show specific parameter
python3 configure.py --get CLASSIFICATION_THRESHOLD
```

**When to use:** Whenever you need to change configuration

---

### `requirements.txt`
**Purpose:** Python package dependencies

**What it contains:**
- List of required Python packages
- Version specifications
- Installation instructions

**Usage:**
```bash
pip install -r requirements.txt
```

**When to modify:** When adding new Python dependencies

---

## Documentation Files

### `README.md`
**Purpose:** Project overview and quick start

**What it contains:**
- Project description
- Quick start guide
- Installation instructions
- Basic usage examples
- Links to detailed documentation

**Audience:** New users, developers

**When to modify:** When project scope or setup changes

---

### `USER_GUIDE.md`
**Purpose:** Comprehensive user documentation

**What it contains:**
- Detailed parameter explanations
- All configuration options
- Output file descriptions
- Metric definitions
- Common workflows
- Troubleshooting guide

**Sections:**
- System Overview
- Configuration Parameters (detailed)
- Running the System
- Output Files Reference
- Understanding Metrics
- Common Workflows

**Audience:** System operators, analysts

**When to use:** For detailed information on any aspect

---

### `QUICK_REFERENCE.md`
**Purpose:** Quick lookup guide

**What it contains:**
- Common commands
- Key metrics table
- File reference table
- Quick diagnostics
- Troubleshooting table

**Audience:** Daily users who need fast answers

**When to use:** For quick command or metric lookup

---

### `CONFIGURATION.md`
**Purpose:** Configuration parameter reference

**What it contains:**
- All configuration parameters
- Parameter descriptions
- Default values
- Valid ranges
- Examples

**Audience:** System administrators

**When to use:** When configuring the system

---

### `FILE_REFERENCE.md` (this file)
**Purpose:** Complete file reference

**What it contains:**
- Description of every file
- Purpose and usage
- When to modify
- Relationships between files

**Audience:** Developers, maintainers

**When to use:** To understand project structure

---

### `CLEANUP_SUMMARY.md`
**Purpose:** Cleanup history and maintenance guide

**What it contains:**
- Files that were removed
- Files that were kept
- Maintenance recommendations
- Project structure

**Audience:** Maintainers

**When to use:** For project maintenance

---

## Test Scripts

### `test/run_full_test.py`
**Purpose:** Main test pipeline orchestrator

**What it does:**
- Runs complete test workflow
- Executes all test steps in sequence
- Generates comprehensive results
- Provides progress feedback

**Steps executed:**
1. Extract data from order history
2. Generate predictions
3. Compare with validation data
4. Analyze overall results
5. Calculate customer-facility accuracy
6. Perform dual prediction analysis

**Usage:**
```bash
python3 test/run_full_test.py
```

**Duration:** ~5-10 minutes

**Generates:** All output files in test/data/

**When to use:** After configuration changes or for complete analysis

---

### `test/extract_data.py`
**Purpose:** Extract and prepare data from order history

**What it does:**
- Loads order history CSV
- Filters for specified date range
- Engineers features
- Splits into test and validation sets
- Saves prepared datasets

**Usage:**
```bash
python3 test/extract_data.py
```

**Generates:**
- `test/data/test_data.csv` - Test period data
- `test/data/val_data.csv` - Validation period data

**When to use:** When you have new order history data

---

### `test/run_predictions.py`
**Purpose:** Generate predictions using ensemble model

**What it does:**
- Loads test data
- Initializes ensemble predictor
- Generates predictions for all items
- Compares with actuals
- Saves predictions

**Usage:**
```bash
python3 test/run_predictions.py
```

**Generates:**
- `test/data/predictions.csv` - All predictions with actuals
- `test/data/prediction_summary.csv` - Per-item summary

**When to use:** After extracting data or changing model weights

---

### `test/verify_data.py`
**Purpose:** Validate extracted data quality

**What it does:**
- Checks for required columns
- Validates data types
- Checks for null values
- Verifies date ranges
- Shows sample records

**Usage:**
```bash
python3 test/verify_data.py
```

**When to use:** After data extraction to ensure quality

---

### `test/compare_with_validation.py`
**Purpose:** Compare predictions with validation period

**What it does:**
- Matches predictions with validation actuals
- Calculates forward-looking accuracy
- Analyzes by volume category
- Identifies top errors

**Usage:**
```bash
python3 test/compare_with_validation.py
```

**Generates:**
- `test/data/validation_comparison.csv`

**When to use:** To measure real-world prediction accuracy

---

### `test/analyze_results.py`
**Purpose:** Comprehensive results analysis

**What it does:**
- Calculates overall accuracy metrics
- Analyzes by volume category
- Compares model performance
- Generates customer analysis
- Creates summary reports

**Usage:**
```bash
python3 test/analyze_results.py
```

**Generates:**
- `test/data/customer_analysis.csv`
- `test/data/volume_analysis.csv`
- `test/data/daily_trends.csv`
- `test/data/ANALYSIS_SUMMARY.txt`

**When to use:** For detailed performance analysis

---

### `test/customer_facility_analysis.py`
**Purpose:** Customer-facility level metrics

**What it does:**
- Calculates metrics for each customer-facility pair
- Computes precision, recall, F1 scores
- Identifies best and worst performers
- Generates confusion matrices

**Usage:**
```bash
python3 test/customer_facility_analysis.py
```

**Generates:**
- `test/data/customer_facility_metrics.csv`
- `test/data/customer_facility_summary.txt`

**When to use:** For facility-level performance analysis

---

### `test/dual_prediction_analysis.py`
**Purpose:** Product-level and quantity-level analysis

**What it does:**
- Analyzes binary classification (will order?)
- Analyzes regression (how much?)
- Calculates both types of metrics
- Provides business interpretation

**Usage:**
```bash
python3 test/dual_prediction_analysis.py
```

**Generates:**
- `test/data/dual_analysis_items.csv` - Product-level details
- `test/data/dual_analysis_customers.csv` - Customer summary
- `test/data/dual_analysis_summary.txt`

**When to use:** For detailed product-level precision analysis

---

### `test/apply_calibration.py`
**Purpose:** Apply calibration factors to predictions

**What it does:**
- Applies customer-specific calibration
- Applies volume-based safety multipliers
- Updates predictions with calibrated values
- Shows calibration impact

**Usage:**
```bash
python3 test/apply_calibration.py
```

**Modifies:**
- `test/data/predictions.csv` - Updates with calibrated values

**When to use:** After changing calibration factors

---

### `test/optimize_calibration.py`
**Purpose:** Calculate optimal calibration factors

**What it does:**
- Analyzes prediction errors by customer
- Calculates correction factors
- Recommends calibration multipliers
- Shows expected improvement

**Usage:**
```bash
python3 test/optimize_calibration.py
```

**Generates:**
- Recommended calibration factors (console output)

**When to use:** When customers show systematic over/under-prediction

---

### `test/test_thresholds.py`
**Purpose:** Optimize classification threshold

**What it does:**
- Tests thresholds from 4 to 15
- Calculates precision, recall, F1 for each
- Recommends optimal threshold
- Optionally applies optimal threshold

**Usage:**
```bash
python3 test/test_thresholds.py
```

**Interactive:** Prompts to apply optimal threshold

**When to use:** To optimize precision/recall balance

---

### `test/customer_precision.py`
**Purpose:** Calculate customer-specific precision metrics

**What it does:**
- Calculates precision, recall, F1 by customer
- Shows confusion matrix
- Provides business interpretation
- Compares customers

**Usage:**
```bash
# Specific customer
python3 test/customer_precision.py scionhealth

# All customers
python3 test/customer_precision.py
```

**Generates:**
- `test/data/customer_precision_analysis.csv`

**When to use:** For customer-specific performance analysis

---

### `test/explain_precision_calculation.py`
**Purpose:** Explain how precision is calculated

**What it does:**
- Shows precision calculation methodology
- Provides real examples
- Demonstrates product-level classification
- Shows sample TP, FP, FN items

**Usage:**
```bash
python3 test/explain_precision_calculation.py
```

**When to use:** To understand precision metrics

---

### `test/extract_customer_predictions.py`
**Purpose:** Extract predictions for specific customers and date

**What it does:**
- Filters predictions by customer and date
- Shows top predictions
- Provides summary statistics
- Saves to separate file

**Usage:**
```bash
# Default: ScionHealth & Mercy on 10/15
python3 test/extract_customer_predictions.py

# Custom date
python3 test/extract_customer_predictions.py 2025-10-20

# Custom customers
python3 test/extract_customer_predictions.py 2025-10-20 scionhealth,ibji
```

**Generates:**
- `test/data/predictions_YYYY-MM-DD_customer1_customer2.csv`

**When to use:** For daily prediction review

---

### `test/extract_all_facilities.py`
**Purpose:** Extract all predictions for specific customers

**What it does:**
- Extracts all predictions across entire test period
- Includes all facilities
- Shows facility-level summaries
- Provides complete history

**Usage:**
```bash
# Default: ScionHealth & Mercy
python3 test/extract_all_facilities.py

# Custom customers
python3 test/extract_all_facilities.py scionhealth,ibji,gcvs
```

**Generates:**
- `test/data/predictions_all_customer1_customer2.csv`

**When to use:** For complete customer analysis

---

### `test/extract_validation_only.py`
**Purpose:** Extract only validation period data

**What it does:**
- Extracts validation period from order history
- Useful for forward-looking analysis
- Provides clean validation dataset

**Usage:**
```bash
python3 test/extract_validation_only.py
```

**Generates:**
- `test/data/val_data.csv`

**When to use:** When you only need validation data

---

### `test/quick_stats.py`
**Purpose:** Quick statistics and summary

**What it does:**
- Shows quick performance metrics
- Displays key statistics
- Provides fast overview

**Usage:**
```bash
python3 test/quick_stats.py
```

**When to use:** For quick performance check

---

## Data Files

### `test/data/test_data.csv`
**Size:** ~183 MB  
**Records:** ~798K  
**Purpose:** Test period data with features

**Contains:**
- Order history for test period (Aug 6 - Oct 21)
- All engineered features
- Target values (actual orders)
- Customer, facility, product information

**Columns:** 38 columns including:
- CustomerID, FacilityID, ProductID, ProductName
- timestamp, target_value
- Rolling statistics (7d, 30d)
- Lag features (7, 14, 30 days)
- Seasonal features
- Encoded categorical variables

**Used by:** Prediction generation scripts

---

### `test/data/val_data.csv`
**Size:** ~29 MB  
**Records:** ~120K  
**Purpose:** Validation period data (forward-looking)

**Contains:**
- Order history for validation period (Oct 22 - Nov 3)
- Same features as test data
- Used to measure real-world accuracy

**Used by:** Validation and comparison scripts

---

### `test/data/predictions.csv`
**Size:** ~221 MB  
**Records:** ~798K  
**Purpose:** Complete predictions with actuals

**Contains:**
- All test period records
- Predicted values (calibrated)
- Actual values
- LightGBM predictions
- DeepAR predictions
- All features

**Key Columns:**
- `predicted_value` - Final calibrated prediction
- `target_value` - Actual quantity ordered
- `lightgbm_prediction` - Raw LightGBM output
- `deepar_prediction` - Raw DeepAR output

**Used by:** All analysis scripts

---

### `test/data/prediction_summary.csv`
**Size:** ~34 MB  
**Records:** ~285K unique items  
**Purpose:** Per-item prediction summary

**Contains:**
- Aggregated predictions by item
- Mean, sum, count of predictions
- Error metrics per item

**Used by:** Item-level analysis

---

### `test/data/validation_comparison.csv`
**Size:** ~10 MB  
**Records:** ~78K items  
**Purpose:** Forward-looking accuracy comparison

**Contains:**
- Items common to both test and validation
- Predicted vs actual comparison
- Error metrics

**Used by:** Accuracy measurement

---

### `test/data/dual_analysis_items.csv`
**Size:** ~12 MB  
**Records:** 78,228 items  
**Purpose:** Product-level precision analysis

**Contains:**
- Each unique Customer+Facility+Product
- Predicted and actual totals
- Binary classifications
- Error metrics

**Key Columns:**
- `item_id` - Unique identifier
- `predicted_value`, `target_value`
- `actual_binary`, `predicted_binary`
- `volume_error`, `volume_error_pct`

**Classification:**
- TP: predicted_binary=1 AND actual_binary=1
- FP: predicted_binary=1 AND actual_binary=0
- FN: predicted_binary=0 AND actual_binary=1
- TN: predicted_binary=0 AND actual_binary=0

**Used by:** Precision analysis, product-level review

---

### `test/data/product_precision_scionhealth_mercy.csv`
**Size:** ~2 MB  
**Records:** 15,757 products  
**Purpose:** ScionHealth & Mercy product precision

**Contains:**
- Same as dual_analysis_items.csv
- Filtered for ScionHealth and Mercy
- Includes classification labels (TP/FP/FN/TN)

**Used by:** Customer-specific precision analysis

---

### `test/data/customer_analysis.csv`
**Size:** ~50 KB  
**Records:** 378 customers  
**Purpose:** Customer-level performance

**Contains:**
- Total actual and predicted volumes
- Volume error percentage
- MAE, MAPE metrics
- Customer-level accuracy

**Used by:** Customer performance review

---

### `test/data/customer_facility_metrics.csv`
**Size:** ~1.7 MB  
**Records:** 5,824 customer-facility pairs  
**Purpose:** Detailed facility-level metrics

**Contains:**
- Metrics for each customer-facility combination
- Precision, recall, F1 scores
- Confusion matrix values
- MAE, RMSE, MAPE, R²

**Used by:** Facility-level analysis

---

### `test/data/customer_precision_analysis.csv`
**Size:** ~30 KB  
**Records:** 307 customers  
**Purpose:** Customer precision metrics

**Contains:**
- Precision, recall, F1 by customer
- Volume metrics
- Number of items analyzed

**Used by:** Customer precision comparison

---

### `test/data/volume_analysis.csv`
**Size:** Small  
**Purpose:** Performance by volume category

**Contains:**
- Metrics for Low/Medium/High/Very High volume
- Count of items in each category
- MAE, MAPE by category

**Used by:** Volume-based analysis

---

### `test/data/daily_trends.csv`
**Size:** Small  
**Purpose:** Daily aggregated metrics

**Contains:**
- Daily totals of actual and predicted
- Daily error metrics
- Trend analysis

**Used by:** Time-series analysis

---

### `test/data/dual_analysis_customers.csv`
**Size:** Small  
**Purpose:** Customer-level dual analysis

**Contains:**
- Product and quantity metrics by customer
- Binary classification performance
- Regression performance

**Used by:** Customer comparison

---

### `test/data/predictions_all_scionhealth_mercy.csv`
**Size:** ~50 MB  
**Records:** 220,052  
**Purpose:** All predictions for ScionHealth & Mercy

**Contains:**
- Complete prediction history
- All facilities (141 ScionHealth, 730 Mercy)
- Entire test period (Aug 6 - Oct 21)

**Used by:** Complete customer analysis

---

### `test/data/predictions_YYYY-MM-DD_customer1_customer2.csv`
**Size:** Varies  
**Purpose:** Date-specific customer predictions

**Contains:**
- Predictions for specific date
- Specified customers only
- Daily prediction review

**Used by:** Daily operations

---

### Summary Report Files

#### `test/data/ANALYSIS_SUMMARY.txt`
**Purpose:** Executive summary of test results

**Contains:**
- Overall accuracy metrics
- Volume accuracy
- Top customers
- Performance by category

---

#### `test/data/customer_facility_summary.txt`
**Purpose:** Customer-facility performance summary

**Contains:**
- Overall statistics
- Performance distribution
- Top and worst performers
- Precision/recall analysis

---

#### `test/data/dual_analysis_summary.txt`
**Purpose:** Product and quantity analysis summary

**Contains:**
- Binary classification metrics
- Confusion matrix
- Regression metrics
- Customer breakdown

---

## Model Files

### `model/lightgbm_model.pkl`
**Size:** Varies  
**Purpose:** Trained LightGBM model

**Contains:**
- Serialized LightGBM model
- Feature importance
- Model parameters

**Created by:** Model training process (external)

**Used by:** ensemble_predictor.py, model_loader.py

**When to update:** After model retraining

---

### `model/feature_config.json`
**Size:** Small  
**Purpose:** Feature configuration

**Contains:**
- List of features used by model
- Feature names and types
- Feature engineering parameters

**Used by:** data_loader.py, ensemble_predictor.py

**When to update:** When features change

---

## IDE Configuration

### `.kiro/steering/ensemble_forecasting_context.md`
**Purpose:** Kiro IDE steering rules and context

**Contains:**
- System architecture decisions
- Configuration philosophy
- Testing framework
- Development guidelines
- Business context
- Deployment considerations

**Used by:** Kiro IDE for AI assistance

**When to modify:** When system architecture or guidelines change

---

## File Relationships

### Data Flow
```
order_history.csv
    ↓
extract_data.py
    ↓
test_data.csv + val_data.csv
    ↓
run_predictions.py
    ↓
predictions.csv
    ↓
apply_calibration.py
    ↓
predictions.csv (calibrated)
    ↓
analyze_results.py + dual_prediction_analysis.py
    ↓
All analysis files
```

### Configuration Flow
```
config_defaults.py (defaults)
    ↓
.env (user settings)
    ↓
env_config.py (loader)
    ↓
All scripts (use configuration)
```

### Model Flow
```
model/lightgbm_model.pkl + DeepAR endpoint
    ↓
model_loader.py
    ↓
ensemble_predictor.py
    ↓
run_predictions.py
    ↓
predictions.csv
```

---

## Quick File Finder

### Need to...

**Change configuration?**
- Edit: `.env` (using `configure.py`)
- Reference: `CONFIGURATION.md`, `USER_GUIDE.md`

**Run predictions?**
- Run: `test/run_full_test.py` (complete)
- Or: `test/run_predictions.py` (predictions only)

**Analyze results?**
- View: `test/data/ANALYSIS_SUMMARY.txt`
- Detailed: `test/data/dual_analysis_items.csv`
- Customer: `test/data/customer_analysis.csv`

**Optimize performance?**
- Threshold: `test/test_thresholds.py`
- Calibration: `test/optimize_calibration.py`
- Apply: `test/apply_calibration.py`

**Extract customer data?**
- All dates: `test/extract_all_facilities.py`
- Specific date: `test/extract_customer_predictions.py`

**Understand metrics?**
- Guide: `USER_GUIDE.md` (Understanding Metrics section)
- Quick: `QUICK_REFERENCE.md`
- Precision: `test/explain_precision_calculation.py`

**Learn the system?**
- Start: `README.md`
- Detailed: `USER_GUIDE.md`
- Quick: `QUICK_REFERENCE.md`
- Files: `FILE_REFERENCE.md` (this file)

---

## File Maintenance

### Regular Cleanup
```bash
# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Clean old extractions (optional)
rm -f test/data/predictions_2025-*.csv
```

### Backup Important Files
```bash
# Configuration
cp .env .env.backup

# Models
cp -r model/ model_backup/

# Results (if needed)
cp -r test/data/ test/data_backup/
```

### File Size Management
Large files (can be regenerated):
- `test/data/predictions.csv` (~221 MB)
- `test/data/test_data.csv` (~183 MB)
- `test/data/predictions_all_*.csv` (~50 MB each)

Small files (keep):
- All configuration files
- All documentation
- All scripts
- Summary reports

---

**Last Updated:** November 2025  
**Version:** 1.0
