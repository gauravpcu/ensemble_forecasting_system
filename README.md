# Ensemble Forecasting System

Healthcare supply chain inventory prediction using LightGBM (95%) + DeepAR (5%) ensemble.

## Quick Start

```bash
# Install
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths

# Extract data (generate context & validation files)
python scripts/extract.py

# Predict
python scripts/predict.py scionhealth
python scripts/predict.py --date 2025-11-15 --customers scionhealth --days 14
python scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth

# Validate predictions
python tests/validate.py predictions.csv tests/data/val_data.csv

# Configure
python scripts/config.py --show
python scripts/config.py --calibrations
```

## Common Commands

```bash
# Data Extraction
python scripts/extract.py                                       # Generate context & validation files

# Predictions
python scripts/predict.py scionhealth                           # One customer
python scripts/predict.py --date 2025-11-15 --customers scionhealth,mercy --days 14
python scripts/predict.py --use-test-data --date 2025-10-26 --customers scionhealth
python scripts/predict.py scionhealth predictions.csv           # Save to file
python scripts/predict.py --use-test-data --customers scionhealth  # Use test_data.csv (faster)

# Validation
python tests/validate.py predictions.csv tests/data/val_data.csv
python tests/validate.py predictions.csv tests/data/val_data.csv --output results.txt

# Configuration
python scripts/config.py --show                                 # Show config
python scripts/config.py --calibrations                         # List calibrations
python scripts/config.py --update mercy 0.85 --status verified  # Update calibration

# Training
python scripts/train.py                                         # Train new model
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
├── model/                 # Trained models
└── docs/                  # Documentation
```

## Key Features

- **90-Day Context**: Uses 90 days of history for predictions
- **Ensemble Model**: LightGBM (95%) + DeepAR (5%)
- **Customer Calibration**: Adjust predictions per customer
- **Multi-Day Forecasting**: Predict 1-30 days ahead
- **No Data Leakage**: Context ends before prediction starts

## Configuration

All settings in `.env` file:
```bash
SOURCE_DATA_FILE=/path/to/order_history.csv
LIGHTGBM_MODEL_PATH=./model/lightgbm_model.pkl
DEEPAR_ENDPOINT_NAME=hybrent-nov
DEEPAR_REGION=us-east-1
CLASSIFICATION_THRESHOLD=4
LIGHTGBM_WEIGHT=0.95
DEEPAR_WEIGHT=0.05
```

## Performance

- **MAE**: 4-6 units (with 90-day context)
- **Precision**: >90% (low false alarm rate)
- **Best for**: Medium-volume items (5-20 units)

## Customer Calibrations

- **ScionHealth**: 1.05x (reduce stockouts)
- **Mercy**: 0.85x (reduce over-ordering)
- **IBJI**: 1.575x (high demand)

## Documentation

- **README.md** (this file) - Quick start and common commands
- **docs/GUIDE.md** - Complete reference guide

## Help

```bash
python scripts/extract.py --help
python scripts/predict.py --help
python scripts/config.py --help
python scripts/train.py --help
python tests/validate.py --help
```

## License

Proprietary - Internal use only
