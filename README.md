# Ensemble Forecasting System

Healthcare supply chain inventory prediction using LightGBM (95%) + DeepAR (5%) ensemble.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your paths
```

### Generate Predictions
```bash
# One customer
python scripts/predict.py scionhealth

# Specific date
python scripts/predict.py 2025-11-15 scionhealth

# Multiple customers
python scripts/predict.py 2025-11-15 scionhealth,mercy

# With facility filter
python scripts/predict.py scionhealth 287 2025-11-15

# Save to file
python scripts/predict.py scionhealth output.csv

# See all options
python scripts/predict.py --help
```

### Train Model
```bash
python scripts/train.py
```

### Run Tests
```bash
python tests/run_test.py --quick          # Quick test
python tests/run_test.py --full           # Full test suite
python tests/run_test.py --customer scionhealth  # Customer-specific
```

### Configuration Management
```bash
# Environment config
python scripts/config.py --show
python scripts/config.py --set BATCH_SIZE 2000

# Customer calibrations
python scripts/config.py --calibrations
python scripts/config.py --customer scionhealth
python scripts/config.py --update mercy 0.85
```

## Project Structure

```
├── src/                    # Source code
│   ├── core/              # Prediction logic
│   ├── models/            # Model loading & ensemble
│   ├── data/              # Data processing
│   ├── config/            # Configuration
│   ├── calibration/       # Customer calibrations
│   └── utils/             # Utilities
├── scripts/               # Executable scripts
│   ├── predict.py         # Generate predictions
│   ├── train.py           # Train models
│   └── manage_calibrations.py
├── tests/                 # Test suite
├── config/                # Config files
├── model/                 # Trained models
└── docs/                  # Documentation
```

## Configuration

All settings in `.env` file:
```bash
SOURCE_DATA_FILE=/path/to/order_history.csv
TEST_DATA_DIR=./tests/data
LIGHTGBM_MODEL_PATH=./model/lightgbm_model.pkl
DEEPAR_ENDPOINT_NAME=hybrent-nov
LIGHTGBM_WEIGHT=0.95
DEEPAR_WEIGHT=0.05
```

## Key Features

- **Ensemble Model**: LightGBM (95%) + DeepAR (5%)
- **Customer Calibration**: Adjust predictions per customer
- **Prediction Date Tracking**: Every prediction includes date metadata
- **Flexible Predictions**: Single/multiple customers, facility filtering
- **90-Day Context**: Uses 90 days of history for predictions
- **Classification**: Automatic reorder recommendations (≥4 units)

## Prediction Output

Every prediction includes:
```csv
prediction_date,prediction_generated_at,CustomerID,FacilityID,ProductID,ProductName,predicted_value,predicted_reorder,reorder_recommendation
2025-11-15,2025-11-11 14:30:00,scionhealth,287,12345,Surgical Gloves,25.5,1,ORDER
```

## Model Performance

- **MAE**: 4-6 units (with 90-day context)
- **Precision**: >90% (low false alarm rate)
- **Best for**: Medium-volume items (5-20 units)
- **Challenge**: Very low volume (0-5 units) and sporadic orders

## Customer Calibrations

Adjust predictions per customer:
- **ScionHealth**: 1.05x (reduce stockouts)
- **Mercy**: 0.85x (reduce over-ordering)
- **IBJI**: 1.575x (high demand)

## Common Tasks

### Daily Predictions
```bash
python scripts/predict.py predictions_$(date +%Y%m%d).csv
```

### Weekly Planning
```bash
for i in {0..6}; do
  date=$(date -v+${i}d +%Y-%m-%d)
  python scripts/predict.py $date predictions_$date.csv
done
```

### Customer Analysis
```bash
python scripts/predict.py scionhealth analysis.csv
python tests/run_test.py --customer scionhealth
```

## Documentation

- **README.md** (this file) - Quick start and overview
- **docs/USAGE_GUIDE.md** - Comprehensive usage guide
- **docs/TRAINING_GUIDE.md** - Model training guide
- **docs/README.md** - System architecture and design

## Support

- Configuration issues: Check `.env` file
- Import errors: Ensure `src/` is in Python path
- Model errors: Verify model files in `model/` directory
- Data errors: Check `SOURCE_DATA_FILE` path in `.env`

## License

Proprietary - Internal use only
