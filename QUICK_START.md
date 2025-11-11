# Quick Start Guide

## Generate Predictions

```bash
# One customer
python scripts/predict.py scionhealth

# Specific date
python scripts/predict.py 2025-11-15 scionhealth

# Multiple customers
python scripts/predict.py 2025-11-15 scionhealth,mercy

# With facility
python scripts/predict.py scionhealth 287 2025-11-15

# Save to file
python scripts/predict.py scionhealth output.csv

# All options
python scripts/predict.py --help
```

## Run Tests

```bash
# Quick test
python tests/run_test.py --quick

# Full test with validation
python tests/run_test.py --full

# Customer-specific test
python tests/run_test.py --customer scionhealth
```

## Train Model

```bash
python scripts/train.py
```

## Configuration Management

```bash
# Environment config
python scripts/config.py --show
python scripts/config.py --set SOURCE_DATA_FILE /path/to/data.csv
python scripts/config.py --set BATCH_SIZE 2000

# Customer calibrations
python scripts/config.py --calibrations
python scripts/config.py --customer scionhealth
python scripts/config.py --update mercy 0.85 --status verified
```

## Project Structure

```
├── src/              # Source code
├── scripts/          # Executable scripts
│   ├── predict.py    # Generate predictions
│   ├── train.py      # Train models
│   └── config.py     # Configuration management
├── tests/            # Test suite
│   └── run_test.py   # Universal test runner
├── config/           # Config files
├── model/            # Trained models
└── docs/             # Documentation
```

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

- **README.md** - Overview and quick start
- **docs/USAGE_GUIDE.md** - Comprehensive usage
- **docs/TRAINING_GUIDE.md** - Model training
- **docs/README.md** - System architecture

## Help

```bash
python scripts/predict.py --help
python tests/run_test.py --help
python scripts/config.py --help
```
