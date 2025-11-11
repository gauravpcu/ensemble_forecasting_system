# Project Structure

This document describes the reorganized project structure.

## Directory Layout

```
ensemble_forecasting_system/
├── src/                          # Source code
│   ├── core/                     # Core business logic
│   │   └── prediction_generator.py
│   ├── models/                   # Model loading and prediction
│   │   ├── ensemble_predictor.py
│   │   └── model_loader.py
│   ├── data/                     # Data processing
│   │   ├── data_loader.py
│   │   └── evaluator.py
│   ├── config/                   # Configuration management
│   │   ├── env_config.py
│   │   └── config_defaults.py
│   ├── calibration/              # Calibration logic
│   │   └── calibration_manager.py
│   └── utils/                    # Utilities
│       └── visualizer.py
│
├── scripts/                      # Executable scripts
│   ├── train.py                  # Train models
│   ├── predict.py                # Generate predictions
│   ├── manage_calibrations.py   # Manage customer calibrations
│   └── configure.py              # Configure system parameters
│
├── tests/                        # All test files
│   ├── run_full_test.py
│   ├── generate_forward_predictions.py
│   ├── analyze_results.py
│   ├── customer_precision.py
│   ├── customer_facility_analysis.py
│   ├── extract_data.py
│   ├── verify_data.py
│   ├── quick_stats.py
│   └── run_predictions.py
│
├── docs/                         # Documentation
│   ├── README.md
│   ├── USAGE_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   ├── MIGRATION_GUIDE.md
│   ├── PRECISION_RECALL_EXPLAINED.md
│   └── ... (all other .md files)
│
├── config/                       # Configuration files
│   └── customer_calibrations.json
│
├── model/                        # Trained models
│   ├── lightgbm_model.pkl
│   └── feature_config.json
│
├── .env                          # Environment configuration
├── .env.example                  # Example environment file
├── requirements.txt              # Python dependencies
└── .gitignore
```

## Module Organization

### src/core/
Core business logic and main prediction workflows.
- `prediction_generator.py` - Unified prediction generation for all scenarios

### src/models/
Model loading, prediction, and ensemble logic.
- `model_loader.py` - Load LightGBM and DeepAR models
- `ensemble_predictor.py` - Combine predictions from multiple models

### src/data/
Data loading, feature engineering, and evaluation.
- `data_loader.py` - Load and prepare data for predictions
- `evaluator.py` - Calculate accuracy metrics and evaluate models

### src/config/
Configuration management and defaults.
- `env_config.py` - Load configuration from .env file
- `config_defaults.py` - Default values for all parameters

### src/calibration/
Customer-specific calibration management.
- `calibration_manager.py` - Manage customer calibration multipliers

### src/utils/
Utility functions and helpers.
- `visualizer.py` - Visualization utilities

## Scripts

### Training
```bash
python scripts/train.py
```

### Prediction
```bash
# Predict for specific customer
python scripts/predict.py scionhealth

# Predict for customer and facility
python scripts/predict.py scionhealth 287

# Predict for specific date
python scripts/predict.py scionhealth 287 2025-11-05

# Save to file
python scripts/predict.py scionhealth 287 2025-11-05 output.csv
```

### Calibration Management
```bash
# List all calibrations
python scripts/manage_calibrations.py list

# Show customer details
python scripts/manage_calibrations.py show scionhealth

# Update calibration
python scripts/manage_calibrations.py update mercy 0.85 --status verified
```

### Configuration
```bash
# Update configuration parameters
python scripts/configure.py
```

## Testing

### Full Test Suite
```bash
python tests/run_full_test.py
```

### Forward-Looking Predictions
```bash
python tests/generate_forward_predictions.py
```

### Customer Analysis
```bash
python tests/customer_precision.py
python tests/customer_facility_analysis.py
```

## Import Patterns

All imports now use the `src.` prefix:

```python
# Configuration
from src.config import env_config

# Models
from src.models.model_loader import load_models
from src.models.ensemble_predictor import EnsemblePredictor

# Data
from src.data.data_loader import DataLoader
from src.data.evaluator import Evaluator

# Core
from src.core.prediction_generator import PredictionGenerator

# Calibration
from src.calibration.calibration_manager import CalibrationManager
```

## Migration Notes

If you have existing scripts that import modules, update them:

**Old:**
```python
import env_config
from data_loader import DataLoader
from model_loader import load_models
```

**New:**
```python
from src.config import env_config
from src.data.data_loader import DataLoader
from src.models.model_loader import load_models
```

## Benefits of New Structure

1. **Clear Organization** - Related code grouped together
2. **Proper Python Package** - src/ is a proper package with __init__.py files
3. **Separation of Concerns** - Scripts, source code, tests, and docs are separate
4. **Easier Testing** - All tests in one place
5. **Better Imports** - Clear import paths with src. prefix
6. **Scalability** - Easy to add new modules and features
