# Ensemble Forecasting System

Healthcare supply chain inventory forecasting using ensemble machine learning (LightGBM + DeepAR).

## 🎯 Overview

Predicts inventory reorder needs for healthcare supply chain management by combining:
- **LightGBM (95%)**: Structured feature-based predictions  
- **DeepAR (5%)**: Time series pattern recognition

### Key Features
- ✅ Extended 90-day context for seasonal pattern detection
- ✅ Customer-specific calibration (14 customers optimized)
- ✅ Volume-based safety multipliers
- ✅ Product-level precision tracking (78K+ items)
- ✅ Comprehensive performance metrics

### Current Performance (ScionHealth)
- **Precision:** 82.2% - When we predict order, we're right 82% of time
- **Recall:** 92.4% - We catch 92.4% of all actual orders  
- **F1 Score:** 87.0% - Excellent balance
- **MAE:** 24.92 units average error per item

---

## 📚 Documentation

| Document | Use When You Need |
|----------|-------------------|
| **[README.md](README.md)** | Quick start, installation, basic usage |
| **[USER_GUIDE.md](USER_GUIDE.md)** | Detailed parameters, metrics, workflows, troubleshooting |
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | Fast command lookup, metric tables |
| **[FILE_REFERENCE.md](FILE_REFERENCE.md)** | Understanding what each file does |

---

## �  Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings
```

### 2. Essential Configuration
Edit `.env`:
```bash
SOURCE_DATA_FILE=/path/to/order_history.csv
DEEPAR_ENDPOINT_NAME=your-endpoint
DEEPAR_REGION=us-east-1
CLASSIFICATION_THRESHOLD=4
```

### 3. Run Test
```bash
python3 test/run_full_test.py
```

### 4. View Results
```bash
cat test/data/ANALYSIS_SUMMARY.txt
```

---

## 🎯 Common Tasks

```bash
# Optimize threshold
python3 test/test_thresholds.py

# Apply calibration
python3 test/apply_calibration.py

# Analyze customer
python3 test/customer_precision.py scionhealth

# Extract predictions
python3 test/extract_all_facilities.py

# Change config
python3 configure.py --set CLASSIFICATION_THRESHOLD 5
```

---

## 🤖 Model Training

### Train New LightGBM Model

```bash
# Set training data
python3 configure.py --set SOURCE_DATA_FILE "/path/to/order_history.csv"

# Train model
python3 train_model.py
```

### Training Data Split

The training process uses a **3-way temporal split**:

```
All Historical Data (e.g., 730 days)
│
├─────────────────────────────────────────────────────────┤
│  [────────── Training (716 days) ──────]  [Test (14d)] │
│  [─ Train (90%) ─]  [─ Val (10%) ─]                    │
└─────────────────────────────────────────────────────────┘
```

**Example with 2 years of data:**
- **Train Set (90%):** Nov 2023 - Aug 2025 (~645 days) - Fit model
- **Validation Set (10%):** Aug 2025 - Oct 2025 (~71 days) - Early stopping
- **Test Set:** Last 14 days - Final evaluation (held out)

**Configuration:**
```bash
# Adjust test period
python3 configure.py --set TRAINING_TEST_DAYS 30

# Adjust validation split
python3 configure.py --set TRAINING_VALIDATION_SPLIT 0.2  # 20%

# Tune hyperparameters
python3 configure.py --set LGBM_LEARNING_RATE 0.01
python3 configure.py --set LGBM_NUM_BOOST_ROUND 5000
```

**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for complete training documentation.**

---

## 📊 Key Metrics

| Metric | What It Means | Good Value | Current |
|--------|---------------|------------|---------|
| **Precision** | % correct when we predict order | >80% | 82.2% ✅ |
| **Recall** | % of actual orders we catch | >90% | 92.4% ✅ |
| **F1 Score** | Balance of precision & recall | >80% | 87.0% ✅ |
| **MAE** | Average error in units | <10 | 24.92 ⚠️ |

---

## 📁 Project Structure

```
ensemble_forecasting_system/
├── 📄 Documentation
│   ├── README.md              # Quick start (this file)
│   ├── USER_GUIDE.md          # Comprehensive guide
│   ├── QUICK_REFERENCE.md     # Fast lookup
│   └── FILE_REFERENCE.md      # File reference
│
├── ⚙️ Configuration
│   ├── .env                   # Active config
│   ├── configure.py           # Config management
│   └── config_defaults.py     # Defaults
│
├── 🔧 Core System
│   ├── ensemble_predictor.py # Main predictor
│   ├── data_loader.py         # Data processing
│   └── model_loader.py        # Model loading
│
├── 🗂️ Models
│   └── model/                 # LightGBM + config
│
└── 🧪 Testing
    ├── run_full_test.py       # Main pipeline
    ├── test_thresholds.py     # Optimize threshold
    ├── apply_calibration.py   # Apply calibration
    └── data/                  # Results
```

---

## 🔄 Typical Workflow

### Initial Setup
```bash
1. Configure .env
2. python3 test/run_full_test.py
3. Review results
```

### Optimization
```bash
1. python3 test/test_thresholds.py
2. python3 test/optimize_calibration.py
3. python3 configure.py --set PARAMETER value
4. python3 test/apply_calibration.py
```

### Daily Operations
```bash
python3 test/extract_customer_predictions.py YYYY-MM-DD
```

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| High volume error (>150%) | Reduce safety multipliers |
| Low recall (<80%) | Decrease threshold |
| Low precision (<50%) | Increase threshold |

**See [USER_GUIDE.md](USER_GUIDE.md#troubleshooting) for detailed troubleshooting.**

---

## 📞 Getting Help

1. **Quick commands**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Detailed guide**: [USER_GUIDE.md](USER_GUIDE.md)
3. **File questions**: [FILE_REFERENCE.md](FILE_REFERENCE.md)

---

**Version:** 2.0  
**Last Updated:** November 2025  
**Optimizations:** Threshold (7→4), Customer Calibration (14), Volume Safety Multipliers
