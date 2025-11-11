# PredictionGenerator - Detailed Explanation

## Overview

`PredictionGenerator` is the core prediction engine that orchestrates the entire forecasting pipeline from data loading to final predictions.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PredictionGenerator                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. load_data()          → Load historical order data        │
│  2. prepare_features()   → Engineer features                 │
│  3. load_models()        → Load LightGBM + DeepAR           │
│  4. generate_predictions()→ Create ensemble predictions      │
│  5. save_predictions()   → Save to CSV (optional)           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Initialization (`__init__`)

**Purpose**: Set up configuration parameters

**Parameters**:
- `customers`: List of customer IDs (None = all)
- `target_date`: Date to predict for (YYYY-MM-DD)
- `context_days`: Days of history to use (default: 90)
- `source_data`: Path to data file
- `use_preprocessed`: If True, data already has features
- `classification_threshold`: Threshold for "ORDER" recommendation (default: 4)
- `verbose`: Print progress messages

**What it does**:
1. Stores all configuration parameters
2. Converts date strings to datetime objects
3. Initializes placeholders for models and data
4. Validates parameters for logical consistency

**Example**:
```python
gen = PredictionGenerator(
    customers=['scionhealth'],
    target_date='2025-11-15',
    context_days=90,
    verbose=True
)
```

### 2. Data Loading (`load_data`)

**Purpose**: Load and filter historical order data

**Process**:
1. **Load data** from CSV file
   - If `use_preprocessed=True`: Load data with features already engineered
   - If `use_preprocessed=False`: Load raw order history

2. **Parse dates** (if raw data)
   - Convert 'CreateDate' column to datetime

3. **Filter by customers** (if specified)
   - Keep only rows for specified customer IDs

**Input**: CSV file with columns:
- `CreateDate`: Order date
- `CustomerID`: Customer identifier
- `FacilityID`: Facility identifier
- `ProductID`: Product identifier
- `OrderUnits`: Quantity ordered
- `Price`: Unit price

**Output**: `self.raw_data` DataFrame with filtered data

### 3. Feature Engineering (`prepare_features`)

**Purpose**: Create features for ML models

**If preprocessed data**: Skip feature engineering

**If raw data**: Engineer features in these steps:

#### Step 1: Determine Date Range
```python
# Example: Predict for 2025-11-15 with 90 days context
max_date = 2025-11-15
min_date = 2025-08-17  # 90 days before
```

#### Step 2: Filter Context Period
Keep only data within the context window

#### Step 3: Create item_id
```python
item_id = CustomerID + '_' + FacilityID + '_' + ProductID
# Example: "scionhealth_287_12345"
```

#### Step 4: Aggregate by Item and Date
Group by `item_id` and `CreateDate`, sum quantities

#### Step 5: Engineer Features

**Time-based features**:
- `day_of_week`: 0-6 (Monday-Sunday)
- `day_of_month`: 1-31
- `month`: 1-12
- `quarter`: 1-4
- `is_month_end`: 1 if last days of month
- `is_quarter_end`: 1 if last days of quarter

**Rolling features** (7-day and 30-day windows):
- `rolling_mean_7d`: Average of last 7 days
- `rolling_std_7d`: Standard deviation of last 7 days
- `rolling_mean_30d`: Average of last 30 days
- `rolling_std_30d`: Standard deviation of last 30 days

**Lag features** (historical values):
- `lag_7`: Value from 7 days ago
- `lag_14`: Value from 14 days ago
- `lag_30`: Value from 30 days ago

**Seasonal features**:
- `seasonal_trend`: Overall trend for item
- `seasonal_seasonal`: Day-of-week pattern

**Business features**:
- `vendor_reliability`: How often vendor appears
- `price_volatility`: Price variation
- `order_frequency`: How often item is ordered

**Categorical encodings**:
- `customer_encoded`: Numeric encoding of CustomerID
- `facility_encoded`: Numeric encoding of FacilityID
- `product_encoded`: Numeric encoding of ProductID

#### Step 6: Get Latest Data
For each item, get the most recent row (for prediction)

**Output**: `self.processed_data` DataFrame with all features

### 4. Model Loading (`load_models`)

**Purpose**: Load trained ML models

**Process**:
1. Load LightGBM model from pickle file
2. Connect to DeepAR SageMaker endpoint
3. Create EnsemblePredictor with both models

**Models**:
- **LightGBM**: Gradient boosting model (95% weight)
- **DeepAR**: Deep learning time series model (5% weight)

**Output**: 
- `self.models`: Dictionary with both models
- `self.predictor`: EnsemblePredictor instance

### 5. Prediction Generation (`generate_predictions`)

**Purpose**: Generate ensemble predictions

**Process**:

#### Step 1: Prepare Features for LightGBM
Use DataLoader to format features correctly

#### Step 2: Generate Predictions
Call `predictor.predict()` which:
1. Gets LightGBM predictions
2. Gets DeepAR predictions
3. Combines: `ensemble = 0.95 * lightgbm + 0.05 * deepar`

#### Step 3: Add Prediction Metadata
```python
prediction_date = target_date or datetime.now()
prediction_generated_at = datetime.now()
```

#### Step 4: Apply Classification
```python
if predicted_value >= 4:
    predicted_reorder = 1
    reorder_recommendation = "ORDER"
else:
    predicted_reorder = 0
    reorder_recommendation = "NO ORDER"
```

#### Step 5: Sort Results
Sort by CustomerID and predicted_value (descending)

**Output**: `self.predictions` DataFrame with:
- `prediction_date`: Date predictions are for
- `prediction_generated_at`: When predictions were made
- `CustomerID`, `FacilityID`, `ProductID`, `ProductName`
- `predicted_value`: Predicted quantity
- `predicted_reorder`: 1 or 0
- `reorder_recommendation`: "ORDER" or "NO ORDER"
- All original features

### 6. Save Predictions (`save_predictions`)

**Purpose**: Save predictions to CSV file

**Process**:
1. Generate filename if not provided
   - Format: `predictions_YYYYMMDD.csv`
2. Reorder columns (important ones first)
3. Save to CSV

**Column Order**:
1. `prediction_date`
2. `prediction_generated_at`
3. `CustomerID`, `FacilityID`, `ProductID`, `ProductName`
4. `predicted_value`, `predicted_reorder`, `reorder_recommendation`
5. All other columns

### 7. Main Pipeline (`generate`)

**Purpose**: Run the complete pipeline

**Process**:
```python
def generate(save=True, output_path=None):
    self.load_data()              # Step 1
    self.prepare_features()       # Step 2
    self.load_models()            # Step 3
    self.generate_predictions()   # Step 4
    if save:
        self.save_predictions(output_path)  # Step 5
    return self.predictions
```

**Usage**:
```python
gen = PredictionGenerator(customers=['scionhealth'])
predictions = gen.generate()  # Runs all steps
```

## Helper Methods

### `_validate_parameters()`
Checks for invalid parameter combinations:
- Can't have both `target_date` and date range
- Date range must have both start and end
- Start must be before end

### `_print(message)`
Prints message only if `verbose=True`

### `get_summary()`
Returns summary statistics by customer

### `print_summary()`
Prints formatted summary to console

### `get_top_items(customer, n=10)`
Returns top N items by predicted volume for a customer

## Data Flow Example

### Input
```csv
CreateDate,CustomerID,FacilityID,ProductID,OrderUnits,Price
2025-08-17,scionhealth,287,12345,50,10.50
2025-08-18,scionhealth,287,12345,45,10.50
...
```

### After Feature Engineering
```csv
item_id,timestamp,target_value,rolling_mean_7d,lag_7,day_of_week,...
scionhealth_287_12345,2025-11-14,47.5,48.2,50,0,...
```

### After Prediction
```csv
prediction_date,prediction_generated_at,CustomerID,FacilityID,ProductID,predicted_value,predicted_reorder,reorder_recommendation
2025-11-15,2025-11-11 14:30:00,scionhealth,287,12345,49.2,1,ORDER
```

## Performance Characteristics

- **Speed**: 30-60 seconds for typical customer
- **Memory**: 2-4 GB for large datasets
- **Accuracy**: MAE 4-6 units, Precision >90%
- **Scalability**: Can handle millions of records

## Configuration

Key parameters from `env_config`:
- `SOURCE_DATA_FILE`: Path to order history
- `CLASSIFICATION_THRESHOLD`: Reorder threshold (default: 4)
- `ROLLING_WINDOW_SHORT`: Short window (default: 7)
- `ROLLING_WINDOW_LONG`: Long window (default: 30)

## Error Handling

The class handles:
- Missing data files
- Invalid date formats
- Missing features
- Model loading failures
- Empty datasets

## Best Practices

1. **Use 90 days context** for best accuracy
2. **Specify target_date** for production predictions
3. **Use preprocessed data** for testing (faster)
4. **Enable verbose** for debugging
5. **Save predictions** for audit trail

## Summary

`PredictionGenerator` is the heart of the forecasting system:
- ✅ Flexible data sources
- ✅ Automatic feature engineering
- ✅ Ensemble predictions
- ✅ Customer calibrations
- ✅ Prediction tracking
- ✅ Easy to use API

It handles all the complexity so you can focus on using the predictions!
