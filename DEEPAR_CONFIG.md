# DeepAR Training Configuration

## Model Details

**Endpoint**: `hybrent-nov`  
**Region**: `us-east-1`  
**Training Job**: `hybrent-deepar-forecast-2025-10-31-16-04-11-711`  
**Status**: InService  
**Created**: 2025-10-31

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `cardinality` | [88, 165] | Categorical feature cardinalities (2 features) |
| `context_length` | 90 | Historical context window (days) |
| `prediction_length` | 14 | Forecast horizon (days) |
| `time_freq` | D | Daily frequency |
| `num_dynamic_feat` | 2 | Number of dynamic features |
| `num_cells` | 64 | LSTM cells |
| `num_layers` | 3 | LSTM layers |
| `dropout_rate` | 0.1 | Dropout rate |
| `embedding_dimension` | 16 | Categorical embedding dimension |
| `epochs` | 100 | Training epochs |
| `learning_rate` | 0.001 | Learning rate |
| `mini_batch_size` | 32 | Batch size |
| `early_stopping_patience` | 20 | Early stopping patience |

## Input Data Format

### Required Fields

```json
{
  "start": "2023-11-01",
  "target": [1, 2, 3, ...],
  "cat": [0, 42],
  "dynamic_feat": [
    [0.5, 0.3, ...],
    [0.9, 1.0, ...]
  ],
  "item_id": "customer_facility_product"
}
```

### Field Descriptions

1. **start** (string): Start timestamp in format "YYYY-MM-DD"

2. **target** (array of numbers): Historical values
   - Length: 90-154 values (context_length + prediction_length)
   - Values: Actual order quantities

3. **cat** (array of 2 integers): Categorical features
   - `cat[0]`: Customer category (0-87, cardinality 88)
   - `cat[1]`: Facility category (0-164, cardinality 165)
   - **Important**: Values must be within cardinality limits

4. **dynamic_feat** (array of 2 arrays): Time-varying features
   - Feature 1: Day of week normalized (0-1 range, appears to be cyclical)
   - Feature 2: Month progress normalized (0-1 range, appears to be cyclical)
   - Length: Must match target length
   - Values: Normalized to 0-1 range

5. **item_id** (string): Unique identifier
   - Format: `{customer}_{facility}_{product}`
   - Example: "abrahealth_74_132574"

## Dynamic Features Explanation

Based on the training data samples:

### Feature 1: Day of Week (Cyclical)
- Values cycle through: 0.0, 0.16, 0.33, 0.5, 0.66, 1.0
- Represents day of week normalized
- Formula: `day_of_week / 6.0`

### Feature 2: Month Progress (Cyclical)
- Values cycle through: 0.0, 0.09, 0.18, 0.27, ..., 0.91, 1.0
- Represents progress through month
- Formula: `day_of_month / 11.0` (appears to be 11-day cycles)

## Categorical Feature Mapping

### Customer Categories (cardinality 88)
- Map customer IDs to integers 0-87
- If customer_id > 87, use modulo: `customer_id % 88`

### Facility Categories (cardinality 165)
- Map facility IDs to integers 0-164
- If facility_id > 164, use modulo: `facility_id % 165`

## Implementation Requirements

To successfully call the DeepAR endpoint, you must provide:

1. ✅ `start`: Timestamp string
2. ✅ `target`: Array of historical values (90+ values)
3. ✅ `cat`: Array of exactly 2 integers within cardinality limits
4. ✅ `dynamic_feat`: Array of exactly 2 arrays, each matching target length
5. ⚠️ `item_id`: Optional but recommended for tracking

## Example Request

```json
{
  "instances": [
    {
      "start": "2025-07-28",
      "target": [10, 12, 15, 8, 20, ...],
      "cat": [0, 42],
      "dynamic_feat": [
        [0.0, 0.16, 0.33, 0.5, 0.66, 0.0, 0.16, ...],
        [0.09, 0.18, 0.27, 0.36, 0.45, 0.54, 0.63, ...]
      ]
    }
  ],
  "configuration": {
    "num_samples": 100,
    "output_types": ["mean"],
    "quantiles": ["0.5"]
  }
}
```

## Training Data Location

- **S3 Bucket**: `sagemaker-us-east-1-236357498302`
- **Training Data**: `s3://sagemaker-us-east-1-236357498302/hybrent-deepar-forecast/data/train.json`
- **Model Output**: `s3://sagemaker-us-east-1-236357498302/hybrent-deepar-forecast/output/hybrent-deepar-forecast-2025-10-31-16-04-11-711/output/model.tar.gz`

## Notes

- The model expects exactly 90 days of context
- Predictions are for 14 days ahead
- All categorical values must be within their cardinality limits
- Dynamic features must be normalized to 0-1 range
- Dynamic features must have the same length as target values
