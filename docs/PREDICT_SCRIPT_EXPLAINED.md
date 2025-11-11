# predict.py - Detailed Explanation

## Overview

The `predict.py` script is a universal prediction tool that handles all forecasting scenarios with smart parameter detection. It's designed to be flexible and easy to use.

## How It Works

### 1. Main Flow

```
User runs script
    ↓
Parse arguments (smart detection)
    ↓
Parse customer list
    ↓
Generate predictions (PredictionGenerator)
    ↓
Filter by facility (optional)
    ↓
Save to file (optional)
    ↓
Display summary
```

### 2. Key Components

#### A. `predict()` Function
The main prediction function that:
1. **Parses customers** - Converts "scionhealth,mercy" to ["scionhealth", "mercy"]
2. **Generates predictions** - Uses PredictionGenerator with 90 days of history
3. **Filters** - Optionally filters by facility
4. **Saves** - Optionally saves to CSV
5. **Displays** - Shows summary statistics

#### B. `parse_args()` Function
Smart argument parser that automatically detects:
- **Dates**: Contains `-` and starts with digit (e.g., "2025-11-15")
- **Facilities**: All digits (e.g., "287")
- **Output files**: Ends with .csv or .json (e.g., "output.csv")
- **Customers**: Everything else (e.g., "scionhealth")

This allows flexible argument order:
```bash
python predict.py scionhealth 2025-11-15  # ✓ Works
python predict.py 2025-11-15 scionhealth  # ✓ Also works
```

#### C. `print_usage()` Function
Displays help information with examples.

### 3. Prediction Generation Process

When you run `generator.generate()`, it:

1. **Loads data** - Reads historical order data
2. **Engineers features** - Creates:
   - Rolling averages (7-day, 30-day)
   - Lag features (7, 14, 30 days back)
   - Seasonal patterns
   - Customer/facility/product encodings
3. **Loads models** - LightGBM and DeepAR
4. **Generates predictions** - Ensemble (95% LightGBM + 5% DeepAR)
5. **Applies calibrations** - Customer-specific adjustments
6. **Returns DataFrame** - With predictions and metadata

### 4. Output Structure

Every prediction includes:
```csv
prediction_date,prediction_generated_at,CustomerID,FacilityID,ProductID,ProductName,predicted_value,predicted_reorder,reorder_recommendation
```

- **prediction_date**: Date you're predicting FOR
- **prediction_generated_at**: When prediction was made
- **predicted_value**: Predicted quantity (units)
- **predicted_reorder**: 1 if should order, 0 if not
- **reorder_recommendation**: "ORDER" or "NO ORDER"

## Usage Examples

### Basic Usage
```bash
# One customer
python predict.py scionhealth
```

**What happens:**
1. Parses "scionhealth" as customer
2. Uses today as prediction date
3. Loads 90 days of history
4. Generates predictions
5. Shows top 10 items

### With Date
```bash
# Specific date
python predict.py 2025-11-15 scionhealth
```

**What happens:**
1. Detects "2025-11-15" as date
2. Detects "scionhealth" as customer
3. Uses 90 days before 2025-11-15 as context
4. Generates predictions for 2025-11-15

### Multiple Customers
```bash
# Multiple customers
python predict.py 2025-11-15 scionhealth,mercy
```

**What happens:**
1. Splits "scionhealth,mercy" into ["scionhealth", "mercy"]
2. Generates predictions for both
3. Shows summary by customer

### With Facility Filter
```bash
# One facility
python predict.py scionhealth 287 2025-11-15
```

**What happens:**
1. Generates predictions for all scionhealth facilities
2. Filters to only facility 287
3. Shows results for that facility only

### Save to File
```bash
# Save results
python predict.py scionhealth output.csv
```

**What happens:**
1. Generates predictions
2. Saves to output.csv
3. Still shows summary on screen

## Smart Parameter Detection

### How It Works

The script uses helper functions to detect argument types:

```python
def is_date(s):
    """Check if string looks like a date"""
    return '-' in s and len(s) >= 8 and s[0].isdigit()

def is_numeric(s):
    """Check if string is all digits (facility ID)"""
    return s.isdigit()

def is_output_file(s):
    """Check if string is an output file"""
    return s.endswith('.csv') or s.endswith('.json')
```

### Detection Logic

| Argument | Detection | Example |
|----------|-----------|---------|
| Date | Contains `-` and starts with digit | `2025-11-15` |
| Facility | All digits | `287` |
| Output | Ends with `.csv` or `.json` | `output.csv` |
| Customer | Everything else | `scionhealth` |

### Argument Patterns

| Pattern | Interpretation |
|---------|----------------|
| `predict.py` | All customers, today |
| `predict.py 2025-11-15` | All customers, specific date |
| `predict.py scionhealth` | One customer, today |
| `predict.py scionhealth 2025-11-15` | One customer, specific date |
| `predict.py 2025-11-15 scionhealth` | Same (order doesn't matter) |
| `predict.py scionhealth 287` | One customer, one facility |
| `predict.py scionhealth 287 2025-11-15` | Customer + facility + date |
| `predict.py 2025-11-15 scionhealth,mercy` | Multiple customers, date |

## Summary Display

### Single Customer
Shows top 10 items:
```
Top 10 Items by Predicted Volume:
  Surgical Gloves Size L                              125.5 units  [ORDER]
  Bandages 2x2                                         98.3 units  [ORDER]
  ...
```

### Multiple Customers
Shows summary by customer:
```
By Customer:
              Items  Total_Units  Avg_Units  To_Order
scionhealth   3,456       30,000       8.68     1,500
mercy         2,222       15,678       7.05       845
```

## Error Handling

The script handles errors gracefully:

```bash
$ python predict.py invalid_customer

❌ Error: No data found for customer 'invalid_customer'

Run 'python predict.py --help' for usage examples
```

## Performance

- **Speed**: ~30-60 seconds for typical customer
- **Memory**: ~2-4 GB for large datasets
- **Accuracy**: MAE 4-6 units, Precision >90%

## Tips

1. **Use specific dates** for planning:
   ```bash
   python predict.py 2025-11-18 scionhealth
   ```

2. **Save results** for analysis:
   ```bash
   python predict.py scionhealth predictions.csv
   ```

3. **Filter by facility** for detailed view:
   ```bash
   python predict.py scionhealth 287
   ```

4. **Use named arguments** for clarity:
   ```bash
   python predict.py --date 2025-11-15 --customers scionhealth --output file.csv
   ```

## Summary

The `predict.py` script is:
- ✅ **Flexible** - Handles all scenarios
- ✅ **Smart** - Auto-detects argument types
- ✅ **Fast** - Generates predictions in seconds
- ✅ **Accurate** - Uses ensemble model with calibrations
- ✅ **Easy** - Simple command line interface

It's the main tool for generating predictions in the system!
