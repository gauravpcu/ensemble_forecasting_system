# Validation Results Comparison

## LightGBM Only vs Ensemble (LightGBM 95% + DeepAR 5%)

### Overall Metrics Comparison

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| **MAE** | 10.39 units | 10.54 units | +0.15 (+1.4%) |
| **RMSE** | 46.33 units | 46.35 units | +0.02 (+0.04%) |
| **MAPE** | 65.8% | 75.4% | +9.6% |
| **R²** | 0.416 | 0.416 | 0.0 |

### Classification Metrics Comparison

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| **Precision** | 95.6% | 93.6% | -2.0% |
| **Recall** | 92.8% | 94.3% | +1.5% |
| **F1 Score** | 94.2% | 94.0% | -0.2% |
| **Accuracy** | 91.3% | 90.8% | -0.5% |

### Confusion Matrix Comparison

#### LightGBM Only
```
                    Actual: No Order    Actual: Order
Predicted: No          119,410            31,436
Predicted: Yes          18,410           403,414
```

#### With DeepAR
```
                    Actual: No Order    Actual: Order
Predicted: No          109,826            24,717
Predicted: Yes          27,994           410,133
```

#### Changes
- **True Positives**: 403,414 → 410,133 (+6,719) ✅ More orders caught
- **False Positives**: 18,410 → 27,994 (+9,584) ⚠️ More false alarms
- **False Negatives**: 31,436 → 24,717 (-6,719) ✅ Fewer missed orders
- **True Negatives**: 119,410 → 109,826 (-9,584) ⚠️ Fewer correct no-orders

### Performance by Volume Category

#### Very Low Volume (0-5 units)

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| Count | 207,076 | 207,076 | - |
| MAE | 2.56 | 2.97 | +0.41 |
| Precision | 72.2% | 65.1% | -7.1% |
| Recall | 69.0% | 75.2% | +6.2% |

#### Low Volume (5-20 units)

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| Count | 229,380 | 229,380 | - |
| MAE | 4.89 | 5.00 | +0.11 |
| Precision | 100.0% | 100.0% | 0.0% |
| Recall | 95.8% | 96.9% | +1.1% |

#### Medium Volume (20-100 units)

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| Count | 116,095 | 116,095 | - |
| MAE | 16.73 | 16.56 | -0.17 ✅ |
| Precision | 100.0% | 100.0% | 0.0% |
| Recall | 99.8% | 99.8% | 0.0% |

#### High Volume (100+ units)

| Metric | LightGBM Only | With DeepAR | Change |
|--------|---------------|-------------|--------|
| Count | 20,119 | 20,119 | - |
| MAE | 117.00 | 116.77 | -0.23 ✅ |
| Precision | 100.0% | 100.0% | 0.0% |
| Recall | 99.1% | 99.1% | 0.0% |

## Analysis

### Strengths of Adding DeepAR

1. **Better Recall** (+1.5%)
   - Catches 6,719 more actual orders
   - Reduces missed orders from 31,436 to 24,717
   - Better for preventing stockouts

2. **Improved Low-Volume Recall** (+6.2%)
   - Significantly better at catching low-volume orders
   - Important for items with sporadic ordering patterns

3. **Slightly Better MAE for High Volume**
   - Medium volume: -0.17 units
   - High volume: -0.23 units

### Trade-offs

1. **Lower Precision** (-2.0%)
   - More false alarms (18,410 → 27,994)
   - 9,584 more unnecessary order recommendations
   - May lead to slight over-ordering

2. **Slightly Higher MAE** (+0.15 units overall)
   - Minimal impact on average error
   - Trade-off for better recall

3. **Lower Precision for Very Low Volume** (-7.1%)
   - More false alarms for low-volume items
   - May need threshold adjustment for this category

## Business Impact

### Positive Impacts

✅ **Fewer Stockouts**
- 6,719 fewer missed orders
- 21% reduction in false negatives (31,436 → 24,717)
- Better availability for customers

✅ **Better Low-Volume Detection**
- 6.2% improvement in recall for sporadic items
- Catches items that might be missed by LightGBM alone

✅ **Time Series Expertise**
- DeepAR adds seasonal and cyclical pattern detection
- Better for items with regular ordering cycles

### Negative Impacts

⚠️ **More False Alarms**
- 9,584 more false positives
- 52% increase in false alarms (18,410 → 27,994)
- May lead to slight over-ordering

⚠️ **Lower Precision**
- 2% drop in precision (95.6% → 93.6%)
- Still excellent, but slightly less reliable

## Recommendations

### 1. Keep DeepAR Enabled
- **Reason**: Better recall is more valuable than precision in healthcare supply chain
- **Impact**: Preventing stockouts is more critical than avoiding over-ordering
- **Trade-off**: Acceptable increase in false alarms for better availability

### 2. Consider Threshold Adjustment
- **Current**: 4 units
- **Suggestion**: Test 5 units to reduce false alarms
- **Expected**: Improve precision while maintaining good recall

### 3. Volume-Specific Strategies
- **Very Low Volume**: Consider higher threshold or LightGBM-only
- **Low-Medium-High Volume**: Keep ensemble as-is

### 4. Monitor Customer-Specific Performance
- Check if some customers benefit more from DeepAR
- Consider customer-specific ensemble weights
- Analyze which items benefit most from time series patterns

### 5. Optimize Ensemble Weights
- **Current**: 95% LightGBM, 5% DeepAR
- **Test**: 90% LightGBM, 10% DeepAR for more DeepAR influence
- **Test**: 98% LightGBM, 2% DeepAR for less false alarms

## Conclusion

**Verdict**: ✅ **Keep DeepAR Enabled**

The ensemble with DeepAR provides:
- ✅ Better recall (94.3% vs 92.8%)
- ✅ Fewer missed orders (21% reduction)
- ✅ Better low-volume detection
- ⚠️ Acceptable trade-off in precision (93.6% vs 95.6%)

For healthcare supply chain, **preventing stockouts is more critical than avoiding over-ordering**, making the improved recall worth the slight decrease in precision.

### Next Steps

1. ✅ Deploy with DeepAR enabled
2. Monitor real-world performance
3. Test threshold adjustments (4 → 5 units)
4. Analyze customer-specific patterns
5. Consider volume-specific strategies
6. Fine-tune ensemble weights based on feedback

---

**Generated**: 2025-11-12  
**Test Period**: 2025-10-26 to 2025-11-08 (14 days)  
**Items Validated**: 572,670  
**Customer**: ScionHealth
