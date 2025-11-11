# Precision and Recall - Complete Explanation

## The Problem

We're predicting whether items need to be reordered. For each item, we make a prediction:
- **"ORDER"** (predicted_reorder = 1) - We think this item needs reordering
- **"DON'T ORDER"** (predicted_reorder = 0) - We think this item doesn't need reordering

Then we compare with reality (actual orders from validation data).

## The Four Outcomes (Confusion Matrix)

```
                        ACTUAL REALITY
                    Needs Order    Doesn't Need Order
                    (Actual = 1)   (Actual = 0)
                    ─────────────────────────────────
PREDICTION         │
Says "ORDER"       │  TRUE          FALSE
(Predicted = 1)    │  POSITIVE      POSITIVE
                   │  (TP)          (FP)
                   │  ✅ Correct    ❌ False Alarm
                   │
Says "DON'T ORDER" │  FALSE         TRUE
(Predicted = 0)    │  NEGATIVE      NEGATIVE
                   │  (FN)          (TN)
                   │  ❌ Missed     ✅ Correct
```

### Definitions

1. **True Positive (TP)** ✅
   - We said "ORDER" and it actually needed ordering
   - **Correct prediction**

2. **False Positive (FP)** ❌
   - We said "ORDER" but it didn't actually need ordering
   - **False alarm / Over-ordering**

3. **False Negative (FN)** ❌
   - We said "DON'T ORDER" but it actually needed ordering
   - **Missed order / Stockout risk**

4. **True Negative (TN)** ✅
   - We said "DON'T ORDER" and it didn't need ordering
   - **Correct prediction**

## Precision Formula

```
Precision = TP / (TP + FP)
```

**What it measures:** "When we predict ORDER, how often are we correct?"

**In other words:** Of all the items we recommended to order, what percentage actually needed ordering?

### ScionHealth Example (92.5% Precision)

```
Items we predicted to ORDER: 4,881
  ├─ True Positives (TP):   4,513 ✅ (actually needed ordering)
  └─ False Positives (FP):    368 ❌ (didn't need ordering - false alarms)

Precision = 4,513 / (4,513 + 368)
Precision = 4,513 / 4,881
Precision = 0.9246 = 92.46%
```

**Interpretation:** When the model says "ORDER" for ScionHealth, it's correct 92.5% of the time. Only 7.5% are false alarms.

### Mercy Example (45.9% Precision)

```
Items we predicted to ORDER: 697
  ├─ True Positives (TP):   320 ✅ (actually needed ordering)
  └─ False Positives (FP):  377 ❌ (didn't need ordering - false alarms)

Precision = 320 / (320 + 377)
Precision = 320 / 697
Precision = 0.4591 = 45.91%
```

**Interpretation:** When the model says "ORDER" for Mercy, it's only correct 46% of the time. 54% are false alarms!

## Recall Formula

```
Recall = TP / (TP + FN)
```

**What it measures:** "Of all items that actually needed ordering, how many did we catch?"

**In other words:** What percentage of actual orders did we successfully predict?

### ScionHealth Example (50.9% Recall)

```
Items that ACTUALLY needed ordering: 8,858
  ├─ True Positives (TP):   4,513 ✅ (we predicted correctly)
  └─ False Negatives (FN):  4,345 ❌ (we missed these)

Recall = 4,513 / (4,513 + 4,345)
Recall = 4,513 / 8,858
Recall = 0.5095 = 50.95%
```

**Interpretation:** The model catches 51% of items that need ordering. It misses 49% (potential stockouts).

### Mercy Example (28.5% Recall)

```
Items that ACTUALLY needed ordering: 1,124
  ├─ True Positives (TP):   309 ✅ (we predicted correctly)
  └─ False Negatives (FN):  815 ❌ (we missed these)

Recall = 309 / (309 + 815)
Recall = 309 / 1,124
Recall = 0.2847 = 28.47%
```

**Interpretation:** The model only catches 28.5% of items that need ordering. It misses 71.5%!

## Visual Explanation

### ScionHealth (Good Precision, Moderate Recall)

```
All Items: 13,874
├─ Items that DON'T need ordering: 5,016
│  ├─ We correctly said "DON'T ORDER": 4,648 ✅ (TN)
│  └─ We wrongly said "ORDER": 368 ❌ (FP) - False alarms
│
└─ Items that DO need ordering: 8,858
   ├─ We correctly said "ORDER": 4,513 ✅ (TP) - Caught!
   └─ We wrongly said "DON'T ORDER": 4,345 ❌ (FN) - Missed!

Precision = 4,513 / (4,513 + 368) = 92.5%
  → When we say "ORDER", we're right 92.5% of the time

Recall = 4,513 / (4,513 + 4,345) = 50.9%
  → We catch 50.9% of items that need ordering
```

### Mercy (Poor Precision, Poor Recall)

```
All Items: 4,031
├─ Items that DON'T need ordering: 2,907
│  ├─ We correctly said "DON'T ORDER": 2,531 ✅ (TN)
│  └─ We wrongly said "ORDER": 376 ❌ (FP) - False alarms
│
└─ Items that DO need ordering: 1,124
   ├─ We correctly said "ORDER": 309 ✅ (TP) - Caught!
   └─ We wrongly said "DON'T ORDER": 815 ❌ (FN) - Missed!

Precision = 309 / (309 + 376) = 45.9%
  → When we say "ORDER", we're only right 46% of the time

Recall = 309 / (309 + 815) = 28.5%
  → We only catch 28.5% of items that need ordering
```

## The Trade-off

**High Precision vs High Recall** - You usually can't have both!

### High Precision (Conservative)
- **Fewer predictions** → Fewer false alarms
- **More reliable** when you do predict
- **Risk:** Miss many orders (low recall)
- **Example:** ScionHealth (92.5% precision, 50.9% recall)

### High Recall (Aggressive)
- **More predictions** → Catch more orders
- **More false alarms** (lower precision)
- **Risk:** Over-ordering and waste
- **Example:** Original Mercy (35.1% recall, but only 27% precision)

## Business Impact

### High Precision (ScionHealth: 92.5%)
**Benefits:**
- ✅ Very trustworthy recommendations
- ✅ Low waste (only 368 false alarms)
- ✅ High confidence in predictions

**Drawbacks:**
- ⚠️ Misses 49% of orders (need safety stock)
- ⚠️ Potential stockouts

**Solution:** Use 50-60% safety stock for missed items

### Low Precision (Mercy: 45.9%)
**Problems:**
- ❌ 54% of recommendations are wrong
- ❌ 377 false alarms (unnecessary orders)
- ❌ Low confidence in predictions
- ❌ Waste and excess inventory

**Solution:** Reduce calibration to improve precision

## How to Improve

### To Improve Precision (Reduce False Alarms)
1. **Lower calibration multiplier**
   - Mercy: 0.362 → 0.18 improved precision from 27% to 46%
2. **Increase classification threshold**
   - Current: 4 units → Try 5-6 units
3. **Better feature engineering**
   - Add more predictive features
   - Remove noisy features

### To Improve Recall (Catch More Orders)
1. **Increase calibration multiplier**
   - But this usually reduces precision
2. **Lower classification threshold**
   - Current: 4 units → Try 2-3 units
3. **Add safety stock**
   - Compensate for missed orders
   - 50-60% safety stock recommended

## F1 Score - The Balance

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**What it measures:** Harmonic mean of precision and recall

**ScionHealth:**
```
F1 = 2 × (0.925 × 0.509) / (0.925 + 0.509)
F1 = 2 × 0.471 / 1.434
F1 = 0.657
```

**Mercy (with 0.18 calibration):**
```
F1 = 2 × (0.459 × 0.285) / (0.459 + 0.285)
F1 = 2 × 0.131 / 0.744
F1 = 0.351
```

Higher F1 = Better balance between precision and recall

## Real-World Example

Imagine you're ordering medical supplies:

### Scenario 1: High Precision (ScionHealth)
```
Model says: "Order these 100 items"
Reality: 92 actually need ordering, 8 don't

Result:
✅ 92 correct orders (no stockouts for these)
❌ 8 unnecessary orders (small waste)
⚠️ But you missed 90 other items that needed ordering!
```

### Scenario 2: Low Precision (Original Mercy)
```
Model says: "Order these 100 items"
Reality: 27 actually need ordering, 73 don't!

Result:
✅ 27 correct orders
❌ 73 unnecessary orders (huge waste!)
⚠️ And you still missed 97 other items that needed ordering!
```

## Summary Table

| Metric | Formula | ScionHealth | Mercy (0.18) | What It Means |
|--------|---------|-------------|--------------|---------------|
| **Precision** | TP/(TP+FP) | 92.5% ✅ | 45.9% ⚠️ | "When I say ORDER, am I right?" |
| **Recall** | TP/(TP+FN) | 50.9% ⚠️ | 28.5% ❌ | "Do I catch most orders?" |
| **F1 Score** | 2×P×R/(P+R) | 0.657 ✅ | 0.351 ⚠️ | "Overall balance" |
| **Accuracy** | (TP+TN)/Total | 66.0% | 70.7% | "Overall correctness" |

## Key Takeaways

1. **Precision** = Reliability of positive predictions
   - High precision = Few false alarms
   - Low precision = Many false alarms

2. **Recall** = Coverage of actual positives
   - High recall = Catch most orders
   - Low recall = Miss many orders

3. **Trade-off** = Can't maximize both
   - Conservative (high precision) → Miss orders
   - Aggressive (high recall) → Many false alarms

4. **Business Decision**
   - Healthcare: Prefer high precision (avoid waste)
   - Use safety stock to compensate for low recall
   - ScionHealth: Good balance (92.5% precision, 51% recall)
   - Mercy: Needs improvement (46% precision, 29% recall)

5. **Target for Production**
   - Precision: ≥ 60% (preferably 80%+)
   - Recall: ≥ 50%
   - F1 Score: ≥ 0.60
