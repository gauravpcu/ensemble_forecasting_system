# Test Run Summary - ScionHealth Oct 26, 2025

## Command
```bash
python3 scripts/predict.py --date 2025-10-26 --customers scionhealth --days 14
```

## Parameters
- **Customer:** ScionHealth
- **Start Date:** October 26, 2025
- **Prediction Days:** 14
- **Context Length:** 90 days

## Date Breakdown

### Prediction Period (14 days)
```
Start Date:  2025-10-26 (October 26, 2025)
End Date:    2025-11-08 (November 8, 2025)
Duration:    14 days
```

**Dates being predicted:**
```
2025-10-26 (Day 1)
2025-10-27 (Day 2)
2025-10-28 (Day 3)
2025-10-29 (Day 4)
2025-10-30 (Day 5)
2025-10-31 (Day 6)
2025-11-01 (Day 7)
2025-11-02 (Day 8)
2025-11-03 (Day 9)
2025-11-04 (Day 10)
2025-11-05 (Day 11)
2025-11-06 (Day 12)
2025-11-07 (Day 13)
2025-11-08 (Day 14)
```

### Context Period (90 days BEFORE prediction)
```
Start Date:  2025-07-28 (July 28, 2025)
End Date:    2025-10-25 (October 25, 2025)
Duration:    90 days
```

**Context ends 1 day BEFORE prediction starts:**
```
Context Last Day:     Oct 25, 2025
Prediction First Day: Oct 26, 2025
Gap:                  No overlap ✓
Data Leakage:         None ✓
```

## Timeline Visualization

```
July 2025          August 2025        September 2025     October 2025       November 2025
│                  │                  │                  │                  │
│                  │                  │                  │                  │
├─ Jul 28          │                  │                  │                  │
│  (Context        │                  │                  │                  │
│   Start)         │                  │                  │                  │
│                  │                  │                  │                  │
│◄─────────────────────────────────────────────────────►│                  │
│                                                        │                  │
│              90 Days Context Period                   │                  │
│                                                        │                  │
│                                                   Oct 25                  │
│                                                   (Context                │
│                                                    End)                   │
│                                                        │                  │
│                                                        │ Oct 26           │
│                                                        │ (Prediction      │
│                                                        │  Start)          │
│                                                        │                  │
│                                                        │◄────────────────►│
│                                                        │                  │
│                                                        │  14 Days         │
│                                                        │  Prediction      │
│                                                        │                  │
│                                                        │             Nov 8│
│                                                        │             (End)│
```

## Data Statistics

### Source Data
- **File:** order_history_2025-11-08.csv
- **Total Records:** 8,474,062
- **ScionHealth Records:** 2,688,390

### Context Data (Jul 28 - Oct 25)
- **Records:** 276,995
- **Item-Date Combinations:** 228,338
- **Unique Items:** 33,596

### Predictions Generated
- **Total Items:** 33,596
- **Items to Order (≥4):** ~12,500
- **Total Units:** ~270,000

## Verification

### No Data Leakage ✓
```
Context Period:     Jul 28 - Oct 25 (90 days)
Prediction Period:  Oct 26 - Nov 8  (14 days)
Overlap:            NONE ✓
```

### Context Length ✓
```
Start:    Jul 28, 2025
End:      Oct 25, 2025
Duration: 90 days ✓
```

### Prediction Range ✓
```
Start:    Oct 26, 2025
End:      Nov 8, 2025
Duration: 14 days ✓
```

## Model Configuration

### Ensemble Weights
- **LightGBM:** 95%
- **DeepAR:** 5%

### Features Used
- 25 features total
- Rolling averages (7-day, 30-day)
- Lag features (7, 14, 30 days)
- Seasonal patterns
- Business metrics

### Classification
- **Threshold:** ≥4 units
- **Recommendation:** ORDER / NO ORDER

## Summary

✅ **Customer:** ScionHealth
✅ **Prediction Period:** Oct 26 - Nov 8, 2025 (14 days)
✅ **Context Period:** Jul 28 - Oct 25, 2025 (90 days)
✅ **No Data Leakage:** Context ends before prediction starts
✅ **Items Predicted:** 33,596
✅ **Model:** LightGBM (95%) + DeepAR (5%)

**All dates verified and correct!** 🎉
