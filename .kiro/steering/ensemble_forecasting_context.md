---
inclusion: always
---

# Ensemble Forecasting System - Context and Guidelines

## System Overview

This is an ensemble forecasting system that combines LightGBM (95%) and DeepAR (5%) models to predict inventory reorder needs for healthcare supply chain management.

## Key Architecture Decisions

### Context Length Strategy
- **Extended Context:** 90 days total extraction (62 days context + 14 days test + 14 days validation)
- **Rationale:** Provides sufficient historical data for seasonal pattern detection and stable trend analysis
- **Impact:** Significantly improves model accuracy, especially for seasonal and regular ordering patterns

### Model Ensemble Weights
- **LightGBM: 95%** - Primary model for structured feature-based predictions
- **DeepAR: 5%** - Secondary model for time series patterns and seasonality
- **Custom Endpoint:** `hybrent-nov` in `us-east-1` region

### Customer-Specific Calibration
- **ScionHealth:** 1.05x multiplier (slight over-prediction to reduce stockouts)
- **Mercy:** 0.85x multiplier (reduce over-ordering tendency)
- **Rationale:** Account for customer-specific ordering behaviors and risk tolerance

## Configuration Philosophy

### Environment-Driven Configuration
- All parameters configurable via `.env` file
- 80+ configurable parameters covering all aspects
- **Zero hardcoded values** - all defaults in `config_defaults.py`
- Easy A/B testing and optimization
- Centralized default management

### Context Length Optimization
- **Short Context (7-30 days):** Fast-moving items, recent trends
- **Long Context (60-90 days):** Seasonal items, stable patterns
- **Current Setting:** 90 days total for optimal balance

### Feature Engineering Strategy
- **Rolling Windows:** 7-day (short-term) and 30-day (long-term) patterns
- **Lag Features:** 7, 14, 30 days back for trend detection
- **Seasonal Features:** Weekly patterns, day-of-week effects
- **Business Features:** Customer, facility, vendor, price volatility

## Testing and Validation Framework

### Forward-Looking Validation
- **Test Period:** Historical data for feature engineering
- **Validation Period:** Future data for accuracy measurement
- **No Data Leakage:** Model never sees validation data during training

### Comprehensive Metrics
- **Regression:** MAE, RMSE, MAPE, R²
- **Classification:** Precision, Recall, F1 (reorder threshold: 5 units)
- **Business:** Volume errors, customer-facility accuracy

### Performance Expectations
- **Overall MAE:** 4-6 units (with extended context)
- **Precision:** >90% (low false alarm rate)
- **Best Performance:** Medium-volume items (5-20 units)
- **Challenge Areas:** Very low volume (0-5 units) and sporadic orders

## Development Guidelines

### Code Organization
- **Configuration:** `env_config.py` - centralized environment management
- **Data Processing:** `data_loader.py` - feature engineering and preparation
- **Models:** Separate LightGBM and DeepAR prediction modules
- **Testing:** Comprehensive test suite in `test/` directory

### Configuration Management
- **No Hardcoded Values:** All defaults centralized in `config_defaults.py`
- **Environment Override:** `.env` file values take precedence over defaults
- **Easy Updates:** Use `configure.py` script for parameter changes
- **Validation:** Always validate configuration changes with test runs
- **Documentation:** Document configuration rationale in steering files
- **Environment-Specific:** Use different `.env` files for different deployments

### Performance Optimization
- **Batch Processing:** Configurable batch sizes for memory management
- **Parallel Processing:** Configurable worker counts
- **Caching:** Enable for repeated operations
- **Memory Management:** Configurable limits and chunk processing

## Business Context

### Healthcare Supply Chain Specifics
- **Critical Items:** Cannot afford stockouts (apply safety multipliers)
- **Cost Sensitivity:** Minimize over-ordering while ensuring availability
- **Regulatory Requirements:** Maintain audit trails and explainable predictions
- **Customer Diversity:** Different ordering patterns require customization

### Risk Management
- **Conservative Approach:** High precision (few false alarms) preferred over high recall
- **Safety Stock:** Apply volume-based multipliers (2.0x for low volume, 1.2x for high volume)
- **Manual Review:** Flag high-value or high-volume predictions for review

## Deployment Considerations

### Production Readiness Checklist
- [ ] Extended context (90 days) configured and tested
- [ ] Customer calibration factors validated
- [ ] DeepAR endpoint accessible and performing
- [ ] AWS credentials and permissions configured
- [ ] Performance metrics meet business requirements
- [ ] Safety stock multipliers appropriate for risk tolerance

### Monitoring and Maintenance
- **Daily:** Quick validation on new data
- **Weekly:** Full test pipeline with latest data
- **Monthly:** Comprehensive analysis and model retraining consideration
- **Quarterly:** Review customer calibration factors and ensemble weights

### Continuous Improvement
- **A/B Testing:** Use configurable parameters to test improvements
- **Customer Feedback:** Incorporate business user feedback into calibration
- **Seasonal Adjustments:** Monitor and adjust for seasonal business patterns
- **Model Updates:** Regular evaluation of ensemble weights and feature importance

## Common Patterns and Solutions

### When Accuracy is Poor
1. **Check Context Length:** Ensure sufficient historical data (60+ days)
2. **Review Customer Calibration:** Adjust multipliers based on actual vs predicted
3. **Analyze Volume Categories:** Different strategies for low/medium/high volume
4. **Feature Engineering:** Add domain-specific features (promotions, seasonality)

### When Over/Under Predicting
1. **Global Calibration:** Adjust `GLOBAL_CALIBRATION_MULTIPLIER`
2. **Customer-Specific:** Update `CUSTOMER_CALIBRATION` ratios
3. **Safety Stock:** Modify volume-based multipliers
4. **Ensemble Weights:** Consider adjusting LightGBM/DeepAR balance

### When Performance Varies by Customer
1. **Segmentation:** Consider customer-specific models
2. **Feature Engineering:** Add customer-specific features
3. **Calibration:** Fine-tune customer multipliers
4. **Manual Review:** Flag problematic customers for special handling

## Integration Guidelines

### API Integration
- Use configurable endpoints and timeouts
- Implement retry logic with exponential backoff
- Cache predictions appropriately
- Provide confidence intervals and explanations

### Data Pipeline Integration
- Ensure data freshness and quality
- Implement data validation and anomaly detection
- Handle missing data gracefully
- Maintain audit trails for regulatory compliance

### Business System Integration
- Provide actionable recommendations (not just predictions)
- Include safety stock calculations
- Support manual overrides and adjustments
- Generate business-friendly reports and dashboards

## Success Metrics

### Technical Metrics
- **Accuracy:** MAE < 6 units, MAPE < 50% for medium volume
- **Reliability:** >99% uptime, <5 second response time
- **Scalability:** Handle 500K+ predictions per day

### Business Metrics
- **Stockout Reduction:** <2% stockout rate for critical items
- **Inventory Optimization:** 10-15% reduction in excess inventory
- **Cost Savings:** Measurable reduction in carrying costs and expedited shipping

### User Adoption
- **Trust:** High user confidence in predictions
- **Adoption:** >80% of recommendations followed
- **Feedback:** Positive user feedback and continuous improvement suggestions

---

This steering document should be updated as the system evolves and new insights are gained from production usage.