"""
LightGBM Model Training Script
==============================
Complete training pipeline for order prediction model
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime, timedelta
import json
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import env_config
warnings.filterwarnings('ignore')

# Load configuration from env_config
DATA_PATH = env_config.TRAINING_DATA_PATH or env_config.SOURCE_DATA_FILE
MODEL_OUTPUT = env_config.MODEL_OUTPUT_PATH
CONFIG_OUTPUT = env_config.CONFIG_OUTPUT_PATH
CHUNK_SIZE = env_config.TRAINING_CHUNK_SIZE
MAX_CHUNKS = env_config.TRAINING_MAX_CHUNKS
TEST_DAYS = env_config.TRAINING_TEST_DAYS
VALIDATION_SPLIT = env_config.TRAINING_VALIDATION_SPLIT
RANDOM_SEED = env_config.RANDOM_SEED

def load_data():
    """Load and prepare data from CSV"""
    print("🔄 Loading data...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    chunks = []
    total_records = 0
    
    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE), 1):
        if i % 20 == 0:
            print(f"   Processing chunk {i} ({total_records:,} records)")
        
        chunk['CreateDate'] = pd.to_datetime(chunk['CreateDate'], errors='coerce')
        chunk = chunk.dropna(subset=['CreateDate'])
        
        if len(chunk) > 0:
            chunks.append(chunk)
            total_records += len(chunk)
        
        if i >= MAX_CHUNKS:
            break
    
    df = pd.concat(chunks, ignore_index=True)
    
    print(f"✅ Loaded {len(df):,} records")
    print(f"📅 Date range: {df['CreateDate'].min().date()} to {df['CreateDate'].max().date()}")
    
    return df

def create_features(df):
    """Create model features"""
    print("🔧 Creating features...")
    
    # Aggregate to daily level
    df_agg = df.groupby(['CustomerID', 'FacilityID', 'ProductID', 'CreateDate']).agg({
        'OrderUnits': 'sum',
        'Price': 'mean'
    }).reset_index()
    
    df_agg = df_agg.rename(columns={'OrderUnits': 'target_value'})
    df_agg = df_agg.sort_values(['CustomerID', 'FacilityID', 'ProductID', 'CreateDate'])
    
    # Entity encodings
    for entity in ['CustomerID', 'FacilityID', 'ProductID']:
        df_agg[f'{entity}_encoded'] = pd.Categorical(df_agg[entity]).codes
    
    # Calendar features
    df_agg['day_of_week'] = df_agg['CreateDate'].dt.dayofweek
    df_agg['day_of_month'] = df_agg['CreateDate'].dt.day
    df_agg['month'] = df_agg['CreateDate'].dt.month
    df_agg['quarter'] = df_agg['CreateDate'].dt.quarter
    df_agg['year'] = df_agg['CreateDate'].dt.year
    df_agg['is_month_end'] = (df_agg['CreateDate'].dt.day >= 28).astype(int)
    df_agg['is_quarter_end'] = ((df_agg['CreateDate'].dt.month.isin([3,6,9,12])) & 
                                (df_agg['CreateDate'].dt.day >= 28)).astype(int)
    
    # Rolling features
    print("📈 Creating rolling features...")
    
    def add_rolling_features(group):
        group = group.sort_values('CreateDate')
        
        # Rolling statistics
        group['rolling_mean_7d'] = group['target_value'].rolling(window=7, min_periods=1).mean()
        group['rolling_std_7d'] = group['target_value'].rolling(window=7, min_periods=1).std().fillna(0)
        group['rolling_mean_30d'] = group['target_value'].rolling(window=30, min_periods=1).mean()
        group['rolling_std_30d'] = group['target_value'].rolling(window=30, min_periods=1).std().fillna(0)
        group['rolling_mean_90d'] = group['target_value'].rolling(window=90, min_periods=1).mean()
        group['rolling_mean_365d'] = group['target_value'].rolling(window=365, min_periods=1).mean()
        
        # Lag features
        group['lag_7'] = group['target_value'].shift(7)
        group['lag_14'] = group['target_value'].shift(14)
        group['lag_30'] = group['target_value'].shift(30)
        group['lag_365'] = group['target_value'].shift(365)
        
        return group
    
    df_agg = df_agg.groupby(['CustomerID', 'FacilityID', 'ProductID']).apply(
        add_rolling_features
    ).reset_index(drop=True)
    
    # Fill missing lags
    df_agg['lag_7'] = df_agg['lag_7'].fillna(df_agg['rolling_mean_7d'])
    df_agg['lag_14'] = df_agg['lag_14'].fillna(df_agg['rolling_mean_7d'])
    df_agg['lag_30'] = df_agg['lag_30'].fillna(df_agg['rolling_mean_30d'])
    df_agg['lag_365'] = df_agg['lag_365'].fillna(df_agg['rolling_mean_365d'])
    
    # Business features
    df_agg['price_volatility'] = df_agg.groupby(['CustomerID', 'FacilityID', 'ProductID'])['Price'].transform(
        lambda x: x.rolling(window=30, min_periods=1).std().fillna(0.15)
    )
    df_agg['seasonal_trend'] = 1.0 + 0.05 * np.sin(2 * np.pi * df_agg['CreateDate'].dt.dayofyear / 365)
    df_agg['seasonal_seasonal'] = 1.0 + 0.03 * np.cos(2 * np.pi * df_agg['CreateDate'].dt.dayofyear / 365)
    df_agg['order_frequency'] = df_agg.groupby(['CustomerID', 'FacilityID', 'ProductID']).cumcount() / 365.0
    df_agg['order_frequency'] = df_agg['order_frequency'].clip(0, 2)
    df_agg['vendor_reliability'] = 0.92
    
    print(f"✅ Created {len(df_agg):,} feature records")
    
    return df_agg

def train_model(df_features):
    """Train LightGBM model"""
    print("🤖 Training model...")
    
    # Train/test split
    max_date = df_features['CreateDate'].max()
    cutoff_date = max_date - timedelta(days=TEST_DAYS)
    
    train_df = df_features[df_features['CreateDate'] < cutoff_date].copy()
    test_df = df_features[df_features['CreateDate'] >= cutoff_date].copy()
    
    # Remove missing lags from training
    train_df = train_df.dropna(subset=['lag_7', 'lag_14', 'lag_30'])
    
    print(f"📊 Train: {len(train_df):,}, Test: {len(test_df):,}")
    
    # Feature columns
    feature_columns = [
        'rolling_mean_7d', 'rolling_std_7d', 'rolling_mean_30d', 'rolling_std_30d',
        'rolling_mean_90d', 'rolling_mean_365d',
        'lag_7', 'lag_14', 'lag_30', 'lag_365',
        'seasonal_trend', 'seasonal_seasonal',
        'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
        'is_month_end', 'is_quarter_end', 'price_volatility',
        'order_frequency', 'vendor_reliability',
        'CustomerID_encoded', 'FacilityID_encoded', 'ProductID_encoded'
    ]
    
    X_train = train_df[feature_columns]
    y_train = train_df['target_value']
    
    # Validation split
    val_size = int(VALIDATION_SPLIT * len(X_train))
    X_val = X_train.iloc[-val_size:]
    y_val = y_train.iloc[-val_size:]
    X_train_fit = X_train.iloc[:-val_size]
    y_train_fit = y_train.iloc[:-val_size]
    
    # LightGBM datasets
    train_data = lgb.Dataset(X_train_fit, label=y_train_fit)
    valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Model parameters from configuration
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': env_config.LGBM_NUM_LEAVES,
        'learning_rate': env_config.LGBM_LEARNING_RATE,
        'feature_fraction': env_config.LGBM_FEATURE_FRACTION,
        'bagging_fraction': env_config.LGBM_BAGGING_FRACTION,
        'bagging_freq': env_config.LGBM_BAGGING_FREQ,
        'verbose': 0,
        'random_state': RANDOM_SEED,
        'min_child_samples': env_config.LGBM_MIN_CHILD_SAMPLES,
        'reg_alpha': env_config.LGBM_REG_ALPHA,
        'reg_lambda': env_config.LGBM_REG_LAMBDA,
        'max_depth': env_config.LGBM_MAX_DEPTH
    }
    
    # Train
    print("🚀 Training...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        num_boost_round=env_config.LGBM_NUM_BOOST_ROUND,
        callbacks=[
            lgb.early_stopping(stopping_rounds=env_config.LGBM_EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(200)
        ]
    )
    
    print(f"✅ Training complete! Best iteration: {model.best_iteration}")
    
    # Evaluate on test set
    test_df = test_df.copy()
    test_df['lag_7'] = test_df['lag_7'].fillna(test_df['rolling_mean_7d'])
    test_df['lag_14'] = test_df['lag_14'].fillna(test_df['rolling_mean_7d'])
    test_df['lag_30'] = test_df['lag_30'].fillna(test_df['rolling_mean_30d'])
    test_df['lag_365'] = test_df['lag_365'].fillna(test_df['rolling_mean_365d'])
    
    X_test = test_df[feature_columns]
    y_test = test_df['target_value']
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Test Results:")
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   R²: {r2:.4f}")
    
    return model, feature_columns, max_date, cutoff_date

def save_model(model, feature_columns, max_date, cutoff_date):
    """Save trained model and configuration"""
    print("💾 Saving model...")
    
    # Save model
    with open(MODEL_OUTPUT, 'wb') as f:
        pickle.dump(model, f)
    
    # Save config
    config = {
        "target_column": "target_value",
        "timestamp_column": "CreateDate",
        "item_id_column": "ProductID",
        "features": feature_columns,
        "max_date": max_date.strftime('%Y-%m-%d'),
        "cutoff_date": cutoff_date.strftime('%Y-%m-%d'),
        "model_info": {
            "best_iteration": model.best_iteration,
            "num_features": len(feature_columns),
            "data_source": DATA_PATH
        }
    }
    
    with open(CONFIG_OUTPUT, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Saved: {MODEL_OUTPUT}")
    print(f"✅ Saved: {CONFIG_OUTPUT}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print(f"\n🔍 Top 10 Features:")
    print(importance_df.head(10).to_string(index=False))

def main():
    """Main training pipeline"""
    print("🚀 LightGBM Model Training")
    print("=" * 50)
    
    # Load data
    df = load_data()
    
    # Create features
    df_features = create_features(df)
    
    # Train model
    model, feature_columns, max_date, cutoff_date = train_model(df_features)
    
    # Save model
    save_model(model, feature_columns, max_date, cutoff_date)
    
    print(f"\n🎉 Training Complete!")

if __name__ == "__main__":
    main()
