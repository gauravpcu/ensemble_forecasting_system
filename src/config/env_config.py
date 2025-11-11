"""
Environment Configuration Loader
Loads configuration from .env file with fallback to defaults
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
from src.config.config_defaults import *

# Load .env file if it exists
env_path = Path('.env')
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ Loaded configuration from .env")
else:
    print(f"⚠️  No .env file found, using defaults (copy .env.example to .env to customize)")


def get_env(key: str, default: any = None, cast_type: type = str) -> any:
    """Get environment variable with type casting and default"""
    value = os.getenv(key)
    
    if value is None or value == '':
        return default
    
    # Type casting
    if cast_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on')
    elif cast_type == int:
        return int(value)
    elif cast_type == float:
        return float(value)
    elif cast_type == list:
        return [item.strip() for item in value.split(',') if item.strip()]
    elif cast_type == dict:
        # Parse "key1:value1,key2:value2" format
        result = {}
        for pair in value.split(','):
            if ':' in pair:
                k, v = pair.split(':', 1)
                result[k.strip()] = float(v.strip())
        return result
    else:
        return value


# ============================================================================
# DATA PATHS
# ============================================================================

SOURCE_DATA_FILE = get_env('SOURCE_DATA_FILE', DEFAULT_SOURCE_DATA_FILE)
DATA_DIR = get_env('DATA_DIR', DEFAULT_DATA_DIR)
TEST_DATA_DIR = get_env('TEST_DATA_DIR', DEFAULT_TEST_DATA_DIR)
RESULTS_DIR = get_env('RESULTS_DIR', DEFAULT_RESULTS_DIR)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

LIGHTGBM_MODEL_PATH = get_env('LIGHTGBM_MODEL_PATH', 
                             DEFAULT_LIGHTGBM_MODEL_PATH or os.path.join(DATA_DIR, 'lightgbm_model.pkl'))
FEATURE_CONFIG_PATH = get_env('FEATURE_CONFIG_PATH', 
                             DEFAULT_FEATURE_CONFIG_PATH or os.path.join(DATA_DIR, 'feature_config.json'))

DEEPAR_ENDPOINT_NAME = get_env('DEEPAR_ENDPOINT_NAME', DEFAULT_DEEPAR_ENDPOINT_NAME)
DEEPAR_REGION = get_env('DEEPAR_REGION', DEFAULT_DEEPAR_REGION)

# ============================================================================
# ENSEMBLE WEIGHTS
# ============================================================================

LIGHTGBM_WEIGHT = get_env('LIGHTGBM_WEIGHT', DEFAULT_LIGHTGBM_WEIGHT, float)
DEEPAR_WEIGHT = get_env('DEEPAR_WEIGHT', DEFAULT_DEEPAR_WEIGHT, float)

ENSEMBLE_WEIGHTS = {
    'lightgbm': LIGHTGBM_WEIGHT,
    'deepar': DEEPAR_WEIGHT
}

# ============================================================================
# DATA EXTRACTION SETTINGS
# ============================================================================

VALIDATION_DAYS = get_env('VALIDATION_DAYS', DEFAULT_VALIDATION_DAYS, int)
TEST_DAYS = get_env('TEST_DAYS', DEFAULT_TEST_DAYS, int)
TOTAL_EXTRACTION_DAYS = get_env('TOTAL_EXTRACTION_DAYS', DEFAULT_TOTAL_EXTRACTION_DAYS, int)

# ============================================================================
# CLASSIFICATION SETTINGS
# ============================================================================

CLASSIFICATION_THRESHOLD = get_env('CLASSIFICATION_THRESHOLD', DEFAULT_CLASSIFICATION_THRESHOLD, int)

# ============================================================================
# TESTING CONFIGURATION
# ============================================================================

FOCUSED_TEST_CUSTOMERS = get_env('FOCUSED_TEST_CUSTOMERS', DEFAULT_FOCUSED_TEST_CUSTOMERS, list)
TEST_SAMPLE_SIZE = get_env('TEST_SAMPLE_SIZE', DEFAULT_TEST_SAMPLE_SIZE, int)
VERBOSE = get_env('VERBOSE', DEFAULT_VERBOSE, bool)

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

MAPE_EXCELLENT = get_env('MAPE_EXCELLENT', DEFAULT_MAPE_EXCELLENT, float)
MAPE_GOOD = get_env('MAPE_GOOD', DEFAULT_MAPE_GOOD, float)
MAPE_FAIR = get_env('MAPE_FAIR', DEFAULT_MAPE_FAIR, float)

# ============================================================================
# AWS CONFIGURATION
# ============================================================================

AWS_ACCESS_KEY_ID = get_env('AWS_ACCESS_KEY_ID', DEFAULT_AWS_ACCESS_KEY_ID)
AWS_SECRET_ACCESS_KEY = get_env('AWS_SECRET_ACCESS_KEY', DEFAULT_AWS_SECRET_ACCESS_KEY)
AWS_DEFAULT_REGION = get_env('AWS_DEFAULT_REGION', DEFAULT_AWS_DEFAULT_REGION)

# ============================================================================
# BEDROCK LLM CONFIGURATION
# ============================================================================

USE_LLM_AUGMENTATION = get_env('USE_LLM_AUGMENTATION', DEFAULT_USE_LLM_AUGMENTATION, bool)
BEDROCK_REGION = get_env('BEDROCK_REGION', DEFAULT_BEDROCK_REGION)
BEDROCK_MODEL_ID = get_env('BEDROCK_MODEL_ID', DEFAULT_BEDROCK_MODEL_ID)
BEDROCK_MAX_RETRIES = get_env('BEDROCK_MAX_RETRIES', DEFAULT_BEDROCK_MAX_RETRIES, int)
BEDROCK_RETRY_DELAY = get_env('BEDROCK_RETRY_DELAY', DEFAULT_BEDROCK_RETRY_DELAY, int)
LLM_CONFIDENCE_THRESHOLD = get_env('LLM_CONFIDENCE_THRESHOLD', DEFAULT_LLM_CONFIDENCE_THRESHOLD, float)

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

BATCH_SIZE = get_env('BATCH_SIZE', DEFAULT_BATCH_SIZE, int)
NUM_WORKERS = get_env('NUM_WORKERS', DEFAULT_NUM_WORKERS, int)
MEMORY_LIMIT = get_env('MEMORY_LIMIT', DEFAULT_MEMORY_LIMIT, int)

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

SAVE_PREDICTIONS = get_env('SAVE_PREDICTIONS', DEFAULT_SAVE_PREDICTIONS, bool)
GENERATE_PLOTS = get_env('GENERATE_PLOTS', DEFAULT_GENERATE_PLOTS, bool)
OUTPUT_FORMAT = get_env('OUTPUT_FORMAT', DEFAULT_OUTPUT_FORMAT)
OUTPUT_COMPRESSION = get_env('OUTPUT_COMPRESSION', DEFAULT_OUTPUT_COMPRESSION)

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = get_env('LOG_LEVEL', DEFAULT_LOG_LEVEL)
LOG_FILE = get_env('LOG_FILE', DEFAULT_LOG_FILE)

# ============================================================================
# SAFETY MULTIPLIERS
# ============================================================================

SAFETY_MULTIPLIER_LOW = get_env('SAFETY_MULTIPLIER_LOW', DEFAULT_SAFETY_MULTIPLIER_LOW, float)
SAFETY_MULTIPLIER_MEDIUM = get_env('SAFETY_MULTIPLIER_MEDIUM', DEFAULT_SAFETY_MULTIPLIER_MEDIUM, float)
SAFETY_MULTIPLIER_HIGH = get_env('SAFETY_MULTIPLIER_HIGH', DEFAULT_SAFETY_MULTIPLIER_HIGH, float)

VOLUME_LOW_THRESHOLD = get_env('VOLUME_LOW_THRESHOLD', DEFAULT_VOLUME_LOW_THRESHOLD, int)
VOLUME_MEDIUM_THRESHOLD = get_env('VOLUME_MEDIUM_THRESHOLD', DEFAULT_VOLUME_MEDIUM_THRESHOLD, int)
VOLUME_HIGH_THRESHOLD = get_env('VOLUME_HIGH_THRESHOLD', DEFAULT_VOLUME_HIGH_THRESHOLD, int)

# ============================================================================
# CALIBRATION
# ============================================================================

GLOBAL_CALIBRATION_MULTIPLIER = get_env('GLOBAL_CALIBRATION_MULTIPLIER', DEFAULT_GLOBAL_CALIBRATION_MULTIPLIER, float)
CUSTOMER_CALIBRATION = get_env('CUSTOMER_CALIBRATION', DEFAULT_CUSTOMER_CALIBRATION, dict)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

ROLLING_WINDOW_SHORT = get_env('ROLLING_WINDOW_SHORT', DEFAULT_ROLLING_WINDOW_SHORT, int)
ROLLING_WINDOW_LONG = get_env('ROLLING_WINDOW_LONG', DEFAULT_ROLLING_WINDOW_LONG, int)
LAG_PERIODS = get_env('LAG_PERIODS', DEFAULT_LAG_PERIODS, list)
SEASONAL_PERIOD = get_env('SEASONAL_PERIOD', DEFAULT_SEASONAL_PERIOD, int)

# ============================================================================
# VALIDATION SETTINGS
# ============================================================================

MIN_RECORDS_FOR_ANALYSIS = get_env('MIN_RECORDS_FOR_ANALYSIS', DEFAULT_MIN_RECORDS_FOR_ANALYSIS, int)
MIN_VALUE_FOR_MAPE = get_env('MIN_VALUE_FOR_MAPE', DEFAULT_MIN_VALUE_FOR_MAPE, int)

# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================

ENABLE_EMAIL_NOTIFICATIONS = get_env('ENABLE_EMAIL_NOTIFICATIONS', False, bool)
EMAIL_SMTP_SERVER = get_env('EMAIL_SMTP_SERVER', None)
EMAIL_SMTP_PORT = get_env('EMAIL_SMTP_PORT', DEFAULT_EMAIL_SMTP_PORT, int)
EMAIL_FROM = get_env('EMAIL_FROM', None)
EMAIL_TO = get_env('EMAIL_TO', None)
EMAIL_USERNAME = get_env('EMAIL_USERNAME', None)
EMAIL_PASSWORD = get_env('EMAIL_PASSWORD', None)

ENABLE_SLACK_NOTIFICATIONS = get_env('ENABLE_SLACK_NOTIFICATIONS', False, bool)
SLACK_WEBHOOK_URL = get_env('SLACK_WEBHOOK_URL', None)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

PLOT_FIGURE_SIZE_WIDTH = get_env('PLOT_FIGURE_SIZE_WIDTH', DEFAULT_PLOT_FIGURE_SIZE_WIDTH, int)
PLOT_FIGURE_SIZE_HEIGHT = get_env('PLOT_FIGURE_SIZE_HEIGHT', DEFAULT_PLOT_FIGURE_SIZE_HEIGHT, int)
PLOT_SUBPLOT_WIDTH = get_env('PLOT_SUBPLOT_WIDTH', DEFAULT_PLOT_SUBPLOT_WIDTH, int)
PLOT_SUBPLOT_HEIGHT = get_env('PLOT_SUBPLOT_HEIGHT', DEFAULT_PLOT_SUBPLOT_HEIGHT, int)
PLOT_DPI = get_env('PLOT_DPI', DEFAULT_PLOT_DPI, int)
PLOT_ALPHA = get_env('PLOT_ALPHA', DEFAULT_PLOT_ALPHA, float)
PLOT_SCATTER_SIZE = get_env('PLOT_SCATTER_SIZE', DEFAULT_PLOT_SCATTER_SIZE, int)
PLOT_STYLE = get_env('PLOT_STYLE', DEFAULT_PLOT_STYLE)

# ============================================================================
# FILE PROCESSING SETTINGS
# ============================================================================

FILE_ENCODING = get_env('FILE_ENCODING', DEFAULT_FILE_ENCODING)
CSV_SEPARATOR = get_env('CSV_SEPARATOR', DEFAULT_CSV_SEPARATOR)
DECIMAL_PLACES = get_env('DECIMAL_PLACES', DEFAULT_DECIMAL_PLACES, int)
DATE_FORMAT = get_env('DATE_FORMAT', DEFAULT_DATE_FORMAT)
DATETIME_FORMAT = get_env('DATETIME_FORMAT', DEFAULT_DATETIME_FORMAT)

# ============================================================================
# DATA SAMPLING SETTINGS
# ============================================================================

DEFAULT_SAMPLE_RANDOM_STATE = get_env('DEFAULT_SAMPLE_RANDOM_STATE', DEFAULT_SAMPLE_RANDOM_STATE, int)
MAX_RECORDS_FOR_QUICK_TEST = get_env('MAX_RECORDS_FOR_QUICK_TEST', DEFAULT_MAX_RECORDS_FOR_QUICK_TEST, int)
PREDICTION_LENGTH_DAYS = get_env('PREDICTION_LENGTH_DAYS', DEFAULT_PREDICTION_LENGTH_DAYS, int)

# ============================================================================
# ROLLING STATISTICS SETTINGS
# ============================================================================

ROLLING_WINDOW_MIN_PERIODS = get_env('ROLLING_WINDOW_MIN_PERIODS', DEFAULT_ROLLING_WINDOW_MIN_PERIODS, int)
ROLLING_MAX_WINDOW_SIZE = get_env('ROLLING_MAX_WINDOW_SIZE', DEFAULT_ROLLING_MAX_WINDOW_SIZE, int)

# ============================================================================
# FILE SIZE AND MEMORY SETTINGS
# ============================================================================

MAX_FILE_SIZE_MB = get_env('MAX_FILE_SIZE_MB', DEFAULT_MAX_FILE_SIZE_MB, int)
CHUNK_SIZE_ROWS = get_env('CHUNK_SIZE_ROWS', DEFAULT_CHUNK_SIZE_ROWS, int)
MAX_MEMORY_USAGE_PCT = get_env('MAX_MEMORY_USAGE_PCT', DEFAULT_MAX_MEMORY_USAGE_PCT, int)

# ============================================================================
# TIMEOUT AND RETRY SETTINGS
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = get_env('DEFAULT_TIMEOUT_SECONDS', DEFAULT_TIMEOUT_SECONDS, int)
MAX_RETRY_ATTEMPTS = get_env('MAX_RETRY_ATTEMPTS', DEFAULT_MAX_RETRY_ATTEMPTS, int)
RETRY_DELAY_SECONDS = get_env('RETRY_DELAY_SECONDS', DEFAULT_RETRY_DELAY_SECONDS, int)
CONNECTION_TIMEOUT = get_env('CONNECTION_TIMEOUT', DEFAULT_CONNECTION_TIMEOUT, int)

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

RANDOM_SEED = get_env('RANDOM_SEED', DEFAULT_RANDOM_SEED, int)
ENABLE_CACHING = get_env('ENABLE_CACHING', DEFAULT_ENABLE_CACHING, bool)
CACHE_DIR = get_env('CACHE_DIR', DEFAULT_CACHE_DIR)
CACHE_MAX_AGE = get_env('CACHE_MAX_AGE', DEFAULT_CACHE_MAX_AGE, int)
ENABLE_PROFILING = get_env('ENABLE_PROFILING', DEFAULT_ENABLE_PROFILING, bool)
PROFILE_OUTPUT_DIR = get_env('PROFILE_OUTPUT_DIR', DEFAULT_PROFILE_OUTPUT_DIR)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_customer_calibration(customer_id: str) -> float:
    """Get calibration multiplier for a specific customer"""
    return CUSTOMER_CALIBRATION.get(customer_id, GLOBAL_CALIBRATION_MULTIPLIER)

def get_safety_multiplier(volume: float) -> float:
    """Get safety multiplier based on volume"""
    if volume < VOLUME_LOW_THRESHOLD:
        return SAFETY_MULTIPLIER_LOW
    elif volume < VOLUME_MEDIUM_THRESHOLD:
        return SAFETY_MULTIPLIER_MEDIUM
    elif volume < VOLUME_HIGH_THRESHOLD:
        return SAFETY_MULTIPLIER_HIGH
    else:
        return 1.0

def categorize_performance(mape: float) -> str:
    """Categorize performance based on MAPE"""
    if mape < MAPE_EXCELLENT:
        return 'Excellent'
    elif mape < MAPE_GOOD:
        return 'Good'
    elif mape < MAPE_FAIR:
        return 'Fair'
    else:
        return 'Poor'

def print_config_summary():
    """Print configuration summary"""
    print("\n" + "=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\nData Paths:")
    print(f"  Source File:     {SOURCE_DATA_FILE}")
    print(f"  Data Directory:  {DATA_DIR}")
    print(f"  Test Directory:  {TEST_DATA_DIR}")
    print(f"\nModel Configuration:")
    print(f"  LightGBM Weight: {LIGHTGBM_WEIGHT}")
    print(f"  DeepAR Weight:   {DEEPAR_WEIGHT}")
    print(f"  DeepAR Endpoint: {DEEPAR_ENDPOINT_NAME}")
    print(f"\nExtraction Settings:")
    print(f"  Validation Days: {VALIDATION_DAYS}")
    print(f"  Test Days:       {TEST_DAYS}")
    print(f"  Total Days:      {TOTAL_EXTRACTION_DAYS}")
    print(f"\nClassification:")
    print(f"  Threshold:       {CLASSIFICATION_THRESHOLD} units")
    print(f"\nPerformance Thresholds:")
    print(f"  Excellent:       < {MAPE_EXCELLENT}% MAPE")
    print(f"  Good:            < {MAPE_GOOD}% MAPE")
    print(f"  Fair:            < {MAPE_FAIR}% MAPE")
    print(f"\nVisualization Settings:")
    print(f"  Plot Size:       {PLOT_FIGURE_SIZE_WIDTH}x{PLOT_FIGURE_SIZE_HEIGHT}")
    print(f"  Plot DPI:        {PLOT_DPI}")
    print(f"  Plot Style:      {PLOT_STYLE}")
    print(f"\nProcessing Settings:")
    print(f"  Batch Size:      {BATCH_SIZE}")
    print(f"  Memory Limit:    {MEMORY_LIMIT}MB")
    print(f"  Timeout:         {DEFAULT_TIMEOUT_SECONDS}s")
    print(f"\nCalibration:")
    print(f"  Global:          {GLOBAL_CALIBRATION_MULTIPLIER}x")
    if CUSTOMER_CALIBRATION:
        print(f"  Customer-specific:")
        for customer, multiplier in CUSTOMER_CALIBRATION.items():
            print(f"    {customer}: {multiplier}x")
    print("=" * 80 + "\n")

def create_env_template():
    """Create a .env file from .env.example if it doesn't exist"""
    env_path = Path('.env')
    example_path = Path('.env.example')
    
    if not env_path.exists() and example_path.exists():
        import shutil
        shutil.copy(example_path, env_path)
        print(f"✓ Created .env file from .env.example")
        print(f"  Edit .env to customize your configuration")
        return True
    return False

def update_env_value(key: str, value: str):
    """Update a value in the .env file"""
    env_path = Path('.env')
    
    if not env_path.exists():
        if not create_env_template():
            print(f"❌ No .env file found and couldn't create from template")
            return False
    
    # Read current content
    lines = []
    key_found = False
    
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update or add the key
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{key}="):
            lines[i] = f"{key}={value}\n"
            key_found = True
            break
    
    if not key_found:
        lines.append(f"{key}={value}\n")
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(lines)
    
    print(f"✓ Updated {key}={value} in .env")
    return True

# ============================================================================
# MODEL TRAINING SETTINGS
# ============================================================================

TRAINING_DATA_PATH = get_env('TRAINING_DATA_PATH', DEFAULT_TRAINING_DATA_PATH, str)
MODEL_OUTPUT_PATH = get_env('MODEL_OUTPUT_PATH', DEFAULT_MODEL_OUTPUT_PATH, str)
CONFIG_OUTPUT_PATH = get_env('CONFIG_OUTPUT_PATH', DEFAULT_CONFIG_OUTPUT_PATH, str)
TRAINING_CHUNK_SIZE = get_env('TRAINING_CHUNK_SIZE', DEFAULT_TRAINING_CHUNK_SIZE, int)
TRAINING_MAX_CHUNKS = get_env('TRAINING_MAX_CHUNKS', DEFAULT_TRAINING_MAX_CHUNKS, int)
TRAINING_TEST_DAYS = get_env('TRAINING_TEST_DAYS', DEFAULT_TRAINING_TEST_DAYS, int)
TRAINING_VALIDATION_SPLIT = get_env('TRAINING_VALIDATION_SPLIT', DEFAULT_TRAINING_VALIDATION_SPLIT, float)

# LightGBM Training Parameters
LGBM_NUM_LEAVES = get_env('LGBM_NUM_LEAVES', DEFAULT_LGBM_NUM_LEAVES, int)
LGBM_LEARNING_RATE = get_env('LGBM_LEARNING_RATE', DEFAULT_LGBM_LEARNING_RATE, float)
LGBM_FEATURE_FRACTION = get_env('LGBM_FEATURE_FRACTION', DEFAULT_LGBM_FEATURE_FRACTION, float)
LGBM_BAGGING_FRACTION = get_env('LGBM_BAGGING_FRACTION', DEFAULT_LGBM_BAGGING_FRACTION, float)
LGBM_BAGGING_FREQ = get_env('LGBM_BAGGING_FREQ', DEFAULT_LGBM_BAGGING_FREQ, int)
LGBM_MIN_CHILD_SAMPLES = get_env('LGBM_MIN_CHILD_SAMPLES', DEFAULT_LGBM_MIN_CHILD_SAMPLES, int)
LGBM_REG_ALPHA = get_env('LGBM_REG_ALPHA', DEFAULT_LGBM_REG_ALPHA, float)
LGBM_REG_LAMBDA = get_env('LGBM_REG_LAMBDA', DEFAULT_LGBM_REG_LAMBDA, float)
LGBM_MAX_DEPTH = get_env('LGBM_MAX_DEPTH', DEFAULT_LGBM_MAX_DEPTH, int)
LGBM_NUM_BOOST_ROUND = get_env('LGBM_NUM_BOOST_ROUND', DEFAULT_LGBM_NUM_BOOST_ROUND, int)
LGBM_EARLY_STOPPING_ROUNDS = get_env('LGBM_EARLY_STOPPING_ROUNDS', DEFAULT_LGBM_EARLY_STOPPING_ROUNDS, int)

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration"""
    errors = []
    warnings = []
    
    # Check weights sum to 1.0
    weight_sum = LIGHTGBM_WEIGHT + DEEPAR_WEIGHT
    if not (0.99 <= weight_sum <= 1.01):
        errors.append(f"Ensemble weights must sum to 1.0 (current: {weight_sum})")
    
    # Check file paths exist
    if not os.path.exists(SOURCE_DATA_FILE):
        warnings.append(f"Source data file not found: {SOURCE_DATA_FILE}")
    
    if not os.path.exists(LIGHTGBM_MODEL_PATH):
        warnings.append(f"LightGBM model not found: {LIGHTGBM_MODEL_PATH}")
    
    # Check thresholds are in order
    if not (MAPE_EXCELLENT < MAPE_GOOD < MAPE_FAIR):
        errors.append("MAPE thresholds must be in ascending order")
    
    # Check volume thresholds
    if not (VOLUME_LOW_THRESHOLD < VOLUME_MEDIUM_THRESHOLD < VOLUME_HIGH_THRESHOLD):
        errors.append("Volume thresholds must be in ascending order")
    
    # Print results
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  • {error}")
    
    if warnings:
        print("\n⚠️  Configuration Warnings:")
        for warning in warnings:
            print(f"  • {warning}")
    
    if not errors and not warnings:
        print("\n✓ Configuration validated successfully")
    
    return len(errors) == 0

# Auto-validate on import
if __name__ != "__main__":
    validate_config()
