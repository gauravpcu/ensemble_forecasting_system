"""
Configuration Defaults
Centralized default values for the ensemble forecasting system
These can be overridden by environment variables or .env file
"""

# ============================================================================
# DATA PATHS DEFAULTS
# ============================================================================

DEFAULT_SOURCE_DATA_FILE = None  # Must be provided by user
DEFAULT_DATA_DIR = './data'
DEFAULT_TEST_DATA_DIR = './test/data'
DEFAULT_RESULTS_DIR = './results'

# ============================================================================
# MODEL CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_LIGHTGBM_MODEL_PATH = None  # Will be set to DATA_DIR/lightgbm_model.pkl
DEFAULT_FEATURE_CONFIG_PATH = None  # Will be set to DATA_DIR/feature_config.json

DEFAULT_DEEPAR_ENDPOINT_NAME = None  # Must be provided by user
DEFAULT_DEEPAR_REGION = 'us-west-2'

# ============================================================================
# ENSEMBLE WEIGHTS DEFAULTS
# ============================================================================

DEFAULT_LIGHTGBM_WEIGHT = 0.95
DEFAULT_DEEPAR_WEIGHT = 0.05

# ============================================================================
# DATA EXTRACTION DEFAULTS
# ============================================================================

DEFAULT_VALIDATION_DAYS = 14
DEFAULT_TEST_DAYS = 14
DEFAULT_TOTAL_EXTRACTION_DAYS = 28

# ============================================================================
# CLASSIFICATION DEFAULTS
# ============================================================================

DEFAULT_CLASSIFICATION_THRESHOLD = 5

# ============================================================================
# TESTING CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_FOCUSED_TEST_CUSTOMERS = ['scionhealth', 'mercy']
DEFAULT_TEST_SAMPLE_SIZE = None
DEFAULT_VERBOSE = True

# ============================================================================
# PERFORMANCE THRESHOLDS DEFAULTS
# ============================================================================

DEFAULT_MAPE_EXCELLENT = 30.0
DEFAULT_MAPE_GOOD = 50.0
DEFAULT_MAPE_FAIR = 75.0

# ============================================================================
# AWS CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_AWS_ACCESS_KEY_ID = None
DEFAULT_AWS_SECRET_ACCESS_KEY = None
DEFAULT_AWS_DEFAULT_REGION = 'us-west-2'

# ============================================================================
# BEDROCK LLM CONFIGURATION DEFAULTS
# ============================================================================

DEFAULT_USE_LLM_AUGMENTATION = False
DEFAULT_BEDROCK_REGION = 'us-east-1'
DEFAULT_BEDROCK_MODEL_ID = 'anthropic.claude-3-sonnet-20241022-v2:0'
DEFAULT_BEDROCK_MAX_RETRIES = 3
DEFAULT_BEDROCK_RETRY_DELAY = 1
DEFAULT_LLM_CONFIDENCE_THRESHOLD = 0.7

# ============================================================================
# PROCESSING SETTINGS DEFAULTS
# ============================================================================

DEFAULT_BATCH_SIZE = 1000
DEFAULT_NUM_WORKERS = 0
DEFAULT_MEMORY_LIMIT = 0

# ============================================================================
# OUTPUT SETTINGS DEFAULTS
# ============================================================================

DEFAULT_SAVE_PREDICTIONS = True
DEFAULT_GENERATE_PLOTS = False
DEFAULT_OUTPUT_FORMAT = 'csv'
DEFAULT_OUTPUT_COMPRESSION = 'none'

# ============================================================================
# LOGGING DEFAULTS
# ============================================================================

DEFAULT_LOG_LEVEL = 'INFO'
DEFAULT_LOG_FILE = None

# ============================================================================
# SAFETY MULTIPLIERS DEFAULTS
# ============================================================================

DEFAULT_SAFETY_MULTIPLIER_LOW = 2.0
DEFAULT_SAFETY_MULTIPLIER_MEDIUM = 1.5
DEFAULT_SAFETY_MULTIPLIER_HIGH = 1.2

DEFAULT_VOLUME_LOW_THRESHOLD = 5
DEFAULT_VOLUME_MEDIUM_THRESHOLD = 20
DEFAULT_VOLUME_HIGH_THRESHOLD = 100

# ============================================================================
# CALIBRATION DEFAULTS
# ============================================================================

DEFAULT_GLOBAL_CALIBRATION_MULTIPLIER = 1.0
DEFAULT_CUSTOMER_CALIBRATION = {}

# ============================================================================
# FEATURE ENGINEERING DEFAULTS
# ============================================================================

DEFAULT_ROLLING_WINDOW_SHORT = 7
DEFAULT_ROLLING_WINDOW_LONG = 30
DEFAULT_LAG_PERIODS = [7, 14, 30]
DEFAULT_SEASONAL_PERIOD = 7

# ============================================================================
# VALIDATION SETTINGS DEFAULTS
# ============================================================================

DEFAULT_MIN_RECORDS_FOR_ANALYSIS = 10
DEFAULT_MIN_VALUE_FOR_MAPE = 1

# ============================================================================
# NOTIFICATION SETTINGS DEFAULTS
# ============================================================================

DEFAULT_ENABLE_EMAIL_NOTIFICATIONS = False
DEFAULT_EMAIL_SMTP_SERVER = None
DEFAULT_EMAIL_SMTP_PORT = 587
DEFAULT_EMAIL_FROM = None
DEFAULT_EMAIL_TO = None
DEFAULT_EMAIL_USERNAME = None
DEFAULT_EMAIL_PASSWORD = None

DEFAULT_ENABLE_SLACK_NOTIFICATIONS = False
DEFAULT_SLACK_WEBHOOK_URL = None

# ============================================================================
# VISUALIZATION SETTINGS DEFAULTS
# ============================================================================

DEFAULT_PLOT_FIGURE_SIZE_WIDTH = 12
DEFAULT_PLOT_FIGURE_SIZE_HEIGHT = 6
DEFAULT_PLOT_SUBPLOT_WIDTH = 15
DEFAULT_PLOT_SUBPLOT_HEIGHT = 5
DEFAULT_PLOT_DPI = 300
DEFAULT_PLOT_ALPHA = 0.5
DEFAULT_PLOT_SCATTER_SIZE = 10
DEFAULT_PLOT_STYLE = 'whitegrid'

# ============================================================================
# FILE PROCESSING DEFAULTS
# ============================================================================

DEFAULT_FILE_ENCODING = 'utf-8'
DEFAULT_CSV_SEPARATOR = ','
DEFAULT_DECIMAL_PLACES = 2
DEFAULT_DATE_FORMAT = '%Y-%m-%d'
DEFAULT_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# ============================================================================
# DATA SAMPLING DEFAULTS
# ============================================================================

DEFAULT_SAMPLE_RANDOM_STATE = 42
DEFAULT_MAX_RECORDS_FOR_QUICK_TEST = 5000
DEFAULT_PREDICTION_LENGTH_DAYS = 14

# ============================================================================
# ROLLING STATISTICS DEFAULTS
# ============================================================================

DEFAULT_ROLLING_WINDOW_MIN_PERIODS = 1
DEFAULT_ROLLING_MAX_WINDOW_SIZE = 30

# ============================================================================
# FILE SIZE AND MEMORY DEFAULTS
# ============================================================================

DEFAULT_MAX_FILE_SIZE_MB = 1000
DEFAULT_CHUNK_SIZE_ROWS = 10000
DEFAULT_MAX_MEMORY_USAGE_PCT = 80

# ============================================================================
# TIMEOUT AND RETRY DEFAULTS
# ============================================================================

DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_MAX_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_DELAY_SECONDS = 1
DEFAULT_CONNECTION_TIMEOUT = 30

# ============================================================================
# MODEL TRAINING DEFAULTS
# ============================================================================

DEFAULT_TRAINING_DATA_PATH = None  # Must be provided by user
DEFAULT_MODEL_OUTPUT_PATH = './model/lightgbm_model.pkl'
DEFAULT_CONFIG_OUTPUT_PATH = './model/feature_config.json'
DEFAULT_TRAINING_CHUNK_SIZE = 100000
DEFAULT_TRAINING_MAX_CHUNKS = 100
DEFAULT_TRAINING_TEST_DAYS = 14
DEFAULT_TRAINING_VALIDATION_SPLIT = 0.1

# LightGBM Training Parameters
DEFAULT_LGBM_NUM_LEAVES = 63
DEFAULT_LGBM_LEARNING_RATE = 0.03
DEFAULT_LGBM_FEATURE_FRACTION = 0.8
DEFAULT_LGBM_BAGGING_FRACTION = 0.8
DEFAULT_LGBM_BAGGING_FREQ = 5
DEFAULT_LGBM_MIN_CHILD_SAMPLES = 50
DEFAULT_LGBM_REG_ALPHA = 0.1
DEFAULT_LGBM_REG_LAMBDA = 0.1
DEFAULT_LGBM_MAX_DEPTH = 8
DEFAULT_LGBM_NUM_BOOST_ROUND = 2000
DEFAULT_LGBM_EARLY_STOPPING_ROUNDS = 100

# ============================================================================
# ADVANCED SETTINGS DEFAULTS
# ============================================================================

DEFAULT_RANDOM_SEED = 42
DEFAULT_ENABLE_CACHING = True
DEFAULT_CACHE_DIR = './.cache'
DEFAULT_CACHE_MAX_AGE = 24
DEFAULT_ENABLE_PROFILING = False
DEFAULT_PROFILE_OUTPUT_DIR = './profiles'

# ============================================================================
# REQUIRED CONFIGURATION VALIDATION
# ============================================================================

REQUIRED_CONFIG = [
    'SOURCE_DATA_FILE',
    'DEEPAR_ENDPOINT_NAME'
]

CRITICAL_CONFIG = [
    'LIGHTGBM_MODEL_PATH',
    'FEATURE_CONFIG_PATH'
]