#!/usr/bin/env python3
"""
Universal Prediction Script
============================
Handles all prediction scenarios with smart parameter detection.

This script can predict for:
- Single or multiple customers
- Specific dates or today
- Specific facilities (optional filter)
- Save to file or display results

Examples:
    python predict.py scionhealth
    python predict.py 2025-11-15 scionhealth,mercy
    python predict.py scionhealth 287 2025-11-15 output.csv
"""

import sys
import os
# Add parent directory to Python path so we can import from src/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import pandas as pd
from src.core.prediction_generator import PredictionGenerator
from datetime import datetime, timedelta


def predict(date=None, customers=None, facility=None, output=None, days=14, use_test_data=False):
    """
    Generate predictions with flexible parameters
    
    This is the main prediction function that:
    1. Parses customer list (handles single or comma-separated)
    2. Generates predictions using PredictionGenerator
    3. Optionally filters by facility
    4. Displays summary statistics
    5. Optionally saves to file
    
    Args:
        date: Start date for predictions (YYYY-MM-DD) - optional, defaults to today
              Predictions will be generated for this date + next 'days' days
        customers: Customer ID(s) - string or comma-separated list (e.g., "scionhealth,mercy")
        facility: Facility ID - optional filter (e.g., "287")
        output: Output file path - optional (e.g., "predictions.csv")
        days: Number of days to predict (default: 14)
              Example: date='2025-11-15', days=14 → predicts Nov 15-28
        use_test_data: If True, uses test_data.csv if it exists (default: False)
    
    Returns:
        DataFrame with predictions including columns:
        - prediction_date: Date predictions are for
        - prediction_generated_at: When predictions were made
        - CustomerID, FacilityID, ProductID, ProductName
        - predicted_value: Predicted quantity
        - predicted_reorder: 1 if should order, 0 if not
        - reorder_recommendation: "ORDER" or "NO ORDER"
    """
    
    # ========================================================================
    # STEP 1: Parse customer list
    # ========================================================================
    # Convert customer string to list for processing
    # Examples:
    #   "scionhealth" → ["scionhealth"]
    #   "scionhealth,mercy" → ["scionhealth", "mercy"]
    customer_list = None
    if customers:
        if ',' in customers:
            # Multiple customers: split by comma and remove whitespace
            customer_list = [c.strip() for c in customers.split(',')]
        else:
            # Single customer: wrap in list
            customer_list = [customers]
    
    # ========================================================================
    # STEP 2: Calculate date range and print header
    # ========================================================================
    # Calculate start and end dates
    # If date is provided, use it as start_date
    # Otherwise, use today as start_date
    if date:
        start_date = date
    else:
        start_date = datetime.now().strftime('%Y-%m-%d')
    
    # Calculate end_date by adding 'days' to start_date
    start_dt = pd.to_datetime(start_date)
    end_dt = start_dt + timedelta(days=days - 1)  # -1 because we include start_date
    end_date = end_dt.strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    
    # Show date range
    if days == 1:
        print(f"Date:       {start_date}")
    else:
        print(f"Date Range: {start_date} to {end_date} ({days} days)")
    
    # Show which customers we're predicting for
    if customer_list:
        print(f"Customers:  {', '.join(customer_list)}")
    else:
        print(f"Customers:  All")
    
    # Show facility filter if specified
    if facility:
        print(f"Facility:   {facility}")
    
    print("=" * 80)
    
    # ========================================================================
    # STEP 3: Determine data source
    # ========================================================================
    # Check if we should use test_data.csv
    source_data = None
    use_preprocessed = False
    
    if use_test_data:
        from src.config.env_config import TEST_DATA_DIR
        test_data_path = os.path.join(TEST_DATA_DIR, 'test_data.csv')
        if os.path.exists(test_data_path):
            source_data = test_data_path
            use_preprocessed = True
            print(f"Using:      {test_data_path}")
        else:
            print(f"⚠️  test_data.csv not found, using SOURCE_DATA_FILE")
            print(f"    Run 'python scripts/extract.py' to generate test data")
    
    # ========================================================================
    # STEP 4: Generate predictions using PredictionGenerator
    # ========================================================================
    # Create generator with parameters:
    # - customers: list of customer IDs to predict for (None = all customers)
    # - start_date: first date to predict for
    # - end_date: last date to predict for (calculated from start_date + days)
    # - context_days: how many days of history to use (90 days = ~3 months)
    # - source_data: path to data file (None = use SOURCE_DATA_FILE from config)
    # - use_preprocessed: True if data already has features (test_data.csv)
    # - verbose: print progress messages during generation
    generator = PredictionGenerator(
        customers=customer_list,
        start_date=start_date,
        end_date=end_date,
        context_days=90,  # Use 90 days of historical data for patterns
        source_data=source_data,
        use_preprocessed=use_preprocessed,
        verbose=True
    )
    
    # Generate predictions (save=False means don't auto-save to file)
    # This will:
    # 1. Load historical data (or test_data.csv if use_test_data=True)
    # 2. Engineer features (or use preprocessed features from test_data.csv)
    # 3. Load LightGBM and DeepAR models
    # 4. Generate predictions using ensemble (95% LightGBM + 5% DeepAR)
    # 5. Apply customer calibrations
    # 6. Return DataFrame with predictions
    predictions = generator.generate(save=False)
    
    # ========================================================================
    # STEP 5: Filter by facility if specified
    # ========================================================================
    # If user specified a facility ID, filter predictions to only that facility
    # This is useful for facility-specific analysis
    if facility:
        predictions = predictions[predictions['FacilityID'].astype(str) == str(facility)]
        print(f"\n✓ Filtered to facility {facility}: {len(predictions):,} items")
    
    # ========================================================================
    # STEP 6: Save to file if output path specified
    # ========================================================================
    # Save predictions to CSV file if user provided output path
    if output:
        predictions.to_csv(output, index=False)
        print(f"\n✓ Saved to: {output}")
    
    # ========================================================================
    # STEP 7: Print summary statistics
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Show prediction metadata (when predictions were made)
    if 'prediction_date' in predictions.columns:
        pred_date = predictions['prediction_date'].iloc[0]
        gen_at = predictions['prediction_generated_at'].iloc[0]
        print(f"Prediction Date:      {pred_date}")
        print(f"Generated At:         {gen_at}")
        print()
    
    # Show overall statistics
    print(f"Total Items:          {len(predictions):,}")
    print(f"Items to Order (≥4):  {predictions['predicted_reorder'].sum():,}")
    print(f"Total Units:          {predictions['predicted_value'].sum():,.0f}")
    print(f"Average per Item:     {predictions['predicted_value'].mean():.2f} units")
    
    # ========================================================================
    # STEP 8: Show customer-specific or item-specific details
    # ========================================================================
    
    # If multiple customers: show summary by customer
    if customer_list and len(customer_list) > 1:
        print("\nBy Customer:")
        customer_summary = predictions.groupby('CustomerID').agg({
            'predicted_value': ['count', 'sum', 'mean'],
            'predicted_reorder': 'sum'
        }).round(2)
        customer_summary.columns = ['Items', 'Total_Units', 'Avg_Units', 'To_Order']
        print(customer_summary.to_string())
    
    # If single customer: show top 10 items by predicted volume
    elif customer_list and len(customer_list) == 1:
        print("\nTop 10 Items by Predicted Volume:")
        top_10 = predictions.nlargest(10, 'predicted_value')[
            ['ProductName', 'predicted_value', 'reorder_recommendation']
        ]
        for idx, row in top_10.iterrows():
            product_name = str(row['ProductName'])[:50]
            print(f"  {product_name:<50} {row['predicted_value']:>8.1f} units  [{row['reorder_recommendation']}]")
    
    print("=" * 80)
    
    return predictions


def parse_args(args):
    """
    Smart argument parser that handles multiple formats
    
    This function automatically detects what each argument is based on its format:
    - Date: Contains '-' and starts with digit (e.g., "2025-11-15")
    - Facility: All digits (e.g., "287")
    - Output: Ends with .csv or .json (e.g., "output.csv")
    - Customer: Everything else (e.g., "scionhealth" or "scionhealth,mercy")
    
    This allows flexible argument order:
        predict.py scionhealth 2025-11-15  ✓ Works
        predict.py 2025-11-15 scionhealth  ✓ Also works
    
    Supported formats:
        predict.py                                    # All customers, today
        predict.py 2025-11-15                         # All customers, specific date
        predict.py scionhealth                        # One customer, today
        predict.py scionhealth 2025-11-15             # One customer, specific date
        predict.py scionhealth 287                    # One customer, one facility, today
        predict.py scionhealth 287 2025-11-15         # One customer, facility, date
        predict.py 2025-11-15 scionhealth,mercy       # Multiple customers, date
        predict.py --date 2025-11-15 --customers scionhealth --facility 287 --output file.csv
    
    Args:
        args: List of command line arguments (sys.argv[1:])
    
    Returns:
        Dictionary with keys: date, customers, facility, output
    """
    
    # ========================================================================
    # Handle named arguments (--date, --customers, etc.)
    # ========================================================================
    if any(arg.startswith('--') for arg in args):
        params = {
            'date': None,
            'customers': None,
            'facility': None,
            'output': None,
            'days': 14,  # Default to 14 days
            'use_test_data': False
        }
        
        # Parse named arguments
        i = 0
        while i < len(args):
            if args[i] == '--date' and i + 1 < len(args):
                params['date'] = args[i + 1]
                i += 2
            elif args[i] == '--days' and i + 1 < len(args):
                params['days'] = int(args[i + 1])
                i += 2
            elif args[i] == '--customers' and i + 1 < len(args):
                params['customers'] = args[i + 1]
                i += 2
            elif args[i] == '--facility' and i + 1 < len(args):
                params['facility'] = args[i + 1]
                i += 2
            elif args[i] == '--output' and i + 1 < len(args):
                params['output'] = args[i + 1]
                i += 2
            elif args[i] == '--use-test-data':
                params['use_test_data'] = True
                i += 1
            else:
                i += 1
        
        return params
    
    # ========================================================================
    # Handle positional arguments (smart detection)
    # ========================================================================
    params = {
        'date': None,
        'customers': None,
        'facility': None,
        'output': None,
        'days': 14,  # Default to 14 days
        'use_test_data': False
    }
    
    # No arguments: predict for all customers today
    if len(args) == 0:
        return params
    
    # ========================================================================
    # Helper functions to detect argument types
    # ========================================================================
    
    def is_date(s):
        """Check if string looks like a date (YYYY-MM-DD)"""
        return '-' in s and len(s) >= 8 and s[0].isdigit()
    
    def is_numeric(s):
        """Check if string is all digits (facility ID)"""
        return s.isdigit()
    
    def is_output_file(s):
        """Check if string is an output file (.csv or .json)"""
        return s.endswith('.csv') or s.endswith('.json')
    
    # ========================================================================
    # Parse based on number of arguments and their types
    # ========================================================================
    
    # 1 argument: could be date, customer, or output file
    if len(args) == 1:
        if is_date(args[0]):
            params['date'] = args[0]
        elif is_output_file(args[0]):
            params['output'] = args[0]
        else:
            params['customers'] = args[0]
    
    # 2 arguments: various combinations
    elif len(args) == 2:
        if is_date(args[0]):
            # Date first: date + customers
            params['date'] = args[0]
            params['customers'] = args[1]
        elif is_date(args[1]):
            # Date second: customer + date
            params['customers'] = args[0]
            params['date'] = args[1]
        elif is_numeric(args[1]):
            # Numeric second: customer + facility
            params['customers'] = args[0]
            params['facility'] = args[1]
        elif is_output_file(args[1]):
            # File second: customer + output
            params['customers'] = args[0]
            params['output'] = args[1]
        else:
            # Default: assume customer
            params['customers'] = args[0]
    
    # 3 arguments: customer + facility + date/output, or date + customers + output
    elif len(args) == 3:
        if is_date(args[0]):
            # Date first: date + customers + output
            params['date'] = args[0]
            params['customers'] = args[1]
            if is_output_file(args[2]):
                params['output'] = args[2]
        elif is_numeric(args[1]):
            # Numeric second: customer + facility + date/output
            if is_date(args[2]):
                params['customers'] = args[0]
                params['facility'] = args[1]
                params['date'] = args[2]
            elif is_output_file(args[2]):
                params['customers'] = args[0]
                params['facility'] = args[1]
                params['output'] = args[2]
        elif is_date(args[1]):
            # Date second: customer + date + output
            params['customers'] = args[0]
            params['date'] = args[1]
            if is_output_file(args[2]):
                params['output'] = args[2]
    
    # 4 arguments: customer + facility + date + output
    elif len(args) == 4:
        params['customers'] = args[0]
        if is_numeric(args[1]):
            params['facility'] = args[1]
        if is_date(args[2]):
            params['date'] = args[2]
        if is_output_file(args[3]):
            params['output'] = args[3]
    
    return params


def print_usage():
    """Print usage examples and help information"""
    print("Universal Prediction Script")
    print("=" * 80)
    print("\nUsage:")
    print("  python predict.py [date] [customers] [facility] [output]")
    print("\nExamples:")
    print("\n  # All customers, today, next 14 days (default)")
    print("  python predict.py")
    print("\n  # All customers, specific date, next 14 days")
    print("  python predict.py 2025-11-15")
    print("\n  # One customer, today, next 14 days")
    print("  python predict.py scionhealth")
    print("\n  # One customer, specific date, next 14 days")
    print("  python predict.py scionhealth 2025-11-15")
    print("  python predict.py 2025-11-15 scionhealth")
    print("\n  # One customer, one facility")
    print("  python predict.py scionhealth 287")
    print("\n  # One customer, facility, date")
    print("  python predict.py scionhealth 287 2025-11-15")
    print("\n  # Multiple customers, date")
    print("  python predict.py 2025-11-15 scionhealth,mercy")
    print("\n  # Save to file")
    print("  python predict.py scionhealth output.csv")
    print("  python predict.py scionhealth 287 2025-11-15 output.csv")
    print("\n  # Named arguments with custom days")
    print("  python predict.py --date 2025-11-15 --customers scionhealth --days 7")
    print("  python predict.py --date 2025-11-15 --customers scionhealth --days 30 --output file.csv")
    print("\n  # Use test_data.csv (if it exists)")
    print("  python predict.py --use-test-data")
    print("  python predict.py --use-test-data --customers scionhealth")
    print("\nNotes:")
    print("  - Date format: YYYY-MM-DD (start date)")
    print("  - Days: Number of days to predict (default: 14)")
    print("  - Multiple customers: comma-separated (scionhealth,mercy,ibji)")
    print("  - Facility: numeric ID")
    print("  - Output: any .csv or .json file")
    print("  - Predictions are generated for date to date+days")
    print("  - --use-test-data: Uses test_data.csv if available (faster, preprocessed features)")
    print("=" * 80)


if __name__ == "__main__":
    """
    Command line interface
    
    This is the entry point when script is run from command line.
    It handles:
    1. Help requests (--help, -h, help)
    2. Argument parsing
    3. Prediction generation
    4. Error handling
    """
    
    # Show help if requested
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_usage()
        sys.exit(0)
    
    # Parse command line arguments (skip first arg which is script name)
    params = parse_args(sys.argv[1:])
    
    # Generate predictions with parsed parameters
    try:
        predictions = predict(**params)
    except Exception as e:
        # Show error message and help hint
        print(f"\n❌ Error: {e}")
        print("\nRun 'python predict.py --help' for usage examples")
        sys.exit(1)
