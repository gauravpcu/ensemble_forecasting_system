#!/usr/bin/env python3
"""
Configuration Management Script
Easily update configuration values in .env file
"""

import argparse
import sys
from pathlib import Path
from env_config import update_env_value, create_env_template, print_config_summary

def main():
    parser = argparse.ArgumentParser(description='Manage configuration settings')
    parser.add_argument('--init', action='store_true', 
                       help='Create .env file from .env.example')
    parser.add_argument('--show', action='store_true',
                       help='Show current configuration')
    parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'),
                       help='Set configuration value: --set KEY VALUE')
    parser.add_argument('--source-file', type=str,
                       help='Set source data file path')
    parser.add_argument('--plot-dpi', type=int,
                       help='Set plot DPI (e.g., 300)')
    parser.add_argument('--batch-size', type=int,
                       help='Set batch size for processing')
    parser.add_argument('--timeout', type=int,
                       help='Set default timeout in seconds')
    parser.add_argument('--validation-days', type=int,
                       help='Set number of validation days')
    parser.add_argument('--test-days', type=int,
                       help='Set number of test days')
    
    args = parser.parse_args()
    
    if args.init:
        if create_env_template():
            print("✓ Configuration initialized")
        else:
            print("⚠️  .env file already exists")
        return
    
    if args.show:
        print_config_summary()
        return
    
    # Handle specific settings
    if args.source_file:
        update_env_value('SOURCE_DATA_FILE', args.source_file)
    
    if args.plot_dpi:
        update_env_value('PLOT_DPI', str(args.plot_dpi))
    
    if args.batch_size:
        update_env_value('BATCH_SIZE', str(args.batch_size))
    
    if args.timeout:
        update_env_value('DEFAULT_TIMEOUT_SECONDS', str(args.timeout))
    
    if args.validation_days:
        update_env_value('VALIDATION_DAYS', str(args.validation_days))
        # Auto-update total days
        test_days = args.test_days or 14  # Default
        update_env_value('TOTAL_EXTRACTION_DAYS', str(args.validation_days + test_days))
    
    if args.test_days:
        update_env_value('TEST_DAYS', str(args.test_days))
        # Auto-update total days
        validation_days = args.validation_days or 14  # Default
        update_env_value('TOTAL_EXTRACTION_DAYS', str(validation_days + args.test_days))
    
    if args.set:
        key, value = args.set
        update_env_value(key, value)
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        print("\nCommon examples:")
        print("  python configure.py --init                    # Create .env file")
        print("  python configure.py --show                    # Show current config")
        print("  python configure.py --source-file /path/to/data.csv")
        print("  python configure.py --plot-dpi 600           # High quality plots")
        print("  python configure.py --batch-size 2000        # Larger batches")
        print("  python configure.py --validation-days 21     # 3 weeks validation")
        print("  python configure.py --set VERBOSE true       # Enable verbose logging")

if __name__ == "__main__":
    main()