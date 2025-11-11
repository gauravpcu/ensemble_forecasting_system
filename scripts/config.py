#!/usr/bin/env python3
"""
Universal Configuration Management
Manage both system config (.env) and customer calibrations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

CALIBRATION_FILE = 'config/customer_calibrations.json'


# ============================================================================
# CALIBRATION MANAGEMENT
# ============================================================================

def load_calibrations():
    """Load customer calibrations"""
    if not os.path.exists(CALIBRATION_FILE):
        return {
            "version": "1.0",
            "last_updated": datetime.now().strftime('%Y-%m-%d'),
            "default_calibration": 1.0,
            "classification_threshold": 4,
            "customers": {}
        }
    
    with open(CALIBRATION_FILE, 'r') as f:
        return json.load(f)


def save_calibrations(data):
    """Save customer calibrations"""
    data['last_updated'] = datetime.now().strftime('%Y-%m-%d')
    os.makedirs(os.path.dirname(CALIBRATION_FILE), exist_ok=True)
    
    with open(CALIBRATION_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Saved calibrations")


def list_calibrations():
    """List all customer calibrations"""
    data = load_calibrations()
    
    print("=" * 100)
    print("CUSTOMER CALIBRATIONS")
    print("=" * 100)
    print(f"Version:                  {data['version']}")
    print(f"Last Updated:             {data['last_updated']}")
    print(f"Default Calibration:      {data['default_calibration']}")
    print(f"Classification Threshold: {data['classification_threshold']} units")
    print()
    
    if not data['customers']:
        print("No customer calibrations configured.")
        return
    
    print(f"{'Customer':<20} {'Calibration':<12} {'Status':<20} {'Precision':<10} {'Recall':<10} {'MAE':<10}")
    print("-" * 100)
    
    for customer_id, config in sorted(data['customers'].items()):
        calibration = config.get('calibration_multiplier', 'N/A')
        status = config.get('status', 'unknown')
        precision = f"{config.get('precision', 0)*100:.1f}%" if config.get('precision') else 'N/A'
        recall = f"{config.get('recall', 0)*100:.1f}%" if config.get('recall') else 'N/A'
        mae = f"{config.get('mae', 0):.2f}" if config.get('mae') else 'N/A'
        
        print(f"{customer_id:<20} {calibration:<12} {status:<20} {precision:<10} {recall:<10} {mae:<10}")
    
    print()


def show_customer(customer_id):
    """Show detailed customer calibration"""
    data = load_calibrations()
    customer_id = customer_id.lower()
    
    if customer_id not in data['customers']:
        print(f"❌ Customer '{customer_id}' not found")
        return
    
    config = data['customers'][customer_id]
    
    print("=" * 80)
    print(f"CUSTOMER: {customer_id.upper()}")
    print("=" * 80)
    print(f"Calibration Multiplier: {config.get('calibration_multiplier', 'N/A')}")
    print(f"Status:                 {config.get('status', 'unknown')}")
    print(f"Last Tested:            {config.get('last_tested', 'Never')}")
    print()
    
    if config.get('precision'):
        print("Performance Metrics:")
        print(f"  Precision:            {config.get('precision', 0)*100:.1f}%")
        print(f"  Recall:               {config.get('recall', 0)*100:.1f}%")
        print(f"  MAE:                  {config.get('mae', 0):.2f} units")
        print()
    
    if config.get('notes'):
        print(f"Notes: {config['notes']}")
    
    if config.get('recommendation'):
        print(f"Recommendation: {config['recommendation']}")


def update_calibration(customer_id, calibration, status='active', precision=None, recall=None, mae=None, notes=None):
    """Update customer calibration"""
    data = load_calibrations()
    customer_id = customer_id.lower()
    
    if customer_id not in data['customers']:
        data['customers'][customer_id] = {}
    
    data['customers'][customer_id]['calibration_multiplier'] = calibration
    data['customers'][customer_id]['status'] = status
    data['customers'][customer_id]['last_tested'] = datetime.now().strftime('%Y-%m-%d')
    
    if precision is not None:
        data['customers'][customer_id]['precision'] = precision
    if recall is not None:
        data['customers'][customer_id]['recall'] = recall
    if mae is not None:
        data['customers'][customer_id]['mae'] = mae
    if notes is not None:
        data['customers'][customer_id]['notes'] = notes
    
    save_calibrations(data)
    print(f"✓ Updated calibration for {customer_id}: {calibration}")


# ============================================================================
# ENV CONFIGURATION MANAGEMENT
# ============================================================================

def update_env_value(key, value):
    """Update value in .env file"""
    env_path = Path('.env')
    
    if not env_path.exists():
        print("❌ .env file not found. Run: python config.py --init")
        return False
    
    # Read current content
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    # Update or add the key
    key_found = False
    new_lines = []
    
    for line in lines:
        if line.strip().startswith(f'{key}='):
            new_lines.append(f'{key}={value}\n')
            key_found = True
        else:
            new_lines.append(line)
    
    # If key not found, add it
    if not key_found:
        new_lines.append(f'\n{key}={value}\n')
    
    # Write back
    with open(env_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Updated {key}={value}")
    return True


def create_env_template():
    """Create .env from .env.example"""
    env_path = Path('.env')
    example_path = Path('.env.example')
    
    if env_path.exists():
        return False
    
    if not example_path.exists():
        print("❌ .env.example not found")
        return False
    
    # Copy example to .env
    with open(example_path, 'r') as f:
        content = f.read()
    
    with open(env_path, 'w') as f:
        f.write(content)
    
    return True


def show_env_config():
    """Show current environment configuration"""
    from src.config import env_config
    
    print("=" * 80)
    print("ENVIRONMENT CONFIGURATION")
    print("=" * 80)
    print(f"\nData Paths:")
    print(f"  SOURCE_DATA_FILE:     {env_config.SOURCE_DATA_FILE}")
    print(f"  TEST_DATA_DIR:        {env_config.TEST_DATA_DIR}")
    print(f"  MODEL_PATH:           {env_config.LIGHTGBM_MODEL_PATH}")
    
    print(f"\nModel Configuration:")
    print(f"  LIGHTGBM_WEIGHT:      {env_config.LIGHTGBM_WEIGHT}")
    print(f"  DEEPAR_WEIGHT:        {env_config.DEEPAR_WEIGHT}")
    print(f"  DEEPAR_ENDPOINT:      {env_config.DEEPAR_ENDPOINT_NAME}")
    
    print(f"\nData Extraction:")
    print(f"  VALIDATION_DAYS:      {env_config.VALIDATION_DAYS}")
    print(f"  TEST_DAYS:            {env_config.TEST_DAYS}")
    print(f"  TOTAL_DAYS:           {env_config.TOTAL_EXTRACTION_DAYS}")
    
    print(f"\nClassification:")
    print(f"  THRESHOLD:            {env_config.CLASSIFICATION_THRESHOLD} units")
    
    print(f"\nProcessing:")
    print(f"  BATCH_SIZE:           {env_config.BATCH_SIZE}")
    print(f"  VERBOSE:              {env_config.VERBOSE}")
    print("=" * 80)


# ============================================================================
# MAIN CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Universal Configuration Management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Environment Configuration
  python config.py --init                           # Create .env file
  python config.py --show                           # Show env config
  python config.py --set SOURCE_DATA_FILE /path/to/data.csv
  python config.py --set BATCH_SIZE 2000
  
  # Customer Calibrations
  python config.py --calibrations                   # List all calibrations
  python config.py --customer scionhealth           # Show customer details
  python config.py --update scionhealth 1.05        # Update calibration
  python config.py --update mercy 0.85 --status verified --mae 3.5
        """
    )
    
    # Environment config options
    env_group = parser.add_argument_group('Environment Configuration')
    env_group.add_argument('--init', action='store_true', help='Create .env from template')
    env_group.add_argument('--show', action='store_true', help='Show environment config')
    env_group.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set env value')
    
    # Calibration options
    cal_group = parser.add_argument_group('Customer Calibrations')
    cal_group.add_argument('--calibrations', action='store_true', help='List all calibrations')
    cal_group.add_argument('--customer', type=str, help='Show customer calibration')
    cal_group.add_argument('--update', nargs=2, metavar=('CUSTOMER', 'VALUE'), help='Update calibration')
    cal_group.add_argument('--status', type=str, help='Set status (with --update)')
    cal_group.add_argument('--precision', type=float, help='Set precision (with --update)')
    cal_group.add_argument('--recall', type=float, help='Set recall (with --update)')
    cal_group.add_argument('--mae', type=float, help='Set MAE (with --update)')
    cal_group.add_argument('--notes', type=str, help='Set notes (with --update)')
    
    args = parser.parse_args()
    
    # Handle environment config
    if args.init:
        if create_env_template():
            print("✓ Created .env file from template")
        else:
            print("⚠️  .env file already exists")
        return
    
    if args.show:
        show_env_config()
        return
    
    if args.set:
        key, value = args.set
        update_env_value(key, value)
        return
    
    # Handle calibrations
    if args.calibrations:
        list_calibrations()
        return
    
    if args.customer:
        show_customer(args.customer)
        return
    
    if args.update:
        customer_id, calibration = args.update
        update_calibration(
            customer_id,
            float(calibration),
            status=args.status or 'active',
            precision=args.precision,
            recall=args.recall,
            mae=args.mae,
            notes=args.notes
        )
        return
    
    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
