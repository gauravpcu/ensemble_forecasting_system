#!/usr/bin/env python3
"""
Main Testing Pipeline for Ensemble Forecasting System
This is the primary way to test the forecasting models
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")

def print_step(step_num, total_steps, title):
    """Print a step header"""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 80 + "\n")

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"Running: {description}...")
    print(f"Script: {script_path}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✓ {description} complete ({elapsed:.1f}s)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed!")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error in {description}!")
        print(f"Error: {e}")
        return False

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def main():
    """Main testing pipeline"""
    
    print_header("ENSEMBLE FORECASTING SYSTEM - FULL TEST PIPELINE")
    
    print("This script will:")
    print("  1. Extract data from order history")
    print("  2. Generate predictions using ensemble model")
    print("  3. Compare with validation data")
    print("  4. Analyze overall results")
    print("  5. Calculate customer-facility accuracy with precision/recall")
    print("\nPress Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        sys.exit(0)
    
    # Configuration
    source_file = "/Volumes/ESSD-1/M4Downloads/order_history_2025-10-28.csv"
    test_data = "test/data/test_data.csv"
    val_data = "test/data/val_data.csv"
    predictions = "test/data/predictions.csv"
    
    # Track results
    results = {}
    start_time = time.time()
    
    # Step 1: Data Extraction
    print_step(1, 5, "DATA EXTRACTION")
    
    if check_file_exists(test_data) and check_file_exists(val_data):
        print("Test and validation data already exist.")
        response = input("Do you want to re-extract? (y/N): ").strip().lower()
        if response == 'y':
            results['extraction'] = run_script("test/extract_data.py", "Data extraction")
        else:
            print("Skipping extraction, using existing data.")
            results['extraction'] = True
    else:
        if not check_file_exists(source_file):
            print(f"❌ ERROR: Source file not found: {source_file}")
            print("Please update the source_file path in this script.")
            sys.exit(1)
        results['extraction'] = run_script("test/extract_data.py", "Data extraction")
    
    if not results['extraction']:
        print("\n❌ Data extraction failed. Cannot continue.")
        sys.exit(1)
    
    # Verify data
    print("\nVerifying extracted data...")
    run_script("test/verify_data.py", "Data verification")
    
    # Step 2: Generate Predictions
    print_step(2, 5, "GENERATE PREDICTIONS")
    
    if check_file_exists(predictions):
        print("Predictions already exist.")
        response = input("Do you want to re-run predictions? (y/N): ").strip().lower()
        if response == 'y':
            results['predictions'] = run_script("test/run_predictions.py", "Prediction generation")
        else:
            print("Skipping predictions, using existing results.")
            results['predictions'] = True
    else:
        results['predictions'] = run_script("test/run_predictions.py", "Prediction generation")
    
    if not results['predictions']:
        print("\n❌ Prediction generation failed. Cannot continue.")
        sys.exit(1)
    
    # Step 3: Validation Comparison
    print_step(3, 5, "VALIDATION COMPARISON")
    results['validation'] = run_script("test/compare_with_validation.py", "Validation comparison")
    
    # Step 4: Comprehensive Analysis
    print_step(4, 5, "COMPREHENSIVE ANALYSIS")
    results['analysis'] = run_script("test/analyze_results.py", "Overall analysis")
    
    # Step 5: Customer-Facility Analysis
    print_step(5, 5, "CUSTOMER-FACILITY ACCURACY ANALYSIS")
    results['customer_facility'] = run_script("test/customer_facility_analysis.py", 
                                               "Customer-facility analysis with precision/recall")
    results['dual_analysis'] = run_script("test/dual_prediction_analysis.py", 
                                          "Dual prediction analysis (Product + Quantity)")
    
    # Final Summary
    total_time = time.time() - start_time
    
    print_header("TEST PIPELINE COMPLETE!")
    
    print(f"⏱️  Total execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")
    
    print("📊 Results Summary:")
    for step, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"   {step.replace('_', ' ').title():<30} {status}")
    
    print("\n📁 Generated Files:\n")
    print("Data Files (test/data/):")
    files = [
        ("test_data.csv", "Test dataset (Oct 1-15)"),
        ("val_data.csv", "Validation dataset (Oct 16-28)"),
        ("predictions.csv", "Full predictions with actuals"),
        ("prediction_summary.csv", "Per-item summary"),
        ("validation_comparison.csv", "Forward-looking comparison"),
        ("customer_analysis.csv", "Customer-level metrics"),
        ("customer_facility_metrics.csv", "Detailed metrics per customer-facility"),
        ("volume_analysis.csv", "Volume category breakdown"),
        ("daily_trends.csv", "Daily patterns"),
        ("dual_analysis_items.csv", "Product + Quantity analysis per item"),
        ("dual_analysis_customers.csv", "Customer-level dual metrics")
    ]
    
    for filename, description in files:
        filepath = f"test/data/{filename}"
        if check_file_exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)  # MB
            print(f"  ✓ {filename:<35} {description} ({size:.1f}MB)")
        else:
            print(f"  ✗ {filename:<35} {description} (missing)")
    
    print("\nReports (test/data/):")
    reports = [
        ("ANALYSIS_SUMMARY.txt", "Executive summary"),
        ("customer_facility_summary.txt", "Customer-facility summary"),
        ("dual_analysis_summary.txt", "Product + Quantity metrics summary")
    ]
    
    for filename, description in reports:
        filepath = f"test/data/{filename}"
        if check_file_exists(filepath):
            print(f"  ✓ {filename:<35} {description}")
        else:
            print(f"  ✗ {filename:<35} {description} (missing)")
    
    print("\nDocumentation (test/):")
    docs = [
        ("PREDICTION_RESULTS.md", "Detailed results & recommendations"),
        ("CUSTOMER_FACILITY_ANALYSIS.md", "Precision/recall analysis"),
        ("QUICK_START.md", "Quick reference guide")
    ]
    
    for filename, description in docs:
        filepath = f"test/{filename}"
        if check_file_exists(filepath):
            print(f"  ✓ {filename:<35} {description}")
        else:
            print(f"  ✗ {filename:<35} {description} (missing)")
    
    print_header("NEXT STEPS")
    
    print("1. Review overall results:")
    print("   cat test/data/ANALYSIS_SUMMARY.txt\n")
    
    print("2. Review customer-facility metrics:")
    print("   cat test/data/customer_facility_summary.txt\n")
    
    print("3. View detailed analysis:")
    print("   open test/PREDICTION_RESULTS.md")
    print("   open test/CUSTOMER_FACILITY_ANALYSIS.md\n")
    
    print("4. Explore data files:")
    print("   head test/data/customer_facility_metrics.csv")
    print("   head test/data/predictions.csv\n")
    
    print("5. View quick statistics:")
    print("   python3 test/quick_stats.py\n")
    
    print_header("✅ ALL TESTS COMPLETE!")
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        print("\n⚠️  Some steps failed. Please review the output above.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest pipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
