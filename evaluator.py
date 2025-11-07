"""
Model evaluation and metrics calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import config


class Evaluator:
    """Evaluate model predictions against actual values"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics = {}
    
    def calculate_regression_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate regression metrics
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        # Remove any NaN or infinite values
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) == 0:
            return {'error': 'No valid data points'}
        
        metrics = {}
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true_clean, y_pred_clean)
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # Mean Absolute Percentage Error (handle division by zero)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + epsilon))) * 100
        metrics['mape'] = mape
        
        # Symmetric Mean Absolute Percentage Error
        smape = np.mean(2.0 * np.abs(y_pred_clean - y_true_clean) / 
                       (np.abs(y_true_clean) + np.abs(y_pred_clean) + epsilon)) * 100
        metrics['smape'] = smape
        
        # R-squared
        try:
            metrics['r2'] = r2_score(y_true_clean, y_pred_clean)
        except:
            metrics['r2'] = 0.0
        
        # Accuracy (100 - MAPE)
        metrics['accuracy'] = max(0, 100 - mape)
        
        # Mean Bias Error
        metrics['mbe'] = np.mean(y_pred_clean - y_true_clean)
        
        # Mean Absolute Scaled Error (MASE) - using naive forecast as baseline
        naive_error = np.mean(np.abs(np.diff(y_true_clean)))
        if naive_error > 0:
            metrics['mase'] = metrics['mae'] / naive_error
        else:
            metrics['mase'] = np.nan
        
        return metrics
    
    def calculate_classification_metrics(self, y_true: np.ndarray, 
                                         y_pred: np.ndarray,
                                         threshold: float = 5.0) -> Dict:
        """
        Calculate classification metrics by converting to binary classification
        (e.g., demand > threshold vs demand <= threshold)
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            threshold: Threshold for binary classification
            
        Returns:
            Dictionary of classification metrics
        """
        # Convert to binary classification
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        metrics = {}
        
        try:
            # Precision, Recall, F1
            metrics['precision'] = precision_score(y_true_binary, y_pred_binary, 
                                                  zero_division=0)
            metrics['recall'] = recall_score(y_true_binary, y_pred_binary, 
                                            zero_division=0)
            metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            metrics['confusion_matrix'] = cm.tolist()
            
            # True/False Positives/Negatives
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                metrics['true_negatives'] = int(tn)
                metrics['false_positives'] = int(fp)
                metrics['false_negatives'] = int(fn)
                metrics['true_positives'] = int(tp)
                
                # Specificity
                metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # Accuracy
                metrics['classification_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def calculate_demand_category_metrics(self, y_true: np.ndarray, 
                                          y_pred: np.ndarray) -> Dict:
        """
        Calculate metrics for different demand categories
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics by category
        """
        categories = {
            'low': (0, config.DEMAND_THRESHOLDS['low']),
            'medium': (config.DEMAND_THRESHOLDS['low'], config.DEMAND_THRESHOLDS['medium']),
            'high': (config.DEMAND_THRESHOLDS['medium'], np.inf)
        }
        
        results = {}
        
        for category, (lower, upper) in categories.items():
            # Filter data for this category
            mask = (y_true >= lower) & (y_true < upper)
            
            if np.sum(mask) == 0:
                results[category] = {'count': 0, 'message': 'No samples in this category'}
                continue
            
            y_true_cat = y_true[mask]
            y_pred_cat = y_pred[mask]
            
            # Calculate metrics for this category
            metrics = self.calculate_regression_metrics(y_true_cat, y_pred_cat)
            metrics['count'] = int(np.sum(mask))
            metrics['percentage'] = float(np.sum(mask) / len(y_true) * 100)
            
            results[category] = metrics
        
        return results
    
    def evaluate_predictions(self, y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            model_name: str = 'model') -> Dict:
        """
        Comprehensive evaluation of predictions
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model being evaluated
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name.upper()}")
        print(f"{'='*60}")
        
        results = {
            'model_name': model_name,
            'sample_size': len(y_true)
        }
        
        # Regression metrics
        print("\nCalculating regression metrics...")
        results['regression_metrics'] = self.calculate_regression_metrics(y_true, y_pred)
        
        # Classification metrics (for different thresholds)
        print("Calculating classification metrics...")
        results['classification_metrics'] = {}
        for threshold_name, threshold_value in config.DEMAND_THRESHOLDS.items():
            results['classification_metrics'][threshold_name] = \
                self.calculate_classification_metrics(y_true, y_pred, threshold_value)
        
        # Category-specific metrics
        print("Calculating category-specific metrics...")
        results['category_metrics'] = self.calculate_demand_category_metrics(y_true, y_pred)
        
        # Print summary
        self._print_evaluation_summary(results)
        
        return results
    
    def _print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results"""
        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        
        # Regression metrics
        reg_metrics = results.get('regression_metrics', {})
        print("\n📊 Regression Metrics:")
        print(f"  MAE:      {reg_metrics.get('mae', 0):.4f}")
        print(f"  RMSE:     {reg_metrics.get('rmse', 0):.4f}")
        print(f"  MAPE:     {reg_metrics.get('mape', 0):.2f}%")
        print(f"  SMAPE:    {reg_metrics.get('smape', 0):.2f}%")
        print(f"  R²:       {reg_metrics.get('r2', 0):.4f}")
        print(f"  Accuracy: {reg_metrics.get('accuracy', 0):.2f}%")
        
        # Classification metrics (for 'low' threshold)
        if 'classification_metrics' in results and 'low' in results['classification_metrics']:
            cls_metrics = results['classification_metrics']['low']
            print(f"\n🎯 Classification Metrics (threshold={config.DEMAND_THRESHOLDS['low']}):")
            print(f"  Precision: {cls_metrics.get('precision', 0):.4f}")
            print(f"  Recall:    {cls_metrics.get('recall', 0):.4f}")
            print(f"  F1-Score:  {cls_metrics.get('f1', 0):.4f}")
            
            if 'true_positives' in cls_metrics:
                print(f"\n  Confusion Matrix:")
                print(f"    TP: {cls_metrics['true_positives']}, FP: {cls_metrics['false_positives']}")
                print(f"    FN: {cls_metrics['false_negatives']}, TN: {cls_metrics['true_negatives']}")
        
        # Category breakdown
        if 'category_metrics' in results:
            print(f"\n📈 Performance by Demand Category:")
            for category, metrics in results['category_metrics'].items():
                if 'count' in metrics and metrics['count'] > 0:
                    print(f"\n  {category.upper()} demand ({metrics['count']} samples, {metrics['percentage']:.1f}%):")
                    print(f"    MAE:      {metrics.get('mae', 0):.4f}")
                    print(f"    MAPE:     {metrics.get('mape', 0):.2f}%")
                    print(f"    Accuracy: {metrics.get('accuracy', 0):.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def compare_models(self, results_dict: Dict[str, Dict]) -> pd.DataFrame:
        """
        Compare multiple models side by side
        
        Args:
            results_dict: Dictionary of evaluation results by model name
            
        Returns:
            DataFrame with comparison
        """
        comparison_data = []
        
        for model_name, results in results_dict.items():
            reg_metrics = results.get('regression_metrics', {})
            
            row = {
                'Model': model_name,
                'MAE': reg_metrics.get('mae', np.nan),
                'RMSE': reg_metrics.get('rmse', np.nan),
                'MAPE (%)': reg_metrics.get('mape', np.nan),
                'R²': reg_metrics.get('r2', np.nan),
                'Accuracy (%)': reg_metrics.get('accuracy', np.nan)
            }
            
            # Add classification metrics if available
            if 'classification_metrics' in results and 'low' in results['classification_metrics']:
                cls_metrics = results['classification_metrics']['low']
                row['Precision'] = cls_metrics.get('precision', np.nan)
                row['Recall'] = cls_metrics.get('recall', np.nan)
                row['F1-Score'] = cls_metrics.get('f1', np.nan)
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        return df


def evaluate_ensemble_results(predictions_dict: Dict, y_true: np.ndarray) -> Dict:
    """
    Evaluate ensemble predictions and individual model predictions
    
    Args:
        predictions_dict: Dictionary with 'ensemble' and 'individual' predictions
        y_true: Actual values
        
    Returns:
        Dictionary with evaluation results for all models
    """
    evaluator = Evaluator()
    results = {}
    
    # Evaluate ensemble
    if 'ensemble' in predictions_dict:
        results['ensemble'] = evaluator.evaluate_predictions(
            y_true, 
            predictions_dict['ensemble'],
            'Ensemble'
        )
    
    # Evaluate individual models
    if 'individual' in predictions_dict:
        for model_name, predictions in predictions_dict['individual'].items():
            results[model_name] = evaluator.evaluate_predictions(
                y_true,
                predictions,
                model_name
            )
    
    # Create comparison
    comparison_df = evaluator.compare_models(results)
    
    return {
        'detailed_results': results,
        'comparison': comparison_df
    }
