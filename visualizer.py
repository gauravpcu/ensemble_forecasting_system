"""
Visualization utilities for predictions and evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os
import config
from env_config import (
    PLOT_FIGURE_SIZE_WIDTH, PLOT_FIGURE_SIZE_HEIGHT,
    PLOT_SUBPLOT_WIDTH, PLOT_SUBPLOT_HEIGHT,
    PLOT_DPI, PLOT_ALPHA, PLOT_SCATTER_SIZE, PLOT_STYLE
)


class Visualizer:
    """Create visualizations for model predictions and evaluation"""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = output_dir or config.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set style
        sns.set_style(PLOT_STYLE)
        plt.rcParams['figure.figsize'] = (PLOT_FIGURE_SIZE_WIDTH, PLOT_FIGURE_SIZE_HEIGHT)
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   model_name: str = 'Model',
                                   save: bool = True):
        """
        Plot predicted vs actual values
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(PLOT_SUBPLOT_WIDTH, PLOT_SUBPLOT_HEIGHT))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=PLOT_ALPHA, s=PLOT_SCATTER_SIZE)
        axes[0].plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_name}: Predicted vs Actual')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residual plot
        residuals = y_pred - y_true
        axes[1].scatter(y_pred, residuals, alpha=PLOT_ALPHA, s=PLOT_SCATTER_SIZE)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 
                                   f'{model_name.lower().replace(" ", "_")}_predictions.png')
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"✓ Saved plot: {filename}")
        
        plt.close()
    
    def plot_error_distribution(self, y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               model_name: str = 'Model',
                               save: bool = True):
        """
        Plot error distribution
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            save: Whether to save the plot
        """
        errors = y_pred - y_true
        percentage_errors = (errors / (y_true + 1e-10)) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(PLOT_SUBPLOT_WIDTH, PLOT_SUBPLOT_HEIGHT))
        
        # Absolute error distribution
        axes[0].hist(np.abs(errors), bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(np.abs(errors)), color='r', linestyle='--', 
                       lw=2, label=f'Mean: {np.mean(np.abs(errors)):.2f}')
        axes[0].set_xlabel('Absolute Error')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'{model_name}: Absolute Error Distribution')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Percentage error distribution
        axes[1].hist(percentage_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(percentage_errors), color='r', linestyle='--', 
                       lw=2, label=f'Mean: {np.mean(percentage_errors):.2f}%')
        axes[1].set_xlabel('Percentage Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Percentage Error Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 
                                   f'{model_name.lower().replace(" ", "_")}_errors.png')
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"✓ Saved plot: {filename}")
        
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, save: bool = True):
        """
        Plot comparison of multiple models
        
        Args:
            comparison_df: DataFrame with model comparison
            save: Whether to save the plot
        """
        metrics = ['MAE', 'RMSE', 'MAPE (%)', 'Accuracy (%)']
        available_metrics = [m for m in metrics if m in comparison_df.columns]
        
        if not available_metrics:
            print("No metrics available for comparison plot")
            return
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(PLOT_SUBPLOT_HEIGHT*n_metrics, PLOT_SUBPLOT_HEIGHT))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            # Create bar plot
            bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                         edgecolor='black', alpha=0.7)
            
            # Color bars (green for best, red for worst)
            if metric in ['MAE', 'RMSE', 'MAPE (%)']:
                # Lower is better
                best_idx = comparison_df[metric].idxmin()
                worst_idx = comparison_df[metric].idxmax()
            else:
                # Higher is better
                best_idx = comparison_df[metric].idxmax()
                worst_idx = comparison_df[metric].idxmin()
            
            bars[best_idx].set_color('green')
            bars[worst_idx].set_color('red')
            
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(comparison_df) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 'model_comparison.png')
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"✓ Saved plot: {filename}")
        
        plt.close()
    
    def plot_category_performance(self, category_metrics: Dict, 
                                 model_name: str = 'Model',
                                 save: bool = True):
        """
        Plot performance by demand category
        
        Args:
            category_metrics: Dictionary of metrics by category
            model_name: Name of the model
            save: Whether to save the plot
        """
        categories = []
        accuracies = []
        counts = []
        
        for category, metrics in category_metrics.items():
            if 'count' in metrics and metrics['count'] > 0:
                categories.append(category.upper())
                accuracies.append(metrics.get('accuracy', 0))
                counts.append(metrics['count'])
        
        if not categories:
            print("No category data available for plotting")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(PLOT_SUBPLOT_WIDTH, PLOT_SUBPLOT_HEIGHT))
        
        # Accuracy by category
        bars = axes[0].bar(categories, accuracies, edgecolor='black', alpha=0.7)
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].set_title(f'{model_name}: Accuracy by Demand Category')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Color code bars
        colors = ['green' if acc > 80 else 'orange' if acc > 60 else 'red' 
                 for acc in accuracies]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Sample distribution
        axes[1].bar(categories, counts, edgecolor='black', alpha=0.7)
        axes[1].set_ylabel('Number of Samples')
        axes[1].set_title(f'{model_name}: Sample Distribution by Category')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save:
            filename = os.path.join(self.output_dir, 
                                   f'{model_name.lower().replace(" ", "_")}_categories.png')
            plt.savefig(filename, dpi=PLOT_DPI, bbox_inches='tight')
            print(f"✓ Saved plot: {filename}")
        
        plt.close()
    
    def create_all_visualizations(self, y_true: np.ndarray, 
                                 predictions_dict: Dict,
                                 evaluation_results: Dict):
        """
        Create all visualizations
        
        Args:
            y_true: Actual values
            predictions_dict: Dictionary with predictions
            evaluation_results: Dictionary with evaluation results
        """
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Plot for ensemble
        if 'ensemble' in predictions_dict:
            self.plot_predictions_vs_actual(
                y_true, 
                predictions_dict['ensemble'],
                'Ensemble',
                save=True
            )
            self.plot_error_distribution(
                y_true,
                predictions_dict['ensemble'],
                'Ensemble',
                save=True
            )
        
        # Plot for individual models
        if 'individual' in predictions_dict:
            for model_name, predictions in predictions_dict['individual'].items():
                self.plot_predictions_vs_actual(
                    y_true,
                    predictions,
                    model_name.upper(),
                    save=True
                )
                self.plot_error_distribution(
                    y_true,
                    predictions,
                    model_name.upper(),
                    save=True
                )
        
        # Model comparison
        if 'comparison' in evaluation_results:
            self.plot_model_comparison(
                evaluation_results['comparison'],
                save=True
            )
        
        # Category performance
        if 'detailed_results' in evaluation_results:
            for model_name, results in evaluation_results['detailed_results'].items():
                if 'category_metrics' in results:
                    self.plot_category_performance(
                        results['category_metrics'],
                        model_name.upper(),
                        save=True
                    )
        
        print(f"\n✓ All visualizations saved to: {self.output_dir}")
        print("="*60 + "\n")
