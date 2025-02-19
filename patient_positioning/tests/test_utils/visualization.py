# tests/test_utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime

class TestVisualizer:
    def __init__(self, style='darkgrid'):
        """Initialize visualizer with specified style"""
        sns.set_style(style)
        self.colors = sns.color_palette("husl", 8)
        
    def create_performance_plot(self, data, metrics, title="Performance Metrics"):
        """Create multi-metric performance plot"""
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
            
        for ax, (metric, color) in zip(axes, zip(metrics, self.colors)):
            sns.lineplot(data=data, x='timestamp', y=metric, ax=ax, color=color)
            ax.set_title(f"{metric} over Time")
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_comparison_plot(self, data, x, y, hue=None, title="Comparison"):
        """Create comparison plot (boxplot or violin plot)"""
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=data, x=x, y=y, hue=hue)
        plt.title(title)
        plt.xticks(rotation=45)
        return plt.gcf()
    
    def create_heatmap(self, data, title="Correlation Heatmap"):
        """Create correlation heatmap"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        return plt.gcf()
    
    def plot_tracking_accuracy(self, positions, predictions, title="Tracking Accuracy"):
        """Plot actual vs predicted positions"""
        plt.figure(figsize=(12, 6))
        
        # Plot actual positions
        plt.scatter(positions[:, 0], positions[:, 1], 
                   c='blue', label='Actual', alpha=0.6)
        
        # Plot predictions
        plt.scatter(predictions[:, 0], predictions[:, 1], 
                   c='red', label='Predicted', alpha=0.6)
        
        # Draw connecting lines
        for pos, pred in zip(positions, predictions):
            plt.plot([pos[0], pred[0]], [pos[1], pred[1]], 
                    'g--', alpha=0.3)
        
        plt.title(title)
        plt.legend()
        return plt.gcf()
    
    def plot_3d_tracking(self, positions, predictions=None):
        """Create 3D visualization of tracking"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot actual positions
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                  c='blue', label='Actual')
        
        if predictions is not None:
            # Plot predictions
            ax.scatter(predictions[:, 0], predictions[:, 1], predictions[:, 2],
                      c='red', label='Predicted')
            
            # Draw connecting lines
            for pos, pred in zip(positions, predictions):
                ax.plot([pos[0], pred[0]], 
                       [pos[1], pred[1]], 
                       [pos[2], pred[2]], 'g--', alpha=0.3)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        return fig
    
    def create_summary_dashboard(self, test_results):
        """Create comprehensive dashboard of test results"""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 2)
        
        # Performance metrics
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_metrics(test_results, ax1)
        
        # Success rate
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_success_rate(test_results, ax2)
        
        # Resource usage
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_resource_usage(test_results, ax3)
        
        # Statistics
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_statistics_summary(test_results, ax4)
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_metrics(self, results, ax):
        """Plot performance metrics subplot"""
        if 'fps' in results:
            sns.lineplot(data=results, x='timestamp', y='fps', ax=ax)
            ax.set_title('Performance (FPS)')
            ax.tick_params(axis='x', rotation=45)
    
    def _plot_success_rate(self, results, ax):
        """Plot success rate subplot"""
        if 'success' in results:
            success_rate = results.groupby('test_type')['success'].mean()
            success_rate.plot(kind='bar', ax=ax)
            ax.set_title('Success Rate by Test Type')
            ax.set_ylim(0, 1)
    
    def _plot_resource_usage(self, results, ax):
        """Plot resource usage subplot"""
        if 'cpu_usage' in results and 'memory_usage' in results:
            ax2 = ax.twinx()
            
            sns.lineplot(data=results, x='timestamp', y='cpu_usage', 
                        ax=ax, color='blue', label='CPU')
            sns.lineplot(data=results, x='timestamp', y='memory_usage', 
                        ax=ax2, color='red', label='Memory')
            
            ax.set_ylabel('CPU Usage (%)')
            ax2.set_ylabel('Memory Usage (MB)')
            ax.set_title('Resource Usage Over Time')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2)
    
    def _plot_statistics_summary(self, results, ax):
        """Plot statistics summary subplot"""
        if 'processing_time' in results:
            stats = results.groupby('test_type')['processing_time'].agg(
                ['mean', 'std', 'min', 'max'])
            stats.plot(kind='bar', ax=ax)
            ax.set_title('Processing Time Statistics')
            ax.tick_params(axis='x', rotation=45)
    
    def save_plots(self, filename_prefix):
        """Save all open plots"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all current figures
        for i, fig in enumerate(plt.get_fignums()):
            figure = plt.figure(fig)
            figure.savefig(f"{filename_prefix}_{i}_{timestamp}.png")
        
        plt.close('all')
    
    def show_plots(self):
        """Display all plots"""
        plt.show()