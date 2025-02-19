# tests/test_utils/data_logger.py
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime

class TestDataLogger:
    def __init__(self, test_name):
        self.test_name = test_name
        self.base_dir = 'test_results'
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories"""
        dirs = [
            self.base_dir,
            os.path.join(self.base_dir, 'raw_data'),
            os.path.join(self.base_dir, 'statistics'),
            os.path.join(self.base_dir, 'plots'),
            os.path.join(self.base_dir, 'reports')
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_raw_data(self, data, suffix=''):
        """Save raw test data to CSV"""
        filename = f"{self.test_name}_{self.timestamp}{suffix}.csv"
        filepath = os.path.join(self.base_dir, 'raw_data', filename)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_statistics(self, stats, suffix=''):
        """Save statistical analysis"""
        filename = f"{self.test_name}_stats_{self.timestamp}{suffix}.json"
        filepath = os.path.join(self.base_dir, 'statistics', filename)
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=4)
        return filepath
    
    def save_plot(self, plot_figure, suffix=''):
        """Save plot figure"""
        filename = f"{self.test_name}_plot_{self.timestamp}{suffix}.png"
        filepath = os.path.join(self.base_dir, 'plots', filename)
        
        plot_figure.savefig(filepath)
        return filepath
    
    def generate_report(self, test_results, include_plots=True):
        """Generate comprehensive test report"""
        report = {
            'test_name': self.test_name,
            'timestamp': self.timestamp,
            'summary': test_results.get('summary', {}),
            'statistics': test_results.get('statistics', {}),
            'configuration': test_results.get('configuration', {}),
            'recommendations': test_results.get('recommendations', [])
        }
        
        # Save report
        filename = f"{self.test_name}_report_{self.timestamp}.json"
        filepath = os.path.join(self.base_dir, 'reports', filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        
        # Generate HTML report
        html_report = self._generate_html_report(report, include_plots)
        html_filepath = filepath.replace('.json', '.html')
        
        with open(html_filepath, 'w') as f:
            f.write(html_report)
        
        return filepath, html_filepath
    
    def _generate_html_report(self, report, include_plots):
        """Generate HTML formatted report"""
        html = f"""
        <html>
        <head>
            <title>Test Report: {self.test_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin: 20px 0; }}
                .stat-table {{ border-collapse: collapse; width: 100%; }}
                .stat-table td, .stat-table th {{ 
                    border: 1px solid #ddd; padding: 8px; 
                }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Test Report: {self.test_name}</h1>
            <p>Generated: {report['timestamp']}</p>
            
            <div class="section">
                <h2>Summary</h2>
                <pre>{json.dumps(report['summary'], indent=2)}</pre>
            </div>
            
            <div class="section">
                <h2>Statistics</h2>
                <pre>{json.dumps(report['statistics'], indent=2)}</pre>
            </div>
        """
        
        if include_plots:
            html += f"""
            <div class="section">
                <h2>Plots</h2>
                <div class="plot">
                    <img src="../plots/{self.test_name}_plot_{self.timestamp}.png" 
                         alt="Test Results Plot">
                </div>
            </div>
            """
        
        if report['recommendations']:
            html += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
            """
            for rec in report['recommendations']:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def append_to_log(self, message):
        """Append message to running log file"""
        log_file = os.path.join(self.base_dir, f"{self.test_name}_log.txt")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")

class TestDataAnalyzer:
    @staticmethod
    def calculate_basic_stats(data):
        """Calculate basic statistical measures"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data))
        }
    
    @staticmethod
    def analyze_time_series(data, column):
        """Analyze time series data"""
        df = pd.DataFrame(data)
        return {
            'trend': TestDataAnalyzer._calculate_trend(df[column]),
            'variance': float(df[column].var()),
            'stability_score': TestDataAnalyzer._calculate_stability(df[column])
        }
    
    @staticmethod
    def _calculate_trend(series):
        """Calculate linear trend"""
        x = np.arange(len(series))
        slope, _ = np.polyfit(x, series, 1)
        return float(slope)
    
    @staticmethod
    def _calculate_stability(series):
        """Calculate stability score (0-1)"""
        normalized_std = np.std(series) / (np.max(series) - np.min(series))
        return float(1 - normalized_std)