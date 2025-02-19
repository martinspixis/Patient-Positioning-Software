# modules/calibration_analytics.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from ..config.settings import *

class CalibrationAnalytics:
    def __init__(self):
        # Iegūstam pilnu ceļu līdz direktorijām
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))
        self.calibration_dir = os.path.join(base_dir, 'data', 'calibration')
        self.analytics_dir = os.path.join(self.calibration_dir, 'analytics')
        
        # Izveidojam nepieciešamās direktorijas
        os.makedirs(self.analytics_dir, exist_ok=True)
        print(f"Analytics directory created/exists at: {self.analytics_dir}")
        
        # Statistikas fails
        self.stats_file = os.path.join(self.analytics_dir, 'calibration_stats.json')
        print(f"Stats file will be saved at: {self.stats_file}")
        self.load_stats()

    def load_stats(self):
        """Ielādē vai izveido jaunu statistikas failu"""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
                print(f"Loaded existing stats from {self.stats_file}")
            except Exception as e:
                print(f"Error loading stats: {e}")
                self._create_new_stats()
        else:
            print("No existing stats file found, creating new")
            self._create_new_stats()

    def _create_new_stats(self):
        """Izveido jaunu statistikas struktūru"""
        self.stats = {
            'total_attempts': 0,
            'successful_calibrations': 0,
            'failed_calibrations': 0,
            'average_duration': 0,
            'accuracy_history': [],
            'calibration_history': []
        }
        self.save_stats()

    def save_stats(self):
        """Saglabā statistiku failā"""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=4)
            print(f"Stats saved successfully to {self.stats_file}")
        except Exception as e:
            print(f"Error saving stats: {e}")

    def log_calibration_attempt(self, success, duration, accuracy=None, error_message=None):
        """Pievieno jaunu kalibrēšanas mēģinājumu statistikai"""
        timestamp = datetime.now().isoformat()
        
        # Atjaunina pamata statistiku
        self.stats['total_attempts'] += 1
        if success:
            self.stats['successful_calibrations'] += 1
        else:
            self.stats['failed_calibrations'] += 1
        
        # Atjaunina vidējo ilgumu
        n = self.stats['total_attempts']
        old_avg = self.stats['average_duration']
        self.stats['average_duration'] = (old_avg * (n-1) + duration) / n
        
        # Pievieno vēsturei
        history_entry = {
            'timestamp': timestamp,
            'success': success,
            'duration': duration,
            'accuracy': accuracy,
            'error': error_message
        }
        self.stats['calibration_history'].append(history_entry)
        
        if accuracy is not None:
            self.stats['accuracy_history'].append({
                'timestamp': timestamp,
                'accuracy': accuracy
            })
        
        print(f"Logged new calibration attempt: success={success}, duration={duration:.2f}s")
        self.save_stats()

    def generate_report(self):
        """Ģenerē detalizētu atskaiti par kalibrēšanas statistiku"""
        df = pd.DataFrame(self.stats['calibration_history'])
        if df.empty:
            return "Nav pieejamu kalibrēšanas datu."
        
        report = "Kalibrēšanas Statistikas Atskaite\n"
        report += "================================\n\n"
        
        # Pamata statistika
        success_rate = (self.stats['successful_calibrations'] / 
                       self.stats['total_attempts'] * 100 if self.stats['total_attempts'] > 0 else 0)
        
        report += f"Kopējais mēģinājumu skaits: {self.stats['total_attempts']}\n"
        report += f"Veiksmīgas kalibrācijas: {self.stats['successful_calibrations']}\n"
        report += f"Neveiksmīgas kalibrācijas: {self.stats['failed_calibrations']}\n"
        report += f"Veiksmju procents: {success_rate:.1f}%\n"
        report += f"Vidējais kalibrēšanas ilgums: {self.stats['average_duration']:.2f} sek\n\n"
        
        if not df.empty:
            # Laika analīze
            if 'duration' in df.columns:
                report += "Laika Analīze:\n"
                success_times = df[df['success']]['duration'].mean() if 'success' in df.columns else 0
                fail_times = df[~df['success']]['duration'].mean() if 'success' in df.columns else 0
                report += f"Vidējais ilgums veiksmīgām kalibrācijām: {success_times:.2f} sek\n"
                report += f"Vidējais ilgums neveiksmīgām kalibrācijām: {fail_times:.2f} sek\n\n"
            
            # Precizitātes analīze
            if 'accuracy' in df.columns and df['accuracy'].notna().any():
                report += "Precizitātes Analīze:\n"
                report += f"Vidējā precizitāte: {df['accuracy'].mean():.2f}\n"
                report += f"Maksimālā precizitāte: {df['accuracy'].max():.2f}\n"
                report += f"Minimālā precizitāte: {df['accuracy'].min():.2f}\n"
        
        return report

    def plot_analytics(self):
        """Ģenerē vizuālos grafikus par kalibrēšanas statistiku"""
        print("\nStarting plot generation...")
        
        df = pd.DataFrame(self.stats['calibration_history'])
        if df.empty:
            print("No data available for plotting")
            return None
        
        try:
            # Uzstādam grafika stilu
            plt.style.use('default')  # Izmantojam default stilu seaborn vietā
            
            # Uzstādam fonā pelēku režģi
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['grid.color'] = '#cccccc'
            
            fig = plt.figure(figsize=(15, 10))
            
            # Colors
            success_color = '#2ecc71'  # zaļš
            failure_color = '#e74c3c'  # sarkans
            
            # 1. Veiksmju/Neveiksmju sadalījums
            print("Plotting success/failure distribution...")
            ax1 = plt.subplot(221)
            if 'success' in df.columns:
                success_counts = df['success'].value_counts()
                ax1.pie([success_counts.get(True, 0), success_counts.get(False, 0)],
                       labels=['Veiksmīgi', 'Neveiksmīgi'],
                       colors=[success_color, failure_color],
                       autopct='%1.1f%%',
                       shadow=True)
                ax1.set_title('Kalibrēšanas Rezultātu Sadalījums')
            
            # 2. Kalibrēšanas ilguma histogramma
            print("Plotting duration histogram...")
            ax2 = plt.subplot(222)
            if 'duration' in df.columns:
                n, bins, patches = ax2.hist(df['duration'], bins=20, 
                                          color=success_color, alpha=0.7)
                ax2.set_title('Kalibrēšanas Ilguma Sadalījums')
                ax2.set_xlabel('Ilgums (sek)')
                ax2.set_ylabel('Biežums')
            
            # 3. Precizitātes izmaiņas laika gaitā
            print("Plotting accuracy changes...")
            ax3 = plt.subplot(223)
            if 'accuracy' in df.columns and 'timestamp' in df.columns and df['accuracy'].notna().any():
                ax3.plot(df['timestamp'], df['accuracy'], 
                        marker='o', color=success_color, 
                        linestyle='-', linewidth=2, markersize=6)
                ax3.set_title('Precizitātes Izmaiņas Laika Gaitā')
                ax3.set_xlabel('Laiks')
                ax3.set_ylabel('Precizitāte')
                plt.xticks(rotation=45)
            
            # 4. Veiksmju/Neveiksmju tendence laika gaitā
            print("Plotting success trend...")
            ax4 = plt.subplot(224)
            if 'success' in df.columns and 'timestamp' in df.columns:
                df['success_numeric'] = df['success'].astype(int)
                df['success_rate'] = df['success_numeric'].rolling(window=5, min_periods=1).mean()
                ax4.plot(df['timestamp'], df['success_rate'], 
                        color=success_color, linewidth=2)
                ax4.set_title('Veiksmju Tendence Laika Gaitā')
                ax4.set_xlabel('Laiks')
                ax4.set_ylabel('Veiksmju Proporcija')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Saglabā grafiku
            plot_path = os.path.join(self.analytics_dir, 'calibration_analytics.png')
            print(f"Saving plot to: {plot_path}")
            plt.savefig(plot_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            if os.path.exists(plot_path):
                stat_info = os.stat(plot_path)
                print(f"Plot saved successfully. File size: {stat_info.st_size} bytes")
                return plot_path
            else:
                print("Error: Plot file was not created")
                return None
                
        except Exception as e:
            print(f"Error generating plot: {str(e)}")
            import traceback
            traceback.print_exc()
            return None