# tests/performance_test.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import psutil
import threading
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from patient_positioning.modules.face_tracker import FaceTracker
from patient_positioning.modules.pose_tracker import PoseTracker
from patient_positioning.modules.face_recognition_module import PatientRecognition

class PerformanceTest:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.pose_tracker = PoseTracker()
        self.recognition = PatientRecognition()
        
        # Initialize monitoring variables
        self.current_cpu = 0
        self.current_memory = 0
        
        self.test_results = {
            'timestamp': [],
            'test_type': [],
            'resolution': [],
            'fps': [],
            'cpu_usage': [],
            'memory_usage': [],
            'processing_time': [],
            'detection_success': []
        }
        
        # Test configurations
        self.resolutions = [
            (640, 480),   # VGA
            (1280, 720),  # 720p
            (1920, 1080)  # 1080p
        ]
        
        self.test_durations = {
            'face_tracking': 30,    # seconds
            'pose_tracking': 30,
            'combined': 30,
            'recognition': 30
        }

    def monitor_system_resources(self, stop_event):
        """Monitor CPU and memory usage in a separate thread"""
        while not stop_event.is_set():
            self.current_cpu = psutil.cpu_percent(interval=1)
            self.current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            time.sleep(0.1)

    def test_camera_performance(self, resolution):
        """Test camera performance at specific resolution"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return None
            
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Requested resolution: {resolution}")
        print(f"Actual resolution: {actual_width}x{actual_height}")
        
        return cap

    def run_tracking_performance_test(self):
        """Test performance of different tracking modes"""
        # Start resource monitoring thread
        stop_monitoring = threading.Event()
        monitor_thread = threading.Thread(target=self.monitor_system_resources, args=(stop_monitoring,))
        monitor_thread.start()
        
        try:
            for resolution in self.resolutions:
                print(f"\nTesting resolution: {resolution[0]}x{resolution[1]}")
                
                # Setup camera
                cap = self.test_camera_performance(resolution)
                if not cap:
                    continue
                    
                # Test each tracking mode
                for test_type in ['face_tracking', 'pose_tracking', 'combined']:
                    print(f"\nTesting {test_type}...")
                    
                    start_time = time.time()
                    frames_processed = 0
                    frame_times = []
                    
                    while time.time() - start_time < self.test_durations[test_type]:
                        loop_start = time.time()
                        
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame based on test type
                        if test_type == 'face_tracking':
                            success = self.process_face_tracking(frame)
                        elif test_type == 'pose_tracking':
                            success = self.process_pose_tracking(frame)
                        else:  # combined
                            success = self.process_combined_tracking(frame)
                        
                        # Calculate frame processing time
                        processing_time = time.time() - loop_start
                        frame_times.append(processing_time)
                        
                        # Record metrics
                        self.test_results['timestamp'].append(datetime.now())
                        self.test_results['test_type'].append(test_type)
                        self.test_results['resolution'].append(f"{resolution[0]}x{resolution[1]}")
                        self.test_results['fps'].append(1.0 / processing_time)
                        self.test_results['cpu_usage'].append(self.current_cpu)
                        self.test_results['memory_usage'].append(self.current_memory)
                        self.test_results['processing_time'].append(processing_time)
                        self.test_results['detection_success'].append(success)
                        
                        frames_processed += 1
                        
                        # Show progress
                        cv2.imshow('Performance Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    avg_fps = frames_processed / self.test_durations[test_type]
                    print(f"Average FPS: {avg_fps:.2f}")
                    print(f"Average processing time: {np.mean(frame_times)*1000:.2f}ms")
                
                # Cleanup camera
                cap.release()
                
        finally:
            # Cleanup
            stop_monitoring.set()
            monitor_thread.join()
            cv2.destroyAllWindows()

    def process_face_tracking(self, frame):
        """Process frame with face tracking"""
        face_detected, face_box, face_landmarks = self.face_tracker.detect_face(frame)
        return face_detected

    def process_pose_tracking(self, frame):
        """Process frame with pose tracking"""
        pose_detected, pose_landmarks = self.pose_tracker.detect_pose(frame)
        return pose_detected

    def process_combined_tracking(self, frame):
        """Process frame with combined tracking"""
        face_success = self.process_face_tracking(frame)
        pose_success = self.process_pose_tracking(frame)
        return face_success and pose_success

    def generate_report(self):
        """Generate performance test report"""
        df = pd.DataFrame(self.test_results)
        
        # Create results directory
        os.makedirs('test_results', exist_ok=True)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'test_results/performance_test_{timestamp}.csv', index=False)
        
        # Calculate statistics
        stats = df.groupby(['test_type', 'resolution']).agg({
            'fps': ['mean', 'std', 'min', 'max'],
            'cpu_usage': ['mean', 'max'],
            'memory_usage': ['mean', 'max'],
            'processing_time': ['mean', 'std'],
            'detection_success': 'mean'
        }).round(2)
        
        # Generate plots
        plt.figure(figsize=(15, 10))
        
        # FPS by resolution and test type
        plt.subplot(221)
        df.boxplot(column='fps', by=['test_type', 'resolution'])
        plt.title('FPS Distribution')
        plt.xticks(rotation=45)
        
        # CPU usage
        plt.subplot(222)
        df.boxplot(column='cpu_usage', by=['test_type', 'resolution'])
        plt.title('CPU Usage Distribution')
        plt.xticks(rotation=45)
        
        # Memory usage over time
        plt.subplot(223)
        for test_type in df['test_type'].unique():
            test_data = df[df['test_type'] == test_type]
            plt.plot(test_data.index, test_data['memory_usage'], 
                    label=test_type, alpha=0.7)
        plt.title('Memory Usage Over Time')
        plt.legend()
        
        # Success rate
        plt.subplot(224)
        success_rate = df.groupby(['test_type', 'resolution'])['detection_success'].mean()
        success_rate.unstack().plot(kind='bar')
        plt.title('Detection Success Rate')
        plt.xticks(rotation=45)
        
        # Save plots
        plt.tight_layout()
        plt.savefig(f'test_results/performance_test_plots_{timestamp}.png')
        
        # Save statistics
        stats.to_csv(f'test_results/performance_test_stats_{timestamp}.csv')
        
        return stats

def main():
    test = PerformanceTest()
    
    print("Performance Test Suite")
    print("====================")
    
    # Run performance tests
    print("\nStarting performance tests...")
    test.run_tracking_performance_test()
    
    # Generate and display report
    print("\nGenerating performance report...")
    stats = test.generate_report()
    print("\nTest Statistics:")
    print(stats)
    
    print("\nTest complete. Results saved in test_results directory.")

if __name__ == "__main__":
    main()