# tests/position_accuracy_test.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from patient_positioning.modules.face_tracker import FaceTracker
from patient_positioning.modules.pose_tracker import PoseTracker
from patient_positioning.utils.visualization import draw_info, draw_pose_info

class PositionAccuracyTest:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.pose_tracker = PoseTracker()
        
        # Initialize results storage
        self.test_results = {
            'timestamp': [],
            'test_type': [],
            'position': [],
            'deviation_x': [],
            'deviation_y': [],
            'deviation_z': [],
            'total_deviation': []
        }
        
        # Predefined test positions (in cm)
        self.test_positions = {
            'center': {'x': 0, 'y': 0, 'z': 100},
            'left': {'x': -10, 'y': 0, 'z': 100},
            'right': {'x': 10, 'y': 0, 'z': 100},
            'up': {'x': 0, 'y': 10, 'z': 100},
            'down': {'x': 0, 'y': -10, 'z': 100}
        }
        
        # Create results directory
        os.makedirs('test_results', exist_ok=True)

    def run_face_tracking_test(self, duration=10):
        """Test face tracking accuracy for each predefined position"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            for position_name, coords in self.test_positions.items():
                print(f"\nTesting position: {position_name}")
                print(f"Please move to position: X={coords['x']}, Y={coords['y']}, Z={coords['z']}")
                input("Press Enter when ready...")

                frames_collected = 0
                frame_start_time = datetime.now()

                while frames_collected < duration * 30:  # 30 fps * duration
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break

                    face_detected, face_box, face_landmarks = self.face_tracker.detect_face(frame)
                    if face_detected:
                        current_points = self.face_tracker.extract_anatomical_points(
                            face_landmarks, frame.shape)
                        
                        # Calculate deviations
                        self._record_measurement('face', position_name, current_points, coords)
                        
                        # Display realtime feedback
                        self._draw_feedback(frame, current_points, coords)

                    cv2.imshow('Face Tracking Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frames_collected += 1
                    
                    # Show progress
                    elapsed_time = (datetime.now() - frame_start_time).seconds
                    print(f"\rCollecting data: {frames_collected}/{duration * 30} frames "
                          f"({elapsed_time}/{duration} seconds)", end='')

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run_pose_tracking_test(self, duration=10):
        """Test pose tracking accuracy for each predefined position"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            for position_name, coords in self.test_positions.items():
                print(f"\nTesting position: {position_name}")
                print(f"Please move to position: X={coords['x']}, Y={coords['y']}, Z={coords['z']}")
                input("Press Enter when ready...")

                frames_collected = 0
                frame_start_time = datetime.now()

                while frames_collected < duration * 30:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break

                    pose_detected, pose_landmarks = self.pose_tracker.detect_pose(frame)
                    if pose_detected:
                        # Calculate deviations
                        self._record_measurement('pose', position_name, pose_landmarks, coords)
                        
                        # Display realtime feedback
                        frame = draw_pose_info(frame, self.pose_tracker, pose_landmarks, True)

                    cv2.imshow('Pose Tracking Test', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    frames_collected += 1
                    
                    # Show progress
                    elapsed_time = (datetime.now() - frame_start_time).seconds
                    print(f"\rCollecting data: {frames_collected}/{duration * 30} frames "
                          f"({elapsed_time}/{duration} seconds)", end='')

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def _record_measurement(self, test_type, position_name, points, target_coords):
        """Record measurement data"""
        if isinstance(points, dict):  # Face tracking points
            # Calculate average deviation for all tracked points
            x_dev = []
            y_dev = []
            z_dev = []
            for point_name, (x, y, z) in points.items():
                x_dev.append(abs(x - target_coords['x']))
                y_dev.append(abs(y - target_coords['y']))
                z_dev.append(abs(z - target_coords['z']))

            dev_x = np.mean(x_dev)
            dev_y = np.mean(y_dev)
            dev_z = np.mean(z_dev)
        else:  # Pose tracking points
            # Use central body point for pose tracking
            central_point = points.landmark[0]  # Use nose as reference
            dev_x = abs(central_point.x - target_coords['x'])
            dev_y = abs(central_point.y - target_coords['y'])
            dev_z = abs(central_point.z - target_coords['z'])

        total_dev = np.sqrt(dev_x**2 + dev_y**2 + dev_z**2)

        self.test_results['timestamp'].append(datetime.now())
        self.test_results['test_type'].append(test_type)
        self.test_results['position'].append(position_name)
        self.test_results['deviation_x'].append(dev_x)
        self.test_results['deviation_y'].append(dev_y)
        self.test_results['deviation_z'].append(dev_z)
        self.test_results['total_deviation'].append(total_dev)

    def _draw_feedback(self, frame, points, target_coords):
        """Draw visual feedback on frame"""
        for point_name, (x, y, z) in points.items():
            # Draw point
            cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            
            # Calculate deviation from target
            dev_x = x - target_coords['x']
            dev_y = y - target_coords['y']
            dev_z = z - target_coords['z']
            total_dev = np.sqrt(dev_x**2 + dev_y**2 + dev_z**2)
            
            # Draw deviation info
            cv2.putText(frame, f"{point_name}: {total_dev:.1f}mm", 
                       (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def generate_report(self):
        """Generate test report with statistics and visualizations"""
        df = pd.DataFrame(self.test_results)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'test_results/accuracy_test_{timestamp}.csv', index=False)
        
        # Calculate statistics
        stats = df.groupby(['test_type', 'position']).agg({
            'deviation_x': ['mean', 'std', 'min', 'max'],
            'deviation_y': ['mean', 'std', 'min', 'max'],
            'deviation_z': ['mean', 'std', 'min', 'max'],
            'total_deviation': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        # Generate plots
        plt.figure(figsize=(15, 10))
        
        # Total deviation boxplot
        plt.subplot(221)
        df.boxplot(column='total_deviation', by=['test_type', 'position'])
        plt.title('Total Deviation by Position')
        plt.xticks(rotation=45)
        
        # Deviation components
        plt.subplot(222)
        df.boxplot(column=['deviation_x', 'deviation_y', 'deviation_z'])
        plt.title('Deviation Components')
        
        # Time series
        plt.subplot(223)
        for test_type in df['test_type'].unique():
            test_data = df[df['test_type'] == test_type]
            plt.plot(test_data.index, test_data['total_deviation'], 
                    label=test_type, alpha=0.7)
        plt.title('Deviation Over Time')
        plt.legend()
        
        # Position accuracy heatmap
        plt.subplot(224)
        accuracy_matrix = df.pivot_table(
            values='total_deviation', 
            index='test_type', 
            columns='position', 
            aggfunc='mean'
        )
        sns.heatmap(accuracy_matrix, annot=True, cmap='RdYlGn_r')
        plt.title('Position Accuracy Heatmap')
        
        # Save plots
        plt.tight_layout()
        plt.savefig(f'test_results/accuracy_test_plots_{timestamp}.png')
        
        # Save statistics
        stats.to_csv(f'test_results/accuracy_test_stats_{timestamp}.csv')
        
        return stats

def main():
    test = PositionAccuracyTest()
    
    print("Position Accuracy Test")
    print("=====================")
    
    try:
        # Run face tracking test
        print("\nStarting Face Tracking Test...")
        test.run_face_tracking_test()
        
        # Run pose tracking test
        print("\nStarting Pose Tracking Test...")
        test.run_pose_tracking_test()
        
        # Generate and display report
        print("\nGenerating Test Report...")
        stats = test.generate_report()
        print("\nTest Statistics:")
        print(stats)
        
        print("\nTest complete. Results saved in test_results directory.")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()