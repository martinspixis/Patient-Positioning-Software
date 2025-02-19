# tests/recognition_test.py
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from patient_positioning.modules.face_recognition_module import PatientRecognition

class RecognitionAccuracyTest:
    def __init__(self):
        self.recognition = PatientRecognition()
        self.test_results = {
            'timestamp': [],
            'test_type': [],
            'patient_id': [],
            'actual_patient': [],
            'confidence': [],
            'lighting_condition': [],
            'angle': [],
            'success': []
        }
        
        # Test conditions
        self.lighting_conditions = ['normal', 'dim', 'bright']
        self.angles = ['front', 'left_30', 'right_30', 'up_20', 'down_20']
        
        # Create results directory
        os.makedirs('test_results', exist_ok=True)
        
    def setup_test_patient(self):
        """Register a test patient for accuracy testing"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return False
            
            patient_id = input("Enter test patient ID: ")
            print("\nCapturing reference image. Please look directly at camera.")
            print("Press 's' to capture, 'q' to quit")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Show frame with guidance
                self._draw_capture_guidance(frame)
                cv2.imshow('Setup Test Patient', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s'):
                    success, result = self.recognition.add_new_patient(frame, patient_id)
                    if success:
                        print(f"Test patient {patient_id} registered successfully")
                        return patient_id
                    else:
                        print("Failed to register patient, please try again")
                elif key == ord('q'):
                    break
                    
        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()
            
        return None

    def run_recognition_test(self, test_patient_id, duration_per_condition=5):
        """Run recognition tests under different conditions"""
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open camera")
                return

            for lighting in self.lighting_conditions:
                print(f"\nTesting under {lighting} lighting conditions")
                input(f"Please adjust lighting and press Enter to continue...")
                
                for angle in self.angles:
                    print(f"\nTesting at {angle} angle")
                    print("Please adjust position according to the angle")
                    input("Press Enter when ready...")
                    
                    start_time = time.time()
                    frames_processed = 0
                    
                    while time.time() - start_time < duration_per_condition:
                        ret, frame = cap.read()
                        if not ret:
                            print("Error: Could not read frame")
                            break
                            
                        # Run recognition
                        patient_id, confidence = self.recognition.recognize_patient(frame)
                        
                        # Record results
                        self.test_results['timestamp'].append(datetime.now())
                        self.test_results['test_type'].append('recognition')
                        self.test_results['patient_id'].append(patient_id)
                        self.test_results['actual_patient'].append(test_patient_id)
                        self.test_results['confidence'].append(confidence)
                        self.test_results['lighting_condition'].append(lighting)
                        self.test_results['angle'].append(angle)
                        self.test_results['success'].append(
                            patient_id == test_patient_id if patient_id else False)
                        
                        # Display realtime feedback
                        self._draw_recognition_feedback(frame, patient_id, confidence,
                                                     lighting, angle)
                        
                        cv2.imshow('Recognition Test', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
                        frames_processed += 1
                        
                        # Show progress
                        elapsed = time.time() - start_time
                        print(f"\rProcessed {frames_processed} frames "
                              f"({elapsed:.1f}/{duration_per_condition}s)", end='')
                
                print(f"\nCompleted {frames_processed} frames for {lighting} lighting")

        finally:
            if 'cap' in locals():
                cap.release()
            cv2.destroyAllWindows()

    def _draw_capture_guidance(self, frame):
        """Draw guidance for patient capture"""
        h, w = frame.shape[:2]
        
        # Draw face area guide
        face_box_size = min(w, h) // 2
        x = (w - face_box_size) // 2
        y = (h - face_box_size) // 2
        cv2.rectangle(frame, (x, y), (x + face_box_size, y + face_box_size), 
                     (0, 255, 0), 2)
        
        # Draw instructions
        cv2.putText(frame, "Position face within the box", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def _draw_recognition_feedback(self, frame, patient_id, confidence, lighting, angle):
        """Draw recognition feedback on frame"""
        info_text = [
            f"Patient ID: {patient_id if patient_id else 'Not recognized'}",
            f"Confidence: {confidence:.2f}",
            f"Lighting: {lighting}",
            f"Angle: {angle}"
        ]
        
        for i, text in enumerate(info_text):
            y = 30 + (i * 30)
            color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
            cv2.putText(frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def generate_report(self):
        """Generate test report with statistics and visualizations"""
        df = pd.DataFrame(self.test_results)
        
        # Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'test_results/recognition_test_{timestamp}.csv', index=False)
        
        # Calculate statistics
        stats = {
            'overall_accuracy': df['success'].mean(),
            'by_lighting': df.groupby('lighting_condition')['success'].mean(),
            'by_angle': df.groupby('angle')['success'].mean(),
            'average_confidence': df['confidence'].mean(),
            'false_positives': len(df[(df['patient_id'].notna()) & 
                                    (df['patient_id'] != df['actual_patient'])])
        }
        
        # Generate plots
        plt.figure(figsize=(15, 10))
        
        # Success rate by lighting condition
        plt.subplot(221)
        df.groupby('lighting_condition')['success'].mean().plot(kind='bar')
        plt.title('Success Rate by Lighting')
        plt.ylabel('Success Rate')
        
        # Success rate by angle
        plt.subplot(222)
        df.groupby('angle')['success'].mean().plot(kind='bar')
        plt.title('Success Rate by Angle')
        
        # Confidence distribution
        plt.subplot(223)
        plt.hist(df['confidence'], bins=20)
        plt.title('Confidence Distribution')
        
        # Time series of confidence
        plt.subplot(224)
        plt.plot(df.index, df['confidence'])
        plt.title('Confidence Over Time')
        
        # Save plots
        plt.tight_layout()
        plt.savefig(f'test_results/recognition_test_plots_{timestamp}.png')
        
        # Save statistics
        pd.DataFrame(stats).to_csv(
            f'test_results/recognition_test_stats_{timestamp}.csv')
        
        return stats

def main():
    try:
        test = RecognitionAccuracyTest()
        
        print("Face Recognition Accuracy Test")
        print("=============================")
        
        # Setup test patient
        test_patient_id = test.setup_test_patient()
        if not test_patient_id:
            print("Failed to setup test patient")
            return
        
        # Run recognition tests
        print("\nStarting recognition tests...")
        test.run_recognition_test(test_patient_id)
        
        # Generate and display report
        print("\nGenerating test report...")
        stats = test.generate_report()
        print("\nTest Statistics:")
        print(pd.DataFrame(stats))
        
        print("\nTest complete. Results saved in test_results directory.")
        
    except Exception as e:
        print(f"\nError during test: {str(e)}")
        raise
        
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()