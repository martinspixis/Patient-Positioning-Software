# main.py
# Disable warnings (must be at the very top)
import os
import sys

# Pievienot pilnu ceļu līdz testiem
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('mediapipe').setLevel(logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

# Regular imports
import cv2
from datetime import datetime
from patient_positioning.modules.face_tracker import FaceTracker
from patient_positioning.modules.pose_tracker import PoseTracker
from patient_positioning.modules.face_recognition_module import PatientRecognition
from patient_positioning.modules.calibration import CalibrationSystem
from patient_positioning.utils.visualization import draw_info, draw_pose_info
from patient_positioning.config.settings import *

class PatientPositioningSystem:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.pose_tracker = PoseTracker()
        self.recognition = PatientRecognition()
        self.calibration = CalibrationSystem()
        
        # Load calibration if exists
        self.calibration_loaded = self.calibration.load_calibration()
        if not self.calibration_loaded:
            print("Warning: System not calibrated")
        
        # Create data directories if they don't exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary data directories"""
        dirs = ['data', 'data/patients', 'data/calibration', 'data/logs']
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def check_calibration(self):
        """Check if system is calibrated and offer calibration"""
        if not self.calibration_loaded:
            print("\nSystem is not calibrated.")
            if input("Would you like to calibrate now? (y/n): ").lower() == 'y':
                self.run_calibration()
                return True
            print("Warning: Measurements may not be accurate without calibration")
            return False
        return True
    
    def run_calibration(self):
        """Run system calibration"""
        print("\nStarting System Calibration")
        print("==========================")
        print("Please place the calibration phantom in view")
        input("Press Enter when ready...")
        
        self.calibration.run_calibration()
        self.calibration_loaded = True
    
    def convert_measurements(self, points):
        """Convert pixel measurements to real-world coordinates"""
        if self.calibration_loaded:
            return self.calibration.convert_to_real_coordinates(points)
        return points
    
    def start_face_tracking(self):
        """Start face tracking with calibrated measurements"""
        self.check_calibration()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        current_patient = None
        tracking = False
        show_points = True
        
        print("\nFace Tracking Controls:")
        print("'s': Save reference position")
        print("'c': Clear visualization")
        print("'r': Reset tracking")
        print("'h': Hide/show points")
        print("'q': Return to menu")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Try to recognize patient
                if not current_patient:
                    patient_id, confidence = self.recognition.recognize_patient(frame)
                    if patient_id:
                        current_patient = patient_id
                        print(f"Recognized patient: {patient_id}")
                    frame = self.recognition.draw_recognition_info(frame, patient_id, confidence)

                # Face tracking
                face_detected, face_box, face_landmarks = self.face_tracker.detect_face(frame)
                
                if face_detected:
                    points = self.face_tracker.extract_anatomical_points(face_landmarks, frame.shape)
                    
                    if self.calibration_loaded:
                        points = self.convert_measurements(points)
                    
                    if show_points:
                        frame = draw_info(frame, self.face_tracker, points, tracking)
                        
                        # Log tracking data if active
                        if tracking and current_patient:
                            self._log_tracking_data(current_patient, 'face', points)

                cv2.imshow('Face Tracking', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    tracking = True
                    self.face_tracker.save_reference_position(points)
                elif key == ord('r'):
                    tracking = False
                    current_patient = None
                elif key == ord('h'):
                    show_points = not show_points
                elif key == ord('c'):
                    tracking = False
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def start_pose_tracking(self):
        """Start pose tracking with calibrated measurements"""
        self.check_calibration()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        tracking = False
        show_points = True
        
        print("\nPose Tracking Controls:")
        print("'s': Save reference position")
        print("'c': Clear visualization")
        print("'r': Reset tracking")
        print("'h': Hide/show points")
        print("'q': Return to menu")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                pose_detected, pose_landmarks = self.pose_tracker.detect_pose(frame)
                
                if pose_detected:
                    points = self.pose_tracker.extract_key_points(pose_landmarks, frame.shape)
                    
                    if self.calibration_loaded:
                        points = self.convert_measurements(points)
                    
                    if show_points:
                        frame = draw_pose_info(frame, self.pose_tracker, points, tracking)

                cv2.imshow('Pose Tracking', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    tracking = True
                    self.pose_tracker.save_reference_position(points)
                elif key == ord('r'):
                    tracking = False
                elif key == ord('h'):
                    show_points = not show_points
                elif key == ord('c'):
                    tracking = False
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def start_combined_tracking(self):
        """Start combined tracking with calibrated measurements"""
        self.check_calibration()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        current_patient = None
        tracking = False
        show_points = True
        show_face = True
        show_pose = True
        
        print("\nCombined Tracking Controls:")
        print("'s': Save reference position")
        print("'c': Clear visualization")
        print("'r': Reset tracking")
        print("'h': Hide/show points")
        print("'f': Toggle face tracking")
        print("'p': Toggle pose tracking")
        print("'q': Return to menu")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()

                # Patient recognition
                if not current_patient:
                    patient_id, confidence = self.recognition.recognize_patient(frame)
                    if patient_id:
                        current_patient = patient_id
                        print(f"Recognized patient: {patient_id}")
                    display_frame = self.recognition.draw_recognition_info(
                        display_frame, patient_id, confidence)

                # Face tracking
                face_points = None
                if show_face:
                    face_detected, face_box, face_landmarks = self.face_tracker.detect_face(frame)
                    if face_detected:
                        face_points = self.face_tracker.extract_anatomical_points(
                            face_landmarks, frame.shape)
                        if self.calibration_loaded:
                            face_points = self.convert_measurements(face_points)

                # Pose tracking
                pose_points = None
                if show_pose:
                    pose_detected, pose_landmarks = self.pose_tracker.detect_pose(frame)
                    if pose_detected:
                        pose_points = self.pose_tracker.extract_key_points(
                            pose_landmarks, frame.shape)
                        if self.calibration_loaded:
                            pose_points = self.convert_measurements(pose_points)

                # Draw combined visualization
                if show_points:
                    if face_points and show_face:
                        display_frame = draw_info(
                            display_frame, self.face_tracker, face_points, tracking)
                    if pose_points and show_pose:
                        display_frame = draw_pose_info(
                            display_frame, self.pose_tracker, pose_points, tracking)
                    
                    # Log data if tracking active
                    if tracking and current_patient:
                        if face_points and show_face:
                            self._log_tracking_data(current_patient, 'face', face_points)
                        if pose_points and show_pose:
                            self._log_tracking_data(current_patient, 'pose', pose_points)

                cv2.imshow('Combined Tracking', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    tracking = True
                    if face_points and show_face:
                        self.face_tracker.save_reference_position(face_points)
                    if pose_points and show_pose:
                        self.pose_tracker.save_reference_position(pose_points)
                elif key == ord('r'):
                    tracking = False
                    current_patient = None
                elif key == ord('h'):
                    show_points = not show_points
                elif key == ord('f'):
                    show_face = not show_face
                    print(f"Face tracking: {'ON' if show_face else 'OFF'}")
                elif key == ord('p'):
                    show_pose = not show_pose
                    print(f"Pose tracking: {'ON' if show_pose else 'OFF'}")
                elif key == ord('c'):
                    tracking = False
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def _log_tracking_data(self, patient_id, tracking_type, points):
        """Log tracking data to patient's file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = f"data/patients/{patient_id}_tracking_log.csv"
        
        # Create file with headers if it doesn't exist
        if not os.path.exists(log_file):
            with open(log_file, 'w', newline='') as f:
                f.write("timestamp,tracking_type,points_data\n")
        
        # Append tracking data
        with open(log_file, 'a', newline='') as f:
            points_str = str(points)  # Convert points to string representation
            f.write(f"{timestamp},{tracking_type},{points_str}\n")

    def add_new_patient(self):
        """Add new patient to the recognition system"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("\nNew Patient Registration")
        patient_id = input("Enter new patient ID: ")
        
        print("\nPosition patient's face in frame and press 's' to capture")
        print("Press 'q' to cancel")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Try to detect face for preview
            success, face_roi, face_rect = self.recognition.detect_face(frame)
            if success and face_rect is not None:
                x, y, w, h = face_rect
                cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_GREEN, 2)
                cv2.putText(frame, "Face detected", (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
            
            cv2.putText(frame, "Position face and press 's' to capture", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
            
            cv2.imshow('New Patient Registration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                success, result = self.recognition.add_new_patient(frame, patient_id)
                if success:
                    print(f"Successfully registered patient {patient_id}")
                    print(f"Photo saved: {result}")
                    break
                else:
                    print(f"Registration failed: {result}")
            elif key == ord('q'):
                print("Registration cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def view_patient_history(self):
        """View patient tracking history"""
        patient_id = input("Enter patient ID to view history: ")
        log_path = self.recognition.get_patient_log_path(patient_id)
        
        if os.path.exists(log_path):
            print(f"\nTracking history for patient {patient_id}:")
            with open(log_path, 'r') as f:
                print(f.read())
        else:
            print(f"No history found for patient {patient_id}")
        
        input("\nPress Enter to continue...")
    
    def show_menu(self):
        """Display main menu"""
        print("\n=== Patient Position Tracking System ===")
        print("1. Face Tracking with Recognition")
        print("2. Body Pose Tracking")
        print("3. Combined Tracking")
        print("4. Add New Patient")
        print("5. View Patient History")
        print("6. System Calibration")
        print("7. View Calibration Analytics")
        print("\nTest Suite:")
        print("8. Position Accuracy Test")
        print("9. Recognition Test")
        print("10. Performance Test")
        print("q. Quit")
        return input("Select option: ")
    
    def run(self):
        """Main program loop"""
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.start_face_tracking()
            elif choice == '2':
                self.start_pose_tracking()
            elif choice == '3':
                self.start_combined_tracking()
            elif choice == '4':
                self.add_new_patient()
            elif choice == '5':
                self.view_patient_history()
            elif choice == '6':
                self.run_calibration()
            elif choice == '7':
                self.calibration.show_analytics()
            elif choice == '8':
                try:
                    from patient_positioning.tests.position_accuracy_test import main as position_accuracy_main
                    position_accuracy_main()
                except Exception as e:
                    print(f"Error loading position accuracy test: {str(e)}")
            elif choice == '9':
                try:
                    from patient_positioning.tests.recognition_test import main as recognition_test_main
                    recognition_test_main()
                except Exception as e:
                    print(f"Error loading recognition test: {str(e)}")
            elif choice == '10':
                try:
                    from patient_positioning.tests.performance_test import main as performance_test_main
                    performance_test_main()
                except Exception as e:
                    print(f"Error loading performance test: {str(e)}")
            elif choice.lower() == 'q':
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    """Program entry point"""
    # Disable unnecessary warnings
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    # Do not filter CalibrationError warnings
    
    try:
        system = PatientPositioningSystem()
        system.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()