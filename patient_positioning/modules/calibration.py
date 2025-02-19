# modules/calibration.py
import cv2
import numpy as np
import json
import os
from datetime import datetime
import time
from patient_positioning.config.settings import *
from patient_positioning.modules.calibration_analytics import CalibrationAnalytics

class CalibrationError(Exception):
    """Custom exception for calibration errors"""
    pass

class CalibrationSystem:
    def __init__(self):
        # Configuration parameters
        self.config = {
            'min_frames': 5,
            'calibration_duration': 10,
            'validation_threshold': 0.8,
            'calibration_validity_days': 30
        }
        
        # Calibration data structure
        self.calibration_data = {
            'timestamp': None,
            'camera_matrix': None,
            'distortion_coeffs': None,
            'pixel_to_mm_ratio': None,
            'reference_points': None,
            'validation_results': None
        }
        
        # Calibration phantom specifications
        self.marker_grid = (3, 3)  # 3x3 grid
        self.marker_spacing = 100  # mm
        self.marker_diameter = 20  # mm
        
        # Initialize analytics
        self.analytics = CalibrationAnalytics()
        
        # Create calibration directory if it doesn't exist
        os.makedirs('data/calibration', exist_ok=True)

    def detect_calibration_markers(self, frame):
        """
        Detect calibration markers in the frame
        Returns: numpy array of detected circles or None
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 2)
            
            # Detect circular markers
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1,
                minDist=50,
                param1=50,
                param2=30,
                minRadius=int(self.marker_diameter/2 * 0.8),  # 20% tolerance
                maxRadius=int(self.marker_diameter/2 * 1.2)
            )
            
            if circles is not None:
                circles = np.uint16(np.around(circles[0]))
                # Sort circles by position (left to right, top to bottom)
                circles = sorted(circles, key=lambda x: (x[1], x[0]))
                return circles
                
        except Exception as e:
            raise CalibrationError(f"Marker detection failed: {str(e)}")
            
        return None

    def calibrate_camera(self, frame, circles):
        """
        Calculate camera matrix and distortion coefficients
        Returns: calibration success (bool)
        """
        try:
            h, w = frame.shape[:2]
            
            # Define real-world coordinates of markers
            object_points = np.zeros((self.marker_grid[0] * self.marker_grid[1], 3), np.float32)
            object_points[:, :2] = np.mgrid[0:self.marker_grid[0], 
                                          0:self.marker_grid[1]].T.reshape(-1, 2) * self.marker_spacing
            
            # Image points from detected circles
            image_points = np.array([circle[:2] for circle in circles], dtype=np.float32)
            
            # Calculate camera matrix
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                [object_points], [image_points], (w, h), None, None
            )
            
            if ret:
                self.calibration_data['camera_matrix'] = mtx.tolist()
                self.calibration_data['distortion_coeffs'] = dist.tolist()
                return True
                
        except Exception as e:
            raise CalibrationError(f"Camera calibration failed: {str(e)}")
            
        return False

    def calculate_pixel_ratio(self, circles):
        """
        Calculate pixel to millimeter ratio
        Returns: calculated ratio (float)
        """
        try:
            pixel_distances = []
            for i in range(len(circles)-1):
                for j in range(i+1, len(circles)):
                    pixel_dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 + 
                        (circles[i][1] - circles[j][1])**2
                    )
                    pixel_distances.append(pixel_dist)
            
            # Known distance between markers
            real_distance = self.marker_spacing  # mm
            
            # Calculate average ratio
            pixel_to_mm = real_distance / np.mean(pixel_distances)
            self.calibration_data['pixel_to_mm_ratio'] = float(pixel_to_mm)
            
            return pixel_to_mm
            
        except Exception as e:
            raise CalibrationError(f"Pixel ratio calculation failed: {str(e)}")

    def validate_calibration(self, frame, circles):
        """
        Validate calibration accuracy
        Returns: validation results dictionary
        """
        try:
            validation_results = {
                'distance_error': [],
                'angle_error': [],
                'marker_detection_accuracy': 0,
                'is_valid': False
            }
            
            # Check number of detected markers
            expected_markers = self.marker_grid[0] * self.marker_grid[1]
            detected_markers = len(circles)
            validation_results['marker_detection_accuracy'] = detected_markers / expected_markers
            
            # Validate distances
            for i in range(len(circles)-1):
                for j in range(i+1, len(circles)):
                    measured_dist = np.sqrt(
                        (circles[i][0] - circles[j][0])**2 +
                        (circles[i][1] - circles[j][1])**2
                    ) * self.calibration_data['pixel_to_mm_ratio']
                    
                    # Compare with known distance
                    error = abs(measured_dist - self.marker_spacing)
                    validation_results['distance_error'].append(error)
            
            # Set validation status
            avg_error = np.mean(validation_results['distance_error'])
            validation_results['is_valid'] = (
                validation_results['marker_detection_accuracy'] >= self.config['validation_threshold'] and
                avg_error < 5.0  # 5mm error threshold
            )
            
            self.calibration_data['validation_results'] = validation_results
            return validation_results
            
        except Exception as e:
            raise CalibrationError(f"Calibration validation failed: {str(e)}")

    def run_calibration(self, duration=None):
        """
        Run complete calibration procedure
        Returns: calibration success (bool)
        """
        if duration is None:
            duration = self.config['calibration_duration']
            
        start_time = time.time()
        success = False
        error_message = None
        accuracy = None
        cap = None
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise CalibrationError("Could not open camera")

            print("\nStarting calibration procedure...")
            print("Place calibration phantom in view")
            
            frames_collected = 0
            successful_frames = 0
            all_circles = []
            
            start_collection = datetime.now()
            
            while (datetime.now() - start_collection).seconds < duration:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                circles = self.detect_calibration_markers(frame)
                frames_collected += 1
                
                if circles is not None and len(circles) == 9:  # All markers detected
                    successful_frames += 1
                    all_circles.append(circles)
                    self._draw_calibration_feedback(frame, circles, True)
                else:
                    self._draw_calibration_feedback(frame, circles, False)
                
                cv2.imshow('Calibration', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Show progress
                progress = (datetime.now() - start_collection).seconds / duration * 100
                print(f"\rProgress: {progress:.1f}% - Successful frames: {successful_frames}", 
                      end='')
            
            if successful_frames < self.config['min_frames']:
                raise CalibrationError(
                    f"Not enough successful calibration frames (got {successful_frames}, "
                    f"need {self.config['min_frames']})"
                )
            
            # Use average of all successful detections
            mean_circles = np.mean(all_circles, axis=0)
            
            # Perform calibration steps
            if not self.calibrate_camera(frame, mean_circles):
                raise CalibrationError("Camera calibration failed")
                
            self.calculate_pixel_ratio(mean_circles)
            validation_results = self.validate_calibration(frame, mean_circles)
            
            if not validation_results['is_valid']:
                raise CalibrationError("Calibration validation failed")
            
            accuracy = validation_results['marker_detection_accuracy']
            
            # Save calibration data
            self.calibration_data['timestamp'] = datetime.now().isoformat()
            self._save_calibration()
            
            success = True
            print("\nCalibration completed successfully!")
            print(f"Validation results: {validation_results}")
            
        except CalibrationError as e:
            error_message = str(e)
            print(f"\nCalibration failed: {error_message}")
            return False
        except Exception as e:
            error_message = str(e)
            print(f"\nUnexpected error during calibration: {error_message}")
            return False
        finally:
            duration = time.time() - start_time
            self.analytics.log_calibration_attempt(
                success=success,
                duration=duration,
                accuracy=accuracy,
                error_message=error_message
            )
            
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            
        return success

    def _draw_calibration_feedback(self, frame, circles, success):
        """Draw calibration feedback on frame"""
        if circles is not None:
            for circle in circles:
                cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
                cv2.circle(frame, (circle[0], circle[1]), 2, (0, 0, 255), 3)
        
        status = "ALIGNED" if success else "NOT ALIGNED"
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.putText(frame, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def _save_calibration(self):
        """Save calibration data to file"""
        try:
            # Save with timestamp
            filename = f"calibration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join('data/calibration', filename)
            
            with open(filepath, 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
                
            # Save as current calibration
            with open('data/calibration/current_calibration.json', 'w') as f:
                json.dump(self.calibration_data, f, indent=4)
                
        except Exception as e:
            raise CalibrationError(f"Failed to save calibration data: {str(e)}")

    def load_calibration(self):
        """
        Load most recent calibration data
        Returns: loading success (bool)
        """
        try:
            with open('data/calibration/current_calibration.json', 'r') as f:
                self.calibration_data = json.load(f)
            return self.is_calibration_valid()
        except Exception:
            return False

    def convert_to_real_coordinates(self, points):
        """
        Convert pixel coordinates to real-world coordinates (mm)
        Returns: converted coordinates
        """
        try:
            if not self.is_calibration_valid():
                return points
                
            real_points = []
            for point in points:
                real_point = [
                    point[0] * self.calibration_data['pixel_to_mm_ratio'],
                    point[1] * self.calibration_data['pixel_to_mm_ratio']
                ]
                real_points.append(real_point)
                
            return real_points
            
        except Exception as e:
            print(f"Coordinate conversion error: {str(e)}")
            return points

    def is_calibration_valid(self):
        """Check if current calibration is valid and not expired"""
        try:
            if not self.calibration_data['timestamp']:
                return False
                
            # Check calibration age
            calibration_time = datetime.fromisoformat(self.calibration_data['timestamp'])
            if (datetime.now() - calibration_time).days > self.config['calibration_validity_days']:
                return False
                
            # Check validation results
            validation = self.calibration_data['validation_results']
            if not validation or not validation.get('is_valid', False):
                return False
                
            return True
            
        except Exception:
            return False

    def show_analytics(self):
        """Show calibration analytics"""
        print("\nCalibration Analytics Report")
        print("==========================")
        print(self.analytics.generate_report())
        
        plot_path = self.analytics.plot_analytics()
        if plot_path and os.path.exists(plot_path):
            print(f"\nAnalytics plot saved to: {plot_path}")
            
            # Parāda grafiku
            img = cv2.imread(plot_path)
            if img is not None:
                cv2.imshow('Calibration Analytics', img)
                cv2.waitKey(0)
                cv2.destroyWindow('Calibration Analytics')
