# modules/face_tracker.py
import cv2
import mediapipe as mp
import numpy as np
from patient_positioning.config.settings import *

class FaceTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=MAX_NUM_FACES,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Define key anatomical points for medical positioning
        self.key_points = {
            'nose_tip': 1,        # Nose tip
            'chin': 152,          # Chin
            'left_ear': 234,      # Left ear
            'right_ear': 454,     # Right ear
            'forehead': 10,       # Forehead
            'left_eye': 33,       # Left eye
            'right_eye': 263,     # Right eye
            'left_cheek': 206,    # Left cheek
            'right_cheek': 426,   # Right cheek
        }
        
        self.reference_points = None
    
    def detect_face(self, frame):
        """Detect face landmarks in the frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get face bounding box
            h, w = frame.shape[:2]
            x_min, x_max = w, 0
            y_min, y_max = h, 0
            
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                x_min = min(x_min, x)
                x_max = max(x_max, x)
                y_min = min(y_min, y)
                y_max = max(y_max, y)
            
            # Draw face mesh
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=self.drawing_spec,
                connection_drawing_spec=self.drawing_spec)
            
            return True, (x_min, y_min, x_max, y_max), face_landmarks
        
        return False, None, None
    
    def extract_anatomical_points(self, face_landmarks, frame_shape):
        """Extract key anatomical points from face landmarks"""
        h, w = frame_shape[:2]
        points = {}
        
        for point_name, index in self.key_points.items():
            landmark = face_landmarks.landmark[index]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            points[point_name] = (x, y, z)
            
        return points
    
    def save_reference_position(self, points):
        """Save current position as reference"""
        self.reference_points = points.copy()
        return True
    
    def calculate_alignment(self, current_points):
        """Calculate alignment with reference position"""
        if not self.reference_points or not current_points:
            return None
        
        alignments = {}
        total_deviation = 0
        
        for point_name in self.key_points.keys():
            if point_name in current_points and point_name in self.reference_points:
                curr = current_points[point_name]
                ref = self.reference_points[point_name]
                
                deviation = np.sqrt(
                    (curr[0] - ref[0])**2 + 
                    (curr[1] - ref[1])**2 + 
                    (curr[2] - ref[2])**2
                )
                
                alignments[point_name] = {
                    'deviation': deviation,
                    'direction': (
                        curr[0] - ref[0],  # X deviation
                        curr[1] - ref[1],  # Y deviation
                        curr[2] - ref[2]   # Z deviation
                    )
                }
                total_deviation += deviation
        
        return {
            'points': alignments,
            'total_deviation': total_deviation,
            'average_deviation': total_deviation / len(alignments)
        }
    
    def generate_correction_instructions(self, alignment_data):
        """Generate text instructions for position correction"""
        if not alignment_data:
            return []
        
        instructions = []
        for point_name, data in alignment_data['points'].items():
            if data['deviation'] > DEVIATION_THRESHOLD_LOW:
                dx, dy, dz = data['direction']
                
                if abs(dx) > DEVIATION_THRESHOLD_LOW:
                    direction = "right" if dx > 0 else "left"
                    instructions.append(f"Move {point_name} {direction}")
                if abs(dy) > DEVIATION_THRESHOLD_LOW:
                    direction = "down" if dy > 0 else "up"
                    instructions.append(f"Move {point_name} {direction}")
                if abs(dz) > 0.1:  # Z-axis threshold
                    direction = "back" if dz > 0 else "forward"
                    instructions.append(f"Move {point_name} {direction}")
                    
        return instructions