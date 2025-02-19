# modules/face_recognition_module.py
import cv2
import os
import json
import numpy as np
from datetime import datetime
from patient_positioning.config.settings import *

class PatientRecognition:
    def __init__(self):
        # Ensure directories exist
        for directory in [PATIENTS_DIR, PHOTOS_DIR, FEATURES_DIR, LOGS_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Load face detection cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Initialize face recognizer
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load existing patient data
        self.patient_ids = {}  # Map numeric IDs to patient IDs
        self.load_patients()
        
    def load_patients(self):
        """Load existing patient data and train recognizer"""
        features_file = os.path.join(FEATURES_DIR, 'features.yml')
        patients_file = os.path.join(FEATURES_DIR, 'patients.json')
        
        if os.path.exists(features_file) and os.path.exists(patients_file):
            try:
                self.recognizer.read(features_file)
                with open(patients_file, 'r') as f:
                    self.patient_ids = json.load(f)
                print(f"Loaded {len(self.patient_ids)} patients")
            except Exception as e:
                print(f"Error loading patient data: {e}")
    
    def detect_face(self, frame):
        """Detect face in frame and return processed face image"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  # Take the first face
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))  # Normalize size
            return True, face_roi, (x, y, w, h)
        return False, None, None
    
    def add_new_patient(self, frame, patient_id):
        """Add new patient to the database"""
        success, face_roi, face_rect = self.detect_face(frame)
        if not success:
            return False, "No face detected in frame"
        
        # Save photo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        photo_path = os.path.join(PHOTOS_DIR, f"{patient_id}_{timestamp}.jpg")
        cv2.imwrite(photo_path, frame)
        
        # Assign numeric ID for the recognizer
        numeric_id = len(self.patient_ids) + 1
        self.patient_ids[str(numeric_id)] = patient_id
        
        # Train recognizer with new face
        faces = [face_roi]
        ids = np.array([numeric_id])
        self.recognizer.update(faces, ids) if len(self.patient_ids) > 1 else self.recognizer.train(faces, ids)
        
        # Save updated recognizer and patient mapping
        features_file = os.path.join(FEATURES_DIR, 'features.yml')
        patients_file = os.path.join(FEATURES_DIR, 'patients.json')
        
        self.recognizer.write(features_file)
        with open(patients_file, 'w') as f:
            json.dump(self.patient_ids, f)
        
        # Create patient log file
        self.create_patient_log(patient_id)
        
        return True, photo_path
    
    def create_patient_log(self, patient_id):
        """Create new log file for patient"""
        log_path = self.get_patient_log_path(patient_id)
        if not os.path.exists(log_path):
            with open(log_path, 'w', newline='') as f:
                f.write("Timestamp,TrackingType,AverageDeviation,Instructions\n")
    
    def recognize_patient(self, frame):
        """Try to recognize patient in frame"""
        success, face_roi, face_rect = self.detect_face(frame)
        if not success:
            return None, 0
        
        try:
            numeric_id, confidence = self.recognizer.predict(face_roi)
            # Convert confidence to 0-1 range (lower is better in OpenCV)
            confidence = max(0, min(100 - confidence, 100)) / 100
            
            if confidence > FACE_RECOGNITION_THRESHOLD:
                patient_id = self.patient_ids.get(str(numeric_id))
                if patient_id:
                    return patient_id, confidence
        except Exception as e:
            print(f"Recognition error: {e}")
        
        return None, 0
    
    def get_patient_log_path(self, patient_id):
        """Get path to patient's log file"""
        return os.path.join(LOGS_DIR, f"{patient_id}_tracking.csv")
    
    def log_tracking_data(self, patient_id, tracking_type, deviation, instructions):
        """Log tracking data to patient's file"""
        log_path = self.get_patient_log_path(patient_id)
        with open(log_path, 'a', newline='') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp},{tracking_type},{deviation},{instructions}\n")
    
    def draw_recognition_info(self, frame, patient_id=None, confidence=None):
        """Draw recognition information on frame"""
        if patient_id:
            cv2.putText(frame, f"Patient ID: {patient_id}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
        else:
            cv2.putText(frame, "No patient recognized", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
        
        return frame