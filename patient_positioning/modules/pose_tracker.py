# modules/pose_tracker.py
import cv2
import mediapipe as mp
import numpy as np
from patient_positioning.config.settings import *

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
            model_complexity=2,  # Palielināts uz 2 lielākai precizitātei
            smooth_landmarks=True  # Stabilākai sekošanai
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.reference_points = None
        
        # Paplašināta punktu definīcija
        self.key_points = {
            # Galva un kakls
            'head': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Visi sejas punkti
            'neck': [0, 11, 12],  # Kakla līnija
            
            # Pleci un rokas
            'shoulders': [11, 12],  # Plecu līnija
            'left_shoulder_complex': [11, 13, 15],  # Kreisais plecs ar roku
            'right_shoulder_complex': [12, 14, 16],  # Labais plecs ar roku
            'left_arm_upper': [11, 13],  # Kreisā augšdelma
            'right_arm_upper': [12, 14],  # Labā augšdelma
            'left_arm_lower': [13, 15],  # Kreisā apakšdelma
            'right_arm_lower': [14, 16],  # Labā apakšdelma
            'left_wrist': [15, 17, 19, 21],  # Kreisā plauksta
            'right_wrist': [16, 18, 20, 22],  # Labā plauksta
            
            # Mugurkauls un torss
            'spine_full': [0, 11, 12, 23, 24],  # Pilna mugurkaula līnija
            'spine_upper': [0, 11, 12],  # Augšējā mugurkaula daļa
            'spine_mid': [11, 23],  # Vidējā mugurkaula daļa
            'spine_lower': [23, 24],  # Apakšējā mugurkaula daļa
            'torso': [11, 12, 23, 24],  # Ķermeņa centrālā daļa
            
            # Gurni un kājas
            'hips': [23, 24],  # Gurnu līnija
            'left_hip_complex': [23, 25, 27],  # Kreisais gurns ar kāju
            'right_hip_complex': [24, 26, 28],  # Labais gurns ar kāju
            'left_leg_upper': [23, 25],  # Kreisais augšstilbs
            'right_leg_upper': [24, 26],  # Labais augšstilbs
            'left_leg_lower': [25, 27],  # Kreisā apakšstilbs
            'right_leg_lower': [26, 28],  # Labā apakšstilbs
            'left_ankle': [27, 29, 31],  # Kreisā pēda
            'right_ankle': [28, 30, 32],  # Labā pēda
            
            # Ķermeņa simetrijas līnijas
            'body_left_side': [11, 23, 25, 27],  # Kreisā puse
            'body_right_side': [12, 24, 26, 28],  # Labā puse
            'body_central_line': [0, 11, 23],  # Centrālā līnija
            
            # Papildus punktu grupas
            'upper_body': [0, 11, 12, 13, 14, 15, 16],  # Augšējā ķermeņa daļa
            'lower_body': [23, 24, 25, 26, 27, 28],  # Apakšējā ķermeņa daļa
            'extremities': [15, 16, 27, 28]  # Ekstremitāšu gali
        }

    def detect_pose(self, frame):
        """Detektē ķermeņa pozu kadrā un zīmē punktus"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Zīmē visus punktus un savienojumus
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=COLOR_GREEN, thickness=2, circle_radius=2),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=COLOR_WHITE, thickness=1)
            )
            return True, results.pose_landmarks
        return False, None

    def extract_key_points(self, landmarks, frame_shape=None):
        """Extract key points from pose landmarks"""
        if landmarks:
            points = {}
            for name, indices in self.key_points.items():
                points[name] = self._get_points_coordinates(landmarks, indices)
            return points
        return None

    def save_reference_position(self, points):
        """Saglabā pašreizējo pozīciju kā references pozīciju"""
        if points:
            self.reference_points = points
            return True
        return False

    def _get_points_coordinates(self, landmarks, indices):
        """Iegūst 3D koordinātas norādītajiem punktiem"""
        return [(landmarks.landmark[idx].x,
                landmarks.landmark[idx].y,
                landmarks.landmark[idx].z) for idx in indices]

    def calculate_alignment(self, current_points):
        """Aprēķina novirzes no references pozīcijas"""
        if not self.reference_points or not current_points:
            return None

        alignments = {}
        total_deviation = 0
        num_points = 0

        for point_name in self.key_points.keys():
            if point_name not in current_points or point_name not in self.reference_points:
                continue

            curr_points = current_points[point_name]
            ref_points = self.reference_points[point_name]
            
            deviations = []
            directions = []
            
            for ref, curr in zip(ref_points, curr_points):
                dev = np.sqrt(
                    (curr[0] - ref[0])**2 + 
                    (curr[1] - ref[1])**2 + 
                    (curr[2] - ref[2])**2
                )
                deviations.append(dev)
                directions.append((
                    curr[0] - ref[0],
                    curr[1] - ref[1],
                    curr[2] - ref[2]
                ))

            avg_deviation = np.mean(deviations)
            alignments[point_name] = {
                'deviation': avg_deviation,
                'directions': directions,
                'max_deviation': max(deviations),
                'min_deviation': min(deviations)
            }
            
            total_deviation += sum(deviations)
            num_points += len(deviations)

        if num_points == 0:
            return None

        return {
            'points': alignments,
            'total_deviation': total_deviation,
            'average_deviation': total_deviation / num_points
        }

    def generate_positioning_instructions(self, alignment_data):
        """Ģenerē detalizētas korekcijas instrukcijas"""
        if not alignment_data:
            return []

        instructions = []
        
        # Pārbauda ķermeņa simetriju
        symmetry_pairs = [
            ('body_left_side', 'body_right_side', 'Body symmetry'),
            ('left_shoulder_complex', 'right_shoulder_complex', 'Shoulder'),
            ('left_hip_complex', 'right_hip_complex', 'Hip'),
            ('left_leg_upper', 'right_leg_upper', 'Upper leg'),
            ('left_leg_lower', 'right_leg_lower', 'Lower leg')
        ]

        for left_part, right_part, name in symmetry_pairs:
            if (left_part in alignment_data['points'] and 
                right_part in alignment_data['points']):
                left_dev = alignment_data['points'][left_part]['deviation']
                right_dev = alignment_data['points'][right_part]['deviation']
                if abs(left_dev - right_dev) > DEVIATION_THRESHOLD_LOW:
                    instructions.append(
                        f"Adjust {name} symmetry (L/R diff: {abs(left_dev - right_dev):.1f})")

        # Pārbauda mugurkaula līniju
        spine_parts = ['spine_upper', 'spine_mid', 'spine_lower']
        for part in spine_parts:
            if part in alignment_data['points']:
                dev = alignment_data['points'][part]['deviation']
                if dev > DEVIATION_THRESHOLD_LOW:
                    instructions.append(f"Correct {part.replace('_', ' ')} alignment (dev: {dev:.1f})")

        # Pārbauda ekstremitātes
        if 'extremities' in alignment_data['points']:
            dev = alignment_data['points']['extremities']['deviation']
            if dev > DEVIATION_THRESHOLD_LOW:
                instructions.append(f"Check extremity positions (dev: {dev:.1f})")

        # Pārbauda ķermeņa centrālo līniju
        if 'body_central_line' in alignment_data['points']:
            dev = alignment_data['points']['body_central_line']['deviation']
            if dev > DEVIATION_THRESHOLD_LOW:
                instructions.append(f"Align central body line (dev: {dev:.1f})")

        return instructions

    def get_key_measurements(self, landmarks):
        """Aprēķina galvenos ķermeņa mērījumus"""
        if not landmarks:
            return None

        measurements = {}
        
        # Plecu platums
        if 'shoulders' in self.key_points:
            left, right = self._get_points_coordinates(landmarks, self.key_points['shoulders'])
            measurements['shoulder_width'] = np.sqrt(
                (right[0] - left[0])**2 + 
                (right[1] - left[1])**2 + 
                (right[2] - left[2])**2
            )

        # Mugurkaula garums
        if 'spine_full' in self.key_points:
            spine_points = self._get_points_coordinates(landmarks, self.key_points['spine_full'])
            spine_length = 0
            for i in range(len(spine_points)-1):
                spine_length += np.sqrt(
                    (spine_points[i+1][0] - spine_points[i][0])**2 +
                    (spine_points[i+1][1] - spine_points[i][1])**2 +
                    (spine_points[i+1][2] - spine_points[i][2])**2
                )
            measurements['spine_length'] = spine_length

        return measurements