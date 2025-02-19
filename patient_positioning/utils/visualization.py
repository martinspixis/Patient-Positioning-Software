# utils/visualization.py
import cv2
import numpy as np
from patient_positioning.config.settings import *

def draw_info(frame, anatomical_tracker, current_points, tracking):
    """Draw facial anatomical points and alignment information"""
    if anatomical_tracker.reference_points:
        alignment_data = anatomical_tracker.calculate_alignment(current_points)
        
        if alignment_data:
            # Draw anatomical points with alignment information
            for point_name, (x, y, _) in current_points.items():
                # Draw point
                cv2.circle(frame, (x, y), 3, COLOR_GREEN, -1)
                
                # Draw label
                cv2.putText(frame, point_name, (x + 5, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
                
                if point_name in alignment_data['points']:
                    deviation = alignment_data['points'][point_name]['deviation']
                    direction = alignment_data['points'][point_name]['direction']
                    
                    # Color based on deviation
                    if deviation < DEVIATION_THRESHOLD_LOW:
                        color = COLOR_GREEN
                    elif deviation < DEVIATION_THRESHOLD_HIGH:
                        color = COLOR_YELLOW
                    else:
                        color = COLOR_RED
                    
                    # Draw deviation arrow
                    end_x = int(x + direction[0] * 50)  # Scale for visibility
                    end_y = int(y + direction[1] * 50)
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), color, 2)
                    
                    # Draw deviation value
                    cv2.putText(frame, f"{deviation:.1f}px", (x + 5, y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Generate and display correction instructions
            instructions = anatomical_tracker.generate_correction_instructions(
                alignment_data)
            
            # Display instructions
            for i, instruction in enumerate(instructions):
                y_pos = 30 + i * 20
                cv2.putText(frame, instruction, 
                          (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX,
                          0.6, COLOR_RED, 2)
            
            # Display average deviation
            avg_dev = alignment_data['average_deviation']
            color = COLOR_GREEN if avg_dev < DEVIATION_THRESHOLD_LOW else (
                COLOR_YELLOW if avg_dev < DEVIATION_THRESHOLD_HIGH else COLOR_RED)
            
            cv2.putText(frame, 
                      f"Average deviation: {avg_dev:.1f}px",
                      (10, frame.shape[0] - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # Just draw points without alignment
        for point_name, (x, y, _) in current_points.items():
            cv2.circle(frame, (x, y), 3, COLOR_GREEN, -1)
            cv2.putText(frame, point_name, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
        
        if not tracking:
            cv2.putText(frame, "Press 's' to save reference position",
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    return frame

def draw_pose_info(frame, pose_tracker, pose_points, tracking):
    """Draw pose information and alignment data"""
    if pose_tracker.reference_points:
        alignment_data = pose_tracker.calculate_alignment(pose_points)
        
        if alignment_data:
            # Draw anatomical points with alignment information
            for point_name, points_list in pose_points.items():
                # Ņemam pirmo punktu no katra punktu komplekta kā atskaites punktu
                x, y, _ = points_list[0]
                x, y = int(x * frame.shape[1]), int(y * frame.shape[0])
                
                if point_name in alignment_data['points']:
                    deviation = alignment_data['points'][point_name]['deviation']
                    
                    # Color based on deviation
                    if deviation < DEVIATION_THRESHOLD_LOW:
                        color = COLOR_GREEN
                    elif deviation < DEVIATION_THRESHOLD_HIGH:
                        color = COLOR_YELLOW
                    else:
                        color = COLOR_RED
                    
                    # Draw point and label for key points
                    if point_name in ['shoulders', 'hips', 'spine_mid', 'neck']:
                        cv2.circle(frame, (x, y), 3, color, -1)
                        cv2.putText(frame, f"{point_name}: {deviation:.1f}",
                                  (x + 5, y + 15),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Generate and display correction instructions
            instructions = pose_tracker.generate_positioning_instructions(
                alignment_data)
            
            for i, instruction in enumerate(instructions):
                y_pos = 30 + i * 25
                cv2.putText(frame, instruction, (10, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
            
            # Display average deviation
            avg_dev = alignment_data['average_deviation']
            color = COLOR_GREEN if avg_dev < DEVIATION_THRESHOLD_LOW else (
                COLOR_YELLOW if avg_dev < DEVIATION_THRESHOLD_HIGH else COLOR_RED)
            
            cv2.putText(frame, f"Average deviation: {avg_dev:.1f}",
                       (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    elif not tracking:
        cv2.putText(frame, "Press 's' to save reference position",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_WHITE, 2)
    
    return frame

def draw_combined_info(frame, face_tracker, pose_tracker, face_landmarks, 
                      pose_landmarks, tracking, show_face=True, show_pose=True):
    """Draw combined face and pose tracking information"""
    display_frame = frame.copy()
    
    if show_face and face_landmarks:
        points = face_tracker.extract_anatomical_points(face_landmarks, frame.shape)
        display_frame = draw_info(display_frame, face_tracker, points, tracking)
    
    if show_pose and pose_landmarks:
        points = pose_tracker.extract_key_points(pose_landmarks, frame.shape)
        display_frame = draw_pose_info(display_frame, pose_tracker, points, tracking)
    
    return display_frame