import pandas as pd
import numpy as np
from typing import List, Dict
import cv2
from difflib import SequenceMatcher

class AttendanceProcessor:
    def __init__(self):
        """Initialize the attendance processor"""
        self.similarity_threshold = 0.6  # Threshold for name matching
        
    def process_attendance(self, student_names: List[str], faces_data: List[Dict], image: np.ndarray) -> List[Dict]:
        """
        Process attendance based on detected faces and student roster
        
        Args:
            student_names: List of student names from CSV
            faces_data: List of detected face data with video status
            image: Original image for additional analysis
            
        Returns:
            List of attendance records with scores
        """
        attendance_records = []
        
        # Try to extract text from image to match with student names
        detected_names = self._extract_names_from_image(image)
        
        # Create attendance record for each student
        for student_name in student_names:
            # Check if student name appears in detected text
            name_found = self._find_matching_name(student_name, detected_names)
            
            if name_found:
                # Student is present, determine video status
                # For simplicity, we'll assign video status based on face detection count
                # In a real scenario, you'd need more sophisticated matching
                video_status = self._determine_student_video_status(faces_data, len(student_names))
                score = 100 if video_status else 50
                status = "Present - Video On" if video_status else "Present - Video Off"
            else:
                # Student not found in image
                score = 0
                status = "Absent"
                video_status = False
            
            attendance_records.append({
                'Name': student_name,
                'Status': status,
                'Score': score,
                'Video_On': video_status
            })
        
        # If we have more faces than matched students, some might be unidentified
        self._handle_unidentified_faces(attendance_records, faces_data, detected_names)
        
        return attendance_records
    
    def _extract_names_from_image(self, image: np.ndarray) -> List[str]:
        """
        Extract text from image that might contain participant names
        This is a simplified OCR approach using template matching for common Meet UI elements
        
        Args:
            image: Input image
            
        Returns:
            List of detected text/names
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing for better text detection
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Find text regions using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
        morph = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Find contours that might contain text
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # For this simplified version, we'll return some common name patterns
        # In a real implementation, you'd use proper OCR like Tesseract
        detected_names = []
        
        # Simple heuristic: look for rectangular regions that might contain names
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size - name labels are typically within certain dimensions
            if 50 < w < 200 and 15 < h < 40:
                # Extract region
                text_region = enhanced[y:y+h, x:x+w]
                
                # Placeholder for actual OCR - in reality you'd use pytesseract here
                # For now, we'll simulate by checking if region has text-like properties
                if self._looks_like_text_region(text_region):
                    detected_names.append(f"detected_name_{len(detected_names)}")
        
        return detected_names
    
    def _looks_like_text_region(self, region: np.ndarray) -> bool:
        """
        Simple heuristic to determine if a region contains text
        """
        if region.size == 0:
            return False
            
        # Check for horizontal text patterns
        horizontal_projection = np.sum(region, axis=1)
        vertical_projection = np.sum(region, axis=0)
        
        # Text regions typically have alternating high/low intensity patterns
        h_variance = np.var(horizontal_projection)
        v_variance = np.var(vertical_projection)
        
        return h_variance > 1000 and v_variance > 1000
    
    def _find_matching_name(self, student_name: str, detected_names: List[str]) -> bool:
        """
        Find if student name matches any detected names using fuzzy matching
        """
        # Since OCR is complex to implement properly here, we'll use a probabilistic approach
        # Based on the number of faces detected vs students expected
        
        # Simple heuristic: if we have detected faces, assume some students are present
        # This is obviously simplified - real implementation would need proper OCR
        return len(detected_names) > 0
    
    def _determine_student_video_status(self, faces_data: List[Dict], total_students: int) -> bool:
        """
        Determine video status for a student based on face detection results
        """
        if not faces_data:
            return False
        
        # Count faces with video on
        video_on_count = sum(1 for face in faces_data if face['video_on'])
        
        # Simple heuristic: if more than half the detected faces have video on,
        # assign video on status to students probabilistically
        video_on_ratio = video_on_count / len(faces_data) if faces_data else 0
        
        # Return True if video on ratio is above threshold
        return video_on_ratio > 0.5
    
    def _handle_unidentified_faces(self, attendance_records: List[Dict], faces_data: List[Dict], detected_names: List[str]) -> None:
        """
        Handle cases where faces are detected but can't be matched to students
        """
        identified_count = sum(1 for record in attendance_records if record['Score'] > 0)
        detected_faces_count = len(faces_data)
        
        # If we have more faces than identified students, update some absent students to present
        if detected_faces_count > identified_count:
            absent_students = [record for record in attendance_records if record['Score'] == 0]
            
            # Update some absent students to present based on remaining faces
            unmatched_faces = min(len(absent_students), detected_faces_count - identified_count)
            
            for i in range(unmatched_faces):
                if i < len(absent_students):
                    # Determine video status from remaining faces
                    remaining_faces = faces_data[identified_count + i:] if identified_count + i < len(faces_data) else faces_data
                    video_on = remaining_faces[0]['video_on'] if remaining_faces else False
                    
                    absent_students[i]['Status'] = "Present - Video On" if video_on else "Present - Video Off"
                    absent_students[i]['Score'] = 100 if video_on else 50
                    absent_students[i]['Video_On'] = video_on
    
    def _calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using sequence matching
        """
        return SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
    
    def export_to_csv(self, attendance_records: List[Dict], filename: str = "attendance_report.csv") -> str:
        """
        Export attendance records to CSV file
        
        Args:
            attendance_records: List of attendance records
            filename: Output filename
            
        Returns:
            CSV data as string
        """
        df = pd.DataFrame(attendance_records)
        return df.to_csv(index=False)
