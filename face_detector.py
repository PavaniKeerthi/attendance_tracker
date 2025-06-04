import cv2
import numpy as np
from typing import List, Dict, Tuple

class FaceDetector:
    def __init__(self):
        """Initialize the face detector with OpenCV's pre-trained models"""
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load eye detection cascade to help distinguish live video vs profile pictures
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_faces_and_video_status(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in the image and determine if they represent live video or profile pictures
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries with face information and video status
        """
        faces_data = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            face_roi_color = image[y:y+h, x:x+w]
            
            # Determine if this is a live video feed or profile picture
            video_status = self._determine_video_status(face_roi, face_roi_color)
            
            # Draw rectangle around face (green for video on, yellow for video off)
            color = (0, 255, 0) if video_status else (0, 255, 255)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # Add label
            label = f"Video {'ON' if video_status else 'OFF'}"
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            faces_data.append({
                'face_id': i,
                'bbox': (x, y, w, h),
                'video_on': video_status,
                'face_region': face_roi_color
            })
        
        return faces_data
    
    def _determine_video_status(self, face_roi_gray: np.ndarray, face_roi_color: np.ndarray) -> bool:
        """
        Determine if a detected face represents a live video feed or a profile picture
        
        This uses several heuristics:
        1. Eye detection - live video more likely to have detectable eyes
        2. Image variance - live video typically has more variation
        3. Color distribution - profile pictures often have more uniform backgrounds
        4. Edge density - live video typically has more edge information
        
        Args:
            face_roi_gray: Grayscale face region
            face_roi_color: Color face region
            
        Returns:
            True if likely live video, False if likely profile picture
        """
        # Heuristic 1: Eye detection
        eyes = self.eye_cascade.detectMultiScale(face_roi_gray)
        eye_score = min(len(eyes) / 2.0, 1.0)  # Normalize to 0-1, ideal is 2 eyes
        
        # Heuristic 2: Image variance (live video typically has more variation)
        variance = np.var(face_roi_gray)
        variance_score = min(variance / 1000.0, 1.0)  # Normalize
        
        # Heuristic 3: Edge density
        edges = cv2.Canny(face_roi_gray, 50, 150)
        edge_density = np.sum(edges > 0) / (face_roi_gray.shape[0] * face_roi_gray.shape[1])
        edge_score = min(edge_density * 10, 1.0)  # Normalize
        
        # Heuristic 4: Color distribution analysis
        if len(face_roi_color.shape) == 3:
            # Check for uniform background colors (common in profile pictures)
            hsv = cv2.cvtColor(face_roi_color, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # More uniform color distribution suggests profile picture
            h_uniformity = np.max(hist_h) / np.sum(hist_h)
            s_uniformity = np.max(hist_s) / np.sum(hist_s)
            color_score = 1.0 - min((h_uniformity + s_uniformity) / 2.0, 1.0)
        else:
            color_score = 0.5
        
        # Combine scores with weights
        final_score = (
            eye_score * 0.3 +
            variance_score * 0.25 +
            edge_score * 0.25 +
            color_score * 0.2
        )
        
        # Threshold: if score > 0.4, consider it live video
        return final_score > 0.4
    
    def get_face_embedding(self, face_region: np.ndarray) -> np.ndarray:
        """
        Extract a simple feature vector from face region for basic matching
        This is a simplified approach - in production, you'd use proper face recognition
        
        Args:
            face_region: Face region as numpy array
            
        Returns:
            Feature vector representing the face
        """
        # Resize to standard size
        face_resized = cv2.resize(face_region, (64, 64))
        
        # Convert to grayscale if needed
        if len(face_resized.shape) == 3:
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        else:
            face_gray = face_resized
        
        # Extract LBP (Local Binary Pattern) features
        # This is a simple texture descriptor
        lbp = self._compute_lbp(face_gray)
        
        # Compute histogram of LBP
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        
        # Normalize histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-7)
        
        return hist
    
    def _compute_lbp(self, image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
        """
        Compute Local Binary Pattern for texture analysis
        """
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        
        for i in range(radius, rows - radius):
            for j in range(radius, cols - radius):
                center = image[i, j]
                binary_string = ""
                
                # Sample points around the center
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + radius * np.cos(angle)
                    y = j + radius * np.sin(angle)
                    
                    # Bilinear interpolation
                    x1, y1 = int(x), int(y)
                    x2, y2 = x1 + 1, y1 + 1
                    
                    if x2 < rows and y2 < cols:
                        # Interpolate pixel value
                        dx, dy = x - x1, y - y1
                        pixel_value = (
                            image[x1, y1] * (1 - dx) * (1 - dy) +
                            image[x1, y2] * (1 - dx) * dy +
                            image[x2, y1] * dx * (1 - dy) +
                            image[x2, y2] * dx * dy
                        )
                        
                        binary_string += '1' if pixel_value >= center else '0'
                
                lbp[i, j] = int(binary_string, 2)
        
        return lbp
