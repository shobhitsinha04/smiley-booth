"""
Face and Smile Detection Module
Uses MediaPipe Face Mesh for landmark detection
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
import math


@dataclass
class FaceData:
    bbox: tuple  # (x, y, w, h)
    center: tuple
    is_centered: bool
    is_smiling: bool
    smile_confidence: float
    landmarks: object = None


class FaceDetector:
    def __init__(self):
        # setup mediapipe face mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # thresholds
        self.smile_threshold = 0.55
        self.min_confidence = 0.5
        self.center_tolerance = 0.12
        
        # smoothing
        self.smile_history = []
        self.history_size = 8
        
    def _get_point(self, landmarks, idx, w, h):
        lm = landmarks.landmark[idx]
        return (lm.x * w, lm.y * h)
    
    def _dist(self, p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def get_face_bbox(self, landmarks, w, h):
        # face oval landmark indices
        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        
        xs = [self._get_point(landmarks, i, w, h)[0] for i in FACE_OVAL]
        ys = [self._get_point(landmarks, i, w, h)[1] for i in FACE_OVAL]
        
        pad = 20
        x1 = max(0, int(min(xs)) - pad)
        y1 = max(0, int(min(ys)) - pad)
        x2 = min(w, int(max(xs)) + pad)
        y2 = min(h, int(max(ys)) + pad)
        
        return (x1, y1, x2 - x1, y2 - y1)
    
    def detect_smile(self, frame, landmarks):
        if landmarks is None:
            return False, 0.0
        
        h, w = frame.shape[:2]
        
        try:
            # key landmark indices
            left_corner = self._get_point(landmarks, 61, w, h)
            right_corner = self._get_point(landmarks, 291, w, h)
            upper_lip = self._get_point(landmarks, 13, w, h)
            lower_lip = self._get_point(landmarks, 17, w, h)
            upper_inner = self._get_point(landmarks, 13, w, h)
            lower_inner = self._get_point(landmarks, 14, w, h)
            nose = self._get_point(landmarks, 4, w, h)
            left_eye = self._get_point(landmarks, 33, w, h)
            right_eye = self._get_point(landmarks, 263, w, h)
            
            # mouth aspect ratio - wider = smile
            mouth_width = self._dist(left_corner, right_corner)
            face_width = self._dist(left_eye, right_eye)
            if face_width == 0:
                return False, 0.0
            mar = mouth_width / face_width
            
            # corner lift - corners go up when smiling
            mouth_center_y = (upper_lip[1] + lower_lip[1]) / 2
            left_lift = mouth_center_y - left_corner[1]
            right_lift = mouth_center_y - right_corner[1]
            avg_lift = (left_lift + right_lift) / 2
            
            face_height = self._dist(nose, (nose[0], 0))
            normalized_lift = avg_lift / (face_height * 0.1) if face_height > 0 else 0
            
            # mouth opening
            opening = self._dist(upper_inner, lower_inner)
            norm_opening = opening / face_width if face_width > 0 else 0
            
            # corner angles
            left_angle = math.atan2(mouth_center_y - left_corner[1],
                                    left_corner[0] - (left_corner[0] + right_corner[0])/2)
            right_angle = math.atan2(mouth_center_y - right_corner[1],
                                     (left_corner[0] + right_corner[0])/2 - right_corner[0])
            angle_score = (left_angle + right_angle) / 2
            
            # combine scores
            mar_score = max(0, (mar - 0.35) * 3)
            lift_score = max(0, normalized_lift * 2)
            opening_score = max(0, min(norm_opening * 2, 0.3))
            angle_bonus = max(0, angle_score * 0.5) if angle_score > 0 else 0
            
            # penalties for frowns
            asymmetry = abs(left_lift - right_lift) / (face_width * 0.1 + 0.001)
            asymmetry_penalty = min(asymmetry * 0.3, 0.3)
            frown_penalty = 0.5 if avg_lift < -2 else 0
            
            # final score
            score = (mar_score * 0.35 + lift_score * 0.40 + 
                    opening_score * 0.15 + angle_bonus * 0.10) - asymmetry_penalty - frown_penalty
            
            confidence = max(0, min(score, 1.0))
            return confidence > self.smile_threshold, confidence
            
        except:
            return False, 0.0
    
    def check_centered(self, face_center, frame_shape):
        fh, fw = frame_shape[:2]
        cx, cy = fw // 2, fh // 2
        tol_x = fw * self.center_tolerance
        tol_y = fh * self.center_tolerance
        dx = abs(face_center[0] - cx)
        dy = abs(face_center[1] - cy)
        return dx < tol_x and dy < tol_y
    
    def get_direction(self, face_center, frame_shape):
        fh, fw = frame_shape[:2]
        cx, cy = fw // 2, fh // 2
        dx = face_center[0] - cx
        dy = face_center[1] - cy
        tol_x = fw * self.center_tolerance
        tol_y = fh * self.center_tolerance
        
        dirs = []
        if abs(dx) > tol_x:
            dirs.append("LEFT" if dx > 0 else "RIGHT")
        if abs(dy) > tol_y:
            dirs.append("UP" if dy > 0 else "DOWN")
        return " & ".join(dirs) if dirs else "CENTERED"
    
    def smooth_detection(self, is_smiling, confidence):
        self.smile_history.append((is_smiling, confidence))
        if len(self.smile_history) > self.history_size:
            self.smile_history.pop(0)
        
        if len(self.smile_history) < self.history_size // 2:
            return False, confidence
        
        avg_conf = sum(c for _, c in self.smile_history) / len(self.smile_history)
        smile_count = sum(1 for s, _ in self.smile_history if s)
        
        # need 70% of frames smiling
        needed = int(len(self.smile_history) * 0.7)
        result = smile_count >= needed and avg_conf >= self.min_confidence
        return result, avg_conf
    
    def detect(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        if not results.multi_face_landmarks:
            self.smile_history.clear()
            return None
        
        landmarks = results.multi_face_landmarks[0]
        bbox = self.get_face_bbox(landmarks, w, h)
        x, y, bw, bh = bbox
        center = (x + bw // 2, y + bh // 2)
        
        is_centered = self.check_centered(center, frame.shape)
        
        if not is_centered:
            self.smile_history.clear()
            return FaceData(bbox, center, False, False, 0.0, landmarks)
        
        is_smiling, conf = self.detect_smile(frame, landmarks)
        is_smiling, conf = self.smooth_detection(is_smiling, conf)
        
        return FaceData(bbox, center, is_centered, is_smiling, conf, landmarks)
    
    def draw_detection_overlay(self, frame, face_data):
        overlay = frame.copy()
        
        if face_data is None:
            cv2.putText(overlay, "No face detected", (50, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return overlay
        
        x, y, w, h = face_data.bbox
        
        # box color based on state
        if face_data.is_centered and face_data.is_smiling:
            color = (0, 255, 0)  # green
        elif face_data.is_centered:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 165, 255)  # orange
        
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
        
        # center crosshair
        fh, fw = frame.shape[:2]
        cx, cy = fw // 2, fh // 2
        cv2.line(overlay, (cx-30, cy), (cx+30, cy), (255, 255, 255), 2)
        cv2.line(overlay, (cx, cy-30), (cx, cy+30), (255, 255, 255), 2)
        
        # center zone
        tx = int(fw * self.center_tolerance)
        ty = int(fh * self.center_tolerance)
        cv2.rectangle(overlay, (cx-tx, cy-ty), (cx+tx, cy+ty), (100, 100, 100), 1)
        
        # face center dot
        cv2.circle(overlay, face_data.center, 8, color, -1)
        
        # direction text
        direction = self.get_direction(face_data.center, frame.shape)
        cv2.putText(overlay, direction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # smile status
        if face_data.is_centered:
            if face_data.is_smiling:
                txt = "Smile: YES!"
                clr = (0, 255, 0)
            else:
                txt = f"Smile: No (need {self.smile_threshold:.0%})"
                clr = (0, 0, 255)
        else:
            txt = "Center first!"
            clr = (0, 165, 255)
        cv2.putText(overlay, txt, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)
        
        # confidence bar
        bx, by, bw, bh = 50, 115, 200, 20
        cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (50, 50, 50), -1)
        thresh_x = bx + int(bw * self.smile_threshold)
        cv2.line(overlay, (thresh_x, by), (thresh_x, by+bh), (255, 255, 255), 2)
        fill = int(bw * face_data.smile_confidence)
        fill_color = (0, 255, 0) if face_data.smile_confidence >= self.smile_threshold else (0, 100, 255)
        cv2.rectangle(overlay, (bx, by), (bx+fill, by+bh), fill_color, -1)
        cv2.rectangle(overlay, (bx, by), (bx+bw, by+bh), (255, 255, 255), 2)
        cv2.putText(overlay, f"{face_data.smile_confidence:.0%}", (bx+bw+10, by+15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay


class CaptureController:
    def __init__(self, required_smile_frames=80, cooldown_frames=60):
        self.required_smile_frames = required_smile_frames
        self.cooldown_frames = cooldown_frames
        self.consecutive_frames = 0
        self.cooldown = 0
    
    def update(self, face_data):
        if self.cooldown > 0:
            self.cooldown -= 1
            return False
        
        if face_data is None:
            self.consecutive_frames = 0
            return False
        
        if face_data.is_centered and face_data.is_smiling:
            self.consecutive_frames += 1
            if self.consecutive_frames >= self.required_smile_frames:
                self.consecutive_frames = 0
                self.cooldown = self.cooldown_frames
                return True
        else:
            self.consecutive_frames = 0
        
        return False
    
    def get_countdown(self):
        if self.consecutive_frames > 0:
            return self.required_smile_frames - self.consecutive_frames
        return -1
    
    def reset(self):
        self.consecutive_frames = 0
        self.cooldown = 0
