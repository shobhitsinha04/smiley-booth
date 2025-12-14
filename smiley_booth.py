#!/usr/bin/env python3
"""
Smiley Booth - Smart Photobooth Application
CS445 Final Project

Authors: Shobhit Sinha, Jay Goenka, Adit Agarwal
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Optional, Tuple

from detection import FaceDetector, CaptureController, FaceData
from filters import FilterManager, create_filter_preview_strip


class SmileyBooth:
    def __init__(self, camera_id: int = 0, output_dir: str = "captured_photos"):
        # Initialize camera
        self.camera_id = camera_id
        self.cap = None
        
        # Initialize modules
        self.detector = FaceDetector()
        self.capture_controller = CaptureController(
            required_smile_frames=80,  # About 5 seconds of smiling
            cooldown_frames=45  # 1.5 seconds between captures
        )
        self.filter_manager = FilterManager()
        
        # Output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # UI state
        self.show_preview_strip = True
        
        # Window settings
        self.window_name = "Smiley Booth - Smart Photobooth"
        
        # Capture animation
        self.flash_alpha = 0
        self.last_capture_time = 0
        self.captured_image = None
        self.show_captured_preview = False
        self.captured_preview_start = 0
        
    def init_camera(self) -> bool:
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual resolution
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
        return True
    
    def save_photo(self, frame: np.ndarray, filtered: bool = True) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filter_name = self.filter_manager.get_current_filter_name() if filtered else "original"
        filename = f"smiley_booth_{timestamp}_{filter_name}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Photo saved: {filepath}")
        
        return filepath
    
    def trigger_capture(self, frame: np.ndarray):
        # Apply current filter
        filtered_frame = self.filter_manager.apply_current_filter(frame)
        
        # Save both original and filtered
        self.save_photo(frame, filtered=False)
        self.save_photo(filtered_frame, filtered=True)
        
        # Store for preview
        self.captured_image = filtered_frame.copy()
        self.show_captured_preview = True
        self.captured_preview_start = time.time()
        
        # Trigger flash effect
        self.flash_alpha = 1.0
        self.last_capture_time = time.time()
        
        print("Photo captured!")
    
    def draw_ui(self, frame: np.ndarray, face_data: Optional[FaceData]) -> np.ndarray:
        ui_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw detection overlay
        ui_frame = self.detector.draw_detection_overlay(ui_frame, face_data)
        
        # Draw filter name
        filter_name = self.filter_manager.get_current_filter_name().upper()
        cv2.putText(ui_frame, f"Filter: {filter_name}",
                   (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw countdown to capture
        if face_data:
            countdown = self.capture_controller.get_countdown()
            if countdown > 0:
                # Draw countdown circle
                progress = 1 - (countdown / self.capture_controller.required_smile_frames)
                end_angle = int(360 * progress)
                
                center = (w // 2, h - 80)
                cv2.ellipse(ui_frame, center, (40, 40), -90, 0, end_angle, (0, 255, 0), 5)
                cv2.putText(ui_frame, f"{countdown}", (center[0] - 10, center[1] + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Draw flash effect
        if self.flash_alpha > 0:
            flash_overlay = np.ones_like(ui_frame) * 255
            ui_frame = cv2.addWeighted(ui_frame, 1 - self.flash_alpha,
                                       flash_overlay, self.flash_alpha, 0)
            self.flash_alpha = max(0, self.flash_alpha - 0.1)
        
        # Draw captured preview
        if self.show_captured_preview and self.captured_image is not None:
            preview_time = time.time() - self.captured_preview_start
            if preview_time < 2.0:  
                # Draw small preview in corner
                preview_h = 150
                preview_w = int(preview_h * w / h)
                preview = cv2.resize(self.captured_image, (preview_w, preview_h))
                
                # Position in bottom right
                x_offset = w - preview_w - 20
                y_offset = h - preview_h - 20
                
                # Add border
                cv2.rectangle(ui_frame, (x_offset - 5, y_offset - 5),
                             (x_offset + preview_w + 5, y_offset + preview_h + 5),
                             (0, 255, 0), 3)
                
                ui_frame[y_offset:y_offset + preview_h, x_offset:x_offset + preview_w] = preview
                
                cv2.putText(ui_frame, "SAVED!",
                           (x_offset, y_offset - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.show_captured_preview = False
        
        # Draw filter preview strip
        if self.show_preview_strip:
            strip = create_filter_preview_strip(frame, self.filter_manager, preview_height=60)
            strip_h = strip.shape[0]
            ui_frame[-strip_h:, :] = strip
        
        return ui_frame
    
    def handle_key(self, key: int, frame: np.ndarray) -> bool:
        if key == -1:
            return True
        
        key = key & 0xFF
        
        # Quit
        if key == ord('q') or key == 27:  # q or ESC
            return False
        
        # Next filter 
        elif key == ord('.') or key == 3 or key == 83:  # Right arrow
            self.filter_manager.next_filter()
        
        # Previous filter 
        elif key == ord(',') or key == 2 or key == 81:  # Left arrow
            self.filter_manager.prev_filter()
        
        # Space - manual capture
        elif key == ord(' '):
            self.trigger_capture(frame)
        
        # Number keys for quick filter selection
        elif ord('1') <= key <= ord('9'):
            filter_idx = key - ord('1')
            if filter_idx < len(self.filter_manager.filter_names):
                self.filter_manager.current_filter_index = filter_idx
        
        return True
    
    def run(self):

        print("\n" + "=" * 60)
        print("       SMILEY BOOTH - Smart Photobooth")
        print("       CS445 Final Project")
        print("=" * 60)
        print("\nStarting camera...")
        
        if not self.init_camera():
            return
        
        print("\nControls:")
        print("  [SPACE] Take photo")
        print("  [LEFT/RIGHT] or [,/.] Change filter")
        print("  [1-9] Quick filter selection")
        print("  [Q] Quit")
        print("\nPhotos will be saved to:", os.path.abspath(self.output_dir))
        print("\n" + "-" * 60 + "\n")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Mirror the frame for natural interaction
                frame = cv2.flip(frame, 1)
                
                # Detect face and smile
                face_data = self.detector.detect(frame)
                
                # Check for auto-capture
                if self.capture_controller.update(face_data):
                    self.trigger_capture(frame)
                
                # Apply current filter for preview
                filtered_frame = self.filter_manager.apply_current_filter(frame)
                
                # Draw UI overlay
                display_frame = self.draw_ui(filtered_frame, face_data)
                
                # Show frame
                cv2.imshow(self.window_name, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1)
                if not self.handle_key(key, frame):
                    break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Goodbye!")


class DemoMode:
    def __init__(self):
        self.filter_manager = FilterManager()
        self.window_name = "Smiley Booth - Filter Demo"
    
    def create_test_image(self, width: int = 640, height: int = 480) -> np.ndarray:
        """Create a colorful test image"""
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Gradient background
        for y in range(height):
            for x in range(width):
                img[y, x] = [
                    int(255 * x / width),
                    int(255 * y / height),
                    int(255 * (1 - x / width))
                ]
        
        # Add some shapes
        cv2.circle(img, (width // 2, height // 2), 100, (255, 255, 255), -1)
        cv2.rectangle(img, (50, 50), (150, 150), (0, 255, 0), -1)
        cv2.rectangle(img, (width - 150, 50), (width - 50, 150), (255, 0, 0), -1)
        
        return img
    
    def run(self, image_path: Optional[str] = None):
        """Run filter demo"""
        if image_path and os.path.exists(image_path):
            test_image = cv2.imread(image_path)
        else:
            test_image = self.create_test_image()
        
        print("Filter Demo Mode")
        print("Press LEFT/RIGHT to change filters, Q to quit")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        while True:
            # Apply current filter
            filtered = self.filter_manager.apply_current_filter(test_image)
            
            # Add filter name
            filter_name = self.filter_manager.get_current_filter_name()
            cv2.putText(filtered, f"Filter: {filter_name}",
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow(self.window_name, filtered)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('.') or key == 3 or key == 83:
                self.filter_manager.next_filter()
            elif key == ord(',') or key == 2 or key == 81:
                self.filter_manager.prev_filter()
        
        cv2.destroyAllWindows()


def main():

    import argparse
    
    parser = argparse.ArgumentParser(
        description="Smiley Booth - Smart Photobooth Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python smiley_booth.py                    # Run with default camera
  python smiley_booth.py --camera 1         # Use camera ID 1
  python smiley_booth.py --demo             # Run filter demo mode
  python smiley_booth.py --output photos    # Save to 'photos' folder
        """
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='captured_photos',
        help='Output directory for captured photos (default: captured_photos)'
    )
    
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run in demo mode (test filters without camera)'
    )
    
    parser.add_argument(
        '--image', '-i',
        type=str,
        help='Image file for demo mode'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        demo = DemoMode()
        demo.run(args.image)
    else:
        booth = SmileyBooth(
            camera_id=args.camera,
            output_dir=args.output
        )
        booth.run()


if __name__ == "__main__":
    main()