"""
Smiley Booth - Smart Photobooth
CS445 Final Project
Shobhit Sinha, Jay Goenka, Adit Agarwal
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
import argparse

from detection import FaceDetector, CaptureController
from filters import FilterManager, create_filter_preview_strip


class SmileyBooth:
    def __init__(self, camera_id=0, output_dir="captured_photos"):
        self.camera_id = camera_id
        self.cap = None
        
        self.detector = FaceDetector()
        self.capture_controller = CaptureController(required_smile_frames=80, cooldown_frames=45)
        self.filter_manager = FilterManager()
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.window_name = "Smiley Booth"
        
        # for flash effect and preview
        self.flash_alpha = 0
        self.captured_image = None
        self.show_preview = False
        self.preview_start = 0
        
    def init_camera(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera initialized: {w}x{h}")
        return True
    
    def save_photo(self, frame, filtered=True):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if filtered:
            name = self.filter_manager.get_current_filter_name()
        else:
            name = "original"
        filename = f"smiley_booth_{timestamp}_{name}.jpg"
        path = os.path.join(self.output_dir, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"Saved: {path}")
        return path
    
    def capture(self, frame):
        # apply filter and save both versions
        filtered = self.filter_manager.apply_current_filter(frame)
        self.save_photo(frame, filtered=False)
        self.save_photo(filtered, filtered=True)
        
        # setup preview
        self.captured_image = filtered.copy()
        self.show_preview = True
        self.preview_start = time.time()
        self.flash_alpha = 1.0
        print("Photo captured!")
    
    def draw_ui(self, frame, face_data):
        ui = frame.copy()
        h, w = frame.shape[:2]
        
        # draw face detection stuff
        ui = self.detector.draw_detection_overlay(ui, face_data)
        
        # show current filter name
        fname = self.filter_manager.get_current_filter_name().upper()
        cv2.putText(ui, f"Filter: {fname}", (w - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # countdown circle when capturing
        if face_data:
            countdown = self.capture_controller.get_countdown()
            if countdown > 0:
                progress = 1 - (countdown / self.capture_controller.required_smile_frames)
                angle = int(360 * progress)
                center = (w // 2, h - 80)
                cv2.ellipse(ui, center, (40, 40), -90, 0, angle, (0, 255, 0), 5)
                cv2.putText(ui, str(countdown), (center[0] - 10, center[1] + 10),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # flash effect
        if self.flash_alpha > 0:
            white = np.ones_like(ui) * 255
            ui = cv2.addWeighted(ui, 1 - self.flash_alpha, white, self.flash_alpha, 0)
            self.flash_alpha = max(0, self.flash_alpha - 0.1)
        
        # show preview of captured photo
        if self.show_preview and self.captured_image is not None:
            if time.time() - self.preview_start < 2.0:
                ph = 150
                pw = int(ph * w / h)
                preview = cv2.resize(self.captured_image, (pw, ph))
                x = w - pw - 20
                y = h - ph - 20
                cv2.rectangle(ui, (x-5, y-5), (x+pw+5, y+ph+5), (0, 255, 0), 3)
                ui[y:y+ph, x:x+pw] = preview
                cv2.putText(ui, "SAVED!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                self.show_preview = False
        
        # filter strip at bottom
        strip = create_filter_preview_strip(frame, self.filter_manager, preview_height=60)
        ui[-strip.shape[0]:, :] = strip
        
        return ui
    
    def handle_key(self, key, frame):
        if key == -1:
            return True
        key = key & 0xFF
        
        if key == ord('q') or key == 27:
            return False
        elif key == ord('.') or key == 83:  # right arrow
            self.filter_manager.next_filter()
        elif key == ord(',') or key == 81:  # left arrow
            self.filter_manager.prev_filter()
        elif key == ord(' '):
            self.capture(frame)
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(self.filter_manager.filter_names):
                self.filter_manager.current_filter_index = idx
        return True
    
    def run(self):
        print("\n" + "=" * 50)
        print("  SMILEY BOOTH - CS445 Final Project")
        print("=" * 50)
        
        if not self.init_camera():
            return
        
        print("\nControls: SPACE=capture, arrows=change filter, Q=quit")
        print(f"Photos saved to: {os.path.abspath(self.output_dir)}\n")
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error reading frame")
                    break
                
                frame = cv2.flip(frame, 1)  # mirror
                
                face_data = self.detector.detect(frame)
                
                # auto capture when smiling
                if self.capture_controller.update(face_data):
                    self.capture(frame)
                
                # apply filter and draw ui
                filtered = self.filter_manager.apply_current_filter(frame)
                display = self.draw_ui(filtered, face_data)
                
                cv2.imshow(self.window_name, display)
                
                if not self.handle_key(cv2.waitKey(1), frame):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            print("Bye!")


class DemoMode:
    """For testing filters without a camera"""
    def __init__(self):
        self.filter_manager = FilterManager()
        
    def run(self, image_path=None):
        if image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
        else:
            # generate test pattern
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            for y in range(480):
                for x in range(640):
                    img[y, x] = [int(255*x/640), int(255*y/480), int(255*(1-x/640))]
            cv2.circle(img, (320, 240), 100, (255, 255, 255), -1)
        
        print("Demo mode - LEFT/RIGHT to change filters, Q to quit")
        cv2.namedWindow("Filter Demo", cv2.WINDOW_NORMAL)
        
        while True:
            filtered = self.filter_manager.apply_current_filter(img)
            name = self.filter_manager.get_current_filter_name()
            cv2.putText(filtered, f"Filter: {name}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Filter Demo", filtered)
            
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('.') or key == 83:
                self.filter_manager.next_filter()
            elif key == ord(',') or key == 81:
                self.filter_manager.prev_filter()
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smiley Booth - Smart Photobooth")
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera ID')
    parser.add_argument('--output', '-o', type=str, default='captured_photos', help='Output folder')
    parser.add_argument('--demo', '-d', action='store_true', help='Demo mode (no camera)')
    parser.add_argument('--image', '-i', type=str, help='Image for demo mode')
    
    args = parser.parse_args()
    
    if args.demo:
        DemoMode().run(args.image)
    else:
        SmileyBooth(camera_id=args.camera, output_dir=args.output).run()


if __name__ == "__main__":
    main()
