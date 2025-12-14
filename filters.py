"""
Image Filters for Smiley Booth
15 creative photo effects
"""

import cv2
import numpy as np
import random


class FilterManager:
    def __init__(self):
        # all available filters
        self.filters = {
            'normal': self.normal,
            'pencil_sketch': self.pencil_sketch,
            'color_sketch': self.color_sketch,
            'glitch': self.glitch,
            'thermal': self.thermal,
            'pinhole': self.pinhole,
            'vintage': self.vintage,
            'pop_art': self.pop_art,
            'neon': self.neon,
            'cartoon': self.cartoon,
            'emboss': self.emboss,
            'watercolor': self.watercolor,
            'noir': self.noir,
            'cyberpunk': self.cyberpunk,
            'vaporwave': self.vaporwave,
        }
        self.filter_names = list(self.filters.keys())
        self.current_filter_index = 0
        
    def get_current_filter_name(self):
        return self.filter_names[self.current_filter_index]
    
    def next_filter(self):
        self.current_filter_index = (self.current_filter_index + 1) % len(self.filter_names)
    
    def prev_filter(self):
        self.current_filter_index = (self.current_filter_index - 1) % len(self.filter_names)
    
    def apply_current_filter(self, frame):
        func = self.filters[self.filter_names[self.current_filter_index]]
        return func(frame)
    
    def apply_filter(self, frame, name):
        if name in self.filters:
            return self.filters[name](frame)
        return frame
    
    # helper for sepia tone
    def _sepia(self, img, strength=1.0):
        # sepia transformation matrix
        mat = np.array([[0.272, 0.534, 0.131],
                        [0.349, 0.686, 0.168],
                        [0.393, 0.769, 0.189]], dtype=np.float32)
        sep = cv2.transform(img.astype(np.float32), mat)
        sep = np.clip(sep, 0, 255)
        if strength < 1.0:
            out = img.astype(np.float32) * (1-strength) + sep * strength
        else:
            out = sep
        return out.astype(np.uint8)
    
    # helper for vignette effect
    def _vignette(self, img, power=1.5, strength=0.6):
        h, w = img.shape[:2]
        # create distance map from center
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        cx, cy = w/2, h/2
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        # darken edges based on distance
        mask = 1 - (dist/max_dist)**power * strength
        mask = np.clip(mask, 0, 1)
        mask = np.dstack([mask]*3)
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)
    
    # no filter, just return original
    def normal(self, frame):
        return frame.copy()
    
    # pencil sketch using dodge blend technique
    def pencil_sketch(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        # dodge blend: gray / (255 - blur)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    
    # colored version of pencil sketch
    def color_sketch(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
        # blend sketch with original colors
        blend = cv2.addWeighted(frame, 0.4, sketch_bgr, 0.6, 0)
        # boost saturation a bit
        hsv = cv2.cvtColor(blend, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # digital glitch effect
    def glitch(self, frame):
        result = frame.copy()
        h, w = frame.shape[:2]
        
        # shift random horizontal slices
        for _ in range(random.randint(3, 8)):
            y = random.randint(0, h-20)
            sh = random.randint(5, 30)
            shift = random.randint(-30, 30)
            ye = min(y + sh, h)
            if shift > 0:
                result[y:ye, shift:w] = frame[y:ye, :w-shift]
            elif shift < 0:
                result[y:ye, :w+shift] = frame[y:ye, -shift:w]
        
        # offset rgb channels for chromatic aberration
        b, g, r = cv2.split(result)
        M = np.float32([[1, 0, random.randint(-10, 10)], [0, 1, 0]])
        r = cv2.warpAffine(r, M, (w, h))
        M = np.float32([[1, 0, random.randint(-10, 10)], [0, 1, 0]])
        b = cv2.warpAffine(b, M, (w, h))
        result = cv2.merge([b, g, r])
        
        # add scanlines
        for y in range(0, h, 4):
            result[y:y+1, :] = (result[y:y+1, :] * 0.7).astype(np.uint8)
        
        # sometimes add a random noise block
        if random.random() > 0.7:
            bx, by = random.randint(0, w-50), random.randint(0, h-30)
            bw, bh = random.randint(30, 100), random.randint(10, 40)
            noise = np.random.randint(0, 255, (bh, bw, 3), dtype=np.uint8)
            ye, xe = min(by+bh, h), min(bx+bw, w)
            result[by:ye, bx:xe] = noise[:ye-by, :xe-bx]
        
        return result
    
    # thermal/heat camera look
    def thermal(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # map grayscale to heat colormap
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        # boost contrast with CLAHE
        lab = cv2.cvtColor(thermal, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    # pinhole camera / lomography look
    def pinhole(self, frame):
        h, w = frame.shape[:2]
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        cx, cy = w//2, h//2
        dist = np.sqrt((X-cx)**2 + (Y-cy)**2)
        max_dist = np.sqrt(cx**2 + cy**2)
        
        # blur edges more than center
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        mask = (dist / max_dist).clip(0, 1)
        mask = np.dstack([mask]*3)
        result = (frame * (1 - mask*0.5) + blur * mask*0.5).astype(np.uint8)
        # add vignette and slight sepia
        result = self._vignette(result, 1.5, 0.9)
        result = self._sepia(result, 0.3)
        return result
    
    # old photo / vintage film look
    def vintage(self, frame):
        result = self._sepia(frame, 1.0).astype(np.float32)
        # warm color cast
        result[:,:,2] = np.clip(result[:,:,2] * 1.1, 0, 255)  # more red
        result[:,:,0] = np.clip(result[:,:,0] * 0.9, 0, 255)  # less blue
        result = result.astype(np.uint8)
        # reduce saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = (hsv[:,:,1] * 0.7).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        result = self._vignette(result, 2.0, 0.5)
        # add film grain
        noise = np.random.normal(0, 15, result.shape).astype(np.float32)
        return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # pop art / comic book style
    def pop_art(self, frame):
        # posterize to reduce colors
        result = (frame // 42) * 42
        # crank up saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 2.0, 0, 255).astype(np.uint8)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # add black edges
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
        edges = cv2.dilate(edges, None)
        result[edges > 0] = [0, 0, 0]
        return result
    
    # neon glow edges
    def neon(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        # create colored glow layers
        dilated = cv2.dilate(edges, kernel, iterations=2)
        glow = np.zeros_like(frame)
        glow[:,:,0] = dilated  # blue
        glow[:,:,1] = cv2.dilate(edges, kernel, iterations=1)  # green
        glow[:,:,2] = edges  # red
        glow = cv2.GaussianBlur(glow, (15, 15), 0)
        # overlay on dark background
        dark = (frame * 0.2).astype(np.uint8)
        return cv2.addWeighted(dark, 1, glow, 2, 0)
    
    # cartoon effect using bilateral filter
    def cartoon(self, frame):
        # smooth colors while keeping edges
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        gray = cv2.medianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 7)
        # get edges
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)
        # posterize colors
        color = (color // 32) * 32
        return cv2.bitwise_and(color, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    
    # emboss / relief effect
    def emboss(self, frame):
        # 3x3 emboss kernel
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        result = cv2.filter2D(frame, -1, kernel) + 128
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # watercolor painting look
    def watercolor(self, frame):
        result = frame.copy()
        # multiple passes of bilateral filter for smooth look
        for _ in range(3):
            result = cv2.bilateralFilter(result, 9, 75, 75)
        # reduce saturation slightly
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = (hsv[:,:,1] * 0.8).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # add paper texture noise
        h, w = result.shape[:2]
        noise = cv2.GaussianBlur(np.random.normal(0, 10, (h, w)).astype(np.float32), (5, 5), 0)
        noise = np.dstack([noise]*3)
        return np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    # film noir black and white
    def noir(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # high contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # push contrast further
        gray = np.clip(gray * 1.3 - 30, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # slight cold/blue tint
        result[:,:,0] = np.clip(result[:,:,0] * 1.1, 0, 255).astype(np.uint8)
        return self._vignette(result, 1.5, 0.6)
    
    # cyberpunk neon city look
    def cyberpunk(self, frame):
        # boost contrast
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # increase saturation
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.5, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # push toward cyan and magenta
        b, g, r = cv2.split(result)
        b = np.clip(b.astype(np.float32) + 30, 0, 255).astype(np.uint8)
        g = np.clip(g.astype(np.float32) + 15, 0, 255).astype(np.uint8)
        mask = result.mean(axis=2) > 128
        r[mask] = np.clip(r[mask].astype(np.float32) + 40, 0, 255).astype(np.uint8)
        result = cv2.merge([b, g, r])
        # scanlines
        h, w = result.shape[:2]
        for y in range(0, h, 3):
            result[y:y+1, :] = (result[y:y+1, :] * 0.8).astype(np.uint8)
        return result
    
    # vaporwave aesthetic
    def vaporwave(self, frame):
        # shift hue toward pink/purple
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + 150) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.4, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        # vertical gradient overlay
        h, w = result.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.float32)
        for y in range(h):
            r = y / h
            gradient[y, :] = [255*r, 100*(1-r), 255*(1-r)]
        result = cv2.addWeighted(result, 0.7, gradient.astype(np.uint8), 0.3, 0)
        # scanlines for that retro feel
        for y in range(0, h, 4):
            result[y:y+2, :] = (result[y:y+2, :] * 0.85).astype(np.uint8)
        return result


# creates the filter preview strip at bottom of screen
def create_filter_preview_strip(frame, filter_manager, preview_height=80):
    h, w = frame.shape[:2]
    pw = w // len(filter_manager.filter_names)
    small = cv2.resize(frame, (pw, preview_height))
    strip = np.zeros((preview_height + 30, w, 3), dtype=np.uint8)
    
    for i, name in enumerate(filter_manager.filter_names):
        x = i * pw
        filtered = filter_manager.apply_filter(small, name)
        strip[:preview_height, x:x+pw] = filtered
        # highlight current filter
        if i == filter_manager.current_filter_index:
            cv2.rectangle(strip, (x, 0), (x+pw, preview_height), (0, 255, 0), 3)
        cv2.putText(strip, name[:8], (x+5, preview_height+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return strip
