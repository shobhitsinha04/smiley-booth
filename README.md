# Smiley Booth - Smart Photobooth

**CS445 Computational Photography - Final Project**

**Team:** Shobhit Sinha (ss194), Jay Goenka (jgoenka2), Adit Agarwal (adit3)

---

## Overview

Smiley Booth is a smart photobooth application that automatically captures photos when the user is centered in the frame and smiling. The system uses real-time face detection and expression analysis to determine the optimal moment for capture. It also includes 15 creative image filters that can be applied to photos.

---

## Installation

```bash
# Install required packages
pip install -r requirements.txt

# Run the application
python smiley_booth.py
```

### Command line options:
```bash
python smiley_booth.py --camera 1      # Use different camera
python smiley_booth.py --output photos # Change output folder
python smiley_booth.py --demo          # Test filters without camera
```

---

## Controls

| Key | Action |
|-----|--------|
| SPACE | Take photo manually |
| Left/Right arrows | Change filter |
| 1-9 | Quick filter selection |
| Q | Quit application |

---

## Project Structure

### smiley_booth.py
The main application file that ties everything together. It handles:
- Camera initialization and frame capture
- Calling the detection and filter modules
- Rendering the UI overlay
- Saving captured photos

### detection.py
Handles all face and smile detection using MediaPipe Face Mesh, which provides 468 facial landmark points.

**Smile detection algorithm:**

We analyze 4 geometric features of the mouth:

1. **Mouth Aspect Ratio (MAR)** - Distance between mouth corners normalized by face width. Smiles make the mouth wider.

2. **Corner Lift** - Measures if mouth corners are above the center of the lips. When smiling, corners lift upward.

3. **Mouth Opening** - Distance between inner lips. Smiles often involve slight mouth opening.

4. **Corner Angle** - Angle of mouth corners relative to horizontal. Positive angles indicate upturned corners.

These features are combined into a weighted score:
```
score = (MAR * 0.35) + (lift * 0.40) + (opening * 0.15) + (angle * 0.10)
```

Penalties are applied for asymmetric expressions and frowns. A score above 55% triggers smile detection.

**Centering detection:**
The face center is compared to the frame center. If within 12% tolerance, the face is considered centered.

### filters.py
Contains 15 image filter implementations:

| Filter | Technique |
|--------|-----------|
| Normal | No effect |
| Pencil Sketch | Grayscale + inversion + Gaussian blur + dodge blend |
| Color Sketch | Pencil sketch blended with original colors, saturation boost |
| Glitch | RGB channel separation and shifting, random slice displacement, scanlines |
| Thermal | Grayscale to JET colormap, CLAHE contrast enhancement |
| Pinhole | Radial blur at edges, vignette darkening, sepia tint |
| Vintage | Sepia tone, warm color cast, reduced saturation, film grain noise |
| Pop Art | Color posterization (6 levels), saturation boost, Canny edge overlay |
| Neon | Canny edges with colored glow on dark background |
| Cartoon | Bilateral filter + adaptive threshold edges + posterized colors |
| Emboss | 3x3 convolution kernel for relief effect |
| Watercolor | Multiple bilateral filter passes, soft noise texture |
| Noir | High contrast B&W with CLAHE, blue tint, strong vignette |
| Cyberpunk | CLAHE contrast, saturated cyan/magenta, scanlines |
| Vaporwave | Hue rotation toward pink/purple, vertical gradient overlay |

---

## How It Works

```
[Webcam] --> [Face Detection] --> [Smile Analysis] --> [Apply Filter] --> [Display]
                   |                     |
                   v                     v
            Check centering        Count frames
                   |                     |
                   v                     v
            Show direction         Auto-capture
```

**Auto-capture process:**
1. MediaPipe detects face and extracts 468 landmarks
2. System checks if face center is within 12% of frame center
3. If centered, mouth landmarks are analyzed for smile
4. Smile must be detected for 80 consecutive frames (~3 seconds)
5. When triggered, both original and filtered photos are saved
6. 45-frame cooldown before next capture

---

## Technical Implementation

**Libraries used:**
- OpenCV - Camera capture, image processing, drawing
- MediaPipe - Face mesh detection with 468 landmarks
- NumPy - Array operations for filter effects

**Color spaces:**
- BGR: Default OpenCV format for color images
- Grayscale: Used for edge detection, sketches, thermal
- HSV: Hue/saturation manipulation (vaporwave, pop art)
- LAB: Lightness channel for CLAHE contrast (thermal, noir, cyberpunk)

**Key techniques:**
- Gaussian blur for smoothing
- Canny edge detection for outlines
- Bilateral filter for edge-preserving smoothing (cartoon, watercolor)
- Affine transforms for channel shifting (glitch)
- Color mapping for thermal effect
- Convolution kernels for emboss

---

## Output

Photos are saved to `captured_photos/` directory:
```
smiley_booth_20241208_143052_original.jpg
smiley_booth_20241208_143052_vintage.jpg
```

Format: `smiley_booth_DATE_TIME_FILTERNAME.jpg`

Both original and filtered versions are saved for each capture.

---

## Dependencies

```
opencv-python
opencv-contrib-python  
numpy
mediapipe
Pillow
```

---

## Troubleshooting

**Camera not detected:**
Try specifying a different camera ID: `python smiley_booth.py --camera 1`

**Smile not being detected:**
- Make sure your face is well-lit
- Look directly at the camera
- Try a natural smile rather than forced


---

## References

- OpenCV documentation: https://docs.opencv.org/
- MediaPipe Face Mesh: https://google.github.io/mediapipe/solutions/face_mesh.html
