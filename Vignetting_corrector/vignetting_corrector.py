import numpy as np
import cv2
import os

class VignettingCorrector():
    def __init__(self):
        self.correction_matrix = None
        self.matrix_shape = None

    def load_reference(self, path):
        if not path or not os.path.exists(path):
            return False

        if os.path.isdir(path):
            return False

        # Load with ANYDEPTH to preserve 12/16-bit if available
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        if img is None:
            return False
        
        # Create float representation
        img_float = img.astype(np.float32)
        
        # Apply slight blur to find true peak without noise
        img_blur = cv2.GaussianBlur(img_float, (21, 21), 0)
        max_val = np.max(img_blur)
        
        if max_val == 0:
            max_val = 1.0

        # Percent of the brightest pixel
        self.correction_matrix = img_blur / max_val
        # Prevent division by zero
        self.correction_matrix[self.correction_matrix == 0] = 1.0
        self.matrix_shape = img.shape
        return True

    def get_correction_preview(self, depth_str):
        if self.correction_matrix is None:
            return None
            
        if depth_str == "16-bit":
            max_val = 65535
            dtype = np.uint16
        elif depth_str == "12-bit":
            max_val = 4095
            dtype = np.uint16
        else:
            max_val = 255
            dtype = np.uint8
            
        # Matrix is 0-1, so multiply by max_val to get the visual
        preview = (self.correction_matrix * max_val).astype(dtype)
        return preview

    def correct_frame(self, frame, enabled=True, depth_str="8-bit"):
        if not enabled or self.correction_matrix is None or frame is None:
            return frame
        
        is_tuple = False
        if isinstance(frame, tuple):
            if len(frame) == 2:
                img = frame[0]
                name = frame[1]
                is_tuple = True
            else:
                img = frame[0]
                name = None
        else:
            img = frame
            name = None
            
        dtype = img.dtype
        # Handle cases where frame is smaller or different shape than the matrix
        if img.shape[:2] != self.matrix_shape:
            # Ensure safe resizing (e.g. width, height)
            matrix = cv2.resize(self.correction_matrix, (img.shape[1], img.shape[0]))
        else:
            matrix = self.correction_matrix

        # Apply correction (Corrected = Input / Correction_percent)
        # This increases the signal in periphery
        if len(img.shape) == 3:
            matrix_3d = np.expand_dims(matrix, axis=2)
            corrected = img.astype(np.float32) / matrix_3d
        else:
            corrected = img.astype(np.float32) / matrix
            
        if depth_str == "16-bit":
            max_val = 65535
        elif depth_str == "12-bit":
            max_val = 4095
        else:
            # fallback for 8-bit
            max_val = 255
            
        corrected = np.clip(corrected, 0, max_val).astype(dtype)
        
        if is_tuple:
            return (corrected, name)
        return corrected
