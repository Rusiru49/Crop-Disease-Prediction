import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
import tensorflow as tf


class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        self.img_size = img_size
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size"""
        return cv2.resize(image, self.img_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Normalize image pixel values"""
        if method == 'standard':
            # Scale to [0, 1]
            return image.astype(np.float32) / 255.0
        elif method == 'imagenet':
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image.astype(np.float32) / 255.0
            return (image - mean) / std
        elif method == 'zscore':
            # Z-score normalization
            mean = np.mean(image)
            std = np.std(image)
            return (image - mean) / (std + 1e-7)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def denormalize_image(self, image: np.ndarray, method: str = 'standard') -> np.ndarray:
        """Denormalize image for visualization"""
        if method == 'imagenet':
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = (image * std + mean) * 255.0
        else:
            image = image * 255.0
        
        return np.clip(image, 0, 255).astype(np.uint8)
    
    def remove_background(self, image: np.ndarray, method: str = 'grabcut') -> np.ndarray:
        """Remove background from leaf images"""
        if method == 'grabcut':
            mask = np.zeros(image.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            h, w = image.shape[:2]
            rect = (10, 10, w-10, h-10)
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            result = image * mask2[:, :, np.newaxis]
            return result
        
        elif method == 'threshold':
            # Simple HSV-based segmentation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([90, 255, 255])
            
            mask = cv2.inRange(hsv, lower_green, upper_green)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            result = cv2.bitwise_and(image, image, mask=mask)
            return result
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def augment_brightness(self, image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """Adjust image brightness"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def crop_center(self, image: np.ndarray, crop_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Crop center of image"""
        if crop_size is None:
            crop_size = self.img_size
        
        h, w = image.shape[:2]
        crop_h, crop_w = crop_size
        
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        
        return image[start_h:start_h+crop_h, start_w:start_w+crop_w]
    
    def preprocess_for_model(self, image: np.ndarray, 
                            normalize_method: str = 'standard',
                            enhance: bool = False) -> np.ndarray:
        """Complete preprocessing pipeline for model input"""
        # Resize
        processed = self.resize_image(image)
        
        # Enhance if requested
        if enhance:
            processed = self.enhance_image(processed)
        
        # Normalize
        processed = self.normalize_image(processed, method=normalize_method)
        
        return processed
    
    def preprocess_batch(self, images: list, 
                        normalize_method: str = 'standard',
                        enhance: bool = False) -> np.ndarray:
        """Preprocess a batch of images"""
        processed_images = []
        
        for img in images:
            processed = self.preprocess_for_model(img, normalize_method, enhance)
            processed_images.append(processed)
        
        return np.array(processed_images)


class TFImagePreprocessor:
    """TensorFlow-based image preprocessing"""
    
    @staticmethod
    def preprocess_input(image: tf.Tensor, img_size: Tuple[int, int]) -> tf.Tensor:
        """Preprocess image for model input"""
        image = tf.image.resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    
    @staticmethod
    def augment_image(image: tf.Tensor) -> tf.Tensor:
        """Apply augmentation to image"""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image


def load_and_preprocess_image(image_path: str, 
                              img_size: Tuple[int, int] = (224, 224),
                              normalize: bool = True) -> np.ndarray:
    """Utility function to load and preprocess a single image"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    preprocessor = ImagePreprocessor(img_size)
    processed = preprocessor.preprocess_for_model(
        image, 
        normalize_method='standard' if normalize else None,
        enhance=True
    )
    
    return processed