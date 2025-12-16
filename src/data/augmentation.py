import albumentations as A
from albumentations.pytorch import ToTensorV2
import imgaug.augmenters as iaa
import numpy as np
from typing import Dict


class AugmentationPipeline:
    """Handles data augmentation for training"""
    
    def __init__(self, augmentation_type: str = 'medium'):
        self.augmentation_type = augmentation_type
        self.transform = self._build_transform()
    
    def _build_transform(self):
        """Build augmentation pipeline based on type"""
        if self.augmentation_type == 'light':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        
        elif self.augmentation_type == 'medium':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.2,
                    rotate_limit=30,
                    p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.4
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MotionBlur(blur_limit=5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        
        elif self.augmentation_type == 'heavy':
            return A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Transpose(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.3,
                    rotate_limit=45,
                    p=0.7
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
                A.GridDistortion(p=0.3),
                A.OpticalDistortion(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.3,
                    contrast_limit=0.3,
                    p=0.7
                ),
                A.HueSaturationValue(
                    hue_shift_limit=30,
                    sat_shift_limit=40,
                    val_shift_limit=30,
                    p=0.5
                ),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 100.0)),
                    A.GaussianBlur(blur_limit=(3, 9)),
                    A.MotionBlur(blur_limit=7),
                    A.MedianBlur(blur_limit=5),
                ], p=0.5),
                A.OneOf([
                    A.CLAHE(clip_limit=4.0),
                    A.Sharpen(),
                    A.Emboss(),
                ], p=0.3),
                A.CoarseDropout(
                    max_holes=8,
                    max_height=16,
                    max_width=16,
                    fill_value=0,
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
        
        else:  # 'none'
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ])
    
    def __call__(self, image: np.ndarray) -> Dict:
        """Apply augmentation to image"""
        return self.transform(image=image)


class ImgAugPipeline:
    """Alternative augmentation using imgaug library"""
    
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.3),
            iaa.Sometimes(0.5, iaa.Affine(
                rotate=(-30, 30),
                scale=(0.8, 1.2),
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            )),
            iaa.Sometimes(0.3, iaa.OneOf([
                iaa.GaussianBlur(sigma=(0, 3.0)),
                iaa.AverageBlur(k=(2, 7)),
                iaa.MedianBlur(k=(3, 7)),
            ])),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(
                loc=0,
                scale=(0.0, 0.05*255),
            )),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),
        ], random_order=True)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation to image"""
        return self.aug(image=image)


class ValidationTransform:
    """Augmentation for validation/test (only normalization)"""
    
    def __init__(self):
        self.transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225]),
        ])
    
    def __call__(self, image: np.ndarray) -> Dict:
        """Apply normalization to image"""
        return self.transform(image=image)


class TestTimeAugmentation:
    """Test-time augmentation for improved predictions"""
    
    def __init__(self, n_augments: int = 5):
        self.n_augments = n_augments
        self.augmentations = [
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.Rotate(limit=15, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.Rotate(limit=-15, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
            A.Compose([
                A.RandomBrightnessContrast(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
            ]),
        ]
    
    def __call__(self, image: np.ndarray) -> list:
        """Generate multiple augmented versions of the image"""
        augmented_images = [image]  # Original
        
        for aug in self.augmentations[:self.n_augments - 1]:
            aug_img = aug(image=image)['image']
            augmented_images.append(aug_img)
        
        return augmented_images


def get_training_augmentation(img_size: tuple = (224, 224), 
                              augmentation_level: str = 'medium'):
    """Get training augmentation pipeline"""
    return AugmentationPipeline(augmentation_type=augmentation_level)


def get_validation_augmentation():
    """Get validation augmentation pipeline (normalization only)"""
    return ValidationTransform()


def get_tta_augmentation(n_augments: int = 5):
    """Get test-time augmentation pipeline"""
    return TestTimeAugmentation(n_augments=n_augments)