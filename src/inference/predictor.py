import numpy as np
import tensorflow as tf
import torch
from typing import Dict, List, Tuple, Union
import json
from pathlib import Path
from datetime import datetime
import cv2


class DiseasePredictor:
    """Handles disease prediction from plant images"""
    
    def __init__(self, model: Union[tf.keras.Model, torch.nn.Module], 
                 class_names: List[str] = None,
                 confidence_threshold: float = 0.7,
                 framework: str = 'tensorflow'):
        self.model = model
        self.class_names = class_names or self._load_class_names()
        self.confidence_threshold = confidence_threshold
        self.framework = framework
        self.num_classes = len(self.class_names)
        
    def _load_class_names(self) -> List[str]:
        """Load class names from metadata"""
        metadata_path = Path('data/metadata/class_labels.json')
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                return data['classes']
        except FileNotFoundError:
            return [f"Class_{i}" for i in range(38)]
    
    def predict(self, image: np.ndarray, return_probs: bool = False) -> Dict:
        """Predict disease from image"""
        # Ensure image is in correct format
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        if self.framework == 'tensorflow':
            predictions = self.model.predict(image, verbose=0)
        else:  # pytorch
            with torch.no_grad():
                image_tensor = torch.FloatTensor(image).permute(0, 3, 1, 2)
                predictions = self.model(image_tensor)
                predictions = torch.softmax(predictions, dim=1).numpy()
        
        # Get top prediction
        pred_class = np.argmax(predictions[0])
        confidence = float(predictions[0][pred_class])
        disease_name = self.class_names[pred_class]
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[::-1][:5]
        top_predictions = [
            (self.class_names[idx], float(predictions[0][idx]))
            for idx in top_indices
        ]
        
        result = {
            'disease': disease_name,
            'class_id': int(pred_class),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'is_confident': confidence >= self.confidence_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        if return_probs:
            result['all_probabilities'] = predictions[0].tolist()
        
        return result
    
    def predict_batch(self, images: np.ndarray) -> List[Dict]:
        """Predict diseases for multiple images"""
        if self.framework == 'tensorflow':
            predictions = self.model.predict(images, verbose=0)
        else:  # pytorch
            with torch.no_grad():
                images_tensor = torch.FloatTensor(images).permute(0, 3, 1, 2)
                predictions = self.model(images_tensor)
                predictions = torch.softmax(predictions, dim=1).numpy()
        
        results = []
        for i, pred in enumerate(predictions):
            pred_class = np.argmax(pred)
            confidence = float(pred[pred_class])
            disease_name = self.class_names[pred_class]
            
            top_indices = np.argsort(pred)[::-1][:5]
            top_predictions = [
                (self.class_names[idx], float(pred[idx]))
                for idx in top_indices
            ]
            
            results.append({
                'disease': disease_name,
                'class_id': int(pred_class),
                'confidence': confidence,
                'top_predictions': top_predictions,
                'is_confident': confidence >= self.confidence_threshold
            })
        
        return results
    
    def predict_with_tta(self, image: np.ndarray, n_augments: int = 5) -> Dict:
        """Predict with test-time augmentation"""
        from src.data.augmentation import TestTimeAugmentation
        
        tta = TestTimeAugmentation(n_augments=n_augments)
        augmented_images = tta(image)
        
        # Stack images
        images_batch = np.stack(augmented_images)
        
        # Get predictions for all augmented versions
        batch_predictions = self.predict_batch(images_batch)
        
        # Average predictions
        all_probs = np.array([
            pred['all_probabilities'] if 'all_probabilities' in pred 
            else self._get_probs_from_pred(pred)
            for pred in batch_predictions
        ])
        avg_probs = np.mean(all_probs, axis=0)
        
        # Get final prediction
        pred_class = np.argmax(avg_probs)
        confidence = float(avg_probs[pred_class])
        disease_name = self.class_names[pred_class]
        
        top_indices = np.argsort(avg_probs)[::-1][:5]
        top_predictions = [
            (self.class_names[idx], float(avg_probs[idx]))
            for idx in top_indices
        ]
        
        return {
            'disease': disease_name,
            'class_id': int(pred_class),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'is_confident': confidence >= self.confidence_threshold,
            'tta_used': True,
            'n_augments': n_augments
        }
    
    def _get_probs_from_pred(self, pred: Dict) -> np.ndarray:
        """Reconstruct probability distribution from prediction"""
        probs = np.zeros(self.num_classes)
        for class_name, prob in pred['top_predictions']:
            class_id = self.class_names.index(class_name)
            probs[class_id] = prob
        return probs
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = threshold


class EnsemblePredictor:
    """Ensemble of multiple models for improved predictions"""
    
    def __init__(self, models: List, class_names: List[str], 
                 weights: List[float] = None):
        self.models = models
        self.class_names = class_names
        self.weights = weights or [1.0 / len(models)] * len(models)
        
        if len(self.weights) != len(self.models):
            raise ValueError("Number of weights must match number of models")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def predict(self, image: np.ndarray) -> Dict:
        """Predict using ensemble of models"""
        all_predictions = []
        
        for model in self.models:
            predictor = DiseasePredictor(model, self.class_names)
            pred = predictor.predict(image, return_probs=True)
            all_predictions.append(pred['all_probabilities'])
        
        # Weighted average of predictions
        weighted_probs = np.zeros(len(self.class_names))
        for probs, weight in zip(all_predictions, self.weights):
            weighted_probs += np.array(probs) * weight
        
        # Get final prediction
        pred_class = np.argmax(weighted_probs)
        confidence = float(weighted_probs[pred_class])
        disease_name = self.class_names[pred_class]
        
        top_indices = np.argsort(weighted_probs)[::-1][:5]
        top_predictions = [
            (self.class_names[idx], float(weighted_probs[idx]))
            for idx in top_indices
        ]
        
        return {
            'disease': disease_name,
            'class_id': int(pred_class),
            'confidence': confidence,
            'top_predictions': top_predictions,
            'ensemble_used': True,
            'num_models': len(self.models)
        }


def load_predictor(model_path: str, 
                  class_names_path: str = None,
                  confidence_threshold: float = 0.7) -> DiseasePredictor:
    """Load predictor from saved model"""
    # Load model
    if model_path.endswith('.h5') or model_path.endswith('.keras'):
        model = tf.keras.models.load_model(model_path)
        framework = 'tensorflow'
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        model = torch.load(model_path)
        model.eval()
        framework = 'pytorch'
    else:
        raise ValueError(f"Unknown model format: {model_path}")
    
    # Load class names
    class_names = None
    if class_names_path:
        with open(class_names_path, 'r') as f:
            data = json.load(f)
            class_names = data['classes']
    
    return DiseasePredictor(
        model=model,
        class_names=class_names,
        confidence_threshold=confidence_threshold,
        framework=framework
    )