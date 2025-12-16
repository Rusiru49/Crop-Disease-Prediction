import os
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import tensorflow as tf
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split


class PlantDiseaseDataset(Dataset):
    """PyTorch Dataset for Plant Disease Images"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform=None, img_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.img_size = img_size
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.img_size)
        image = np.array(image) / 255.0
        
        if self.transform:
            image = self.transform(image=image)['image']
        
        label = self.labels[idx]
        return torch.FloatTensor(image).permute(2, 0, 1), torch.LongTensor([label])


class DataLoaderManager:
    """Manages data loading and preprocessing"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.img_size = tuple(config['dataset']['img_size'])
        self.batch_size = config['dataset']['batch_size']
        self.data_dir = Path(config['dataset']['raw_path'])
        self.class_names = []
        self.class_to_idx = {}
        
    def load_class_mapping(self):
        """Load or create class mapping"""
        metadata_path = Path('data/metadata/class_labels.json')
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                self.class_names = data['classes']
                self.class_to_idx = data['class_to_idx']
        else:
            # Create from directory structure
            self.class_names = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            
            # Save mapping
            os.makedirs(metadata_path.parent, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump({
                    'classes': self.class_names,
                    'class_to_idx': self.class_to_idx,
                    'num_classes': len(self.class_names)
                }, f, indent=2)
    
    def prepare_dataset(self) -> Tuple[List, List]:
        """Prepare image paths and labels"""
        self.load_class_mapping()
        
        image_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_file in class_dir.glob('*.jpg'):
                image_paths.append(str(img_file))
                labels.append(class_idx)
            for img_file in class_dir.glob('*.png'):
                image_paths.append(str(img_file))
                labels.append(class_idx)
        
        return image_paths, labels
    
    def split_dataset(self, image_paths: List, labels: List) -> Dict:
        """Split dataset into train, validation, and test sets"""
        split_ratio = self.config['dataset']['split_ratio']
        
        # First split: train and temp (val+test)
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels, 
            test_size=(1 - split_ratio['train']),
            stratify=labels,
            random_state=42
        )
        
        # Second split: val and test
        val_size = split_ratio['val'] / (split_ratio['val'] + split_ratio['test'])
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=42
        )
        
        return {
            'train': (train_paths, train_labels),
            'val': (val_paths, val_labels),
            'test': (test_paths, test_labels)
        }
    
    def create_tf_dataset(self, image_paths: List, labels: List, 
                         augment: bool = False) -> tf.data.Dataset:
        """Create TensorFlow dataset"""
        
        def load_and_preprocess(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.img_size)
            img = tf.cast(img, tf.float32) / 255.0
            
            if augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_brightness(img, 0.2)
                img = tf.image.random_contrast(img, 0.8, 1.2)
            
            return img, label
        
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(1000) if augment else dataset
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def get_data_statistics(self, labels: List) -> Dict:
        """Calculate dataset statistics"""
        unique, counts = np.unique(labels, return_counts=True)
        
        stats = {
            'total_images': len(labels),
            'num_classes': len(unique),
            'class_distribution': {
                self.class_names[idx]: int(count) 
                for idx, count in zip(unique, counts)
            },
            'min_samples': int(np.min(counts)),
            'max_samples': int(np.max(counts)),
            'mean_samples': float(np.mean(counts)),
            'std_samples': float(np.std(counts))
        }
        
        return stats


def save_dataset_splits(data_splits: Dict, output_dir: str):
    """Save dataset splits to disk"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, (paths, labels) in data_splits.items():
        split_data = {
            'image_paths': paths,
            'labels': labels,
            'count': len(labels)
        }
        
        with open(output_path / f'{split_name}_split.json', 'w') as f:
            json.dump(split_data, f, indent=2)
    
    print(f"Dataset splits saved to {output_dir}")