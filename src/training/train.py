import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    TensorBoard, CSVLogger
)
import numpy as np
from pathlib import Path
import yaml
import json
from datetime import datetime
from typing import Dict, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.data_loader import DataLoaderManager
from src.models.transfer_learning import TransferLearningModel
from src.models.cnn_model import build_custom_cnn
from src.utils.logger import setup_logger


class ModelTrainer:
    """Handles model training pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_logger('ModelTrainer')
        self.model = None
        self.history = None
        
        # Create directories
        self.models_dir = Path(config['paths']['models_dir'])
        self.logs_dir = Path(config['paths']['logs_dir'])
        self.tensorboard_dir = Path(config['paths']['tensorboard_dir'])
        
        for directory in [self.models_dir, self.logs_dir, self.tensorboard_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def build_model(self):
        """Build model based on configuration"""
        model_config = self.config['model']
        architecture = model_config['architecture']
        
        self.logger.info(f"Building {architecture} model...")
        
        if architecture.startswith('resnet') or architecture.startswith('efficientnet') \
           or architecture.startswith('mobilenet') or architecture in ['vgg16', 'vgg19']:
            # Transfer learning model
            model_builder = TransferLearningModel(
                architecture=architecture,
                num_classes=model_config['num_classes'],
                freeze_backbone=model_config['freeze_backbone'],
                dropout_rate=model_config['dropout_rate']
            )
            self.model = model_builder.build_model()
        else:
            # Custom CNN
            self.model = build_custom_cnn(
                num_classes=model_config['num_classes'],
                architecture=architecture
            )
        
        self.logger.info("Model built successfully")
        return self.model
    
    def compile_model(self):
        """Compile model with optimizer and loss"""
        training_config = self.config['training']
        
        # Optimizer
        optimizer_name = training_config['optimizer'].lower()
        learning_rate = training_config['learning_rate']
        
        if optimizer_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == 'sgd':
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate,
                momentum=0.9,
                nesterov=True
            )
        elif optimizer_name == 'adamw':
            optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        # Loss function
        loss = training_config['loss_function']
        
        # Metrics
        metrics = ['accuracy', 
                  tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                  tf.keras.metrics.Precision(name='precision'),
                  tf.keras.metrics.Recall(name='recall')]
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        
        self.logger.info("Model compiled successfully")
    
    def get_callbacks(self) -> list:
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint
        checkpoint_config = self.config['training']['checkpoint']
        checkpoint_path = self.models_dir / 'best_model.h5'
        checkpoint = ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=checkpoint_config['monitor'],
            save_best_only=checkpoint_config['save_best_only'],
            mode=checkpoint_config['mode'],
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Early stopping
        es_config = self.config['training']['early_stopping']
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=es_config['patience'],
            min_delta=es_config['min_delta'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Reduce learning rate
        rlr_config = self.config['training']['reduce_lr']
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=rlr_config['factor'],
            patience=rlr_config['patience'],
            min_lr=rlr_config['min_lr'],
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = TensorBoard(
            log_dir=str(self.tensorboard_dir / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1,
            write_graph=True,
            write_images=True
        )
        callbacks.append(tensorboard)
        
        # CSV Logger
        csv_logger = CSVLogger(
            str(self.logs_dir / 'training_log.csv'),
            append=True
        )
        callbacks.append(csv_logger)
        
        return callbacks
    
    def train(self, train_dataset, val_dataset) -> Dict:
        """Train the model"""
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        epochs = self.config['training']['epochs']
        
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        # Train model
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        self.logger.info("Training completed")
        
        # Save final model
        final_model_path = self.models_dir / 'final_model.h5'
        self.model.save(str(final_model_path))
        self.logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        history_path = self.logs_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            history_dict = {k: [float(v) for v in val] 
                          for k, val in self.history.history.items()}
            json.dump(history_dict, f, indent=2)
        
        return self.history.history
    
    def evaluate(self, test_dataset) -> Dict:
        """Evaluate model on test set"""
        self.logger.info("Evaluating model on test set...")
        
        results = self.model.evaluate(test_dataset, verbose=1, return_dict=True)
        
        self.logger.info(f"Test Results: {results}")
        
        # Save evaluation results
        eval_path = self.logs_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        
        return results
    
    def fine_tune(self, train_dataset, val_dataset, 
                  num_layers_to_unfreeze: int = 10) -> Dict:
        """Fine-tune the model by unfreezing top layers"""
        self.logger.info(f"Fine-tuning model (unfreezing top {num_layers_to_unfreeze} layers)...")
        
        # Unfreeze top layers
        base_model = self.model.layers[1]  # Assuming first layer is input
        base_model.trainable = True
        
        for layer in base_model.layers[:-num_layers_to_unfreeze]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        lower_lr = self.config['training']['learning_rate'] / 10
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lower_lr),
            loss=self.config['training']['loss_function'],
            metrics=['accuracy']
        )
        
        # Train
        fine_tune_epochs = self.config['training']['epochs'] // 2
        
        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=fine_tune_epochs,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        # Save fine-tuned model
        ft_model_path = self.models_dir / 'fine_tuned_model.h5'
        self.model.save(str(ft_model_path))
        self.logger.info(f"Fine-tuned model saved to {ft_model_path}")
        
        return self.history.history


def main():
    """Main training script"""
    # Load config
    config_path = Path('config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize data loader
    data_loader = DataLoaderManager(config)
    
    # Prepare dataset
    print("Preparing dataset...")
    image_paths, labels = data_loader.prepare_dataset()
    splits = data_loader.split_dataset(image_paths, labels)
    
    # Create TensorFlow datasets
    train_dataset = data_loader.create_tf_dataset(
        splits['train'][0], 
        splits['train'][1],
        augment=True
    )
    
    val_dataset = data_loader.create_tf_dataset(
        splits['val'][0],
        splits['val'][1],
        augment=False
    )
    
    test_dataset = data_loader.create_tf_dataset(
        splits['test'][0],
        splits['test'][1],
        augment=False
    )
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Build and train model
    trainer.build_model()
    trainer.compile_model()
    
    print("\nModel Summary:")
    trainer.model.summary()
    
    # Train
    history = trainer.train(train_dataset, val_dataset)
    
    # Evaluate
    results = trainer.evaluate(test_dataset)
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()