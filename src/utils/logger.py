import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(name: str, 
                log_file: Optional[str] = None,
                level: int = logging.INFO,
                format_string: Optional[str] = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TrainingLogger:
    """Logger for training metrics and progress"""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f'training_{timestamp}.log'
        
        self.logger = setup_logger(
            'TrainingLogger',
            str(self.log_file),
            level=logging.INFO
        )
    
    def log_epoch(self, epoch: int, metrics: dict):
        """Log epoch metrics"""
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Epoch {epoch}: {metrics_str}')
    
    def log_best_model(self, epoch: int, metric: str, value: float):
        """Log best model information"""
        self.logger.info(f'Best model at epoch {epoch}: {metric}={value:.4f}')
    
    def log_config(self, config: dict):
        """Log configuration"""
        self.logger.info('='*50)
        self.logger.info('Training Configuration:')
        for key, value in config.items():
            self.logger.info(f'{key}: {value}')
        self.logger.info('='*50)
    
    def log_data_info(self, train_size: int, val_size: int, test_size: int):
        """Log dataset information"""
        self.logger.info('Dataset Information:')
        self.logger.info(f'Training samples: {train_size}')
        self.logger.info(f'Validation samples: {val_size}')
        self.logger.info(f'Test samples: {test_size}')
    
    def log_model_summary(self, total_params: int, trainable_params: int):
        """Log model summary"""
        self.logger.info('Model Summary:')
        self.logger.info(f'Total parameters: {total_params:,}')
        self.logger.info(f'Trainable parameters: {trainable_params:,}')
        self.logger.info(f'Non-trainable parameters: {total_params - trainable_params:,}')


class PredictionLogger:
    """Logger for prediction results"""
    
    def __init__(self, log_dir: str = 'logs/predictions'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f'predictions_{timestamp}.log'
        
        self.logger = setup_logger(
            'PredictionLogger',
            str(self.log_file),
            level=logging.INFO
        )
    
    def log_prediction(self, image_path: str, prediction: dict):
        """Log single prediction"""
        self.logger.info(f'Image: {image_path}')
        self.logger.info(f'Prediction: {prediction["disease"]}')
        self.logger.info(f'Confidence: {prediction["confidence"]:.4f}')
        
        if 'top_predictions' in prediction:
            self.logger.info('Top 3 predictions:')
            for i, (disease, conf) in enumerate(prediction['top_predictions'][:3], 1):
                self.logger.info(f'  {i}. {disease}: {conf:.4f}')
        
        self.logger.info('-'*50)
    
    def log_batch_predictions(self, predictions: list):
        """Log batch predictions"""
        self.logger.info(f'Batch prediction: {len(predictions)} images')
        
        correct = sum(1 for p in predictions if p.get('is_correct', False))
        accuracy = correct / len(predictions) if predictions else 0
        
        self.logger.info(f'Batch accuracy: {accuracy:.4f}')
        
        avg_confidence = sum(p['confidence'] for p in predictions) / len(predictions)
        self.logger.info(f'Average confidence: {avg_confidence:.4f}')


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger"""
    return logging.getLogger(name)


# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)