import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, VGG16, VGG19, 
    InceptionV3, MobileNetV2, EfficientNetB0, 
    DenseNet121, Xception
)
from tensorflow.keras import layers, Model, Sequential
import torch
import torchvision.models as models


class TransferLearningModel:
    """Wrapper for transfer learning models"""
    
    def __init__(self, architecture: str, num_classes: int, 
                 input_shape: tuple = (224, 224, 3),
                 weights: str = 'imagenet',
                 freeze_backbone: bool = True,
                 dropout_rate: float = 0.3):
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.weights = weights
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
        self.model = None
        
    def build_model(self) -> Model:
        """Build transfer learning model"""
        base_model = self._get_base_model()
        
        # Freeze base model if specified
        if self.freeze_backbone:
            base_model.trainable = False
        
        # Build complete model
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False if self.freeze_backbone else True)
        
        # Add custom classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def _get_base_model(self):
        """Get pretrained base model"""
        model_dict = {
            'resnet50': ResNet50,
            'resnet101': ResNet101,
            'vgg16': VGG16,
            'vgg19': VGG19,
            'inceptionv3': InceptionV3,
            'mobilenetv2': MobileNetV2,
            'efficientnetb0': EfficientNetB0,
            'densenet121': DenseNet121,
            'xception': Xception,
        }
        
        if self.architecture not in model_dict:
            raise ValueError(f"Unknown architecture: {self.architecture}")
        
        model_class = model_dict[self.architecture]
        
        # Special handling for InceptionV3 and Xception (different input size)
        if self.architecture in ['inceptionv3', 'xception']:
            input_shape = (299, 299, 3)
        else:
            input_shape = self.input_shape
        
        base_model = model_class(
            include_top=False,
            weights=self.weights,
            input_shape=input_shape
        )
        
        return base_model
    
    def unfreeze_top_layers(self, num_layers: int = 10):
        """Unfreeze top N layers for fine-tuning"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() first.")
        
        base_model = self.model.layers[1]  # Get base model layer
        
        # Freeze all layers first
        base_model.trainable = True
        
        # Freeze bottom layers
        for layer in base_model.layers[:-num_layers]:
            layer.trainable = False
        
        print(f"Unfroze top {num_layers} layers for fine-tuning")
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            self.build_model()
        return self.model.summary()


class PyTorchTransferModel(torch.nn.Module):
    """PyTorch transfer learning model"""
    
    def __init__(self, architecture: str, num_classes: int, 
                 pretrained: bool = True, freeze_backbone: bool = True):
        super(PyTorchTransferModel, self).__init__()
        
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        
        # Get base model
        if architecture == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = torch.nn.Identity()
        elif architecture == 'resnet101':
            self.base_model = models.resnet101(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = torch.nn.Identity()
        elif architecture == 'vgg16':
            self.base_model = models.vgg16(pretrained=pretrained)
            num_features = self.base_model.classifier[0].in_features
            self.base_model.classifier = torch.nn.Identity()
        elif architecture == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = torch.nn.Identity()
        elif architecture == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = torch.nn.Identity()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Freeze base model if specified
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Classification head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        features = self.base_model(x)
        return self.classifier(features)


def create_transfer_model(architecture: str, 
                          num_classes: int,
                          input_shape: tuple = (224, 224, 3),
                          freeze_backbone: bool = True,
                          framework: str = 'tensorflow') -> Model:
    """Factory function to create transfer learning model"""
    
    if framework == 'tensorflow':
        model_builder = TransferLearningModel(
            architecture=architecture,
            num_classes=num_classes,
            input_shape=input_shape,
            freeze_backbone=freeze_backbone
        )
        return model_builder.build_model()
    
    elif framework == 'pytorch':
        return PyTorchTransferModel(
            architecture=architecture,
            num_classes=num_classes,
            freeze_backbone=freeze_backbone
        )
    
    else:
        raise ValueError(f"Unknown framework: {framework}")


def get_recommended_lr(architecture: str, freeze_backbone: bool) -> float:
    """Get recommended learning rate based on architecture and training mode"""
    if freeze_backbone:
        # Higher learning rate for training only the top layers
        return 0.001
    else:
        # Lower learning rate for fine-tuning
        base_lrs = {
            'resnet50': 0.0001,
            'resnet101': 0.0001,
            'vgg16': 0.00001,
            'vgg19': 0.00001,
            'inceptionv3': 0.0001,
            'mobilenetv2': 0.001,
            'efficientnetb0': 0.001,
            'densenet121': 0.0001,
            'xception': 0.0001,
        }
        return base_lrs.get(architecture.lower(), 0.0001)