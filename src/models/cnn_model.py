import tensorflow as tf
from tensorflow.keras import layers, models, Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCNN(Model):
    """Custom CNN architecture for plant disease classification"""
    
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.3):
        super(CustomCNN, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional blocks
        self.conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        self.conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')
        self.bn4 = layers.BatchNormalization()
        self.pool4 = layers.MaxPooling2D((2, 2))
        
        self.conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')
        self.bn5 = layers.BatchNormalization()
        self.pool5 = layers.MaxPooling2D((2, 2))
        
        # Dense layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(256, activation='relu')
        self.dropout2 = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x, training=training)
        x = self.pool5(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        return self.output_layer(x)


class LightweightCNN(Model):
    """Lightweight CNN for mobile deployment"""
    
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.2):
        super(LightweightCNN, self).__init__()
        
        self.conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.pool1 = layers.MaxPooling2D((2, 2))
        
        self.conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.pool2 = layers.MaxPooling2D((2, 2))
        
        self.conv3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool3 = layers.MaxPooling2D((2, 2))
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.global_pool(x)
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        
        return self.output_layer(x)


class PyTorchCNN(nn.Module):
    """PyTorch implementation of custom CNN"""
    
    def __init__(self, num_classes: int = 38, dropout_rate: float = 0.3):
        super(PyTorchCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ResidualBlock(layers.Layer):
    """Residual block for custom ResNet-like architecture"""
    
    def __init__(self, filters, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = layers.Conv2D(filters, (3, 3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        
        self.conv2 = layers.Conv2D(filters, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()
        
        self.shortcut = None
        if stride != 1:
            self.shortcut = nn.Sequential([
                layers.Conv2D(filters, (1, 1), strides=stride),
                layers.BatchNormalization()
            ])
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.shortcut:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        
        x = layers.add([x, shortcut])
        return tf.nn.relu(x)


class CustomResNet(Model):
    """Custom ResNet-style architecture"""
    
    def __init__(self, num_classes: int = 38):
        super(CustomResNet, self).__init__()
        
        self.conv1 = layers.Conv2D(64, (7, 7), strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D((3, 3), strides=2, padding='same')
        
        self.block1 = ResidualBlock(64)
        self.block2 = ResidualBlock(128, stride=2)
        self.block3 = ResidualBlock(256, stride=2)
        self.block4 = ResidualBlock(512, stride=2)
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        
        x = self.global_pool(x)
        return self.output_layer(x)


def build_custom_cnn(num_classes: int = 38, 
                    input_shape: tuple = (224, 224, 3),
                    architecture: str = 'standard') -> Model:
    """Build custom CNN model"""
    
    if architecture == 'standard':
        model = CustomCNN(num_classes=num_classes)
    elif architecture == 'lightweight':
        model = LightweightCNN(num_classes=num_classes)
    elif architecture == 'resnet':
        model = CustomResNet(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Build model
    model.build((None,) + input_shape)
    
    return model