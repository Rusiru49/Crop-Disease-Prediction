# Crop Disease Prediction System

A deep learning-based system for detecting and classifying plant diseases using Convolutional Neural Networks (CNN). The system includes a trained model and a FastAPI backend for real-time disease prediction.

## Overview

This project uses TensorFlow/Keras to train a CNN model on the PlantVillage dataset for identifying plant diseases. The trained model is then deployed using FastAPI to provide a REST API endpoint for disease prediction.

## Features

- **Image Classification**: Predicts plant health status from leaf images
- **Three Classes**: Early Blight, Late Blight, and Healthy
- **REST API**: Easy-to-use FastAPI endpoint for predictions
- **High Accuracy**: Multi-layer CNN architecture for robust predictions
- **Data Augmentation**: Enhanced training with image transformations

## Project Structure

```
.
├── training.py              # Model training script
├── main.py                  # FastAPI server
├── PlantVillage/           # Dataset directory
└── saved_models/           # Trained model storage
    └── 1.keras             # Trained model file
```

## Requirements

```
tensorflow>=2.10.0
fastapi>=0.95.0
uvicorn>=0.21.0
pillow>=9.0.0
numpy>=1.23.0
python-multipart>=0.0.6
matplotlib>=3.5.0
```

## Installation

1. Clone the repository and navigate to the project directory

2. Install required packages:
```bash
pip install tensorflow fastapi uvicorn pillow numpy python-multipart matplotlib
```

3. Download the PlantVillage dataset and place it in the `PlantVillage/` directory

## Model Training

The training script (`training.py`) implements:

- **Dataset Loading**: Automatic image loading from directory structure
- **Data Splitting**: 80% training, 10% validation, 10% testing
- **Preprocessing**: Image resizing (256x256) and normalization
- **Data Augmentation**: Random flips and rotations
- **CNN Architecture**: 6 convolutional layers with max pooling
- **Training**: 50 epochs with Adam optimizer

### Train the Model

```python
python training.py
```

The script will:
- Load and preprocess the PlantVillage dataset
- Split data into train/validation/test sets
- Train the CNN model for 50 epochs
- Save the trained model in `saved_models/` directory
- Display training accuracy and validation metrics

### Model Architecture

- Input: 256x256x3 RGB images
- 6 Conv2D layers (32-64 filters) with ReLU activation
- MaxPooling2D after each convolution
- Flatten layer
- Dense layer (64 units)
- Output layer with softmax activation (3 classes)

## API Server

The FastAPI server (`main.py`) provides endpoints for disease prediction.

### Start the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### API Endpoints

#### Health Check
```
GET /ping
```
Response: `"Hello, I am alive!"`

#### Predict Disease
```
POST /predict
```

**Parameters:**
- `file`: Image file (JPEG, PNG)

**Response:**
```json
{
  "class": "Early Blight",
  "confidence": 0.95,
  "predictions": {
    "Early Blight": 0.95,
    "Late Blight": 0.03,
    "Healthy": 0.02
  }
}
```

### Example Usage

**Using cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@plant_leaf.jpg"
```

**Using Python:**
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("plant_leaf.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Classes

The model predicts one of three classes:

1. **Early Blight**: Fungal disease causing dark spots on leaves
2. **Late Blight**: Serious disease causing rapid plant death
3. **Healthy**: No disease detected

## Configuration

Key parameters in `training.py`:

- `BATCH_SIZE = 32`: Number of images per batch
- `IMAGE_SIZE = 256`: Input image dimensions
- `EPOCHS = 50`: Training epochs
- `CHANNELS = 3`: RGB color channels

## CORS Configuration

The API has CORS enabled for all origins. For production, modify the `allow_origins` parameter in `main.py`:

```python
allow_origins=["https://yourdomain.com"]
```

## Model Versioning

The training script automatically versions saved models by incrementing the version number in the `saved_models/` directory.

## Performance

- The model is evaluated on the test set after training
- Predictions include confidence scores for all classes
- Typical inference time: <100ms per image

## Troubleshooting

**Model not found error:**
- Ensure the model file exists at `../saved_models/1.keras`
- Check that the model was successfully trained and saved

**CORS errors:**
- Verify CORS middleware configuration
- Check that the frontend origin is allowed

**Low prediction accuracy:**
- Ensure images are clear and well-lit
- Use images similar to the training dataset
- Consider retraining with more data

## Future Enhancements

- Add support for more plant diseases
- Implement batch prediction
- Add model performance monitoring
- Create a web frontend interface
- Add authentication and rate limiting

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- PlantVillage dataset for training data
- TensorFlow/Keras for deep learning framework
- FastAPI for API development
