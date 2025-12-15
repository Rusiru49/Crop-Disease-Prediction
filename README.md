# Crop-Disease-Prediction

crop-disease-prediction/
│
├── data/
│   ├── raw/
│   │   └── plantvillage/          # Raw PlantVillage dataset images
│   ├── processed/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── metadata/
│       ├── class_labels.json
│       └── dataset_stats.json
│
├── models/
│   ├── saved_models/              # Trained model checkpoints
│   ├── model_architecture.py      # Model definitions
│   └── pretrained/                # Pre-trained weights
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── preprocessing.py       # Image preprocessing
│   │   └── augmentation.py        # Data augmentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_model.py           # Custom CNN models
│   │   ├── transfer_learning.py   # Transfer learning models
│   │   └── ensemble.py            # Ensemble methods
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py               # Training pipeline
│   │   ├── validate.py            # Validation logic
│   │   └── callbacks.py           # Custom callbacks
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── visualization.py       # Result visualization
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py           # Prediction pipeline
│   │   └── explainability.py      # Model explainability (Grad-CAM, etc.)
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging utilities
│       └── helpers.py              # Helper functions
│
├── streamlit_app/
│   ├── __init__.py
│   ├── app.py                      # Main Streamlit application
│   ├── pages/
│   │   ├── 1_🔍_Prediction.py     # Disease prediction page
│   │   ├── 2_📊_Model_Info.py     # Model information page
│   │   ├── 3_📈_Analytics.py      # Analytics dashboard
│   │   └── 4_ℹ️_About.py          # About page
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── image_uploader.py      # Image upload component
│   │   ├── result_display.py      # Result display component
│   │   ├── charts.py               # Chart components
│   │   └── sidebar.py              # Sidebar component
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── session_state.py       # Session state management
│   │   └── styling.py              # Custom styling
│   │
│   └── assets/
│       ├── images/
│       │   ├── logo.png
│       │   └── banner.png
│       ├── styles/
│       │   └── custom.css
│       └── data/
│           └── disease_info.json   # Disease information & remedies
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_inference.py
│
├── config/
│   ├── config.yaml                 # Main configuration
│   └── model_config.yaml           # Model hyperparameters
│
├── scripts/
│   ├── download_dataset.py         # Dataset download script
│   ├── prepare_data.py             # Data preparation
│   └── train_model.py              # Training script
│
├── docs/
│   ├── README.md
│   ├── data_description.md
│   ├── model_architecture.md
│   └── deployment.md
│
├── .streamlit/
│   └── config.toml                 # Streamlit configuration
│
├── .gitignore
├── requirements.txt
├── setup.py
├── README.md
└── LICENSE
