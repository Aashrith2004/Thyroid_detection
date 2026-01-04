# Thyroid Disorder Detection - Unified Application

This project combines Deep Learning (ResNet50 with Grad-CAM) and Machine Learning (Random Forest) models for comprehensive thyroid disorder detection.

## Features

1. **Deep Learning Analysis**: Image-based detection using ResNet50 with Grad-CAM visualization
2. **Machine Learning Analysis**: Blood test parameter-based prediction using Random Forest
3. **Combined Analysis**: Fusion of both models for comprehensive assessment

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate ML Model Files

The ML model pickle files have already been generated. If you need to regenerate them:

```bash
python generate_ml_model.py
```

This will create:
- `ml_model.pkl` - Trained Random Forest model
- `ml_scaler.pkl` - StandardScaler for feature normalization

### 3. Required Files

Make sure you have these files in the project directory:
- `new_thyroid_resnet50_best.h5` - Deep Learning model
- `ml_model.pkl` - Machine Learning model
- `ml_scaler.pkl` - Scaler for ML model
- `thyroid_dataset.csv` - Dataset (for reference)

## Running the Application

### Unified App (Recommended)

Run the combined application that includes both DL and ML:

```bash
streamlit run unified_app.py
```

### Individual Apps

You can also run the individual components:

**Deep Learning only (Streamlit version):**
```bash
streamlit run dl_streamlit.py
```

**Machine Learning only:**
```bash
streamlit run app.py
```

## Application Structure

### Unified App Pages

1. **Deep Learning (Image)**: 
   - Upload thyroid ultrasound images
   - Get predictions with Grad-CAM visualization
   - Shows original and heatmap overlay images

2. **Machine Learning (Blood Test)**:
   - Input blood test parameters (Age, Sex, TSH, T3, T4, T4U, FTI)
   - Get predictions with probability scores
   - View reference ranges for parameters

3. **Combined Analysis**:
   - Use both image and blood test data
   - Get fused predictions from both models
   - Comprehensive assessment with combined confidence scores

## Model Details

### Deep Learning Model
- **Architecture**: ResNet50
- **Input**: 224x224 RGB images
- **Output**: Benign/Malignant classification
- **Visualization**: Grad-CAM heatmaps

### Machine Learning Model
- **Algorithm**: Random Forest Classifier
- **Features**: Age, Sex, TSH, T3, T4, T4U, FTI
- **Output**: hyperthyroid/hypothyroid/negative classification
- **Accuracy**: 100% on test set

### Fusion Strategy
- Weighted combination of DL (60%) and ML (40%) confidence scores
- Provides comprehensive assessment combining image and clinical data

## Notes

- This tool is for research purposes only
- Always consult healthcare professionals for medical diagnosis
- The models are trained on specific datasets and may not generalize to all cases
