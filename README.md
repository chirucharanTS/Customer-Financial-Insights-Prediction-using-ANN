### Project Name: **Customer Financial Insights Prediction using ANN**

---

### **README**

#### **Problem Statement**
The goal of this project is to predict financial insights based on customer demographic, behavioral, and financial data. By building a regression model, we aim to predict a target financial variable, allowing businesses to understand customer profiles and make data-driven decisions to enhance services.

---

### **Overview**
This project leverages a deep learning approach using an Artificial Neural Network (ANN) for regression tasks. The data comprises customer information, including demographic details, account activity, and financial data. A robust preprocessing pipeline ensures the data is ready for training. The ANN model predicts a target numeric variable, aiding in financial insight generation.

---

### **Workflow**

#### **1. Data Preparation**
- **Dataset**: The dataset contains customer demographics, financial behavior, and account activity.
- **Preprocessing Steps**:
  - Dropped irrelevant columns (`RowNumber`, `CustomerId`, `Surname`).
  - Encoded categorical variables:
    - Used `LabelEncoder` for the `Gender` column.
    - Applied `OneHotEncoder` for the `Geography` column.
  - Combined the processed columns with the original dataset.
  - Normalized numerical features using `StandardScaler`.

#### **2. Train-Test Split**
- Divided the data into training and testing sets in an 80-20 ratio.

#### **3. Model Development**
- Built an ANN regression model using TensorFlow/Keras.
- The architecture includes:
  - Input layer: Matches the number of input features.
  - Two hidden layers with 64 and 32 neurons, using ReLU activation.
  - Output layer with a single neuron for regression predictions.
- Optimized the model using the Adam optimizer and evaluated it using the Mean Absolute Error (MAE).

#### **4. Model Training**
- Utilized Early Stopping to prevent overfitting by monitoring validation loss.
- Integrated TensorBoard for logging training metrics and visualizing performance.

#### **5. Model Evaluation**
- Evaluated the model on the test set to compute test loss and MAE.
- Saved the trained model for future inference.

---

### **Setup Instructions**

#### **1. Prerequisites**
- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `tensorflow`
  - `pickle`
  - `datetime`

Install required libraries using:
```bash
pip install pandas numpy scikit-learn tensorflow
```

#### **2. Steps to Run**

1. **Preprocessing**:
   - Load and preprocess the dataset.
   - Save encoders and scaler as `.pkl` files for later use.

2. **Model Training**:
   - Train the ANN regression model using the training data.
   - Save the trained model as `regression_model.h5`.

3. **Inference**:
   - Load the saved model, encoders, and scaler.
   - Preprocess new input data to match training features.
   - Predict the financial target variable using the trained model.

---

### **Directory Structure**
```
Customer_Financial_Insights_Prediction/
│
├── data/
│   ├── Churn_Modelling.csv         # Input dataset
│
├── models/
│   ├── regression_model.h5         # Trained ANN model
│   ├── label_encoder_gender.pkl    # Gender encoder
│   ├── onehot_encoder_geo.pkl      # Geography encoder
│   ├── scaler.pkl                  # Feature scaler
│
├── logs/
│   ├── fit/                        # TensorBoard logs
│
├── scripts/
│   ├── preprocess.py               # Data preprocessing script
│   ├── train_model.py              # Model training script
│   ├── inference.py                # Inference script
│
└── README.md                       # Project description and instructions
```

---

### **How to Use**

#### **1. Train the Model**
Run the `train_model.py` script:
```bash
python scripts/train_model.py
```

#### **2. Make Predictions**
Use the `inference.py` script to make predictions:
```bash
python scripts/inference.py
```

---

### **Key Features**
- Preprocessing pipeline for handling categorical and numerical data.
- ANN model architecture optimized for regression tasks.
- Saved preprocessing objects (encoders, scaler) for consistent inference.
- TensorBoard integration for training monitoring.

---

### **Future Enhancements**
- Experiment with additional features or feature engineering for improved predictions.
- Optimize model hyperparameters using grid search or random search.
- Extend the pipeline for real-time predictions with a user-friendly interface.

--- 
