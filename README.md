# **Customer Financial Insights Prediction Using ANN**

This project leverages Artificial Neural Networks (ANN) to predict customer financial insights based on various features. It is designed for ease of deployment with a Streamlit application and includes tools for preprocessing, training, and prediction.

---

## **Directory Structure**

```
customer-financial-insights-prediction-using-ann/
│
├── app.py                       # Main Streamlit application script
├── churn_modelling.csv          # Dataset for training and testing
├── experiments.ipynb            # Exploratory Data Analysis (EDA) and model experiments
├── hyperparametertuninggsnn.ipynb # Notebook for hyperparameter tuning of ANN
├── label_encoder_gender.pk1     # Pretrained Label Encoder for 'Gender' feature
├── model.h5                     # Trained ANN model file in HDF5 format
├── onehot_encoder_geo.pk1       # Pretrained One-Hot Encoder for 'Geography' feature
├── prediction.ipynb             # Notebook to demonstrate predictions using the trained model
├── requirements.txt             # List of Python dependencies
├── salaryregression.ipynb       # Regression task notebook for additional experimentation
├── scaler.pk1                   # Pretrained scaler for input data normalization
```

---

## **Project Workflow**

### **1. Data Preparation**
   - The dataset `churn_modelling.csv` contains customer-related features such as demographics, geography, and churn status.
   - Preprocessing includes:
     - Encoding categorical features using `label_encoder_gender.pk1` and `onehot_encoder_geo.pk1`.
     - Scaling numerical features using `scaler.pk1`.

### **2. Model Development**
   - The ANN model is defined and trained using TensorFlow/Keras.
   - Notebooks like `experiments.ipynb` provide insight into exploratory analysis and preliminary testing.
   - `hyperparametertuninggsnn.ipynb` fine-tunes the ANN hyperparameters for improved accuracy.

### **3. Prediction**
   - The trained model (`model.h5`) predicts customer financial insights when given new input data.
   - Predictions are showcased in `prediction.ipynb`.

### **4. Deployment**
   - The Streamlit application (`app.py`) provides an interactive interface for end-users to input customer details and get predictions.

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone https://github.com/<your-username>/customer-financial-insights-prediction-using-ann.git
cd customer-financial-insights-prediction-using-ann
```

### **2. Create a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Run the Application**
```bash
streamlit run app.py
```

---

## **Features**
- **Data Preprocessing:**  
  - Automated handling of categorical and numerical data using pretrained encoders and scalers.
  
- **Model Training:**  
  - Customizable ANN model built with TensorFlow/Keras.
  - Hyperparameter tuning support using Grid Search or Random Search.

- **User-Friendly Deployment:**  
  - Streamlit-powered interactive web application for real-time predictions.

---

## **Dependencies**
The required Python libraries are listed in `requirements.txt`. Key dependencies include:
- TensorFlow
- Streamlit
- Scikit-learn
- Pandas
- NumPy

To ensure smooth execution, ensure your Python version is compatible with the required TensorFlow version. For TensorFlow 2.15.0, Python 3.8-3.11 is recommended.

---

## **Usage**

1. Launch the Streamlit application using `streamlit run app.py`.
2. Upload or input customer details via the interface.
3. Get predictions and insights on customer financial behavior.

---

## **Limitations**
- The model's performance depends on the quality of input data.
- Ensure the Python version aligns with TensorFlow compatibility to avoid installation errors.

---

## **Future Enhancements**
- Integration of advanced deep learning techniques like LSTM or GRU for temporal data.
- Enhanced deployment options using Docker or cloud platforms.
- More robust hyperparameter tuning pipelines with frameworks like Optuna.

---
