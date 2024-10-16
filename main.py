# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
data = pd.read_csv(url, names=columns)

# Data Preprocessing
data = data.replace('?', pd.NA).dropna()  # Handle missing values
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)  # Binary classification (1: heart disease, 0: no heart disease)

X = data.drop(columns=['target'])
y = data['target']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM model
model = svm.SVC(kernel='linear')
model.fit(X_train, y_train)

# Streamlit App

# Title
st.title("Heart Disease Prediction App")

# User Input
st.sidebar.header('Input Features')
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (in mm Hg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Cholesterol (in mg/dl)", min_value=100, max_value=400, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
restecg = st.sidebar.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.sidebar.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (0-2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thal (3 = Normal, 6 = Fixed Defect, 7 = Reversible Defect)", [3, 6, 7])

# Prepare input data for prediction
input_data = pd.DataFrame({
    'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
    'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
    'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope], 'ca': [ca], 'thal': [thal]
})

# Normalize the input data
input_data_scaled = scaler.transform(input_data)

# Make Prediction
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.write("Prediction: You are likely to have Heart Disease Please Consult a Docto.")
    else:
        st.write("Prediction: Congratulations ! You are unlikely to have Heart Disease. Be safe Be healthy")

# Model Accuracy
st.subheader("Model Accuracy")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Accuracy of the SVM model: {accuracy * 100:.2f}%")
