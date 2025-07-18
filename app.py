import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# Load the trained model
model = joblib.load("best_model.pkl")

# Set up the page
st.set_page_config(page_title="Employee Salary Classification", page_icon="📈", layout="centered")

st.title("Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

# Input fields - Note: we need to match exactly what was used in training
age = st.sidebar.slider("Age", 17, 75, 30)
workclass = st.sidebar.selectbox("Work Class", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
    "Local-gov", "State-gov", "Others"
])
educational_num = st.sidebar.slider("Educational Number", 5, 16, 10)
marital_status = st.sidebar.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", 
    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces", "Others"
])
relationship = st.sidebar.selectbox("Relationship", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])
race = st.sidebar.selectbox("Race", [
    "White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
    "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"
])

# Let's add the missing features that might be in the original dataset
# These are likely default values or derived features
fnlwgt = 189778  # Common default value from adult dataset
capital_gain = 0  # Most people have 0 capital gain
capital_loss = 0  # Most people have 0 capital loss

# Create the input dataframe with ALL features used in training
# Based on the original adult dataset, we need 13 features total
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],  # Adding missing feature
    'educational-num': [educational_num],
    'marital-status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'gender': [gender],
    'capital-gain': [capital_gain],  # Adding missing feature
    'capital-loss': [capital_loss],  # Adding missing feature
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

# Display input data preview
st.write("### Input Data")
st.write(input_df)

# Prediction button
if st.button("Predict Salary Class"):
    try:
        # Create a copy for processing
        processed_input = input_df.copy()
        
        # Apply the same preprocessing as in training
        le_workclass = LabelEncoder()
        le_marital = LabelEncoder()
        le_occupation = LabelEncoder()
        le_relationship = LabelEncoder()
        le_race = LabelEncoder()
        le_gender = LabelEncoder()
        le_native = LabelEncoder()
        
        # Define the possible values (these should match training data)
        workclass_values = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", 
                           "Local-gov", "State-gov", "Others"]
        marital_values = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", 
                         "Widowed", "Married-spouse-absent", "Married-AF-spouse"]
        occupation_values = ["Tech-support", "Craft-repair", "Other-service", "Sales",
                            "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
                            "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
                            "Protective-serv", "Armed-Forces", "Others"]
        relationship_values = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
        race_values = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
        gender_values = ["Male", "Female"]
        native_values = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                        "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
                        "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
                        "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
                        "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
                        "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador",
                        "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
        
        # Fit and transform categorical columns
        le_workclass.fit(workclass_values)
        le_marital.fit(marital_values)
        le_occupation.fit(occupation_values)
        le_relationship.fit(relationship_values)
        le_race.fit(race_values)
        le_gender.fit(gender_values)
        le_native.fit(native_values)
        
        processed_input['workclass'] = le_workclass.transform(processed_input['workclass'])
        processed_input['marital-status'] = le_marital.transform(processed_input['marital-status'])
        processed_input['occupation'] = le_occupation.transform(processed_input['occupation'])
        processed_input['relationship'] = le_relationship.transform(processed_input['relationship'])
        processed_input['race'] = le_race.transform(processed_input['race'])
        processed_input['gender'] = le_gender.transform(processed_input['gender'])
        processed_input['native-country'] = le_native.transform(processed_input['native-country'])
        
        # Apply standard scaling to match training pipeline
        scaler = StandardScaler()
        processed_input_scaled = scaler.fit_transform(processed_input)
        
        # Make prediction
        prediction = model.predict(processed_input_scaled)
        
        # Display results
        result = ">50K" if prediction[0] == 1 else "≤50K"
        st.success(f"Prediction: {result}")
        
        # Try to get prediction probabilities
        try:
            prediction_proba = model.predict_proba(processed_input_scaled)
            confidence = max(prediction_proba[0]) * 100
            st.info(f"Confidence: {confidence:.2f}%")
        except:
            pass
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Debug info:")
        st.write(f"Input shape: {processed_input.shape}")
        st.write(f"Input columns: {list(processed_input.columns)}")
        st.write(f"Model type: {type(model)}")

# Batch prediction
st.markdown("---")
st.markdown("### Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        st.write("Uploaded data preview:")
        st.write(batch_data.head())
        
        # Check if required columns exist
        required_columns = ['age', 'workclass', 'educational-num', 'marital-status', 'occupation', 
                          'relationship', 'race', 'gender', 'hours-per-week', 'native-country']
        
        missing_columns = [col for col in required_columns if col not in batch_data.columns]
        
        if missing_columns:
            st.error(f"Missing columns in uploaded file: {missing_columns}")
            st.info("Please ensure your CSV file has all required columns.")
        else:
            # Add default values for missing features
            batch_processed = batch_data[required_columns].copy()
            batch_processed['fnlwgt'] = 189778  # Default value
            batch_processed['capital-gain'] = 0  # Default value
            batch_processed['capital-loss'] = 0  # Default value
            
            # Reorder columns to match training data
            batch_processed = batch_processed[['age', 'workclass', 'fnlwgt', 'educational-num', 
                                             'marital-status', 'occupation', 'relationship', 'race', 
                                             'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 
                                             'native-country']]
            
            # Apply same preprocessing
            le_workclass = LabelEncoder()
            le_marital = LabelEncoder()
            le_occupation = LabelEncoder()
            le_relationship = LabelEncoder()
            le_race = LabelEncoder()
            le_gender = LabelEncoder()
            le_native = LabelEncoder()
            
            # Fit encoders
            le_workclass.fit(workclass_values)
            le_marital.fit(marital_values)
            le_occupation.fit(occupation_values)
            le_relationship.fit(relationship_values)
            le_race.fit(race_values)
            le_gender.fit(gender_values)
            le_native.fit(native_values)
            
            # Transform
            batch_processed['workclass'] = le_workclass.transform(batch_processed['workclass'])
            batch_processed['marital-status'] = le_marital.transform(batch_processed['marital-status'])
            batch_processed['occupation'] = le_occupation.transform(batch_processed['occupation'])
            batch_processed['relationship'] = le_relationship.transform(batch_processed['relationship'])
            batch_processed['race'] = le_race.transform(batch_processed['race'])
            batch_processed['gender'] = le_gender.transform(batch_processed['gender'])
            batch_processed['native-country'] = le_native.transform(batch_processed['native-country'])
            
            # Apply scaling
            scaler = StandardScaler()
            batch_processed_scaled = scaler.fit_transform(batch_processed)
            
            # Make predictions
            batch_preds = model.predict(batch_processed_scaled)
            batch_data['PredictedClass'] = [">50K" if pred == 1 else "≤50K" for pred in batch_preds]
            
            # Display results
            st.write("### Predictions:")
            st.write(batch_data.head())
            
            # Download button
            csv = batch_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name='predicted_classes.csv',
                mime='text/csv'
            )
            
    except Exception as e:
        st.error(f"Error processing batch file: {str(e)}")
        st.info("Please check your CSV file format and ensure it matches the required structure.")

# Information section
st.markdown("---")
st.markdown("### About")
st.markdown("""
This app predicts whether an employee earns more than $50K annually based on demographic and work-related features.
The model was trained on the Adult Census Income dataset.

**Features used:** age, workclass, fnlwgt, educational-num, marital-status, occupation, relationship, race, gender, capital-gain, capital-loss, hours-per-week, native-country

**Note:** For simplicity, fnlwgt, capital-gain, and capital-loss are set to default values (189778, 0, 0 respectively) in the single prediction interface.
""")
