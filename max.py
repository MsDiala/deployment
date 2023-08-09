# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import joblib



# Load the data
train_data = pd.read_csv('Train-set.csv')
test_data = pd.read_csv('Test-set.csv')

# Drop rows with missing values (NaN) for simplicity
train_data.dropna(inplace=True)

# Separate features and target
X = train_data.drop(columns=['Target'])  # Corrected target column name
y = train_data['Target']

# Categorical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day', 'month', 'poutcome']

# Numeric columns
numeric_columns = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']

# Impute missing values in numeric columns
imputer = SimpleImputer(strategy='mean')
X[numeric_columns] = imputer.fit_transform(X[numeric_columns])

# One-hot encode categorical columns
encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X[categorical_columns])

# Combine encoded features with numeric features
X_encoded = np.concatenate((X_encoded, X[numeric_columns]), axis=1)

# Feature selection
selected_features = SelectKBest(f_classif, k=10).fit(X_encoded, y).get_support(indices=True)

# Select the important features
X_selected = X_encoded[:, selected_features]

# Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_selected, y)
# Save the trained model
joblib.dump(clf, 'bank_model.pkl')


# Streamlit App
st.title("Banking Term Deposit Prediction")
st.write("This app predicts if a client will subscribe to a term deposit.")

# Get user input
st.sidebar.header("User Input")

# Add input fields for user's age, job, marital status, education, default, balance, housing, loan, contact, day, month, duration, campaign, pdays, previous, poutcome
user_age = st.sidebar.number_input("Age of the client", min_value=18, max_value=100, value=30)
user_balance = st.sidebar.number_input("Average yearly balance in euros", value=0)
user_duration = st.sidebar.number_input("Last contact duration (seconds)", value=0)
user_campaign = st.sidebar.number_input("Number of contacts performed during this campaign", value=0)
user_pdays = st.sidebar.number_input("Number of days passed by after the client was last contacted from a previous campaign", value=0)
user_previous = st.sidebar.number_input("Number of contacts performed before this campaign and for this client", value=0)
user_job = st.sidebar.selectbox("Type of job", X['job'].unique())
user_marital = st.sidebar.selectbox("Marital status", X['marital'].unique())
user_education = st.sidebar.selectbox("Education", X['education'].unique())
user_default = st.sidebar.selectbox("Has credit in default?", X['default'].unique())
user_housing = st.sidebar.selectbox("Has housing loan?", X['housing'].unique())
user_loan = st.sidebar.selectbox("Has personal loan?", X['loan'].unique())
user_contact = st.sidebar.selectbox("Contact communication type", X['contact'].unique())
user_day = st.sidebar.selectbox("Last contact day of the week", X['day'].unique())
user_month = st.sidebar.selectbox("Last contact month of year", X['month'].unique())
user_poutcome = st.sidebar.selectbox("Outcome of the previous marketing campaign", X['poutcome'].unique())

# Define categorical options for user input
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital_options = ['divorced', 'married', 'single', 'unknown']
education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
default_options = ['no', 'yes', 'unknown']
housing_options = ['no', 'yes', 'unknown']
loan_options = ['no', 'yes', 'unknown']
contact_options = ['cellular', 'telephone']
day_options = ['mon', 'tue', 'wed', 'thu', 'fri']
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcome_options = ['failure', 'nonexistent', 'success']

# Preprocess user input and make a prediction
if st.sidebar.button("Predict"):
    # Collect user input
    user_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
    user_balance = st.sidebar.number_input("Balance", min_value=0, value=0)
    user_duration = st.sidebar.number_input("Duration (seconds)", min_value=0, value=0)
    user_campaign = st.sidebar.number_input("Number of Contacts during Campaign", min_value=0, value=0)
    user_pdays = st.sidebar.number_input("Days since Last Contact", min_value=0, value=0)
    user_previous = st.sidebar.number_input("Number of Previous Contacts", min_value=0, value=0)
    
    # Collect categorical user input
    user_job = st.sidebar.selectbox("Job", job_options)
    user_marital = st.sidebar.selectbox("Marital Status", marital_options)
    user_education = st.sidebar.selectbox("Education", education_options)
    user_default = st.sidebar.selectbox("Has Credit in Default?", default_options)
    user_housing = st.sidebar.selectbox("Has Housing Loan?", housing_options)
    user_loan = st.sidebar.selectbox("Has Personal Loan?", loan_options)
    user_contact = st.sidebar.selectbox("Contact Communication Type", contact_options)
    user_day = st.sidebar.selectbox("Last Contact Day of the Week", day_options)
    user_month = st.sidebar.selectbox("Last Contact Month", month_options)
    user_poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", poutcome_options)
    
    # Preprocess and encode user input (rest of the code remains the same)
    # ...
    # ... (code for collecting numerical and categorical user input)
    
    # Preprocess and encode user input
    user_input = np.array([user_age, user_balance, user_duration, user_campaign, user_pdays, user_previous]).reshape(1, -1)
    
    # Encode categorical features
    user_categorical_input = np.array([user_job, user_marital, user_education, user_default,
                                       user_housing, user_loan, user_contact, user_day,
                                       user_month, user_poutcome]).reshape(1, -1)
    user_input_encoded = encoder.transform(user_categorical_input)
    
    # Select important features
    user_input_selected = user_input_encoded[:, selected_features]
    
    # Make prediction
    prediction = model.predict(user_input_selected)
    
    # Display the prediction
    if prediction[0] == 1:
        st.sidebar.write("Prediction: Client will subscribe to a term deposit")
    else:
        st.sidebar.write("Prediction: Client will not subscribe to a term deposit")

