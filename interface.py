import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('Model.pkl')  
le = joblib.load('LabelEncoder.pkl')
df = pd.read_csv('data\digital_literacy_dataset.csv')

slider_columns = ['Age', 'Gender', 'Education_Level', 'Employment_Status', 'Household_Income', 'Location_Type', 
                  'Basic_Computer_Knowledge_Score', 'Internet_Usage_Score', 'Mobile_Literacy_Score', 
                  'Post_Training_Basic_Computer_Knowledge_Score', 'Post_Training_Internet_Usage_Score', 
                  'Post_Training_Mobile_Literacy_Score', 'Modules_Completed', 'Average_Time_Per_Module', 
                  'Quiz_Performance', 'Session_Count', 'Engagement_Level', 'Adaptability_Score', 
                  'Feedback_Rating', 'Skill_Application', 'Employment_Impact']

st.title('Digital Literacy Score Prediction')

input_data = {}
for column in slider_columns:
    if df[column].dtype == 'object':  
        options = df[column].unique()  
        selected_option = st.selectbox(column, options)
        try:
            input_data[column] = le.transform([selected_option])[0]  
        except ValueError:
            input_data[column] = -1  
    else:
        min_val = int(df[column].min())  
        max_val = int(df[column].max()) 
        step_val = 1
        input_data[column] = st.slider(column, min_val, max_val, int(df[column].mean()), step=step_val)

input_df = pd.DataFrame([input_data])
categorical_columns = ['Gender', 'Education_Level', 'Employment_Status', 'Household_Income', 'Location_Type', 
                       'Engagement_Level', 'Employment_Impact']

for col in categorical_columns:
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        input_df[col] = -1

input_df = input_df[df.columns.drop(['Overall_Literacy_Score', 'User_ID'])]

if st.button('Predict'):
    prediction = model.predict(input_df)
    st.write(f'Predicted Digital Literacy Score: {round(prediction[0], 2)}')
