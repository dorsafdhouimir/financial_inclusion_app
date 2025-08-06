import streamlit as st
import pandas as pd
import joblib

model = joblib.load('model_financial_inclusion.pkl')

st.title("Prédiction d'accès à un compte bancaire")

age = st.number_input("Âge du répondant", min_value=18, max_value=100, value=30)
household_size = st.number_input("Taille du foyer", min_value=1, max_value=20, value=3)
cellphone_access = st.selectbox("Accès à un téléphone portable", [0, 1])

gender = st.selectbox("Genre", ['Male', 'Female'])


input_dict = {
    'age_of_respondent': age,
    'household_size': household_size,
    'cellphone_access': cellphone_access,
    'gender_Male': 1 if gender == 'Male' else 0,
    
}

input_df = pd.DataFrame([input_dict])

for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0


input_df = input_df[model.feature_names_in_]

if st.button("Prédire"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"La personne est susceptible d'avoir un compte bancaire.\nProbabilité : {proba:.2f}")
    else:
        st.warning(f"La personne est moins susceptible d'avoir un compte bancaire.\nProbabilité : {proba:.2f}")
