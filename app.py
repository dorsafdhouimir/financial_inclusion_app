import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.pkl')

st.title("Prédiction d'Inclusion Financière en Afrique")

with st.form("formulaire_donnees"):
    st.header("Entrez les informations personnelles")

    country = st.selectbox("Pays", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    year = st.number_input("Année", min_value=2010, max_value=2025, value=2023)
    location_type = st.selectbox("Type de localisation", ["Urban", "Rural"])
    cellphone_access = st.selectbox("Accès au téléphone portable", ["Yes", "No"])
    household_size = st.number_input("Taille du ménage", min_value=1, max_value=20, value=3)
    age_of_respondent = st.number_input("Âge du répondant", min_value=18, max_value=100, value=30)
    gender_of_respondent = st.selectbox("Genre du répondant", ["Male", "Female"])
    relationship_with_head = st.selectbox("Relation avec le chef de ménage", ["Head of Household", "Spouse", "Child", "Other"])
    marital_status = st.selectbox("État civil", ["Married", "Single", "Divorced", "Widowed", "Other"])
    education_level = st.selectbox("Niveau d'éducation", ["No education", "Primary", "Secondary", "Tertiary"])
    job_type = st.selectbox("Type d'emploi", ["Unemployed", "Self employed", "Government", "Private", "Other"])

    submitted = st.form_submit_button("Prédire")

if submitted:

    input_dict = {
        'country': [country],
        'year': [year],
        'location_type': [location_type],
        'cellphone_access': [cellphone_access],
        'household_size': [household_size],
        'age_of_respondent': [age_of_respondent],
        'gender_of_respondent': [gender_of_respondent],
        'relationship_with_head': [relationship_with_head],
        'marital_status': [marital_status],
        'education_level': [education_level],
        'job_type': [job_type]
    }
    input_df = pd.DataFrame(input_dict)

    input_encoded = pd.get_dummies(input_df, drop_first=True)

    model_features = model.feature_names_in_ 
    for col in model_features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    input_encoded = input_encoded[model_features]

    prediction = model.predict(input_encoded)

    if prediction[0] == 1:
        st.success("Le modèle prédit que cette personne a un compte bancaire.")
    else:
        st.warning("Le modèle prédit que cette personne n'a pas de compte bancaire.")
