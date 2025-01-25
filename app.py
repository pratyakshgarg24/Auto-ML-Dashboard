from operator import index
import streamlit as st
import plotly.express as px
import pandas_profiling
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import os 

if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Auto ML Dashboard")
    choice = st.radio("Navigation", ["Upload","Profiling","Modelling", "Download"])
    st.info("This is an interactive Streamlit application designed to simplify the process of data exploration and ML modeling. Users can upload datasets, generate detailed exploratory data analysis (EDA) reports using profiling tools, and build machine learning models automatically with PyCaret. The app also allows downloading the best-performing model for future use.")

if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "Profiling": 
    st.title("Exploratory Data Analysis")
    if df is not None and not df.empty:
        profile_report = ProfileReport(df)
        st_profile_report(profile_report)
    else:
        st.warning("Please upload a valid dataset first.")


if choice == "Modelling": 
    st.title("Model Building")
    task_type = st.radio("Choose the Task Type", ["Regression", "Classification"])
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'): 
        if task_type == "Regression":
            from pycaret.regression import setup, compare_models, pull, save_model
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.write("Regression Model Setup Summary:")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.write("Regression Model Comparison:")
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')

        elif task_type == "Classification":
            from pycaret.classification import setup, compare_models, pull, save_model
            setup(df, target=chosen_target, verbose=False)
            setup_df = pull()
            st.write("Classification Model Setup Summary:")
            st.dataframe(setup_df)
            best_model = compare_models()
            compare_df = pull()
            st.write("Classification Model Comparison:")
            st.dataframe(compare_df)
            save_model(best_model, 'best_model')


if choice == "Download": 
    with open('best_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")