import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from preprocessing.encode import label_encoding, one_hot_encoding
from preprocessing.scaling import scale_all , scale_one_col
from sklearn.preprocessing import StandardScaler, MinMaxScaler


st.set_page_config(page_title="Advanced Data Preprocessing Tool", layout="wide")
st.title("Advanced Data Preprocessing Tool")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSelectbox, .stRadio {
        margin-bottom: 20px;
    }
    .stAlert {
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])


def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


if uploaded_file:
    df = load_data(uploaded_file)
    st.write("## Original Dataset")
    st.dataframe(df)

    with st.expander("Show Summary Statistics"):
        st.write("### Numerical Features")
        st.write(df.describe())
        st.write("### Categorical Features")
        st.write(df.select_dtypes(include=['object']).describe())

    st.write("## Missing Data Analysis")

    st.write("## Data Visualization")
    col1, col2 = st.columns(2)
    with col1:
        col_to_viz = st.selectbox("Select a column to visualize", df.columns, key="viz_col")
    with col2:
        plot_type = st.selectbox("Select plot type", ["Histogram", "Box Plot", "Bar Plot"], key="plot_type")

    if df[col_to_viz].dtype in ['int64', 'float64']:
        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_to_viz, nbins=20, title=f"Histogram of {col_to_viz}")
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            fig = px.box(df, y=col_to_viz, title=f"Box Plot of {col_to_viz}")
            st.plotly_chart(fig)
        elif plot_type == "Bar Plot":
            st.error("Bar Plot is not suitable for numeric columns. Please select a categorical column.")
    else:
        if plot_type == "Bar Plot":
            fig = px.bar(df[col_to_viz].value_counts(), title=f"Bar Plot of {col_to_viz}")
            st.plotly_chart(fig)
        elif plot_type == "Histogram":
            st.error("Histogram is not suitable for categorical columns. Please select a numeric column.")
        elif plot_type == "Box Plot":
            st.error("Box Plot is not suitable for categorical columns. Please select a numeric column.")

    missing_data = df.isnull().sum().reset_index()
    missing_data.columns = ["Column", "Missing Values"]
    missing_data["Missing Percentage"] = (missing_data["Missing Values"] / len(df)) * 100
    st.write("### Missing Values by Column")
    st.dataframe(missing_data)

    st.write("## Handle Missing Values")
    col_to_handle = st.selectbox("Select column to handle missing values", df.columns, key="missing_col")
    action = st.radio("Choose an action", ["Remove Rows", "Fill with Mean", "Fill with Mode", "Fill with Custom Value"], key="action")

    if action == "Fill with Custom Value":
        custom_value = st.text_input("Enter custom value", key="custom_value")

    if st.button("Apply Missing Values Handling"):
        if action == "Remove Rows":
            df = df.dropna(subset=[col_to_handle])
        elif action == "Fill with Mean":
            if df[col_to_handle].dtype in ['int64', 'float64']:
                df[col_to_handle].fillna(df[col_to_handle].mean(), inplace=True)
            else:
                st.error("Cannot fill non-numeric columns with mean.")
        elif action == "Fill with Mode":
            df[col_to_handle].fillna(df[col_to_handle].mode()[0], inplace=True)
        elif action == "Fill with Custom Value":
            df[col_to_handle].fillna(custom_value, inplace=True)
        st.success("Missing values handled!")
        st.dataframe(df)

    
    st.write("## Encoding")
    categorical_col = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_col) > 0:
        selected_column = st.selectbox("Choose a categorical column to encode:", categorical_col)
        method_encode = st.radio("Choose encoding method:", ["One-Hot Encoding", "Label Encoding"])
        
        if st.button("Apply Encoding"):
            df = one_hot_encoding(df, selected_column) if method_encode == "One-Hot Encoding" else label_encoding(df, selected_column)
            st.dataframe(df)

    st.write("## Feature Scaling")
    method_scaling = st.radio("Choose a scaling method", ["Standard Scaling", "Min-Max Scaling"])
    scaler = StandardScaler() if method_scaling == "Standard Scaling" else MinMaxScaler()
    option = st.radio("Choose type of scaling", ["Scale one column", "Scale all columns"])

    if option == "Scale one column":
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        selected_column = st.selectbox("Select column", num_cols)
        if st.button("Apply Scaling"):
            df = scale_one_col(df, selected_column, scaler)
            st.dataframe(df)
    elif option == "Scale all columns":
        if st.button("Apply Scaling"):
            df = scale_all(df, scaler)
            st.dataframe(df)


    st.write("## Download Processed Data")
    output_format = st.selectbox("Select output format", ["CSV", "Excel"], key="output_format")
    if output_format == "CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")
    elif output_format == "Excel":
        excel_file = BytesIO()
        df.to_excel(excel_file, index=False)
        st.download_button("Download Excel", excel_file, "processed_data.xlsx", "application/vnd.ms-excel")

else:
    st.info("Please upload a dataset to get started.")
