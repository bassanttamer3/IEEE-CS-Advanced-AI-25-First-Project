import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder


def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)


def show_summary_statistics(df):
    st.write("### Numerical Features")
    st.write(df.describe())
    st.write("### Categorical Features")
    st.write(df.select_dtypes(include=['object']).describe())


def visualize_data(df):
    col1, col2 = st.columns(2)
    with col1:
        col_to_viz = st.selectbox("Select a column to visualize", df.columns, key="viz_col")
    with col2:
        plot_type = st.selectbox("Select plot type", ["Auto", "Histogram", "Box Plot", "Bar Plot"], key="plot_type")

    col_dtype = df[col_to_viz].dtype

    if plot_type == "Auto":
        if col_dtype in ['int64', 'float64']:
            plot_type = "Histogram"
        else:
            plot_type = "Bar Plot"

    if col_dtype in ['int64', 'float64']:
        if plot_type == "Histogram":
            fig = px.histogram(df, x=col_to_viz, nbins=20, title=f"Histogram of {col_to_viz}")
            st.plotly_chart(fig)
        elif plot_type == "Box Plot":
            fig = px.box(df, y=col_to_viz, title=f"Box Plot of {col_to_viz}")
            st.plotly_chart(fig)
        elif plot_type == "Bar Plot":  # Convert numeric to categorical by binning
            df_binned = df[col_to_viz].apply(lambda x: round(x, -1))  # Example binning
            fig = px.bar(df_binned.value_counts().sort_index(), title=f"Bar Plot of {col_to_viz} (Binned)")
            st.plotly_chart(fig)
    else:
        if plot_type == "Bar Plot":
            fig = px.bar(df[col_to_viz].value_counts(), title=f"Bar Plot of {col_to_viz}")
            st.plotly_chart(fig)
        else:  # Default to Bar Plot if an incompatible plot type was chosen
            st.warning(f"{plot_type} is not suitable for categorical columns. Defaulting to Bar Plot.")
            fig = px.bar(df[col_to_viz].value_counts(), title=f"Bar Plot of {col_to_viz}")
            st.plotly_chart(fig)  # Function to handle missing values


def auto_handle_missing(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    st.success("Automatically handled missing values!")
    st.dataframe(df)
    return df


def handle_missing_values(df):
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

    if st.button("Auto Handle Missing Values"):
        df = auto_handle_missing(df)

    return df


def auto_drop_correlated_columns(df, threshold=0.7):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    corr_matrix = df[numerical_cols].corr().abs()
    upper = corr_matrix.where(~np.tril(np.ones(corr_matrix.shape, dtype=bool)))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        st.success(f"Automatically removed correlated columns: {', '.join(to_drop)}")
    else:
        st.info("No highly correlated columns found to remove.")

    st.dataframe(df)
    return df


def show_correlation_heatmap(df):
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if st.button("Show Correlation Heatmap"):
        corr = df[numerical_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)


def drop_columns_correlation(df):
    col_to_drop_corr = st.selectbox("Select a column to drop (based on heatmap)", df.columns, key="drop_corr_col")
    if st.button("Drop Column (Correlation)"):
        df.drop(columns=[col_to_drop_corr], inplace=True)
        st.success(f"Column '{col_to_drop_corr}' has been dropped.")
        st.dataframe(df)

    if st.button("Auto Drop Correlated Columns"):
        df = auto_drop_correlated_columns(df)

    return df


def auto_remove_redundant_features(df, threshold=0.95):
    redundant_cols = [col for col in df.columns if df[col].nunique() / len(df) < (1 - threshold)]
    if redundant_cols:
        df.drop(columns=redundant_cols, inplace=True)
        st.success(f"Automatically removed redundant columns: {', '.join(redundant_cols)}")
    else:
        st.info("No redundant columns found to remove.")
    st.dataframe(df)
    return df


def remove_redundant_features(df):
    selected_column_rrf = st.selectbox("Select a column to analyze redundancy", df.columns, key="rrf_col")
    if st.button("Show Value Counts"):
        value_counts = df[selected_column_rrf].value_counts()
        st.write(value_counts)

    col_to_drop = st.selectbox("Select a column to drop", df.columns, key="drop_col")
    if st.button("Drop Column"):
        df.drop(columns=[col_to_drop], inplace=True)
        st.success(f"Column '{col_to_drop}' has been dropped.")
        st.dataframe(df)

    if st.button("Auto Remove Redundant Features"):
        df = auto_remove_redundant_features(df)

    return df


def one_hot_encoding(df, column):
    encoded = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, encoded], axis=1)
    return df


def label_encoding(df, column):
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    return df


def auto_handle_encoding(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            if df[col].nunique() > 2:
                df = one_hot_encoding(df, col)
            else:
                df = label_encoding(df, col)
        st.success("Automatically encoded categorical columns!")
        st.dataframe(df)
    else:
        st.info("No categorical columns found to encode.")
    return df


def handle_encoding(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        selected_column = st.selectbox("Choose a categorical column to encode:", categorical_cols)
        method_encode = st.radio("Choose encoding method:", ["One-Hot Encoding", "Label Encoding"])

        if st.button("Apply Encoding"):
            df = one_hot_encoding(df, selected_column) if method_encode == "One-Hot Encoding" else label_encoding(df, selected_column)
            st.dataframe(df)

    if st.button("Auto Encode Categorical Columns"):
        df = auto_handle_encoding(df)

    return df  # Function to handle feature scaling


def scale_one_col(df, column, scaler):
    df[f'{column}_scaled'] = scaler.fit_transform(df[[column]])
    return df


def scale_all(df, scaler):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    scaled_data = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_data, columns=[f'{col}_scaled' for col in numeric_cols])
    return pd.concat([df.reset_index(drop=True), scaled_df.reset_index(drop=True)], axis=1)


def auto_handle_scaling(df, method="Standard Scaling"):
    scaler = StandardScaler() if method == "Standard Scaling" else MinMaxScaler()
    df = scale_all(df, scaler)
    st.success(f"Automatically applied {method} to all numerical columns!")
    st.dataframe(df)
    return df


def handle_feature_scaling(df):
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

    if st.button("Auto Scale All Columns"):
        df = auto_handle_scaling(df, method_scaling)

    return df


def auto_preprocess(df):
    df = auto_drop_correlated_columns(df)
    df = auto_remove_redundant_features(df)
    df = auto_handle_encoding(df)
    df = auto_handle_scaling(df)
    st.success("Automatically applied all preprocessing steps!")
    return df


def download_processed_data(df):
    output_format = st.selectbox("Select output format", ["CSV"], key="output_format")
    if output_format == "CSV":
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "processed_data.csv", "text/csv")


def main():
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
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        st.write("## Original Dataset")
        st.dataframe(df)

        # Create tabs for each functionality
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
            "Summary Statistics", "Data Visualization", "Missing Values", "Correlation Heatmap",
            "Remove Redundant Features", "Encoding", "Feature Scaling", "Auto Preprocess", "Download Data"
        ])

        with tab1:
            st.write("### Summary Statistics")
            show_summary_statistics(df)

        with tab2:
            st.write("### Data Visualization")
            visualize_data(df)

        with tab3:
            st.write("### Handle Missing Values")
            df = handle_missing_values(df)

        with tab4:
            st.write("### Feature Selection and Correlation Heatmap")
            show_correlation_heatmap(df)
            df = drop_columns_correlation(df)

        with tab5:
            st.write("### Remove Redundant Features")
            df = remove_redundant_features(df)

        with tab6:
            st.write("### Encoding")
            df = handle_encoding(df)

        with tab7:
            st.write("### Feature Scaling")
            df = handle_feature_scaling(df)

        with tab8:
            st.write("### Auto Preprocess")
            if st.button("Run Auto Preprocess"):
                df = auto_preprocess(df)
                st.success("Auto preprocessing completed!")
                st.dataframe(df)

        with tab9:
            st.write("### Download Processed Data")
            download_processed_data(df)
    else:
        st.info("Please upload a dataset to get started.")


if __name__ == "__main__":
    main()
