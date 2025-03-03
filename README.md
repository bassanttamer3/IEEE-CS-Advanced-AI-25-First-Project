# Data Processing and Visualization App 

This Streamlit app allows users to upload datasets, explore data, handle missing values, visualize distributions, and preprocess features with scaling and encoding.

## Features

### 1. Data Loading
- Supports CSV file uploads.
- Displays the dataset for easy inspection.

### 2. Summary Statistics
- Provides descriptive statistics for both numerical and categorical features.

### 3. Data Visualization
- Select columns to visualize with options for:
  - Histograms
  - Box plots
  - Bar plots
- Automatically chooses suitable plots based on data type.

### 4. Missing Value Handling
- Options to:
  - Remove rows with missing values.
  - Fill with mean, mode, or custom values.
  - Auto-handle missing values.

### 5. Correlation Analysis
- Displays correlation heatmaps.
- Drop columns based on correlation threshold.
  - Auto-drop highly correlated columns.

### 6. Redundant Feature Removal
- Identify and drop redundant columns with low variance.
- Auto-remove columns with high redundancy.

### 7. Encoding
- Supports:
  - One-hot encoding for multi-class categorical features.
  - Label encoding for binary categories.
- Auto-encode all categorical columns.

### 8. Feature Scaling
- Options for:
  - Standard Scaling (Z-score normalization)
  - Min-Max Scaling (range normalization)
- Scale single or all numerical columns.

## How to Run the App

1. Clone the repository:
```bash
git clone <https://github.com/bassanttamer3/IEEE-CS-Advanced-AI-25-First-Project>
```
2. Navigate to the project directory:
```bash
cd IEEE-CS-Advanced-AI-25-First-Project
```
3. Run the Streamlit app:
```bash
streamlit run main.py
```

## Dependencies
- Python 3.8+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn


## Team

This project was developed by our team to streamline data preprocessing and visualization.

- Bassant Tamer
- Ezzeldin Nabil
- Mostafa Mohamed


We worked collaboratively to build this app, combining our skills in data science, visualization, and software development.


