import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler, 
                                MinMaxScaler, RobustScaler)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score)
from sklearn.impute import SimpleImputer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
import io
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
    st.session_state.missing_values_handled = False
    st.session_state.scaling_applied = False

# Helper functions
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    if data.isnull().sum().sum() == 0:
        st.session_state.missing_values_handled = True
        return data
    
    st.warning("⚠️ Dataset contains missing values. Please choose how to handle them.")
    
    col1, col2 = st.columns(2)
    method = col1.radio(
        "Imputation method:",
        ["Drop rows with missing values", 
         "Fill with mean (numeric only)",
         "Fill with median (numeric only)",
         "Fill with mode",
         "Fill with constant value"]
    )
    
    fill_value = None
    if method == "Fill with constant value":
        fill_value = col2.text_input("Enter constant value to use:", "0")
        try:
            fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
        except ValueError:
            pass  # Keep as string if not convertible to number
    
    if st.button("Apply Missing Value Treatment"):
        cleaned_data = data.copy()
        
        if method == "Drop rows with missing values":
            cleaned_data = data.dropna()
            st.success(f"Removed {len(data) - len(cleaned_data)} rows with missing values.")
        else:
            if method == "Fill with mean (numeric only)":
                imputer = SimpleImputer(strategy='mean')
                numeric_cols = data.select_dtypes(include=np.number).columns
                cleaned_data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            elif method == "Fill with median (numeric only)":
                imputer = SimpleImputer(strategy='median')
                numeric_cols = data.select_dtypes(include=np.number).columns
                cleaned_data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            elif method == "Fill with mode":
                imputer = SimpleImputer(strategy='most_frequent')
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), 
                                         columns=data.columns)
            elif method == "Fill with constant value":
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), 
                                         columns=data.columns)
        
        st.session_state.processed_data = cleaned_data
        st.session_state.missing_values_handled = True
        st.experimental_rerun()
    
    return data

def apply_feature_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """Apply scaling to continuous variables"""
    continuous_vars = [
        col for col in data.select_dtypes(include=np.number).columns
        if len(data[col].unique()) > 2  # Not binary
    ]
    
    if not continuous_vars:
        return data
    
    scaling_method = st.sidebar.radio(
        "Select scaling method:",
        ["None", "Standardization (Z-score)", 
         "Normalization (Min-Max)", "Robust Scaling"],
        index=0
    )
    
    if scaling_method == "None":
        return data
    
    if st.sidebar.button("Apply Scaling"):
        scaled_data = data.copy()
        
        if scaling_method == "Standardization (Z-score)":
            scaler = StandardScaler()
        elif scaling_method == "Normalization (Min-Max)":
            scaler = MinMaxScaler()
        elif scaling_method == "Robust Scaling":
            scaler = RobustScaler()
        
        scaled_data[continuous_vars] = scaler.fit_transform(data[continuous_vars])
        st.session_state.processed_data = scaled_data
        st.session_state.scaling_applied = True
        st.success(f"Applied {scaling_method} to continuous variables")
        st.experimental_rerun()
    
    return data

def validate_data(data: pd.DataFrame, x_vars: list, y_var: str) -> bool:
    """Validate the input data and selected variables"""
    if data is None or data.empty:
        st.error("No data available for analysis!")
        return False
    
    for var in x_vars + [y_var]:
        if var not in data.columns:
            st.error(f"Variable '{var}' not found in data!")
            return False
        
        if not pd.api.types.is_numeric_dtype(data[var]):
            st.error(f"Variable '{var}' must be numeric!")
            return False
    
    if data[x_vars].isnull().any().any() or data[y_var].isnull().any():
        st.error("Data still contains missing values after preprocessing!")
        return False
    
    return True

# [Keep all other existing helper functions like create_diagnostic_plots(), etc.]

def main():
    st.set_page_config(page_title="Advanced Regression Analysis", layout="wide")
    st.title("📊 Advanced Regression Analysis App")
    
    # Data upload section
    with st.sidebar.expander("📁 Data Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None and st.session_state.data is None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.session_state.processed_data = st.session_state.data.copy()
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Data preprocessing section
    if st.session_state.data is not None and not st.session_state.missing_values_handled:
        st.session_state.processed_data = handle_missing_values(st.session_state.data)
        return  # Stop execution until missing values are handled
    
    if (st.session_state.processed_data is not None and 
        st.session_state.missing_values_handled and 
        not st.session_state.scaling_applied):
        apply_feature_scaling(st.session_state.processed_data)
        return  # Stop execution until scaling is decided
    
    # Main analysis section
    if (st.session_state.processed_data is not None and 
        st.session_state.missing_values_handled):
        
        data = st.session_state.processed_data
        
        # [Rest of your main analysis code remains the same...]
        # Include all the model training, evaluation, and visualization code here
        # Make sure to use the processed_data from session_state

if __name__ == "__main__":
    main()
