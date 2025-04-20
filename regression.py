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
import pickle
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
    st.session_state.processed_data = None
    st.session_state.missing_values_handled = False
    st.session_state.scaling_applied = False
    st.session_state.analysis_ready = False

# Helper functions
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    if data.isnull().sum().sum() == 0:
        st.session_state.missing_values_handled = True
        return data
    
    st.warning("⚠️ Dataset contains missing values. Please choose how to handle them.")
    
    method = st.radio(
        "Imputation method:",
        ["Drop rows with missing values", 
         "Fill with mean (numeric only)",
         "Fill with median (numeric only)",
         "Fill with mode",
         "Fill with constant value"]
    )
    
    fill_value = None
    if method == "Fill with constant value":
        fill_value = st.text_input("Enter constant value to use:", "0")
        try:
            fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
        except ValueError:
            pass
    
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
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
            elif method == "Fill with constant value":
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        
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
        st.session_state.scaling_applied = True
        return data
    
    scaling_method = st.radio(
        "Select scaling method:",
        ["None", "Standardization (Z-score)", 
         "Normalization (Min-Max)", "Robust Scaling"],
        index=0
    )
    
    if scaling_method == "None":
        st.session_state.scaling_applied = True
        return data
    
    if st.button("Apply Scaling"):
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
        st.error("Data contains missing values! Please handle them first.")
        return False
    
    return True

def create_diagnostic_plots(y_true, y_pred, model_type='regression'):
    """Create diagnostic plots for model evaluation"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if model_type == 'regression':
        # Residual plot for regression
        residuals = y_true - y_pred
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='-')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
    else:
        # ROC curve for classification
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
    
    st.pyplot(fig)

def main():
    st.set_page_config(page_title="Regression Analysis App", layout="wide")
    st.title("📊 Regression Analysis App")
    
    # Upload data
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file is not None and st.session_state.data is None:
        try:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.processed_data = st.session_state.data.copy()
            st.success("Data loaded successfully!")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
    
    # Data preprocessing pipeline
    if st.session_state.data is not None:
        # Show raw data preview
        with st.expander("Raw Data Preview"):
            st.dataframe(st.session_state.data.head())
        
        # Missing value handling
        if not st.session_state.missing_values_handled:
            st.header("Step 1: Handle Missing Values")
            handle_missing_values(st.session_state.processed_data)
            return
        
        # Feature scaling
        if not st.session_state.scaling_applied:
            st.header("Step 2: Feature Scaling")
            apply_feature_scaling(st.session_state.processed_data)
            return
        
        # Show processed data
        st.header("Processed Data Ready for Analysis")
        st.dataframe(st.session_state.processed_data.head())
        
        # Model configuration
        st.header("Step 3: Configure Analysis")
        data = st.session_state.processed_data
        
        # Select target variable
        y_var = st.selectbox("Select target variable (Y):", data.columns)
        
        # Select features
        x_vars = st.multiselect(
            "Select features (X):", 
            [col for col in data.columns if col != y_var]
        )
        
        if len(x_vars) == 0:
            st.warning("Please select at least one feature!")
            return
        
        # Select model type
        model_type = st.selectbox(
            "Select model type:",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", 
             "Elastic Net", "Logistic Regression", "Gradient Boosting"]
        )
        
        # Model-specific parameters
        if model_type in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
            alpha = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0)
        
        if model_type == "Elastic Net":
            l1_ratio = st.slider("L1 ratio:", 0.0, 1.0, 0.5)
        
        # Train/test split
        test_size = st.slider("Test set size:", 0.1, 0.5, 0.2)
        
        # Run analysis button
        if st.button("Run Analysis"):
            if not validate_data(data, x_vars, y_var):
                return
            
            X = data[x_vars]
            y = data[y_var]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize and train model
            if model_type == "Linear Regression":
                model = LinearRegression()
            elif model_type == "Ridge Regression":
                model = Ridge(alpha=alpha)
            elif model_type == "Lasso Regression":
                model = Lasso(alpha=alpha)
            elif model_type == "Elastic Net":
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            elif model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_type == "Gradient Boosting":
                if len(np.unique(y)) == 2:  # Binary classification
                    model = GradientBoostingClassifier()
                else:
                    model = GradientBoostingRegressor()
            
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Show results
            st.header("Analysis Results")
            
            if model_type in ["Linear Regression", "Ridge Regression", 
                             "Lasso Regression", "Elastic Net"]:
                st.subheader("Model Coefficients")
                coef_df = pd.DataFrame({
                    "Feature": x_vars,
                    "Coefficient": model.coef_
                })
                st.dataframe(coef_df)
                
                if hasattr(model, 'intercept_'):
                    st.write(f"Intercept: {model.intercept_:.4f}")
                
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.subheader("Model Performance")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
                
            elif model_type == "Logistic Regression":
                # Classification metrics
                accuracy = accuracy_score(y_test, y_pred.round())
                cm = confusion_matrix(y_test, y_pred.round())
                cr = classification_report(y_test, y_pred.round())
                
                st.subheader("Model Performance")
                st.write(f"Accuracy: {accuracy:.4f}")
                
                st.subheader("Confusion Matrix")
                st.write(cm)
                
                st.subheader("Classification Report")
                st.text(cr)
            
            # Diagnostic plots
            st.subheader("Diagnostic Plots")
            create_diagnostic_plots(
                y_test, y_pred,
                'regression' if model_type != "Logistic Regression" else 'classification'
            )
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                importance_df = pd.DataFrame({
                    "Feature": x_vars,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x="Importance", y="Feature", data=importance_df, ax=ax)
                st.pyplot(fig)
            
            # Prediction interface
            st.header("Make Predictions")
            pred_input = {}
            
            cols = st.columns(3)
            for i, var in enumerate(x_vars):
                with cols[i % 3]:
                    pred_input[var] = st.number_input(
                        f"Enter {var}",
                        value=float(data[var].mean())
                    )
            
            if st.button("Predict"):
                input_df = pd.DataFrame([pred_input])
                prediction = model.predict(input_df)
                
                if model_type == "Logistic Regression":
                    st.success(f"Predicted class: {int(prediction[0])} (probability: {prediction[0]:.2f})")
                else:
                    st.success(f"Predicted value: {prediction[0]:.4f}")

if __name__ == "__main__":
    main()
