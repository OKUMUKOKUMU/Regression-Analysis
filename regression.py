import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score)
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
import io
from typing import Tuple, Union
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# App title and config
st.set_page_config(page_title="Regression Analysis App", layout="wide")
st.title("📊 Advanced Regression Analysis App")
st.sidebar.title("Settings")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None

# Helper functions
def validate_data(data: pd.DataFrame, x_vars: list, y_var: str) -> bool:
    """Validate the input data and selected variables"""
    if data.empty:
        st.error("Uploaded data is empty!")
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

def prepare_data(data: pd.DataFrame, x_vars: list, y_var: str, test_size: float = 0.2) -> Tuple:
    """Prepare and split data into train/test sets"""
    X = data[x_vars]
    y = data[y_var]
    
    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test
    return X, None, y, None

def create_diagnostic_plots(y_true: np.ndarray, y_pred: np.ndarray, X_used: np.ndarray) -> plt.Figure:
    """Create diagnostic plots for regression analysis"""
    residuals = y_true - y_pred
    leverage = (X_used * np.linalg.pinv(X_used.T @ X_used) @ X_used.T).sum(axis=1)
    cooks_d = residuals**2 / (X_used.shape[1] * np.var(residuals)) * leverage

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Residuals vs Fitted
    sns.scatterplot(x=y_pred, y=residuals, ax=ax[0])
    ax[0].axhline(0, color='red', linestyle='--')
    ax[0].set_title("Residuals vs Fitted")
    ax[0].set_xlabel("Fitted Values")
    ax[0].set_ylabel("Residuals")
    
    # Residuals Histogram
    sns.histplot(residuals, bins=20, kde=True, ax=ax[1])
    ax[1].set_title("Residuals Distribution")
    
    # Cook's Distance vs Leverage
    sns.scatterplot(x=leverage, y=cooks_d, ax=ax[2])
    ax[2].set_title("Influence Plot")
    ax[2].set_xlabel("Leverage")
    ax[2].set_ylabel("Cook's Distance")
    
    plt.tight_layout()
    return fig

# Main app function
def main():
    # Upload data
    with st.sidebar.expander("📁 Data Upload", expanded=True):
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                st.session_state.data = pd.read_csv(uploaded_file)
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Show data preview
        with st.expander("🔍 Data Preview", expanded=False):
            st.dataframe(data.head())
            st.write(f"Shape: {data.shape}")
            st.write("Summary Statistics:")
            st.dataframe(data.describe())
        
        # Model selection and configuration
        col1, col2 = st.columns(2)
        
        with col1:
            regression_type = st.selectbox(
                "Select regression type:",
                [
                    "Simple Linear Regression",
                    "Multiple Linear Regression",
                    "Polynomial Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "Elastic Net Regression",
                    "Logistic Regression",
                    "Gradient Boosting",
                    "Stepwise Regression",
                ]
            )
            
            y_var = st.selectbox("Select target variable (Y):", data.columns)
        
        with col2:
            x_vars = st.multiselect(
                "Select features (X):", 
                [col for col in data.columns if col != y_var]
            )
            
            test_size = st.slider(
                "Test set size (0 for no split):", 
                0.0, 0.5, 0.2, 0.05
            )
        
        # Model-specific parameters
        model_params = {}
        if regression_type == "Polynomial Regression":
            model_params['degree'] = st.slider("Polynomial degree:", 2, 5, 2)
        elif regression_type in ["Ridge Regression", "Lasso Regression"]:
            model_params['alpha'] = st.slider("Regularization strength:", 0.01, 10.0, 1.0, 0.01)
        elif regression_type == "Elastic Net Regression":
            model_params['alpha'] = st.slider("Alpha:", 0.01, 10.0, 1.0, 0.01)
            model_params['l1_ratio'] = st.slider("L1 ratio:", 0.0, 1.0, 0.5, 0.01)
        elif regression_type == "Gradient Boosting":
            model_params['n_estimators'] = st.slider("Number of trees:", 10, 200, 100)
            model_params['learning_rate'] = st.slider("Learning rate:", 0.01, 1.0, 0.1, 0.01)
        elif regression_type == "Stepwise Regression":
            model_params['direction'] = st.radio("Stepwise direction:", ["forward", "backward"])
        
        # Run analysis button
        if st.button("🚀 Run Analysis"):
            if not x_vars:
                st.error("Please select at least one feature variable!")
                return
            
            if not validate_data(data, x_vars, y_var):
                return
            
            # Prepare data
            X_train, X_test, y_train, y_test = prepare_data(
                data, x_vars, y_var, test_size
            )
            
            # Initialize variables
            model = None
            results_text = ""
            X_used = X_train
            scaler = StandardScaler()
            
            try:
                # Model training
                with st.spinner("Training model..."):
                    if regression_type == "Simple Linear Regression" and len(x_vars) == 1:
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_train)
                        results_text = (
                            f"Intercept: {model.intercept_:.4f}\n"
                            f"Coefficient: {model.coef_[0]:.4f}\n"
                            f"R²: {r2_score(y_train, y_pred):.4f}"
                        )
                        
                    elif regression_type == "Multiple Linear Regression":
                        X_train_const = sm.add_constant(X_train)
                        model = sm.OLS(y_train, X_train_const).fit()
                        y_pred = model.predict(X_train_const)
                        results_text = model.summary().as_text()
                        X_used = X_train_const
                        
                    elif regression_type == "Polynomial Regression":
                        poly = PolynomialFeatures(degree=model_params['degree'])
                        X_poly = poly.fit_transform(X_train)
                        model = LinearRegression()
                        model.fit(X_poly, y_train)
                        y_pred = model.predict(X_poly)
                        results_text = f"R²: {r2_score(y_train, y_pred):.4f}"
                        X_used = X_poly
                        
                    elif regression_type == "Ridge Regression":
                        X_scaled = scaler.fit_transform(X_train)
                        model = Ridge(alpha=model_params['alpha'])
                        model.fit(X_scaled, y_train)
                        y_pred = model.predict(X_scaled)
                        results_text = f"R²: {r2_score(y_train, y_pred):.4f}"
                        X_used = X_scaled
                        
                    elif regression_type == "Lasso Regression":
                        X_scaled = scaler.fit_transform(X_train)
                        model = Lasso(alpha=model_params['alpha'])
                        model.fit(X_scaled, y_train)
                        y_pred = model.predict(X_scaled)
                        results_text = f"R²: {r2_score(y_train, y_pred):.4f}"
                        X_used = X_scaled
                        
                    elif regression_type == "Elastic Net Regression":
                        X_scaled = scaler.fit_transform(X_train)
                        model = ElasticNet(
                            alpha=model_params['alpha'],
                            l1_ratio=model_params['l1_ratio']
                        )
                        model.fit(X_scaled, y_train)
                        y_pred = model.predict(X_scaled)
                        results_text = f"R²: {r2_score(y_train, y_pred):.4f}"
                        X_used = X_scaled
                        
                    elif regression_type == "Gradient Boosting":
                        if data[y_var].nunique() == 2:  # Binary classification
                            model = GradientBoostingClassifier(
                                n_estimators=model_params['n_estimators'],
                                learning_rate=model_params['learning_rate']
                            )
                        else:  # Regression
                            model = GradientBoostingRegressor(
                                n_estimators=model_params['n_estimators'],
                                learning_rate=model_params['learning_rate']
                            )
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_train)
                        if data[y_var].nunique() == 2:
                            acc = accuracy_score(y_train, y_pred)
                            results_text = f"Accuracy: {acc:.4f}"
                        else:
                            results_text = f"R²: {r2_score(y_train, y_pred):.4f}"
                        
                    elif regression_type == "Logistic Regression":
                        X_scaled = scaler.fit_transform(X_train)
                        model = LogisticRegression()
                        model.fit(X_scaled, y_train)
                        y_pred = model.predict(X_scaled)
                        acc = accuracy_score(y_train, y_pred)
                        cm = confusion_matrix(y_train, y_pred)
                        report = classification_report(y_train, y_pred)
                        results_text = (
                            f"Accuracy: {acc:.4f}\n\n"
                            f"Confusion Matrix:\n{cm}\n\n"
                            f"Classification Report:\n{report}"
                        )
                        X_used = X_scaled
                        
                    elif regression_type == "Stepwise Regression":
                        direction = model_params['direction']
                        lin_model = LinearRegression()
                        sfs = SFS(
                            lin_model,
                            k_features='best',
                            forward=(direction == "forward"),
                            floating=False,
                            scoring='r2',
                            cv=0
                        )
                        X_scaled = scaler.fit_transform(X_train)
                        sfs.fit(X_scaled, y_train)
                        selected_feat = list(sfs.k_feature_names_)
                        model = LinearRegression()
                        model.fit(X_train[selected_feat], y_train)
                        y_pred = model.predict(X_train[selected_feat])
                        coefs = dict(zip(selected_feat, model.coef_))
                        results_text = (
                            f"Selected Features: {selected_feat}\n"
                            f"Coefficients: {coefs}\n"
                            f"R²: {r2_score(y_train, y_pred):.4f}"
                        )
                        X_used = X_train[selected_feat]
                
                # Store model in session state
                st.session_state.model = model
                
                # Show results
                st.subheader("📝 Model Results")
                st.text(results_text)
                
                # Diagnostic plots
                if regression_type != "Logistic Regression":
                    st.subheader("📈 Diagnostic Plots")
                    fig = create_diagnostic_plots(y_train, y_pred, X_used)
                    st.pyplot(fig)
                
                # Feature importance for tree-based models
                if regression_type == "Gradient Boosting":
                    st.subheader("🌳 Feature Importance")
                    feat_imp = pd.DataFrame({
                        'Feature': x_vars,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feat_imp, ax=ax)
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                
                # Test set evaluation
                if test_size > 0:
                    st.subheader("🧪 Test Set Evaluation")
                    
                    if regression_type == "Multiple Linear Regression":
                        X_test_const = sm.add_constant(X_test)
                        y_test_pred = model.predict(X_test_const)
                    elif regression_type == "Polynomial Regression":
                        X_test_poly = poly.transform(X_test)
                        y_test_pred = model.predict(X_test_poly)
                    elif regression_type in ["Ridge Regression", "Lasso Regression", 
                                          "Elastic Net Regression", "Logistic Regression"]:
                        X_test_scaled = scaler.transform(X_test)
                        y_test_pred = model.predict(X_test_scaled)
                    elif regression_type == "Stepwise Regression":
                        y_test_pred = model.predict(X_test[selected_feat])
                    else:
                        y_test_pred = model.predict(X_test)
                    
                    if data[y_var].nunique() == 2:  # Classification
                        test_acc = accuracy_score(y_test, y_test_pred)
                        st.write(f"Test Accuracy: {test_acc:.4f}")
                    else:  # Regression
                        test_r2 = r2_score(y_test, y_test_pred)
                        test_mse = mean_squared_error(y_test, y_test_pred)
                        st.write(f"Test R²: {test_r2:.4f}")
                        st.write(f"Test MSE: {test_mse:.4f}")
                
                # Download results
                st.subheader("💾 Download Results")
                buffer = io.StringIO()
                buffer.write(f"Regression Type: {regression_type}\n\n")
                buffer.write(results_text)
                
                st.download_button(
                    label="Download Results as TXT",
                    data=buffer.getvalue(),
                    file_name="regression_results.txt",
                    mime="text/plain"
                )
                
                # Prediction interface
                st.subheader("🔮 Make Predictions")
                pred_cols = st.columns(len(x_vars))
                pred_input = {}
                
                for i, var in enumerate(x_vars):
                    with pred_cols[i]:
                        pred_input[var] = st.number_input(
                            f"Enter {var}",
                            value=float(data[var].mean())
                        )
                
                if st.button("Predict"):
                    input_df = pd.DataFrame([pred_input])
                    
                    if regression_type == "Multiple Linear Regression":
                        input_df = sm.add_constant(input_df)
                        prediction = model.predict(input_df)
                    elif regression_type == "Polynomial Regression":
                        input_poly = poly.transform(input_df)
                        prediction = model.predict(input_poly)
                    elif regression_type in ["Ridge Regression", "Lasso Regression",
                                            "Elastic Net Regression", "Logistic Regression"]:
                        input_scaled = scaler.transform(input_df)
                        prediction = model.predict(input_scaled)
                    elif regression_type == "Stepwise Regression":
                        input_df = input_df[selected_feat]
                        prediction = model.predict(input_df)
                    else:
                        prediction = model.predict(input_df)
                    
                    st.success(f"Predicted {y_var}: {prediction[0]:.4f}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
