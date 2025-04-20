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
from scipy import stats
from sklearn.metrics import roc_curve, auc

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'preprocessing_stage' not in st.session_state:
    st.session_state.preprocessing_stage = 'upload'  # upload → missing → scaling → analysis
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model_results = None

# Helper functions
def handle_missing_values_ui(data: pd.DataFrame):
    """UI for handling missing values"""
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
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), 
                                         columns=data.columns)
            elif method == "Fill with constant value":
                imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
                cleaned_data = pd.DataFrame(imputer.fit_transform(data), 
                                         columns=data.columns)
        
        st.session_state.processed_data = cleaned_data
        st.session_state.preprocessing_stage = 'scaling'
        return True
    
    return False

def apply_feature_scaling_ui(data: pd.DataFrame):
    """UI for applying feature scaling"""
    continuous_vars = [
        col for col in data.select_dtypes(include=np.number).columns
        if len(data[col].unique()) > 2  # Not binary
    ]
    
    if not continuous_vars:
        st.session_state.preprocessing_stage = 'analysis'
        return True
    
    st.sidebar.subheader("Feature Scaling Options")
    scaling_method = st.sidebar.radio(
        "Select scaling method:",
        ["None", "Standardization (Z-score)", 
         "Normalization (Min-Max)", "Robust Scaling"],
        index=0
    )
    
    if scaling_method == "None":
        st.session_state.preprocessing_stage = 'analysis'
        return True
    
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
        st.success(f"Applied {scaling_method} to continuous variables")
        st.session_state.preprocessing_stage = 'analysis'
        return True
    
    return False

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Diagnostic Plots", fontsize=16)
    
    # Residuals vs Fitted
    residuals = y_true - y_pred
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # QQ Plot or ROC Curve
    if model_type == 'regression':
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q')
    else:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, 
                      label=f'ROC curve (area = {roc_auc:.2f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('Receiver Operating Characteristic')
        axes[0, 1].legend(loc="lower right")
    
    # Scale-Location or Confusion Matrix
    if model_type == 'regression':
        sqrt_residuals = np.sqrt(np.abs(residuals))
        axes[1, 0].scatter(y_pred, sqrt_residuals, alpha=0.5)
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('Sqrt(|Residuals|)')
        axes[1, 0].set_title('Scale-Location')
    else:
        cm = confusion_matrix(y_true, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
    
    # Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.5)
    axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    st.pyplot(fig)

def feature_importance_plot(model, feature_names):
    """Create feature importance plot for supported models"""
    importance = None
    
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1 and importance.shape[0] > 1:  # Multi-class case
            importance = importance.mean(axis=0)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    if importance is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        sns.barplot(x='Importance', y='Feature', data=features_df, ax=ax)
        ax.set_title('Feature Importance')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Feature importance not available for this model type")

def main():
    st.set_page_config(page_title="Advanced Regression Analysis", layout="wide")
    st.title("📊 Advanced Regression Analysis App")
    
    # Data upload stage
    if st.session_state.preprocessing_stage == 'upload':
        with st.sidebar.expander("📁 Data Upload", expanded=True):
            uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    st.session_state.raw_data = pd.read_csv(uploaded_file)
                    st.session_state.processed_data = st.session_state.raw_data.copy()
                    
                    if st.session_state.raw_data.isnull().sum().sum() > 0:
                        st.session_state.preprocessing_stage = 'missing'
                    else:
                        st.session_state.preprocessing_stage = 'scaling'
                    
                    st.success("Data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
    
    # Missing value handling stage
    elif st.session_state.preprocessing_stage == 'missing':
        if handle_missing_values_ui(st.session_state.processed_data):
            st.experimental_rerun()
    
    # Feature scaling stage
    elif st.session_state.preprocessing_stage == 'scaling':
        if apply_feature_scaling_ui(st.session_state.processed_data):
            st.experimental_rerun()
    
    # Main analysis stage
    elif st.session_state.preprocessing_stage == 'analysis':
        data = st.session_state.processed_data
        
        # Show data preview
        with st.expander("🔍 Processed Data Preview", expanded=False):
            st.dataframe(data.head())
            st.write(f"Shape: {data.shape}")
            st.write("Summary Statistics:")
            st.dataframe(data.describe())
        
        # Model selection and configuration
        st.header("Model Configuration")
        
        analysis_type = st.radio(
            "Select Analysis Type:",
            ["Regression", "Classification"]
        )
        
        y_var = st.selectbox(
            "Select Target Variable (Y):",
            data.select_dtypes(include=np.number).columns
        )
        
        # X variables selection
        x_vars = st.multiselect(
            "Select Predictor Variables (X):",
            [col for col in data.select_dtypes(include=np.number).columns if col != y_var],
            default=[col for col in data.select_dtypes(include=np.number).columns if col != y_var][:3]
        )
        
        if len(x_vars) == 0:
            st.warning("Please select at least one predictor variable!")
            return
        
        # Model selection and hyperparameters
        st.subheader("Model Selection")
        
        if analysis_type == "Regression":
            model_type = st.selectbox(
                "Select Model Type:",
                ["Linear Regression", "Ridge Regression", "Lasso Regression", 
                 "Elastic Net", "Polynomial Regression", "Gradient Boosting"]
            )
        else:
            model_type = st.selectbox(
                "Select Model Type:",
                ["Logistic Regression", "Gradient Boosting Classifier"]
            )
            
        test_size = st.slider("Test Set Size (%):", 10, 50, 30) / 100
        
        # Advanced options
        with st.expander("Advanced Model Options"):
            if model_type in ["Ridge Regression", "Lasso Regression", "Elastic Net"]:
                alpha = st.slider("Regularization Strength (Alpha):", 0.01, 10.0, 1.0)
            
            if model_type == "Elastic Net":
                l1_ratio = st.slider("L1 Ratio (0=Ridge, 1=Lasso):", 0.0, 1.0, 0.5)
            
            if model_type == "Polynomial Regression":
                poly_degree = st.slider("Polynomial Degree:", 2, 5, 2)
            
            if "Gradient Boosting" in model_type:
                n_estimators = st.slider("Number of Trees:", 50, 500, 100)
                max_depth = st.slider("Max Tree Depth:", 3, 10, 3)
                learning_rate = st.slider("Learning Rate:", 0.01, 0.3, 0.1)
        
        # Feature selection
        with st.expander("Feature Selection"):
            do_feature_selection = st.checkbox("Perform Feature Selection")
            if do_feature_selection:
                selection_method = st.radio(
                    "Selection Method:",
                    ["Forward Selection", "Backward Elimination"]
                )
                k_features = st.slider("Maximum Number of Features:", 
                                     1, len(x_vars), min(5, len(x_vars)))
        
        # Train model button
        if st.button("Train Model"):
            # Validate data
            if not validate_data(data, x_vars, y_var):
                return
            
            # Prepare data
            X = data[x_vars]
            y = data[y_var]
            
            # Feature selection if requested
            if do_feature_selection:
                if analysis_type == "Regression":
                    estimator = LinearRegression()
                    scoring = 'neg_mean_squared_error'
                else:
                    estimator = LogisticRegression(max_iter=1000)
                    scoring = 'accuracy'
                
                sfs = SFS(estimator, 
                         k_features=min(k_features, X.shape[1]), 
                         forward=(selection_method == "Forward Selection"),
                         floating=False, 
                         scoring=scoring,
                         cv=5)
                
                with st.spinner("Performing feature selection..."):
                    sfs.fit(X, y)
                
                selected_features = list(X.columns[list(sfs.k_feature_idx_)])
                X = X[selected_features]
                st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
            
            # Apply polynomial features if selected
            if model_type == "Polynomial Regression":
                poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                X = pd.DataFrame(
                    poly.fit_transform(X),
                    columns=poly.get_feature_names_out(X.columns)
                )
                feature_names = X.columns
            else:
                feature_names = x_vars
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Initialize and train model
            with st.spinner("Training model..."):
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Ridge Regression":
                    model = Ridge(alpha=alpha)
                elif model_type == "Lasso Regression":
                    model = Lasso(alpha=alpha)
                elif model_type == "Elastic Net":
                    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
                elif model_type == "Polynomial Regression":
                    model = LinearRegression()
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                elif model_type == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, random_state=42)
                elif model_type == "Gradient Boosting Classifier":
                    model = GradientBoostingClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                
                model.fit(X_train, y_train)
            
            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Store results in session state
            st.session_state.model_trained = True
            st.session_state.model_results = {
                'model': model,
                'model_type': model_type,
                'analysis_type': analysis_type,
                'feature_names': feature_names,
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred
            }
            
            st.experimental_rerun()
        
        # Results section
        if st.session_state.model_trained and st.session_state.model_results is not None:
            st.header("Model Results")
            
            res = st.session_state.model_results
            model = res['model']
            model_type = res['model_type']
            analysis_type = res['analysis_type']
            feature_names = res['feature_names']
            X_train = res['X_train']
            X_test = res['X_test']
            y_train = res['y_train']
            y_test = res['y_test']
            y_train_pred = res['y_train_pred']
            y_test_pred = res['y_test_pred']
            
            col1, col2 = st.columns(2)
            
            if analysis_type == "Regression":
                # Regression metrics
                with col1:
                    st.subheader("Training Set Metrics")
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    train_r2 = r2_score(y_train, y_train_pred)
                    
                    st.write(f"RMSE: {train_rmse:.4f}")
                    st.write(f"R²: {train_r2:.4f}")
                    st.write(f"Mean Target: {y_train.mean():.4f}")
                
                with col2:
                    st.subheader("Test Set Metrics")
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    test_r2 = r2_score(y_test, y_test_pred)
                    
                    st.write(f"RMSE: {test_rmse:.4f}")
                    st.write(f"R²: {test_r2:.4f}")
                    st.write(f"Mean Target: {y_test.mean():.4f}")
                
                # Coefficients for linear models
                if hasattr(model, 'coef_'):
                    st.subheader("Model Coefficients")
                    coef_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': model.coef_
                    })
                    if hasattr(model, 'intercept_'):
                        st.write(f"Intercept: {model.intercept_:.4f}")
                    
                    st.dataframe(coef_df)
                
                # OLS Summary (for linear models)
                if model_type in ["Linear Regression", "Ridge Regression", "Lasso Regression", 
                                "Elastic Net", "Polynomial Regression"]:
                    with st.expander("Detailed Statistical Summary (OLS)"):
                        X_with_const = sm.add_constant(X_train)
                        ols_model = sm.OLS(y_train, X_with_const).fit()
                        st.text(ols_model.summary())
            
            else:
                # Classification metrics
                with col1:
                    st.subheader("Training Set Metrics")
                    train_acc = accuracy_score(y_train, np.round(y_train_pred))
                    
                    try:
                        train_auc = roc_auc_score(y_train, y_train_pred)
                        st.write(f"AUC: {train_auc:.4f}")
                    except:
                        pass
                    
                    st.write(f"Accuracy: {train_acc:.4f}")
                
                with col2:
                    st.subheader("Test Set Metrics")
                    test_acc = accuracy_score(y_test, np.round(y_test_pred))
                    
                    try:
                        test_auc = roc_auc_score(y_test, y_test_pred)
                        st.write(f"AUC: {test_auc:.4f}")
                    except:
                        pass
                    
                    st.write(f"Accuracy: {test_acc:.4f}")
                
                # Classification report
                st.subheader("Classification Report")
                cr = classification_report(y_test, np.round(y_test_pred), output_dict=True)
                cr_df = pd.DataFrame(cr).transpose()
                st.dataframe(cr_df)
            
            # Diagnostic plots
            st.header("Diagnostic Plots")
            create_diagnostic_plots(y_test, y_test_pred, analysis_type.lower())
            
            # Feature importance
            st.header("Feature Importance")
            feature_importance_plot(model, feature_names)
            
            # Prediction tool
            st.header("Prediction Tool")
            
            with st.expander("Make Predictions"):
                col_values = {}
                
                # Create input fields for each feature
                cols = st.columns(3)
                for i, feature in enumerate(x_vars):
                    col_idx = i % 3
                    feature_min = data[feature].min()
                    feature_max = data[feature].max()
                    feature_mean = data[feature].mean()
                    
                    col_values[feature] = cols[col_idx].slider(
                        f"{feature}:", 
                        float(feature_min), 
                        float(feature_max), 
                        float(feature_mean)
                    )
                
                # Create input DataFrame
                input_df = pd.DataFrame([col_values])
                
                # Handle polynomial features
                if model_type == "Polynomial Regression":
                    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
                    input_df = pd.DataFrame(
                        poly.fit_transform(input_df),
                        columns=poly.get_feature_names_out(input_df.columns)
                    )
                
                if st.button("Predict"):
                    prediction = model.predict(input_df)
                    
                    if analysis_type == "Regression":
                        st.success(f"Predicted {y_var}: {prediction[0]:.4f}")
                    else:
                        prob = prediction[0]
                        class_pred = 1 if prob >= 0.5 else 0
                        st.success(f"Predicted Class: {class_pred} (Probability: {prob:.4f})")
            
            # Download model
            with st.expander("Export Results"):
                # Save model to bytes
                model_bytes = BytesIO()
                pickle.dump(model, model_bytes)
                model_bytes.seek(0)
                
                st.download_button(
                    label="Download Model (.pkl)",
                    data=model_bytes,
                    file_name=f"{model_type.replace(' ', '_').lower()}_model.pkl",
                    mime="application/octet-stream"
                )
                
                # Save predictions to CSV
                predictions_df = pd.DataFrame({
                    'Actual': y_test,
                    'Predicted': y_test_pred
                })
                
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="Download Predictions (.csv)",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv"
                )
                
                # Generate report
                report = f"""
                # Analysis Report: {model_type}

                ## Dataset Information
                - Total observations: {len(data)}
                - Features used: {', '.join(x_vars)}
                - Target variable: {y_var}

                ## Model Performance
                """
                
                if analysis_type == "Regression":
                    report += f"""
                    - Test RMSE: {test_rmse:.4f}
                    - Test R²: {test_r2:.4f}
                    - Training RMSE: {train_rmse:.4f}
                    - Training R²: {train_r2:.4f}
                    """
                else:
                    report += f"""
                    - Test Accuracy: {test_acc:.4f}
                    - Training Accuracy: {train_acc:.4f}
                    """
                    if 'test_auc' in locals():
                        report += f"- Test AUC: {test_auc:.4f}\n"
                    if 'train_auc' in locals():
                        report += f"- Training AUC: {train_auc:.4f}\n"
                
                if hasattr(model, 'coef_'):
                    report += "\n## Model Coefficients\n"
                    for feat, coef in zip(feature_names, model.coef_):
                        report += f"- {feat}: {coef:.4f}\n"
                    
                    if hasattr(model, 'intercept_'):
                        report += f"- Intercept: {model.intercept_:.4f}\n"
                
                st.download_button(
                    label="Download Report (.md)",
                    data=report,
                    file_name="model_report.md",
                    mime="text/markdown"
                )

if __name__ == "__main__":
    main()
