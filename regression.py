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
from statsmodels.formula.api import ols
import statsmodels.stats.diagnostic as sm_diagnostic
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.model_results = None
    
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
    
    scaling_method = st.sidebar.radio(
        "Select scaling method:",
        ["None", "Standardization (Z-score)", 
         "Normalization (Min-Max)", "Robust Scaling"],
        index=0
    )
    
    if scaling_method == "None":
        st.session_state.scaling_applied = True
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

def create_diagnostic_plots(X, y, y_pred, model_type='regression'):
    """Create diagnostic plots for model evaluation"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Model Diagnostic Plots", fontsize=16)
    
    # Residuals vs Fitted
    residuals = y - y_pred
    axes[0, 0].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_xlabel('Fitted values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # QQ Plot
    if model_type == 'regression':
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q')
    else:
        # For classification, show ROC curve instead
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # For multi-class, use first class probability
            y_score = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred
        else:
            y_score = y_pred
            
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y, y_score)
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
    
    # Scale-Location
    if model_type == 'regression':
        sqrt_residuals = np.sqrt(np.abs(residuals))
        axes[1, 0].scatter(y_pred, sqrt_residuals, alpha=0.5)
        axes[1, 0].set_xlabel('Fitted values')
        axes[1, 0].set_ylabel('Sqrt(|Residuals|)')
        axes[1, 0].set_title('Scale-Location')
    else:
        # For classification, show confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, np.round(y_pred))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        axes[1, 0].set_title('Confusion Matrix')
    
    # Actual vs Predicted
    axes[1, 1].scatter(y, y_pred, alpha=0.5)
    axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    axes[1, 1].set_xlabel('Actual')
    axes[1, 1].set_ylabel('Predicted')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    st.pyplot(fig)

def feature_importance_plot(model, X, feature_names):
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

def perform_feature_selection(X, y, model_type='regression', method='forward', k_features=5):
    """Perform feature selection using Sequential Feature Selection"""
    if model_type == 'regression':
        estimator = LinearRegression()
        scoring = 'neg_mean_squared_error'
    else:
        estimator = LogisticRegression(max_iter=1000)
        scoring = 'accuracy'
    
    sfs = SFS(estimator, 
              k_features=min(k_features, X.shape[1]), 
              forward=(method == 'forward'),
              floating=False, 
              scoring=scoring,
              cv=5)
    
    with st.spinner("Performing feature selection..."):
        sfs.fit(X, y)
    
    selected_features = list(X.columns[list(sfs.k_feature_idx_)])
    
    st.success(f"Selected {len(selected_features)} features: {', '.join(selected_features)}")
    
    # Plot performance vs number of features
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = []
    for i in range(1, len(sfs.subsets_) + 1):
        metrics.append(sfs.subsets_[i]['avg_score'])
    
    ax.plot(range(1, len(metrics) + 1), metrics, marker='o')
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Performance Metric')
    ax.set_title('Performance vs Number of Features')
    plt.tight_layout()
    st.pyplot(fig)
    
    return selected_features

def create_cross_validation_results(X, y, model, model_type, cv=5):
    """Perform cross-validation and show results"""
    from sklearn.model_selection import cross_val_score
    
    if model_type == 'regression':
        scoring = 'neg_mean_squared_error'
        scoring_name = 'Negative MSE'
    else:
        scoring = 'accuracy'
        scoring_name = 'Accuracy'
    
    with st.spinner(f"Performing {cv}-fold cross-validation..."):
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    if model_type == 'regression':
        # Convert negative MSE to RMSE
        scores = np.sqrt(-scores)
        scoring_name = 'RMSE'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, cv+1), scores)
    ax.axhline(y=np.mean(scores), color='r', linestyle='--', 
              label=f'Mean: {np.mean(scores):.4f}')
    ax.set_xlabel('Fold')
    ax.set_ylabel(scoring_name)
    ax.set_title(f'{cv}-Fold Cross-Validation Results')
    ax.set_xticks(range(1, cv+1))
    ax.legend()
    plt.tight_layout()
    
    st.subheader("Cross-Validation Results")
    st.pyplot(fig)
    st.write(f"Mean {scoring_name}: {np.mean(scores):.4f}")
    st.write(f"Standard Deviation: {np.std(scores):.4f}")

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
    
    # Reset button
    if st.sidebar.button("Reset App"):
        for key in st.session_state.keys():
            del st.session_state[key]
    
    # Data preprocessing section
    if st.session_state.data is not None and not st.session_state.missing_values_handled:
        st.session_state.processed_data = handle_missing_values(st.session_state.data)
    
    if (st.session_state.processed_data is not None and 
        st.session_state.missing_values_handled and 
        not st.session_state.scaling_applied):
        apply_feature_scaling(st.session_state.processed_data)
    
    # Main analysis section
    if (st.session_state.processed_data is not None and 
        st.session_state.missing_values_handled and
        st.session_state.scaling_applied):
        
        data = st.session_state.processed_data
        
        # Data overview
        st.header("Data Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"Dataset Shape: {data.shape}")
            st.write(f"Number of Numeric Features: {len(data.select_dtypes(include=np.number).columns)}")
            
        with col2:
            if st.button("View Data Sample"):
                st.dataframe(data.head())
        
        # Data statistics
        if st.checkbox("Show Data Statistics"):
            st.write(data.describe())
        
        # Correlation matrix
        if st.checkbox("Show Correlation Matrix"):
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = data.select_dtypes(include=np.number).corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Variable selection
        st.header("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_type = st.radio(
                "Select Analysis Type:",
                ["Regression", "Classification"]
            )
        
        with col2:
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
        col1, col2 = st.columns(2)
        
        with col1:
            if analysis_type == "Regression":
                modeling_approach = st.radio(
                    "Modeling Approach:",
                    ["Traditional Statistics", "Machine Learning"]
                )
                
                if modeling_approach == "Traditional Statistics":
                    model_type = st.selectbox(
                        "Select Model Type:",
                        ["OLS Regression", "Weighted Least Squares"]
                    )
                else:
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
                
        with col2:
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
            
            # Cross-validation options
            do_cross_validation = st.checkbox("Perform Cross-Validation")
            if do_cross_validation:
                cv_folds = st.slider("Number of CV Folds:", 3, 10, 5)
        
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
            
            # For classification, ensure target is binary
            if analysis_type == "Classification":
                if len(np.unique(y)) > 2:
                    st.warning("Target variable has more than 2 classes. Converting to binary (>= median).")
                    median_y = np.median(y)
                    y = (y >= median_y).astype(int)
            
            # Feature selection if requested
            if do_feature_selection:
                selected_features = perform_feature_selection(
                    X, y, 
                    model_type='regression' if analysis_type == "Regression" else 'classification',
                    method='forward' if selection_method == "Forward Selection" else 'backward',
                    k_features=k_features
                )
                X = X[selected_features]
                x_vars = selected_features
            
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
                if analysis_type == "Regression" and modeling_approach == "Traditional Statistics":
                    # Traditional statistical modeling
                    try:
                        if model_type == "OLS Regression":
                            formula = f"{y_var} ~ " + " + ".join(x_vars)
                            model = ols(formula=formula, data=data).fit()
                            
                            # Store predictions
                            y_train_pred = model.predict(X_train)
                            y_test_pred = model.predict(X_test)
                            
                            # Show full statistical summary
                            st.subheader("Statistical Model Summary")
                            st.text(str(model.summary()))
                            
                            # Additional diagnostics
                            st.subheader("Model Diagnostics")
                            
                            # Normality tests
                            st.write("**Normality Tests**")
                            jb_test = sm_diagnostic.het_white(model.resid, model.model.exog)
                            st.write(f"Jarque-Bera Test: p-value = {jb_test[1]:.4f}")
                            
                            # Heteroscedasticity tests
                            st.write("**Heteroscedasticity Tests**")
                            white_test = sm_diagnostic.het_white(model.resid, model.model.exog)
                            st.write(f"White Test: p-value = {white_test[1]:.4f}")
                            
                            # Multicollinearity check
                            st.write("**Multicollinearity Check**")
                            vif_data = pd.DataFrame()
                            vif_data["feature"] = x_vars
                            vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                                            for i in range(len(x_vars))]
                            st.dataframe(vif_data)
                            
                            # Store model for predictions
                            st.session_state.model_trained = True
                            st.session_state.model_results = {
                                'model': model,
                                'model_type': model_type,
                                'analysis_type': analysis_type,
                                'feature_names': x_vars,
                                'X_train': X_train,
                                'X_test': X_test,
                                'y_train': y_train,
                                'y_test': y_test,
                                'y_train_pred': y_train_pred,
                                'y_test_pred': y_test_pred,
                                'modeling_approach': modeling_approach
                            }
                            
                    except Exception as e:
                        st.error(f"Error in traditional modeling: {str(e)}")
                        return
                else:
                    # Machine learning approach
                    try:
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
                        
                        # Cross-validation if requested
                        if do_cross_validation:
                            create_cross_validation_results(
                                X, y, model, 
                                model_type='regression' if analysis_type == "Regression" else 'classification',
                                cv=cv_folds
                            )
                        
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
                            'y_test_pred': y_test_pred,
                            'modeling_approach': modeling_approach
                        }
                        
                    except Exception as e:
                        st.error(f"Error in machine learning modeling: {str(e)}")
                        return
        
        # Results section
        if st.session_state.model_trained and st.session_state.model_results is not None:
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
            modeling_approach = res.get('modeling_approach', 'Machine Learning')
            
            st.header("Model Results")
            
            if modeling_approach == "Traditional Statistics":
                # Already shown summary in training section
                pass
            else:
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
            
            create_diagnostic_plots(
                X_test, y_test, y_test_pred, 
                model_type='regression' if analysis_type == "Regression" else 'classification'
            )
            
            # Feature importance
            st.header("Feature Importance")
            feature_importance_plot(model, X_test, feature_names)
            
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
                    if modeling_approach == "Traditional Statistics":
                        # For statsmodels, we need to create a proper DataFrame
                        pred_df = pd.DataFrame([col_values])
                        pred_df = sm.add_constant(pred_df)
                        prediction = model.predict(pred_df)
                        st.success(f"Predicted {y_var}: {prediction[0]:.4f}")
                        st.write(f"95% Confidence Interval: [{model.conf_int().iloc[0,0]:.4f}, {model.conf_int().iloc[0,1]:.4f}]")
                    else:
                        prediction = model.predict(input_df)
                        
                        if analysis_type == "Regression":
                            st.success(f"Predicted {y_var}: {prediction[0]:.4f}")
                        else:
                            prob = prediction[0]
                            class_pred = 1 if prob >= 0.5 else 0
                            st.success(f"Predicted Class: {class_pred} (Probability: {prob:.4f})")
            
            # Residual analysis for regression
            if analysis_type == "Regression":
                st.header("Residual Analysis")
                
                residuals = y_test - y_test_pred
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Residual Statistics")
                    residual_stats = pd.DataFrame({
                        'Statistic': ['Mean', 'Standard Deviation', 'Minimum', '25th Percentile', 
                                    'Median', '75th Percentile', 'Maximum'],
                        'Value': [np.mean(residuals), np.std(residuals), np.min(residuals), 
                                np.percentile(residuals, 25), np.median(residuals),
                                np.percentile(residuals, 75), np.max(residuals)]
                    })
                    st.dataframe(residual_stats)
                    
                    # Durbin-Watson test for autocorrelation
                    from statsmodels.stats.stattools import durbin_watson
                    dw = durbin_watson(residuals)
                    st.write(f"Durbin-Watson Statistic: {dw:.4f}")
                    if dw < 1.5:
                        st.warning("Positive autocorrelation may be present (DW < 1.5)")
                    elif dw > 2.5:
                        st.warning("Negative autocorrelation may be present (DW > 2.5)")
                    else:
                        st.success("No significant autocorrelation detected (DW ≈ 2)")
                
                with col2:
                    st.subheader("Residual Distribution")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.histplot(residuals, kde=True, ax=ax)
                    ax.set_xlabel('Residuals')
                    ax.set_ylabel('Frequency')
                    ax.set_title('Distribution of Residuals')
                    st.pyplot(fig)
                
                # Heteroscedasticity tests
                st.subheader("Heteroscedasticity Tests")
                
                # Breusch-Pagan test
                from statsmodels.stats.diagnostic import het_breuschpagan
                X_with_const = sm.add_constant(X_test)
                bp_test = het_breuschpagan(residuals, X_with_const)
                
                bp_df = pd.DataFrame({
                    'Test': ['Lagrange multiplier statistic', 'p-value', 
                            'f-value', 'f p-value'],
                    'Value': bp_test
                })
                st.write("Breusch-Pagan Test:")
                st.dataframe(bp_df)
                
                if bp_test[1] < 0.05:
                    st.warning("Breusch-Pagan test indicates potential heteroscedasticity (p < 0.05)")
                else:
                    st.success("No significant heteroscedasticity detected by Breusch-Pagan test")
                
                # Residuals vs each predictor
                st.subheader("Residuals vs Predictors")
                predictor = st.selectbox("Select predictor to plot against residuals:", x_vars)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X_test[predictor], residuals, alpha=0.5)
                ax.axhline(y=0, color='r', linestyle='-')
                ax.set_xlabel(predictor)
                ax.set_ylabel('Residuals')
                ax.set_title(f'Residuals vs {predictor}')
                st.pyplot(fig)
                
                # Normality tests
                st.subheader("Normality Tests")
                
                from scipy import stats
                shapiro_test = stats.shapiro(residuals)
                ks_test = stats.kstest(residuals, 'norm', 
                                     args=(np.mean(residuals), np.std(residuals)))
                
                norm_df = pd.DataFrame({
                    'Test': ['Shapiro-Wilk', 'Kolmogorov-Smirnov'],
                    'Statistic': [shapiro_test[0], ks_test[0]],
                    'p-value': [shapiro_test[1], ks_test[1]]
                })
                st.dataframe(norm_df)
                
                if shapiro_test[1] < 0.05 or ks_test[1] < 0.05:
                    st.warning("Normality tests suggest residuals may not be normally distributed")
                else:
                    st.success("Residuals appear normally distributed based on tests")
        
        # Download results
        if st.session_state.model_trained:
            st.sidebar.header("Export Results")
            
            if st.sidebar.button("Export Model Summary"):
                buffer = io.StringIO()
                
                if analysis_type == "Regression":
                    buffer.write(f"Regression Model Summary\n")
                    buffer.write(f"Model Type: {model_type}\n")
                    buffer.write(f"Target Variable: {y_var}\n")
                    buffer.write(f"Predictor Variables: {', '.join(x_vars)}\n\n")
                    
                    if modeling_approach == "Traditional Statistics":
                        buffer.write(str(model.summary()))
                    else:
                        buffer.write("Training Set Metrics:\n")
                        buffer.write(f"RMSE: {train_rmse:.4f}\n")
                        buffer.write(f"R²: {train_r2:.4f}\n\n")
                        
                        buffer.write("Test Set Metrics:\n")
                        buffer.write(f"RMSE: {test_rmse:.4f}\n")
                        buffer.write(f"R²: {test_r2:.4f}\n\n")
                        
                        if hasattr(model, 'coef_'):
                            buffer.write("Model Coefficients:\n")
                            for feat, coef in zip(feature_names, model.coef_):
                                buffer.write(f"{feat}: {coef:.4f}\n")
                            if hasattr(model, 'intercept_'):
                                buffer.write(f"Intercept: {model.intercept_:.4f}\n")
                else:
                    buffer.write(f"Classification Model Summary\n")
                    buffer.write(f"Model Type: {model_type}\n")
                    buffer.write(f"Target Variable: {y_var}\n")
                    buffer.write(f"Predictor Variables: {', '.join(x_vars)}\n\n")
                    
                    buffer.write("Training Set Metrics:\n")
                    buffer.write(f"Accuracy: {train_acc:.4f}\n")
                    if 'train_auc' in locals():
                        buffer.write(f"AUC: {train_auc:.4f}\n\n")
                    
                    buffer.write("Test Set Metrics:\n")
                    buffer.write(f"Accuracy: {test_acc:.4f}\n")
                    if 'test_auc' in locals():
                        buffer.write(f"AUC: {test_auc:.4f}\n\n")
                    
                    buffer.write("Classification Report:\n")
                    buffer.write(classification_report(y_test, np.round(y_test_pred)))
                
                st.sidebar.download_button(
                    label="Download Summary",
                    data=buffer.getvalue(),
                    file_name="model_summary.txt",
                    mime="text/plain"
                )

if __name__ == "__main__":
    main()
