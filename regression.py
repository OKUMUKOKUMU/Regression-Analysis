import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import (PolynomialFeatures, StandardScaler, 
                                MinMaxScaler, RobustScaler)
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, learning_curve
from sklearn.metrics import (mean_squared_error, r2_score, 
                            accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score, 
                            roc_curve, precision_recall_curve)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.diagnostic as sm_diagnostic
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io
import base64
import pickle
import warnings
from scipy import stats
import time

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up page config
st.set_page_config(
    page_title="Advanced Regression Analysis", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Advanced Regression Analysis App - v2.0"
    }
)

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
if 'models_to_compare' not in st.session_state:
    st.session_state.models_to_compare = {}
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1  # Track progress through the workflow
    
# Decorators for caching
@st.cache_data(ttl=3600)
def load_sample_dataset(dataset_name):
    """Load a sample dataset"""
    if dataset_name == "Boston Housing":
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        data = pd.DataFrame(housing.data, columns=housing.feature_names)
        data['PRICE'] = housing.target
        return data
    elif dataset_name == "Diabetes":
        from sklearn.datasets import load_diabetes
        diabetes = load_diabetes()
        data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        data['disease_progression'] = diabetes.target
        return data
    elif dataset_name == "Wine Quality":
        wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep=';')
        return wine_data
    elif dataset_name == "Iris (Classification)":
        from sklearn.datasets import load_iris
        iris = load_iris()
        data = pd.DataFrame(iris.data, columns=iris.feature_names)
        data['species'] = iris.target
        return data
    
@st.cache_data
def compute_correlation_matrix(data):
    """Compute correlation matrix for numeric columns"""
    return data.select_dtypes(include=np.number).corr()

@st.cache_resource
def train_model_pipeline(X, y, model_type, params=None, cv=5):
    """Train a model and return it along with the pipeline"""
    # Create appropriate processing steps
    numeric_features = X.select_dtypes(include=np.number).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])
    
    # Choose model based on model_type
    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Ridge Regression":
        model = Ridge(alpha=params.get('alpha', 1.0))
    elif model_type == "Lasso Regression":
        model = Lasso(alpha=params.get('alpha', 1.0))
    elif model_type == "Elastic Net":
        model = ElasticNet(alpha=params.get('alpha', 1.0), 
                          l1_ratio=params.get('l1_ratio', 0.5))
    elif model_type == "Polynomial Regression":
        poly_degree = params.get('poly_degree', 2)
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ('regression', LinearRegression())
        ])
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42
        )
    elif model_type == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            random_state=42
        )
    elif model_type == "Logistic Regression":
        model = LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 1000),
            random_state=42
        )
    elif model_type == "Gradient Boosting Classifier":
        model = GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 3),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=42
        )
    
    # Create pipeline
    if model_type != "Polynomial Regression":  # Poly already has a pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
    else:
        # For polynomial regression, preprocessor is already included
        pipeline = model
        
    # Train model
    pipeline.fit(X, y)
    
    return pipeline

# Helper functions
def perform_exploratory_analysis(data):
    """Perform exploratory data analysis"""
    
    # Basic statistics
    st.subheader("Summary Statistics")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.dataframe(data.describe())
    
    with col2:
        st.write("Dataset Information")
        st.write(f"Rows: {data.shape[0]}")
        st.write(f"Columns: {data.shape[1]}")
        st.write(f"Missing Values: {data.isnull().sum().sum()}")
        
        # Data types distribution
        dtypes_count = data.dtypes.value_counts().to_dict()
        st.write("Data Types:")
        for dtype, count in dtypes_count.items():
            st.write(f"- {dtype}: {count}")
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    # Only include numeric columns for correlation
    numeric_data = data.select_dtypes(include=np.number)
    if len(numeric_data.columns) > 1:
        corr_matrix = compute_correlation_matrix(numeric_data)
        
        # Interactive correlation heatmap with Plotly
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        fig.update_layout(
            height=600,
            width=800,
            title="Correlation Matrix Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show strongest correlations
        st.subheader("Top Correlations")
        
        # Calculate absolute correlations and get top pairs
        corr_unstack = corr_matrix.unstack().sort_values(ascending=False)
        # Remove self correlations (which are always 1.0)
        corr_unstack = corr_unstack[corr_unstack < 1.0]
        top_corr = corr_unstack.head(10)
        
        # Display as a table
        top_corr_df = pd.DataFrame({'Correlation': top_corr})
        top_corr_df.index.names = ['Variable 1', 'Variable 2']
        st.dataframe(top_corr_df.reset_index())
    else:
        st.info("Need at least two numeric columns to calculate correlations.")
    
    # Distribution plots
    st.subheader("Distribution Analysis")
    
    selected_var = st.selectbox(
        "Select variable to visualize distribution:",
        options=numeric_data.columns.tolist()
    )
    
    if selected_var:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram with KDE
            fig = px.histogram(
                data, x=selected_var, 
                marginal="box", 
                opacity=0.7,
                nbins=30,
                title=f"Distribution of {selected_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Boxplot for outlier detection
            fig = px.box(
                data, y=selected_var,
                title=f"Boxplot of {selected_var}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Distribution statistics
        st.subheader(f"Statistics for {selected_var}")
        stats_df = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                data[selected_var].mean(),
                data[selected_var].median(),
                data[selected_var].std(),
                data[selected_var].min(),
                data[selected_var].max(),
                stats.skew(data[selected_var].dropna()),
                stats.kurtosis(data[selected_var].dropna())
            ]
        })
        st.dataframe(stats_df)
        
        # Normality test
        st.subheader("Normality Test")
        # Shapiro-Wilk test
        sample = data[selected_var].dropna()
        # Limit to 5000 samples for the test
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        
        stat, p_value = stats.shapiro(sample)
        
        if p_value < 0.05:
            st.warning(f"The distribution of {selected_var} is likely not normal (p-value: {p_value:.4f})")
        else:
            st.success(f"The distribution of {selected_var} is likely normal (p-value: {p_value:.4f})")
    
    # Scatter plot matrix for relationships
    st.subheader("Relationship Explorer")
    
    if len(numeric_data.columns) > 1:
        # Allow selecting multiple variables for scatter plot matrix
        scatter_vars = st.multiselect(
            "Select variables to explore relationships:",
            options=numeric_data.columns.tolist(),
            default=numeric_data.columns[:3].tolist() if len(numeric_data.columns) >= 3 else numeric_data.columns[:2].tolist()
        )
        
        if len(scatter_vars) >= 2:
            # Create scatter plot matrix with plotly
            fig = px.scatter_matrix(
                data, 
                dimensions=scatter_vars,
                opacity=0.7,
                title="Scatter Plot Matrix"
            )
            fig.update_layout(height=700)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least two variables to create a scatter plot matrix.")
    else:
        st.info("Need at least two numeric columns to explore relationships.")
            
def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values in the dataset"""
    if data.isnull().sum().sum() == 0:
        st.session_state.missing_values_handled = True
        return data
    
    st.info("🔍 Dataset contains missing values. Please choose how to handle them.")
    
    # Show missing value statistics
    missing_stats = pd.DataFrame({
        'Column': data.columns,
        'Missing Values': data.isnull().sum(),
        'Percentage': data.isnull().sum() / len(data) * 100
    }).sort_values('Missing Values', ascending=False)
    
    st.dataframe(missing_stats)
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        method = st.radio(
            "Imputation method:",
            ["Drop rows with missing values", 
             "Fill with mean (numeric only)",
             "Fill with median (numeric only)",
             "Fill with mode",
             "Fill with constant value"],
            help="Select how you want to handle missing values in your dataset."
        )
    
    fill_value = None
    if method == "Fill with constant value":
        with col2:
            fill_value = st.text_input("Enter constant value to use:", "0")
            try:
                fill_value = float(fill_value) if '.' in fill_value else int(fill_value)
            except ValueError:
                pass  # Keep as string if not convertible to number
    
    with col3:
        apply_button = st.button("Apply Missing Value Treatment", use_container_width=True)
        
    if apply_button:
        with st.spinner("Processing..."):
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
            st.balloons()
            st.success("Missing values have been handled successfully!")
            # Increment progress step
            st.session_state.current_step = 2
            # Force rerun to update UI
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
    
    st.info("🔄 You can optionally apply scaling to continuous variables.")
    
    scaling_method = st.radio(
        "Select scaling method:",
        ["None", "Standardization (Z-score)", 
         "Normalization (Min-Max)", "Robust Scaling"],
        index=0,
        help="Standardization: Transforms features to have mean=0 and variance=1.\n"
             "Normalization: Scales features to a range between 0 and 1.\n"
             "Robust Scaling: Scales using median and interquartile range, robust to outliers."
    )
    
    if scaling_method == "None":
        st.session_state.scaling_applied = True
        st.success("No scaling applied. Proceeding with original data.")
        # Increment progress step
        st.session_state.current_step = 3
        # Force rerun to update UI
        st.experimental_rerun()
        return data
    
    # Show preview of how scaling affects the data
    if st.checkbox("Preview scaling effect"):
        preview_var = st.selectbox(
            "Select a variable to preview scaling:",
            options=continuous_vars
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Original distribution
            fig = px.histogram(
                data, x=preview_var,
                title=f"Original Distribution: {preview_var}",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Scaled distribution
            if scaling_method == "Standardization (Z-score)":
                scaler = StandardScaler()
            elif scaling_method == "Normalization (Min-Max)":
                scaler = MinMaxScaler()
            elif scaling_method == "Robust Scaling":
                scaler = RobustScaler()
                
            scaled_values = scaler.fit_transform(data[[preview_var]])
            scaled_df = pd.DataFrame({f"Scaled {preview_var}": scaled_values.flatten()})
            
            fig = px.histogram(
                scaled_df, x=f"Scaled {preview_var}",
                title=f"Scaled Distribution: {preview_var}",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)
    
    apply_scaling_button = st.button("Apply Scaling", use_container_width=True)
    
    if apply_scaling_button:
        with st.spinner("Applying scaling..."):
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
            st.success(f"Applied {scaling_method} to continuous variables successfully!")
            # Increment progress step
            st.session_state.current_step = 3
            # Force rerun to update UI
            st.experimental_rerun()
    
    return data

def create_diagnostic_plots(X, y, y_pred, model_type='regression'):
    """Create diagnostic plots for model evaluation using Plotly"""
    if model_type == 'regression':
        residuals = y - y_pred
        
        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Residuals vs Fitted", 
                "Normal Q-Q", 
                "Scale-Location", 
                "Actual vs Predicted"
            )
        )
        
        # 1. Residuals vs Fitted
        fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals,
                mode='markers',
                marker=dict(opacity=0.7),
                name='Residuals'
            ),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_trace(
            go.Scatter(
                x=[min(y_pred), max(y_pred)],
                y=[0, 0],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Zero Line'
            ),
            row=1, col=1
        )
        
        # 2. QQ Plot (manually calculate quantiles)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.arange(0.01, 1, 0.01))
        sorted_residuals = np.sort(residuals)
        sampled_residuals = np.quantile(sorted_residuals, np.arange(0.01, 1, 0.01))
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, 
                y=sampled_residuals,
                mode='markers',
                marker=dict(opacity=0.7),
                name='QQ Plot'
            ),
            row=1, col=2
        )
        
        # Add reference line
        min_val = min(min(theoretical_quantiles), min(sampled_residuals))
        max_val = max(max(theoretical_quantiles), max(sampled_residuals))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='Reference Line'
            ),
            row=1, col=2
        )
        
        # 3. Scale-Location
        sqrt_residuals = np.sqrt(np.abs(residuals))
        fig.add_trace(
            go.Scatter(
                x=y_pred, 
                y=sqrt_residuals,
                mode='markers',
                marker=dict(opacity=0.7),
                name='Sqrt(|Residuals|)'
            ),
            row=2, col=1
        )
        
        # 4. Actual vs Predicted
        fig.add_trace(
            go.Scatter(
                x=y, 
                y=y_pred,
                mode='markers',
                marker=dict(opacity=0.7),
                name='Predictions'
            ),
            row=2, col=2
        )
        
        # Add reference line
        min_val = min(min(y), min(y_pred))
        max_val = max(max(y), max(y_pred))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name='y=x'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Regression Diagnostic Plots",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Fitted Values", row=1, col=1)
        fig.update_yaxes(title_text="Residuals", row=1, col=1)
        
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles", row=1, col=2)
        
        fig.update_xaxes(title_text="Fitted Values", row=2, col=1)
        fig.update_yaxes(title_text="Sqrt(|Residuals|)", row=2, col=1)
        
        fig.update_xaxes(title_text="Actual Values", row=2, col=2)
        fig.update_yaxes(title_text="Predicted Values", row=2, col=2)
        
    else:  # Classification
        # For classification, create different diagnostic plots
        
        # Create subplots: 2 rows, 2 columns
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "ROC Curve", 
                "Precision-Recall Curve", 
                "Confusion Matrix", 
                "Prediction Distribution"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "heatmap"}, {"type": "histogram"}]
            ]
        )
        
        # 1. ROC Curve
        if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Multi-class case - use first class probability for simplicity
            y_score = y_pred[:, 1] if y_pred.shape[1] > 1 else y_pred
        else:
            y_score = y_pred
            
        fpr, tpr, _ = roc_curve(y, y_score)
        roc_auc = roc_auc_score(y, y_score)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                line=dict(color='darkorange', width=2),
                name=f'ROC curve (area = {roc_auc:.2f})'
            ),
            row=1, col=1
        )
        
        # Add diagonal reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(color='navy', dash='dash'),
                name='Random'
            ),
            row=1, col=1
        )
        
        # 2. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y, y_score)
        
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                mode='lines',
                line=dict(color='green', width=2),
                name='Precision-Recall curve'
            ),
            row=1, col=2
        )
        
        # 3. Confusion Matrix
        cm = confusion_matrix(y, np.round(y_score))
        
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=['Predicted 0', 'Predicted 1'],
                y=['Actual 0', 'Actual 1'],
                colorscale='Blues',
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                textfont={"size":14}
            ),
            row=2, col=1
        )
        
        # 4. Prediction Distribution
        fig.add_trace(
            go.Histogram(
                x=y_score,
                nbinsx=30,
                marker_color='rgba(0, 0, 255, 0.5)',
                name='Class 0'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1000,
            title_text="Classification Diagnostic Plots",
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig.update_xaxes(title_text="Predicted", row=2, col=1)
        fig.update_yaxes(title_text="Actual", row=2, col=1)
        
        fig.update_xaxes(title_text="Prediction Score", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)

def feature_importance_plot(model, X, feature_names):
    """Create interactive feature importance plot for supported models using Plotly"""
    importance = None
    
    if hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
        if len(importance.shape) > 1 and importance.shape[0] > 1:  # Multi-class case
            importance = importance.mean(axis=0)
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    
    if importance is not None:
        # Create DataFrame for the plot
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Create bar plot with Plotly
        fig = px.bar(
            features_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Feature Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=600,
            width=800,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance not available for this model type.")

def plot_learning_curve(model, X, y, cv=5):
    """Plot learning curve to analyze model performance with varying training set sizes"""
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error',
        random_state=42
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    fig = go.Figure()
    
    # Add training score line
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode='lines+markers',
            name='Training score',
            line=dict(color='blue')
        )
    )
    
    # Add training score confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([
                train_scores_mean - train_scores_std,
                (train_scores_mean + train_scores_std)[::-1]
            ]),
            fill='toself',
            fillcolor='rgba(0, 0, 255, 0.1)',
            line=dict(color='rgba(0, 0, 255, 0)'),
            showlegend=False
        )
    )
    
    # Add test score line
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode='lines+markers',
            name='Cross-validation score',
            line=dict(color='green')
        )
    )
    
    # Add test score confidence interval
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([train_sizes, train_sizes[::-1]]),
            y=np.concatenate([
                test_scores_mean - test_scores_std,
                (test_scores_mean + test_scores_std)[::-1]
            ]),
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(0, 255, 0, 0)'),
            showlegend=False
        )
    )
    
    fig.update_layout(
        title='Learning Curve',
        xaxis_title='Training examples',
        yaxis_title='Mean Squared Error',
        height=500,
        width=800
    )
    
    st.plotly_chart(fig, use_container_width=True)

def get_download_link(data, filename="processed_data.csv", text="Download processed data"):
    """Generate a download link for dataframe"""
    csv = data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def download_model(model, filename="trained_model.pkl"):
    """Create a download link for the trained model"""
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">Download trained model</a>'
    return href

def main():
    st.title("📊 Advanced Regression Analysis App")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Determine which step the user can access based on progress
    if st.session_state.current_step >= 1:
        data_tab = st.sidebar.checkbox("1. Data Selection & Exploration", value=True)
    else:
        data_tab = True
        
    if st.session_state.current_step >= 2:
        preprocessing_tab = st.sidebar.checkbox("2. Preprocessing", value=False)
    else:
        preprocessing_tab = False
        
    if st.session_state.current_step >= 3:
        modeling_tab = st.sidebar.checkbox("3. Model Building", value=False)
    else:
        modeling_tab = False
        
    if st.session_state.model_trained:
        evaluation_tab = st.sidebar.checkbox("4. Evaluation & Interpretation", value=False)
    else:
        evaluation_tab = False
    
    # 1. DATA SELECTION & EXPLORATION
    if data_tab:
        st.header("1️⃣ Data Selection & Exploration")
        
        # Data upload options
        data_upload_method = st.radio(
            "Choose data source:",
            ["Upload CSV", "Upload Excel file", "Use sample dataset"]
        )
        
        # Handle data upload
        if data_upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload a CSV file:", type=["csv"])
            if uploaded_file is not None:
                try:
                    data = pd.read_csv(uploaded_file)
                    st.session_state.data = data
                    st.session_state.processed_data = data.copy()
                    st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading CSV file: {e}")
        
        elif data_upload_method == "Upload Excel file":
            uploaded_file = st.file_uploader("Upload an Excel file:", type=["xlsx", "xls"])
            if uploaded_file is not None:
                try:
                    data = pd.read_excel(uploaded_file)
                    st.session_state.data = data
                    st.session_state.processed_data = data.copy()
                    st.success(f"Successfully loaded data with {data.shape[0]} rows and {data.shape[1]} columns.")
                except Exception as e:
                    st.error(f"Error loading Excel file: {e}")
        
        else:  # Use sample dataset
            sample_dataset = st.selectbox(
                "Choose a sample dataset:",
                ["Boston Housing", "Diabetes", "Wine Quality", "Iris (Classification)"]
            )
            
            if st.button("Load Sample Dataset"):
                with st.spinner("Loading sample dataset..."):
                    data = load_sample_dataset(sample_dataset)
                    st.session_state.data = data
                    st.session_state.processed_data = data.copy()
                    st.success(f"Successfully loaded {sample_dataset} dataset with {data.shape[0]} rows and {data.shape[1]} columns.")
        
        # If data is loaded, show exploration options
        if st.session_state.data is not None:
            data = st.session_state.data
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(data.head())
            
            # Show column information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", data.shape[0])
            with col2:
                st.metric("Columns", data.shape[1])
            with col3:
                st.metric("Missing Values", data.isnull().sum().sum())
            
            # Exploration options
            if st.checkbox("Show detailed column information"):
                col_info = pd.DataFrame({
                    'Data Type': data.dtypes,
                    'Non-Null Count': data.count(),
                    'Null Count': data.isnull().sum(),
                    'Unique Values': [len(data[col].unique()) for col in data.columns]
                })
                st.dataframe(col_info)
            
            if st.checkbox("Show full exploratory data analysis"):
                perform_exploratory_analysis(data)
            
            # Auto-detect target variable (if not already selected)
            if st.button("Proceed to Preprocessing"):
                st.session_state.current_step = 2
                st.experimental_rerun()
    
    # 2. PREPROCESSING
    if preprocessing_tab:
        st.header("2️⃣ Data Preprocessing")
        
        if st.session_state.data is None:
            st.warning("Please load data first in the Data Selection step.")
            return
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        # Missing values treatment
        st.subheader("Missing Values Treatment")
        if not st.session_state.missing_values_handled:
            data = handle_missing_values(data)
        else:
            st.success("✅ Missing values have been handled.")
        
        # Feature scaling 
        st.subheader("Feature Scaling")
        if st.session_state.missing_values_handled and not st.session_state.scaling_applied:
            data = apply_feature_scaling(data)
        elif st.session_state.scaling_applied:
            st.success("✅ Feature scaling has been applied.")
        
        # If all preprocessing steps are complete, show button to move to modeling
        if st.session_state.missing_values_handled and st.session_state.scaling_applied:
            if st.button("Proceed to Model Building"):
                st.session_state.current_step = 3
                st.experimental_rerun()
    
    # 3. MODEL BUILDING
    if modeling_tab:
        st.header("3️⃣ Model Building")
        
        if st.session_state.processed_data is None:
            st.warning("Please complete the preprocessing steps first.")
            return
        
        data = st.session_state.processed_data
        
        # Target variable selection
        st.subheader("Select Target Variable")
        target_variable = st.selectbox(
            "Choose the target variable to predict:",
            options=data.columns.tolist()
        )
        
        # Feature selection
        st.subheader("Feature Selection")
        
        all_features = [col for col in data.columns if col != target_variable]
        
        feature_selection_method = st.radio(
            "Feature selection method:",
            ["Manual selection", "All features", "Automatic feature selection"]
        )
        
        if feature_selection_method == "Manual selection":
            selected_features = st.multiselect(
                "Select features to include in the model:",
                options=all_features,
                default=all_features
            )
        elif feature_selection_method == "All features":
            selected_features = all_features
            st.info(f"Using all {len(selected_features)} available features.")
        else:  # Automatic feature selection
            # Determine if regression or classification task
            unique_targets = data[target_variable].nunique()
            is_regression = unique_targets > 10 or np.issubdtype(data[target_variable].dtype, np.number)
            
            auto_selection_method = st.selectbox(
                "Choose automatic feature selection method:",
                ["Select K Best", "Recursive Feature Selection"]
            )
            
            k_features = st.slider(
                "Number of features to select:",
                min_value=1,
                max_value=len(all_features),
                value=min(5, len(all_features))
            )
            
            if st.button("Run Feature Selection"):
                with st.spinner("Selecting best features..."):
                    X = data[all_features]
                    y = data[target_variable]
                    
                    if auto_selection_method == "Select K Best":
                        if is_regression:
                            selector = SelectKBest(score_func=f_regression, k=k_features)
                        else:
                            from sklearn.feature_selection import f_classif
                            selector = SelectKBest(score_func=f_classif, k=k_features)
                    else:  # Recursive
                        if is_regression:
                            estimator = LinearRegression()
                        else:
                            estimator = LogisticRegression(max_iter=1000)
                            
                        selector = SFS(
                            estimator,
                            k_features=k_features,
                            forward=True,
                            floating=False,
                            scoring='r2' if is_regression else 'accuracy',
                            cv=5
                        )
                    
                    # Handle different selector APIs
                    if auto_selection_method == "Select K Best":
                        selector.fit(X, y)
                        selected_mask = selector.get_support()
                        selected_features = [all_features[i] for i in range(len(all_features)) if selected_mask[i]]
                    else:
                        selector.fit(X, y)
                        selected_features = [all_features[i] for i in selector.k_feature_idx_]
                    
                    st.success(f"Selected {len(selected_features)} best features.")
            else:
                selected_features = all_features[:k_features]
                st.info(f"Using first {k_features} features by default. Click 'Run Feature Selection' to find optimal features.")
        
        # Show selected features
        st.write("Selected Features:", selected_features)
        
        # Model selection and configuration
        st.subheader("Model Configuration")
        
        # Determine if regression or classification task
        unique_targets = data[target_variable].nunique()
        is_classification = unique_targets <= 10 and not np.issubdtype(data[target_variable].dtype, np.float64)
        
        if is_classification:
            st.info(f"Detected classification task with {unique_targets} classes.")
            model_types = ["Logistic Regression", "Gradient Boosting Classifier"]
        else:
            st.info("Detected regression task.")
            model_types = [
                "Linear Regression", 
                "Ridge Regression", 
                "Lasso Regression", 
                "Elastic Net", 
                "Polynomial Regression",
                "Gradient Boosting",
                "Random Forest"
            ]
        
        model_type = st.selectbox(
            "Select model type:",
            options=model_types
        )
        
        # Show hyperparameters based on model type
        st.subheader("Hyperparameters")
        
        params = {}
        if model_type == "Ridge Regression":
            params['alpha'] = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0)
        elif model_type == "Lasso Regression":
            params['alpha'] = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0)
        elif model_type == "Elastic Net":
            params['alpha'] = st.slider("Regularization strength (alpha):", 0.01, 10.0, 1.0)
            params['l1_ratio'] = st.slider("L1 ratio:", 0.0, 1.0, 0.5)
        elif model_type == "Polynomial Regression":
            params['poly_degree'] = st.slider("Polynomial degree:", 2, 5, 2)
        elif model_type == "Gradient Boosting" or model_type == "Gradient Boosting Classifier":
            params['n_estimators'] = st.slider("Number of trees:", 50, 500, 100)
            params['max_depth'] = st.slider("Maximum tree depth:", 2, 10, 3)
            params['learning_rate'] = st.slider("Learning rate:", 0.01, 0.3, 0.1)
        elif model_type == "Random Forest":
            params['n_estimators'] = st.slider("Number of trees:", 50, 500, 100)
            params['max_depth'] = st.slider("Maximum tree depth:", 2, 20, 10, 1)
        elif model_type == "Logistic Regression":
            params['C'] = st.slider("Inverse of regularization strength (C):", 0.01, 10.0, 1.0)
            params['max_iter'] = st.slider("Maximum iterations:", 100, 2000, 1000, 100)
            
        # Train-test split configuration
        st.subheader("Train-Test Split")
        test_size = st.slider("Test set size (%):", 10, 40, 20) / 100
        random_state = st.number_input("Random state:", 0, 100, 42)
        
        # Cross-validation configuration
        st.subheader("Cross-Validation")
        cv_folds = st.slider("Number of CV folds:", 2, 10, 5)
        
        # Train model button
        if st.button("Train Model", key="train_model_button"):
            with st.spinner(f"Training {model_type}..."):
                start_time = time.time()
                
                # Prepare data
                X = data[selected_features]
                y = data[target_variable]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Train model using pipeline
                model_pipeline = train_model_pipeline(
                    X_train, y_train, model_type, params, cv=cv_folds
                )
                
                # Evaluate model
                y_pred_train = model_pipeline.predict(X_train)
                y_pred_test = model_pipeline.predict(X_test)
                
                # Calculate metrics
                if not is_classification:
                    train_mse = mean_squared_error(y_train, y_pred_train)
                    test_mse = mean_squared_error(y_test, y_pred_test)
                    train_r2 = r2_score(y_train, y_pred_train)
                    test_r2 = r2_score(y_test, y_pred_test)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model_pipeline, X, y, 
                        cv=cv_folds, 
                        scoring='neg_mean_squared_error'
                    )
                    cv_mse = -cv_scores.mean()
                    cv_mse_std = cv_scores.std()
                    
                    metrics = {
                        'Training MSE': train_mse,
                        'Test MSE': test_mse,
                        'Training R²': train_r2,
                        'Test R²': test_r2,
                        'CV MSE': cv_mse,
                        'CV MSE Std': cv_mse_std
                    }
                else:
                    # Classification metrics
                    train_acc = accuracy_score(y_train, np.round(y_pred_train))
                    test_acc = accuracy_score(y_test, np.round(y_pred_test))
                    
                    # Get probabilities if model supports it
                    if hasattr(model_pipeline, "predict_proba"):
                        y_prob_test = model_pipeline.predict_proba(X_test)[:, 1]
                        roc_auc = roc_auc_score(y_test, y_prob_test)
                    else:
                        roc_auc = None
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(
                        model_pipeline, X, y, 
                        cv=cv_folds, 
                        scoring='accuracy'
                    )
                    cv_acc = cv_scores.mean()
                    cv_acc_std = cv_scores.std()
                    
                    metrics = {
                        'Training Accuracy': train_acc,
                        'Test Accuracy': test_acc,
                        'ROC AUC': roc_auc,
                        'CV Accuracy': cv_acc,
                        'CV Accuracy Std': cv_acc_std
                    }
                
                # Save results in session state
                st.session_state.model_trained = True
                st.session_state.model_results = {
                    'model': model_pipeline,
                    'model_type': model_type,
                    'is_classification': is_classification,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'y_pred_train': y_pred_train,
                    'y_pred_test': y_pred_test,
                    'metrics': metrics,
                    'selected_features': selected_features,
                    'target_variable': target_variable,
                    'training_time': time.time() - start_time
                }
                
                # Add this model to comparison
                model_name = f"{model_type}"
                st.session_state.models_to_compare[model_name] = {
                    'metrics': metrics,
                    'model_type': model_type,
                    'is_classification': is_classification
                }
                
                st.balloons()
                st.success(f"Model training completed in {time.time() - start_time:.2f} seconds!")
                # Force rerun to update UI
                st.experimental_rerun()
    
    # 4. EVALUATION
    if evaluation_tab:
        st.header("4️⃣ Model Evaluation & Interpretation")
        
        if not st.session_state.model_trained:
            st.warning("Please train a model first.")
            return
        
        results = st.session_state.model_results
        
        # Display model information
        st.subheader("Model Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Model Type:** {results['model_type']}")
            st.write(f"**Target Variable:** {results['target_variable']}")
            st.write(f"**Features Used:** {len(results['selected_features'])}")
            st.write(f"**Training Time:** {results['training_time']:.2f} seconds")
        
        with col2:
            # Display key metrics
            metrics = results['metrics']
            for metric, value in metrics.items():
                if value is not None:
                    st.metric(metric, f"{value:.4f}")
        
        # Show detailed performance metrics
        st.subheader("Performance Metrics")
        
        if not results['is_classification']:
            # Regression metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Metrics**")
                st.write(f"Mean Squared Error: {metrics['Training MSE']:.4f}")
                st.write(f"R²: {metrics['Training R²']:.4f}")
                st.write(f"RMSE: {np.sqrt(metrics['Training MSE']):.4f}")
            
            with col2:
                st.write("**Test Metrics**")
                st.write(f"Mean Squared Error: {metrics['Test MSE']:.4f}")
                st.write(f"R²: {metrics['Test R²']:.4f}")
                st.write(f"RMSE: {np.sqrt(metrics['Test MSE']):.4f}")
                
            st.write("**Cross-Validation Metrics**")
            st.write(f"Mean MSE: {metrics['CV MSE']:.4f} (±{metrics['CV MSE Std']:.4f})")
            st.write(f"Mean RMSE: {np.sqrt(metrics['CV MSE']):.4f}")
        else:
            # Classification metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Training Metrics**")
                st.write(f"Accuracy: {metrics['Training Accuracy']:.4f}")
            
            with col2:
                st.write("**Test Metrics**")
                st.write(f"Accuracy: {metrics['Test Accuracy']:.4f}")
                if metrics['ROC AUC'] is not None:
                    st.write(f"ROC AUC: {metrics['ROC AUC']:.4f}")
            
            st.write("**Cross-Validation Metrics**")
            st.write(f"Mean Accuracy: {metrics['CV Accuracy']:.4f} (±{metrics['CV Accuracy Std']:.4f})")
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(
                results['y_test'], 
                np.round(results['y_pred_test']),
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(
                results['y_test'], 
                np.round(results['y_pred_test'])
            )
            
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['0', '1'],
                y=['0', '1'],
                text_auto=True,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Diagnostic plots
        st.subheader("Diagnostic Plots")
        
        create_diagnostic_plots(
            results['X_test'], 
            results['y_test'], 
            results['y_pred_test'],
            'classification' if results['is_classification'] else 'regression'
        )
        
        # Feature importance
        st.subheader("Feature Importance")
        
        feature_importance_plot(
            results['model'][-1] if isinstance(results['model'], Pipeline) else results['model'],
            results['X_train'],
            results['selected_features']
        )
        
        # Learning curve
        st.subheader("Learning Curve")
        
        plot_learning_curve(
            results['model'],
            results['X_train'],
            results['y_train']
        )
        
        # Model comparison
        st.subheader("Model Comparison")
        
        if len(st.session_state.models_to_compare) > 0:
            # Create a dataframe to compare models
            comparison_data = []
            
            for model_name, model_data in st.session_state.models_to_compare.items():
                model_metrics = model_data['metrics']
                model_row = {'Model': model_name}
                
                # Add metrics to the row
                for metric, value in model_metrics.items():
                    if value is not None:
                        model_row[metric] = value
                
                comparison_data.append(model_row)
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df.style.highlight_max(subset=[col for col in comparison_df.columns if col != 'Model']))
            
            # Create comparison bar chart
            if not results['is_classification']:
                # For regression models
                fig = make_subplots(rows=1, cols=2, subplot_titles=("MSE Comparison", "R² Comparison"))
                
                # MSE comparison
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['Model'],
                        y=comparison_df['Test MSE'],
                        name='Test MSE',
                        marker_color='indianred'
                    ),
                    row=1, col=1
                )
                
                # R² comparison
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['Model'],
                        y=comparison_df['Test R²'],
                        name='Test R²',
                        marker_color='lightseagreen'
                    ),
                    row=1, col=2
                )
            else:
                # For classification models
                fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy Comparison", "ROC AUC Comparison"))
                
                # Accuracy comparison
                fig.add_trace(
                    go.Bar(
                        x=comparison_df['Model'],
                        y=comparison_df['Test Accuracy'],
                        name='Test Accuracy',
                        marker_color='indianred'
                    ),
                    row=1, col=1
                )
                
                # ROC AUC comparison if available
                if 'ROC AUC' in comparison_df.columns:
                    fig.add_trace(
                        go.Bar(
                            x=comparison_df['Model'],
                            y=comparison_df['ROC AUC'],
                            name='ROC AUC',
                            marker_color='lightseagreen'
                        ),
                        row=1, col=2
                    )
            
            fig.update_layout(height=400, width=800)
            st.plotly_chart(fig, use_container_width=True)
        
        # Download options
        st.subheader("Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(get_download_link(
                st.session_state.processed_data,
                "processed_data.csv",
                "📥 Download processed data"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(download_model(
                results['model'],
                f"{results['model_type'].replace(' ', '_').lower()}_model.pkl"
            ), unsafe_allow_html=True)
            
        # Prediction on new data
        st.subheader("Make Predictions")
        
        if st.checkbox("Make predictions on new data"):
            st.write("Enter values for prediction:")
            
            # Create input fields for each feature
            feature_values = {}
            for feature in results['selected_features']:
                data = st.session_state.processed_data
                if np.issubdtype(data[feature].dtype, np.number):
                    # For numeric features
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                  # For numeric features
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    feature_values[feature] = st.slider(
                        f"{feature}",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val
                    )
                else:
                    # For categorical features
                    unique_values = data[feature].unique().tolist()
                    feature_values[feature] = st.selectbox(
                        f"{feature}",
                        options=unique_values,
                        index=0
                    )
            
            # Make prediction button
            if st.button("Predict"):
                # Create a DataFrame with the input values
                input_data = pd.DataFrame([feature_values])
                
                # Make prediction
                prediction = results['model'].predict(input_data)
                
                st.success(f"Predicted {results['target_variable']}: {prediction[0]:.4f}")
                
                # Show confidence interval for certain models if possible
                model = results['model']
                if hasattr(model, 'predict_proba') and results['is_classification']:
                    proba = model.predict_proba(input_data)
                    st.write(f"Prediction probability: {proba[0][1]:.4f}")
            
            # Batch prediction option
            st.subheader("Batch Prediction")
            uploaded_file = st.file_uploader("Upload a CSV file for batch prediction:", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    batch_data = pd.read_csv(uploaded_file)
                    
                    # Check if all required features are present
                    missing_features = [f for f in results['selected_features'] if f not in batch_data.columns]
                    
                    if missing_features:
                        st.error(f"Missing features in uploaded file: {', '.join(missing_features)}")
                    else:
                        # Make predictions
                        batch_predictions = results['model'].predict(batch_data[results['selected_features']])
                        
                        # Add predictions to the data
                        batch_data['Predicted_' + results['target_variable']] = batch_predictions
                        
                        # Display results
                        st.dataframe(batch_data)
                        
                        # Download predictions
                        st.markdown(get_download_link(
                            batch_data,
                            "predictions.csv",
                            "📥 Download predictions"
                        ), unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Error processing batch prediction: {e}")
    
    # Footer information
    st.sidebar.subheader("About")
    st.sidebar.info(
        """
        **Advanced Regression Analysis App** v2.0
        
        This tool provides a comprehensive workflow for:
        - Data exploration and visualization
        - Preprocessing and feature engineering
        - Model building with multiple algorithms
        - Performance evaluation and interpretation
        
        Created using Streamlit, scikit-learn, and Plotly.
        """
    )
    
    # Reset option
    if st.sidebar.button("Reset Application"):
        # Reset all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
      
