import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import statsmodels.api as sm

# App title
st.title("📊 Regression Analysis App")

# Upload data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Data Preview")
    st.dataframe(data.head())

    # Variable selection
    regression_type = st.selectbox("Select the type of regression:", [
        "Simple Linear Regression",
        "Multiple Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Elastic Net Regression",
        "Logistic Regression",
        "Gradient Boosting",
        "Stepwise Regression",
    ])

    y_var = st.selectbox("Select the dependent (Y) variable:", data.columns)
    x_vars = st.multiselect("Select the independent (X) variable(s):", [col for col in data.columns if col != y_var])

    # Detect binary target for logistic regression
    if regression_type == "Logistic Regression":
        y_unique = data[y_var].nunique()
        if y_unique != 2:
            st.warning(f"The selected Y variable has {y_unique} unique values. Logistic regression requires a binary outcome.")

    # Run regression when ready
    if st.button("Run Regression"):
        X = data[x_vars]
        y = data[y_var]

        # Standardize features for regularized regressions
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if regression_type == "Simple Linear Regression" and len(x_vars) == 1:
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write(f"**Intercept:** {model.intercept_:.4f}")
            st.write(f"**Coefficient:** {model.coef_[0]:.4f}")
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Multiple Linear Regression":
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write(f"**Intercept:** {model.intercept_:.4f}")
            st.write(f"**Coefficients:** {dict(zip(x_vars, model.coef_))}")
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Polynomial Regression":
            degree = st.slider("Select polynomial degree:", 2, 5, 2)
            poly = PolynomialFeatures(degree)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            y_pred = model.predict(X_poly)
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Ridge Regression":
            alpha = st.slider("Select alpha (L2 penalty):", 0.01, 10.0, 1.0)
            model = Ridge(alpha=alpha)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Lasso Regression":
            alpha = st.slider("Select alpha (L1 penalty):", 0.01, 10.0, 1.0)
            model = Lasso(alpha=alpha)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Elastic Net Regression":
            alpha = st.slider("Alpha (penalty strength):", 0.01, 10.0, 1.0)
            l1_ratio = st.slider("L1 ratio (0 = Ridge, 1 = Lasso):", 0.0, 1.0, 0.5)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Gradient Boosting":
            model = GradientBoostingRegressor()
            model.fit(X, y)
            y_pred = model.predict(X)
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        elif regression_type == "Logistic Regression":
            model = LogisticRegression()
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            acc = accuracy_score(y, y_pred)
            cm = confusion_matrix(y, y_pred)
            st.write(f"**Accuracy:** {acc:.4f}")
            st.write("**Confusion Matrix:**")
            st.write(cm)

        elif regression_type == "Stepwise Regression":
            direction = st.radio("Stepwise direction:", ["forward", "backward"])
            lin_model = LinearRegression()
            sfs = SFS(lin_model,
                     k_features='best',
                     forward=True if direction == "forward" else False,
                     floating=False,
                     scoring='r2',
                     cv=0)
            sfs.fit(X_scaled, y)
            selected_feat = list(sfs.k_feature_names_)
            model = LinearRegression()
            model.fit(X[selected_feat], y)
            y_pred = model.predict(X[selected_feat])
            st.write(f"**Selected Features:** {selected_feat}")
            st.write(f"**R²:** {r2_score(y, y_pred):.4f}")

        else:
            st.warning("Please ensure your selection and variables are appropriate for the model.")
