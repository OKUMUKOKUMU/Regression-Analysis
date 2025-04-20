                # Residual analysis
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
