import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from fpdf import FPDF
from io import BytesIO
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_ind, f_oneway, chi2_contingency, zscore
# Set up the app
st.set_page_config(page_title="InsightsDigger", page_icon="🔍")

st.title("🔍InsightsDigger🧐")
st.caption("Advanced Data Analysis and Visualization")


# File uploader for CSV
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read and display the dataset
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Data Preprocessing Section
    st.subheader("Data Preprocessing 🧠")

    # Handling Missing Values
    if st.checkbox("Handle Missing Values❓"):
        missing_values_option = st.selectbox("Choose Method to Handle Missing Values", ["Drop Rows", "Fill with Mean",
                                                                                        "Fill with Median", "Fill with Mode"])

        if missing_values_option == "Drop Rows":
            df = df.dropna()
            st.write("Dataset after dropping missing values:")
            st.dataframe(df.head())
        elif missing_values_option == "Fill with Mean":
            df = df.fillna(df.mean())
            st.write("Dataset after filling missing values with mean:")
            st.dataframe(df.head())
        elif missing_values_option == "Fill with Median":
            df = df.fillna(df.median())
            st.write("Dataset after filling missing values with median:")
            st.dataframe(df.head())
        elif missing_values_option == "Fill with Mode":
            df = df.fillna(df.mode().iloc[0])
            st.write("Dataset after filling missing values with mode:")
            st.dataframe(df.head())

    # Feature Scaling
    if st.checkbox("Apply Feature Scaling💡"):
        scale_method = st.selectbox("Select Scaling Method", ["Standard Scaling", "Min-Max Scaling"])

        if scale_method == "Standard Scaling":
            df_scaled = (df - df.mean()) / df.std()
            st.write("Scaled Data (Standard Scaling):")
            st.dataframe(df_scaled.head())
        elif scale_method == "Min-Max Scaling":
            df_scaled = (df - df.min()) / (df.max() - df.min())
            st.write("Scaled Data (Min-Max Scaling):")
            st.dataframe(df_scaled.head())

    # Principal Component Analysis (PCA)
    if st.checkbox("Apply PCA for Dimensionality Reduction 🧮"):
        df_numeric = df.select_dtypes(include=[np.number])
        df_scaled = (df_numeric - df_numeric.mean()) / df_numeric.std()
        covariance_matrix = np.cov(df_scaled.T)
        eigvals, eigvecs = np.linalg.eig(covariance_matrix)
        eigvals_sorted_idx = np.argsort(eigvals)[::-1]
        eigvecs_sorted = eigvecs[:, eigvals_sorted_idx]
        pca_result = np.dot(df_scaled, eigvecs_sorted[:, :2])
        df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        st.write("PCA Results:")
        st.dataframe(df_pca.head())

        # Plot PCA results
        fig = px.scatter(df_pca, x='PC1', y='PC2', title='PCA Scatter Plot')
        st.plotly_chart(fig)

    # Data Visualization Section
    st.subheader("Data Visualization 📊")
    visualization_type = st.selectbox(
    "Select Visualization Type",
    ["Bar Plot", "Histogram", "Box Plot", "Scatter Plot", "Line Plot","Correlation Heatmap"]
    )
    if visualization_type in ["Bar Plot", "Histogram", "Box Plot"]:
    # Univariate Visualizations (One Parameter)
        column = st.selectbox("Select Column", df.columns)
        if column:
            if visualization_type == "Bar Plot":
                st.write(f"Bar plot for {column}")
                fig = px.bar(df[column].value_counts(), title=f'Bar Plot of {column}')
                fig.update_layout(xaxis_title=column, yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)

            elif visualization_type == "Histogram":
                st.write(f"Histogram for {column}")
                fig = px.histogram(df, x=column, nbins=20, title=f'Histogram of {column}')
                fig.update_layout(xaxis_title=column, yaxis_title='Frequency')
                st.plotly_chart(fig, use_container_width=True)

            elif visualization_type == "Box Plot":
                st.write(f"Box plot for {column}")
                fig = px.box(df, y=column, title=f'Box Plot of {column}')
                fig.update_layout(yaxis_title=column)
                st.plotly_chart(fig, use_container_width=True)

    elif visualization_type in ["Scatter Plot", "Line Plot"]:
        # Bivariate Visualizations (Two Parameters)
        x_col = st.selectbox("Select X Column", df.columns)
        y_col = st.selectbox("Select Y Column", df.columns)
        if x_col and y_col:
            if visualization_type == "Scatter Plot":
                st.write(f"Scatter plot between {x_col} and {y_col}")
                fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot of {x_col} vs {y_col}')
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                st.plotly_chart(fig, use_container_width=True)

            elif visualization_type == "Line Plot":
                st.write(f"Line plot between {x_col} and {y_col}")
                fig = px.line(df, x=x_col, y=y_col, title=f'Line Plot of {x_col} vs {y_col}')
                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
                st.plotly_chart(fig, use_container_width=True)

    # Correlation Heatmap
    elif visualization_type in ["Correlation Heatmap"]:
        st.write("Correlation Heatmap:")
        plt.figure(figsize=(10, 6))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        st.pyplot(plt)

    # Data Summary Section
    st.subheader("Data Summary 📝")
    if st.checkbox("Show Data Summary 📋"):
        st.write("Summary Statistics:")
        st.dataframe(df.describe())
    
    if st.checkbox("Show Column Insights for Prediction 🌐"):
        target_column = st.selectbox("Select Target Column", options=df.columns)

    # Ensure target column selection
        if target_column:
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Handle categorical columns
            X = pd.get_dummies(X, drop_first=True)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a RandomForest model for feature importance
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Calculate feature importance
            feature_importances = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.write("Feature Importance:")
            st.dataframe(feature_importances)

            # Suggest dropping columns based on a threshold
            threshold = st.slider("Set Importance Threshold to Suggest Dropping", 0.0, 1.0, 0.01)
            drop_suggestions = feature_importances[feature_importances["Importance"] < threshold]["Feature"]

            st.write("Suggested Columns to Drop:")
            st.write(list(drop_suggestions))

            # Further analysis or visualization
            st.write("Further Analysis:")
            st.bar_chart(feature_importances.set_index("Feature"))
    
    if st.checkbox("Perform Hypothesis Testing 🔢"):
        st.write("Select Hypothesis Test:")
        test_type = st.radio("Test Type", ["t-test", "ANOVA", "Chi-Square"])

        if test_type == "t-test":
    # Select numeric column
            numeric_col = st.selectbox("Select Numeric Column", df.select_dtypes(include=[np.number]).columns)
            
            # Select categorical column with exactly two unique values
            categorical_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object', 'category']).columns)
            
            # Check if the categorical column has exactly two unique values
            unique_values = df[categorical_col].unique()
            if len(unique_values) != 2:
                st.warning("The selected categorical column must have exactly two unique values for a t-test.")
            else:
                # Split the numeric column into two groups
                group1 = df[df[categorical_col] == unique_values[0]][numeric_col]
                group2 = df[df[categorical_col] == unique_values[1]][numeric_col]
                
                # Perform t-test
                t_stat, p_val = ttest_ind(group1, group2)
                st.write(f"T-statistic: {t_stat}, P-value: {p_val}")
                
                # Interpret results
                if p_val < 0.05:
                    st.success("There is a significant difference between the two groups (p < 0.05).")
                else:
                    st.warning("There is no significant difference between the two groups (p >= 0.05).")

        elif test_type == "ANOVA":
            # Select numeric column
            numeric_col = st.selectbox("Select Numeric Column", df.select_dtypes(include=[np.number]).columns)
            
            # Select categorical column with three or more unique values
            categorical_col = st.selectbox("Select Categorical Column", df.select_dtypes(include=['object', 'category']).columns)
            
            # Check if the categorical column has at least three unique values
            unique_values = df[categorical_col].unique()
            if len(unique_values) < 3:
                st.warning("The selected categorical column must have at least three unique values for ANOVA.")
            else:
                # Split the numeric column into groups
                groups = [df[df[categorical_col] == value][numeric_col] for value in unique_values]
                
                # Perform ANOVA
                f_stat, p_val = f_oneway(*groups)
                st.write(f"F-statistic: {f_stat}, P-value: {p_val}")
                
                # Interpret results
                if p_val < 0.05:
                    st.success("There is a significant difference between the groups (p < 0.05).")
                else:
                    st.warning("There is no significant difference between the groups (p >= 0.05).")

        elif test_type == "Chi-Square":
            # Select two categorical columns
            cols = st.multiselect("Select Two Categorical Columns", df.columns, key="chi")
            
            if len(cols) == 2:
                # Create contingency table
                contingency_table = pd.crosstab(df[cols[0]], df[cols[1]])
                
                # Perform Chi-Square Test
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                
                # Display results
                st.write(f"Chi-Square: {chi2}, P-value: {p}, Degrees of Freedom: {dof}")
                
                # Interpret results
                if p < 0.05:
                    st.success(f"There is a significant association between {cols[0]} and {cols[1]} (p < 0.05).")
                else:
                    st.warning(f"There is no significant association between {cols[0]} and {cols[1]} (p >= 0.05).")
            else:
                st.warning("Please select exactly two categorical columns for the Chi-Square test.")
    
    
    if st.checkbox("Detect Outliers ⚠️"):
        method = st.radio("Select Outlier Detection Method", ["Z-Score", "IQR", "Isolation Forest"])
    
        if method == "Z-Score":
           from scipy.stats import zscore
           z_scores = zscore(df.select_dtypes(include="number"))
           outliers = (z_scores > 3).any(axis=1)
           st.write(f"Number of Outliers: {outliers.sum()}")
        elif method == "IQR":
           Q1 = df.quantile(0.25)
           Q3 = df.quantile(0.75)
           IQR = Q3 - Q1
           outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)
           st.write(f"Number of Outliers: {outliers.sum()}")
        elif method == "Isolation Forest":
           from sklearn.ensemble import IsolationForest
           model = IsolationForest(contamination=0.1)
           preds = model.fit_predict(df.select_dtypes(include="number"))
           outliers = preds == -1
           st.write(f"Number of Outliers: {outliers.sum()}")






    # Export Processed Data Section
    st.subheader("Export Processed Data ⤴️")
    if 'df' in locals() and not df.empty:
        # CSV Export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download Processed Data as CSV📑",
            data=csv_data,
            file_name="processed_data.csv",
            mime="text/csv"
        )

        # Excel Export
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Processed_Data')
        output.seek(0)

        st.download_button(
            label="Download Processed Data as Excel📑",
            data=output,
            file_name="processed_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.write("No data available to export. Please upload and process a dataset first.")

    # Generate Full PDF Report
    # Generate Full PDF Report
    st.subheader("Generate CSV Report 🗂️")

    def generate_csv_report(df):
       # Creating a summary of the dataset
       summary = df.describe()

       # Save the summary to a CSV
       csv_buffer = io.StringIO()
       summary.to_csv(csv_buffer)
       csv_buffer.seek(0)
    
       return csv_buffer.getvalue()

    if st.button("Generate and Download CSV Report 🗂️"):
        csv_report = generate_csv_report(df)
        st.download_button(
        label="Download Dataset Summary as CSV 📑",
        data=csv_report,
        file_name="data_analysis_summary.csv 📑",
        mime="text/csv"
    )
    
