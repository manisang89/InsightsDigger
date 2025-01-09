# InsightsDigger 🔍

**InsightsDigger** is an advanced data analysis and visualization tool built with Streamlit. It allows users to upload CSV files, perform data preprocessing, generate visualizations, and conduct statistical analyses. The app also provides the ability to download processed data and generate comprehensive PDF reports.

---

## Features 🚀

### 1. **Data Preprocessing**
   - Handle missing values:
     - Drop rows with missing values.
     - Fill missing values with mean, median, or mode.
   - Apply feature scaling:
     - Standard Scaling.
     - Min-Max Scaling.
   - Perform Principal Component Analysis (PCA) for dimensionality reduction.

### 2. **Data Visualization**
   - Create interactive visualizations:
     - Bar plots.
     - Histograms.
     - Box plots.
     - Scatter plots.
     - Line plots.
   - Generate correlation heatmaps.

### 3. **Statistical Analysis**
   - Perform hypothesis testing:
     - t-test.
     - ANOVA.
     - Chi-Square test.
   - Detect outliers using:
     - Z-Score.
     - IQR (Interquartile Range).
     - Isolation Forest.

### 4. **Machine Learning Insights**
   - Calculate feature importance using RandomForestClassifier.
   - Suggest columns to drop based on importance thresholds.

### 5. **Export and Reporting**
   - Download processed data as CSV or Excel.
   - Generate and download a comprehensive PDF report with:
     - Dataset preview.
     - Summary statistics.
     - Visualizations.
     - Feature importance.

---

## How to Use 🛠️

### 1. **Upload a CSV File**
   - Click the "Choose a CSV file" button to upload your dataset.

### 2. **Data Preprocessing**
   - Handle missing values and apply feature scaling as needed.
   - Use PCA for dimensionality reduction.

### 3. **Visualize Data**
   - Select the type of visualization (e.g., bar plot, histogram, scatter plot).
   - Choose the columns to visualize.

### 4. **Perform Statistical Analysis**
   - Conduct hypothesis tests (t-test, ANOVA, Chi-Square).
   - Detect outliers using Z-Score, IQR, or Isolation Forest.

### 5. **Generate Insights**
   - View feature importance and get suggestions for dropping columns.

### 6. **Export Data and Reports**
   - Download processed data as CSV or Excel.
   - Generate and download a PDF report summarizing your analysis.

---

## Libraries Used 📚

- **Streamlit**: For building the web app.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For static visualizations.
- **Plotly**: For interactive visualizations.
- **Scikit-learn**: For machine learning and statistical analysis.
- **FPDF**: For generating PDF reports.

---

