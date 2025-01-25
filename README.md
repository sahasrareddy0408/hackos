# hackos
Education Effectiveness Analysis

This Streamlit application, Education Effectiveness Analysis, helps analyze the effectiveness of educational strategies and predicts improvements in student performance using machine learning models.

Features

Upload your dataset (CSV format) or use a fallback file.

Data cleaning and preprocessing:

Removal of non-numeric columns.

Handling missing and infinite values.

Multiple machine learning models to choose from:

Random Forest Regressor

Linear Regression

Performance metrics for each model:

Mean Squared Error (MSE)

R-squared (R²)

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

Visualizations:

Actual vs. Predicted values.

Residuals plot.

Feature importance (for Random Forest model).

Correlation matrix.

Recommendations based on feature importance.

Installation

Clone the repository:

git clone https://github.com/your-repo/education-effectiveness-analysis.git

Navigate to the project directory:

cd education-effectiveness-analysis

Install the required dependencies:

pip install -r requirements.txt

Running the Application

Run the Streamlit app:

streamlit run app.py

Open the provided link in your browser (usually http://localhost:8501).

Dataset Requirements

The dataset should be in CSV format.

Ensure the target column is named Improvement_Percentage.

Non-numeric columns will be automatically removed during preprocessing.

Usage

Upload your dataset in the sidebar or use the fallback file if available.

Select a machine learning model from the dropdown menu.

Click the Evaluate button to view performance metrics and visualizations.

Analyze the results and review recommendations to enhance educational strategies.

Visualizations

Actual vs. Predicted Plot: Shows how well the model's predictions align with actual values.

Residuals Plot: Helps identify patterns in prediction errors.

Feature Importance Pie Chart: Highlights the contributions of each feature to the target variable (for Random Forest only).

Correlation Matrix: Displays relationships between features.

Deployment

The application is deployed and accessible via the following link:
Education Effectiveness Analysis

Technologies Used

Python

Streamlit

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

License

This project is licensed under the MIT License.

Feedback

For questions or feedback, please create an issue in the repository or contact the developer.
