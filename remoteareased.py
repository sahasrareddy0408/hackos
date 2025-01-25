import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Streamlit setup
st.set_page_config(page_title="Education Effectiveness Analysis", page_icon="📊", layout="wide")

# Title and introduction
st.title("📊 Education Effectiveness Analysis")
st.subheader("Predicting Improvement Percentage using Multiple Machine Learning Models")

st.write("""
    This tool helps analyze the effectiveness of educational strategies and predicts improvements in student performance.
    You can upload your own data, explore various models, and get recommendations for improving the effectiveness of learning strategies.
""")

# Sidebar for file upload
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

# Fallback file path for testing
fallback_path = "C:/Users/chava/Downloads/Enhanced_Education_Dataset (1).csv"

# File loading logic
if uploaded_file:
    # Use uploaded file
    file_to_load = uploaded_file
    st.info("Using the uploaded file.")
elif fallback_path:  # Use fallback path if no file is uploaded
    try:
        file_to_load = fallback_path
        st.info(f"Using the fallback file: `{fallback_path}`.")
    except FileNotFoundError:
        st.error("Fallback file not found. Please upload a file.")
        st.stop()
else:
    st.info("Please upload your CSV file to start the analysis.")
    st.stop()

# Load the dataset
try:
    df = pd.read_csv(file_to_load)
    st.write("### Data Overview")
    st.dataframe(df.head())  # Display the first few rows of the dataset

    # Data cleaning
    st.write("### Cleaning Data: Removing Non-Numeric Columns")
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    st.write(f"Removing non-numeric columns: {non_numeric_columns}.")
    df = df.drop(columns=non_numeric_columns)

    # Define features (X) and target variable (y)
    X = df.drop(columns=["Improvement_Percentage"])  # Features
    y = df["Improvement_Percentage"]  # Target variable

    # Handle NaN and infinite values
    st.write("### Handling Missing Data")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())  # Fill NaN with mean in features
    y = y.replace([np.inf, -np.inf], np.nan).fillna(y.mean())  # Fill NaN with mean in target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (important for many models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Initialization
    models = {
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Linear Regression": LinearRegression()
    }

    # Function to train and evaluate models
    def evaluate_model(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        return mse, r2, mae, rmse, y_pred

    # Show performance results and graphs
    selected_model = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model]

    if st.button(f"Evaluate {selected_model}"):
        try:
            st.write(f"### {selected_model} Model Performance")

            mse, r2, mae, rmse, y_pred = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
            st.write(f"- **MSE (Mean Squared Error)**: {mse:.2f}")
            st.write(f"- **R² (R-squared)**: {r2:.2f}")
            st.write(f"- **MAE (Mean Absolute Error)**: {mae:.2f}")
            st.write(f"- **RMSE (Root Mean Squared Error)**: {rmse:.2f}")

            # Show Visualizations
            st.write(f"### {selected_model} Model Results")

            # Actual vs Predicted Graph (Reduced Size)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(y_test, y_pred, alpha=0.6, color='green')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
            ax.set_xlabel("Actual Improvement Percentage")
            ax.set_ylabel("Predicted Improvement Percentage")
            ax.set_title(f"{selected_model} - Actual vs Predicted Improvement Percentage")
            st.pyplot(fig)

            # Residuals Plot (Reduced Size)
            residuals = y_test - y_pred
            st.write(f"### Residuals for {selected_model} Model")
            plt.figure(figsize=(8, 5))
            sns.scatterplot(x=y_pred, y=residuals, color='orange', alpha=0.6)
            plt.axhline(0, color='red', linestyle='--')
            plt.title(f"Residuals vs Predicted Values for {selected_model}")
            plt.xlabel("Predicted Improvement Percentage")
            plt.ylabel("Residuals")
            st.pyplot()

            # Feature Importance (only for models that support it)
            if selected_model == "Random Forest":
                st.write(f"### Feature Importance for {selected_model}")
                feature_importance = model.feature_importances_
                features = X.columns

                # Pie Chart for Feature Contribution
                st.write("### Feature Contribution to Improvement Percentage")
                feature_imp_df = pd.DataFrame({
                    "Feature": X.columns,
                    "Importance": feature_importance
                })
                feature_imp_df = feature_imp_df.sort_values(by="Importance", ascending=False)

                # Pie chart of feature contributions
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.pie(feature_imp_df["Importance"], labels=feature_imp_df["Feature"], autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set3", len(feature_imp_df)))
                ax.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
                st.pyplot(fig)

                # Provide Recommendations Based on Feature Importance
                st.write("### Recommendations:")
                for idx, row in feature_imp_df.iterrows():
                    if row["Importance"] > 0.1:
                        st.write(f"- **{row['Feature']}**: This field is significantly contributing to improvement. Maintain or enhance efforts in this area.")
                    else:
                        st.write(f"- **{row['Feature']}**: This field has a lower contribution. Consider improving this field for better results.")

                # Correlation Matrix for better understanding
                st.write("### Correlation Matrix")
                corr_matrix = df.corr()
                plt.figure(figsize=(10, 6))
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
                plt.title("Correlation Matrix of Features")
                st.pyplot()

        except Exception as e:
            st.error(f"An error occurred during model evaluation: {e}")
        finally:
            st.write("### Model Evaluation Completed.")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
