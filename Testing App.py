import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

st.title("House Price Prediction App")

# Load the trained model
model = joblib.load('linear_regression_model.joblib')

# Load the test data directly
file_path = "test.csv"  # Provide the correct file path
test_data = pd.read_csv(file_path)

# Display the test data
st.subheader("Test Data")
st.write(test_data)

def preprocess_data(test_data):
    # Assume 'numeric_features' is a list of numeric columns in your training data
    numeric_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
    
    # Assuming you didn't save the numeric imputer and scaler separately
    # You need to perform the same preprocessing steps as in your training script
    # If you used specific values for imputation during training, use them here as well
    test_data[numeric_features] = test_data[numeric_features].fillna(0)  # Replace with your imputation strategy
    
    test_data['TotalBathrooms'] = test_data['FullBath'] + 0.5 * test_data['HalfBath']
    selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotalBathrooms']
    test_features = test_data[selected_features]
    
    # Assuming you didn't save the scaler separately
    # Fit and transform the test features using the same scaler used during training
    scaler = StandardScaler()  # Use the same scaler used during training
    test_features = scaler.fit_transform(test_features)
    
    return test_features

def make_predictions(model, test_features):
    return model.predict(test_features)

def main():
    st.subheader("Predictions")
    test_features = preprocess_data(test_data)
    predictions = make_predictions(model, test_features)
    test_data['PredictedSalePrice'] = predictions
    predictions_table = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotalBathrooms', 'PredictedSalePrice']]
    st.write(predictions_table)

    st.subheader("Model Performance Metrics")
    mse = mean_squared_error(test_data['PredictedSalePrice'], test_data['PredictedSalePrice'])
    r2 = r2_score(test_data['PredictedSalePrice'], test_data['PredictedSalePrice'])
    
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"R-squared (R2): {r2}")

    st.subheader("Visualizations")
    
    # Scatter plot for GrLivArea vs. Predicted SalePrice
    st.title("Scatter plot for GrLivArea vs. Predicted SalePrice")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.scatter(test_data['GrLivArea'], test_data['PredictedSalePrice'], color='red', label='Predicted SalePrice')
    ax1.scatter(test_data['GrLivArea'], test_data['PredictedSalePrice'], color='blue', label='Actual SalePrice')
    ax1.set_title('Test Dataset Visualization - GrLivArea vs. SalePrice')
    ax1.set_xlabel('GrLivArea')
    ax1.set_ylabel('SalePrice')
    ax1.legend()
    st.pyplot(fig1)

    # Scatter plot for BedroomAbvGr vs. Predicted SalePrice
    st.title("Scatter plot for BedroomAbvGr vs. Predicted SalePrice")
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.scatter(test_data['BedroomAbvGr'], test_data['PredictedSalePrice'], color='red', label='Predicted SalePrice')
    ax2.scatter(test_data['BedroomAbvGr'], test_data['PredictedSalePrice'], color='blue', label='Actual SalePrice')
    ax2.set_title('Test Dataset Visualization - BedroomAbvGr vs. SalePrice')
    ax2.set_xlabel('BedroomAbvGr')
    ax2.set_ylabel('SalePrice')
    ax2.legend()
    st.pyplot(fig2)

    # Histogram for GrLivArea distribution
    st.title("Histogram for GrLivArea distribution")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.hist(test_data['GrLivArea'], bins=30, color='green', alpha=0.7)
    ax3.set_title('Distribution of GrLivArea in Test Data')
    ax3.set_xlabel('GrLivArea')
    ax3.set_ylabel('Frequency')
    st.pyplot(fig3)

    # Histogram for BedroomAbvGr distribution
    st.title("Histogram for BedroomAbvGr distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    ax4.hist(test_data['BedroomAbvGr'], bins=15, color='purple', alpha=0.7)
    ax4.set_title('Distribution of BedroomAbvGr in Test Data')
    ax4.set_xlabel('BedroomAbvGr')
    ax4.set_ylabel('Frequency')
    st.pyplot(fig4)

    # Residual Analysis
    st.title("Residual Analysis")
    residuals = test_data['PredictedSalePrice'] - test_data['PredictedSalePrice']
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    ax5.scatter(test_data['PredictedSalePrice'], residuals, color='green')
    ax5.axhline(y=0, color='red', linestyle='--', label='Zero Residuals Line')
    ax5.set_title('Residual Analysis')
    ax5.set_xlabel('Actual SalePrice')
    ax5.set_ylabel('Residuals')
    ax5.legend()
    st.pyplot(fig5)

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
        features = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotalBathrooms']].columns
        fig6, ax6 = plt.subplots(figsize=(12, 6))
        ax6.bar(features, feature_importance, color='orange')
        ax6.set_title('Feature Importance')
        ax6.set_xlabel('Features')
        ax6.set_ylabel('Importance')
        st.pyplot(fig6)

    # Correlation Matrix
    st.title("Correlation Matrix")
    correlation_matrix = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath', 'TotalBathrooms', 'PredictedSalePrice']].corr()
    fig7, ax7 = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax7)
    ax7.set_title('Correlation Matrix')
    st.pyplot(fig7)

if __name__ == "__main__":
    main()
