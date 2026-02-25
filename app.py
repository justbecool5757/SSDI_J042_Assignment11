import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
st.title("ML App: Classification & Regression with Column Drop")
st.write("""
Upload your dataset, select problem type, choose target and model.
You can also drop columns you don't want to include in the model.
""")
 

problem_type = st.selectbox("Select Problem Type", ["Classification", "Regression"])
 

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
 
if uploaded_file is not None:
 
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
 
    drop_columns = st.multiselect(
        "Columns to Drop (optional)",
        options=df.columns.tolist()
    )
    df_model = df.drop(columns=drop_columns) if drop_columns else df.copy()
 

    if problem_type == "Classification":
     
        target_options = [col for col in df_model.columns if df_model[col].dtype == 'object' or df_model[col].nunique() <= 10]
        st.info("Columns suitable for Classification: categorical or with few unique values")
        model_options = ["GaussianNB","Error"]
    else:
       
        target_options = [col for col in df_model.columns if np.issubdtype(df_model[col].dtype, np.number)]
        st.info("Columns suitable for Regression: numeric/continuous")
        model_options = ["LinearRegression"]
 
    if len(target_options) == 0:
        st.error("No suitable target columns found for the chosen problem type.")
    else:
        target_column = st.selectbox("Select Target Column", target_options)
        test_size = st.slider("Test Size (Train/Test Split)", 0.1, 0.5, 0.2, step=0.05)
        model_choice = st.selectbox("Select Model", model_options)
 
        if st.button("Train Model"):
 
            X = df_model.drop(columns=[target_column])
            y = df_model[target_column]
 
            # Encode categorical features
            X = pd.get_dummies(X, drop_first=True)
 
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
 
          
            if problem_type == "Classification":
 
                if model_choice == "GaussianNB":
                    model = GaussianNB()
                elif model_choice == "Error":
                    print("This is an error model choice for demonstration purposes.")
                else:
                    st.error("Invalid model choice")
                    st.stop()
 
                model.fit(X_train, y_train)
 
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
 
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
 
                st.subheader("Classification Metrics")
                st.write(f"Training Accuracy: {train_acc:.4f}")
                st.write(f"Testing Accuracy: {test_acc:.4f}")
 
                st.subheader("Confusion Matrix (Test Data)")
                cm = confusion_matrix(y_test, y_test_pred)
                fig, ax = plt.subplots()
                cax = ax.matshow(cm)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.colorbar(cax)
 
                for (i, j), val in np.ndenumerate(cm):
                    ax.text(j, i, f"{val}", ha='center', va='center')
 
                st.pyplot(fig)
 
            # --------------------
            # REGRESSION
            # --------------------
            else:
 
                model = LinearRegression()
                model.fit(X_train, y_train)
 
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
 
                mae = mean_absolute_error(y_test, y_test_pred)
                mse = mean_squared_error(y_test, y_test_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_test_pred)
 
                st.subheader("Regression Metrics")
                st.write(f"MAE: {mae:.4f}")
                st.write(f"MSE: {mse:.4f}")
                st.write(f"RMSE: {rmse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
 
                st.subheader("Actual vs Predicted")
                fig, ax = plt.subplots()
                ax.scatter(y_test, y_test_pred)
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                st.pyplot(fig)
 
else:
    st.info("Please upload a CSV file to start.")