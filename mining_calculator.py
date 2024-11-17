import streamlit as st
import joblib
import numpy as np
import pandas as pd
import io

def load_models():
    loaded_model = joblib.load('xgboost_mining_model.joblib')
    loaded_scaler = joblib.load('scaler.joblib')
    loaded_le = joblib.load('label_encoder.joblib')
    return loaded_model, loaded_scaler, loaded_le

def clean_numeric(x):
    return float(str(x).strip().replace(',', '.'))

def predict_mining_class(feature1, feature2, model, scaler, le):
    features = np.array([[feature1, feature2]])
    features_scaled = scaler.transform(features)
    prediction_encoded = model.predict(features_scaled)
    prediction = le.inverse_transform(prediction_encoded)
    return prediction[0]

def process_file(input_file, model, scaler, le):
    # Read the uploaded file
    input_data = pd.read_csv(input_file)
    
    # Clean the numeric columns
    input_data.iloc[:, 0] = input_data.iloc[:, 0].apply(clean_numeric)
    input_data.iloc[:, 1] = input_data.iloc[:, 1].apply(clean_numeric)
    
    # Make predictions
    predictions = []
    for index, row in input_data.iterrows():
        result = predict_mining_class(row.iloc[0], row.iloc[1], model, scaler, le)
        predictions.append(result)
    
    # Add predictions column
    input_data['Predicted_Class'] = predictions
    
    return input_data

def to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)
    return output.read()

def main():
    st.title('Mining Class Predictor')
    
    # Load the models
    try:
        model, scaler, le = load_models()
        st.success("Models loaded successfully!")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File upload
    st.write("Please upload your CSV file with two columns (feature1, feature2)")
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])

    if uploaded_file is not None:
        try:
            # Process the file
            results = process_file(uploaded_file, model, scaler, le)
            
            # Show preview of results
            st.write("Preview of processed data:")
            st.dataframe(results.head())
            
            # Create download button
            excel_data = to_excel(results)
            st.download_button(
                label="Download Excel file",
                data=excel_data,
                file_name="predictions_output.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please make sure your CSV file has the correct format with two columns.")

if __name__ == "__main__":
    main()